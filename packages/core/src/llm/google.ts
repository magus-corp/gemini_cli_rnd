/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { LlmClient } from './index.js';
import {
  EmbedContentParameters,
  GenerateContentConfig,
  Part,
  SchemaUnion,
  PartListUnion,
  Content,
  Tool,
  GenerateContentResponse,
  GenerateContentParameters,
} from '@google/genai';
import { getFolderStructure } from '../utils/getFolderStructure.js';
import { Config } from '../config/config.js';
import { ReadManyFilesTool } from '../tools/read-many-files.js';
import { getResponseText } from '../utils/generateContentResponseUtilities.js';
import { checkNextSpeaker } from '../utils/nextSpeakerChecker.js';
import { reportError } from '../utils/errorReporting.js';
import { retryWithBackoff } from '../utils/retry.js';
import { getErrorMessage } from '../utils/errors.js';
import { ProxyAgent, setGlobalDispatcher } from 'undici';
import { DEFAULT_GEMINI_FLASH_MODEL } from '../config/models.js';

import { LangfuseService } from './langfuse.js';

// Import from core
import {
  Turn,
  ServerGeminiStreamEvent,
  GeminiEventType,
  ChatCompressionInfo,
} from '../core/turn.js';
import { getCoreSystemPrompt } from '../core/prompts.js';
import { GeminiChat } from '../core/geminiChat.js';
import { tokenLimit } from '../core/tokenLimits.js';
import { AuthType } from '../core/contentGenerator.js';

function isThinkingSupported(model: string) {
  if (model.startsWith('gemini-2.5')) return true;
  return false;
}

export class GoogleClient implements LlmClient {
  private chat?: GeminiChat;
  private contentGenerator?: ContentGenerator;
  private model: string;
  private embeddingModel: string;
  private generateContentConfig: GenerateContentConfig = {
    temperature: 0,
    topP: 1,
  };
  private readonly MAX_TURNS = 100;
  private langfuseService: LangfuseService;

  constructor(private config: Config) {
    if (config.getProxy()) {
      setGlobalDispatcher(new ProxyAgent(config.getProxy() as string));
    }

    this.model = config.getModel();
    this.embeddingModel = config.getEmbeddingModel();
    this.langfuseService = new LangfuseService(config);
  }

  async initialize() {
    this.contentGenerator = await createContentGenerator({
      model: this.model,
      authType: AuthType.USE_GEMINI, // Assuming GoogleClient always uses GEMINI auth
    });
    await this.startChat();
  }
  private getContentGenerator(): ContentGenerator {
    if (!this.contentGenerator) {
      throw new Error('Content generator not initialized');
    }
    return this.contentGenerator;
  }

  async addHistory(content: Content) {
    this.getChat().addHistory(content);
  }

  getChat(): GeminiChat {
    if (!this.chat) {
      throw new Error('Chat not initialized');
    }
    return this.chat;
  }

  async getHistory(): Promise<Content[]> {
    return this.getChat().getHistory();
  }

  async setHistory(history: Content[]): Promise<void> {
    this.getChat().setHistory(history);
  }

  async resetChat(): Promise<void> {
    await this.startChat();
  }

  private async getEnvironment(): Promise<Part[]> {
    const cwd = this.config.getWorkingDir();
    const today = new Date().toLocaleDateString(undefined, {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
    const platform = process.platform;
    const folderStructure = await getFolderStructure(cwd, {
      fileService: this.config.getFileService(),
    });
    const context = `
  Okay, just setting up the context for our chat.
  Today is ${today}.
  My operating system is: ${platform}
  I'm currently working in the directory: ${cwd}
  ${folderStructure}
          `.trim();

    const initialParts: Part[] = [{ text: context }];
    const toolRegistry = await this.config.getToolRegistry();

    // Add full file context if the flag is set
    if (this.config.getFullContext()) {
      try {
        const readManyFilesTool = toolRegistry.getTool(
          'read_many_files',
        ) as ReadManyFilesTool;
        if (readManyFilesTool) {
          // Read all files in the target directory
          const result = await readManyFilesTool.execute(
            {
              paths: ['**/*'], // Read everything recursively
              useDefaultExcludes: true, // Use default excludes
            },
            AbortSignal.timeout(30000),
          );
          if (result.llmContent) {
            initialParts.push({
              text: `\n--- Full File Context ---\n${result.llmContent}`,
            });
          } else {
            console.warn(
              'Full context requested, but read_many_files returned no content.',
            );
          }
        } else {
          console.warn(
            'Full context requested, but read_many_files tool not found.',
          );
        }
      } catch (error) {
        // Not using reportError here as it's a startup/config phase, not a chat/generation phase error.
        console.error('Error reading full file context:', error);
        initialParts.push({
          text: '\n--- Error reading full file context ---',
        });
      }
    }

    return initialParts;
  }

  async startChat(extraHistory?: Content[]): Promise<LlmClient> {
    return this.langfuseService.trace('startChat', async () => {
      const envParts = await this.getEnvironment();
      const toolRegistry = await this.config.getToolRegistry();
      const toolDeclarations = toolRegistry.getFunctionDeclarations();
      const tools: Tool[] = [{ functionDeclarations: toolDeclarations }];
      const initialHistory: Content[] = [
        {
          role: 'user',
          parts: envParts,
        },
        {
          role: 'model',
          parts: [{ text: 'Got it. Thanks for the context!' }],
        },
      ];
      const history = initialHistory.concat(extraHistory ?? []);
      try {
        const userMemory = this.config.getUserMemory();
        const systemInstruction = getCoreSystemPrompt(userMemory);
        const generateContentConfigWithThinking = isThinkingSupported(
          this.model,
        )
          ? {
              ...this.generateContentConfig,
              thinkingConfig: {
                includeThoughts: true,
              },
            }
          : this.generateContentConfig;
        this.chat = new GeminiChat(
          this.config,
          this.getContentGenerator(),
          {
            systemInstruction,
            ...generateContentConfigWithThinking,
            tools,
          },
          history,
        );
        return this;
      } catch (error) {
        await reportError(
          error,
          'Error initializing Gemini chat session.',
          history,
          'startChat',
        );
        throw new Error(`Failed to initialize chat: ${getErrorMessage(error)}`);
      }
    });
  }

  async *sendMessageStream(
    request: PartListUnion,
    signal: AbortSignal,
    turns: number = this.MAX_TURNS,
  ): AsyncGenerator<ServerGeminiStreamEvent, Turn> {
    if (!turns) {
      return new Turn(this.getChat());
    }

    const compressed = await this.tryCompressChat();
    if (compressed) {
      yield { type: GeminiEventType.ChatCompressed, value: compressed };
    }
    const turn = new Turn(this.getChat());
    const resultStream = turn.run(request, signal);
    for await (const event of resultStream) {
      yield event;
    }
    if (!turn.pendingToolCalls.length && signal && !signal.aborted) {
      const nextSpeakerCheck = await checkNextSpeaker(
        this.getChat(),
        this,
        signal,
      );
      if (nextSpeakerCheck?.next_speaker === 'model') {
        const nextRequest = [{ text: 'Please continue.' }];
        // This recursive call's events will be yielded out, but the final
        // turn object will be from the top-level call.
        yield* this.sendMessageStream(nextRequest, signal, turns - 1);
      }
    }
    return turn;
  }

  async generateJson(
    contents: Content[],
    schema: SchemaUnion,
    abortSignal: AbortSignal,
    model: string = DEFAULT_GEMINI_FLASH_MODEL,
    config: GenerateContentConfig = {},
  ): Promise<Record<string, unknown>> {
    return this.langfuseService.trace('generateJson', async () => {
      try {
        const userMemory = this.config.getUserMemory();
        const systemInstruction = getCoreSystemPrompt(userMemory);
        const requestConfig = {
          abortSignal,
          ...this.generateContentConfig,
          ...config,
        };

        const apiCall = () =>
          this.getContentGenerator().generateContent({
            model,
            config: {
              ...requestConfig,
              systemInstruction,
              responseSchema: schema,
              responseMimeType: 'application/json',
            },
            contents,
          });

        const result = await retryWithBackoff(apiCall, {
          onPersistent429: async (authType?: string) =>
            await this.handleFlashFallback(authType),
          authType: this.config.getContentGeneratorConfig()?.authType,
        });

        const text = getResponseText(result);
        if (!text) {
          const error = new Error(
            'API returned an empty response for generateJson.',
          );
          await reportError(
            error,
            'Error in generateJson: API returned an empty response.',
            contents,
            'generateJson-empty-response',
          );
          throw error;
        }
        try {
          return JSON.parse(text);
        } catch (parseError) {
          await reportError(
            parseError,
            'Failed to parse JSON response from generateJson.',
            {
              responseTextFailedToParse: text,
              originalRequestContents: contents,
            },
            'generateJson-parse',
          );
          throw new Error(
            `Failed to parse API response as JSON: ${getErrorMessage(
              parseError,
            )}`,
          );
        }
      } catch (error) {
        if (abortSignal.aborted) {
          throw error;
        }

        // Avoid double reporting for the empty response case handled above
        if (
          error instanceof Error &&
          error.message === 'API returned an empty response for generateJson.'
        ) {
          throw error;
        }

        await reportError(
          error,
          'Error generating JSON content via API.',
          contents,
          'generateJson-api',
        );
        throw new Error(
          `Failed to generate JSON content: ${getErrorMessage(error)}`,
        );
      }
    });
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    return this.langfuseService.trace('generateContent', async () => {
      const modelToUse = this.model;
      const configToUse: GenerateContentConfig = {
        ...this.generateContentConfig,
        ...request.generationConfig,
      };

      try {
        const userMemory = this.config.getUserMemory();
        const systemInstruction =
          request.systemInstruction ?? getCoreSystemPrompt(userMemory);

        const requestConfig = {
          ...configToUse,
          systemInstruction,
        };

        const apiCall = () =>
          this.getContentGenerator().generateContent({
            model: modelToUse,
            config: requestConfig,
            contents: request.contents,
          });

        const result = await retryWithBackoff(apiCall, {
          onPersistent429: async (authType?: string) =>
            await this.handleFlashFallback(authType),
          authType: this.config.getContentGeneratorConfig()?.authType,
        });
        return result;
      } catch (error: unknown) {
        await reportError(
          error,
          `Error generating content via API with model ${modelToUse}.`,
          {
            requestContents: request.contents,
            requestConfig: configToUse,
          },
          'generateContent-api',
        );
        throw new Error(
          `Failed to generate content with model ${modelToUse}: ${getErrorMessage(
            error,
          )}`,
        );
      }
    });
  }

  async generateEmbedding(texts: string[]): Promise<number[][]> {
    return this.langfuseService.trace('generateEmbedding', async () => {
      if (!texts || texts.length === 0) {
        return [];
      }
      const embedModelParams: EmbedContentParameters = {
        model: this.embeddingModel,
        contents: texts,
      };

      const embedContentResponse =
        await this.getContentGenerator().embedContent(embedModelParams);
      if (
        !embedContentResponse.embeddings ||
        embedContentResponse.embeddings.length === 0
      ) {
        throw new Error('No embeddings found in API response.');
      }

      if (embedContentResponse.embeddings.length !== texts.length) {
        throw new Error(
          `API returned a mismatched number of embeddings. Expected ${texts.length}, got ${embedContentResponse.embeddings.length}.`,
        );
      }

      return embedContentResponse.embeddings.map((embedding, index) => {
        const values = embedding.values;
        if (!values || values.length === 0) {
          throw new Error(
            `API returned an empty embedding for input text at index ${index}: "${texts[index]}"`,
          );
        }
        return values;
      });
    });
  }

  async tryCompressChat(
    force: boolean = false,
  ): Promise<ChatCompressionInfo | null> {
    return this.langfuseService.trace('tryCompressChat', async () => {
      const history = this.getChat().getHistory(true); // Get curated history

      // Regardless of `force`, don't do anything if the history is empty.
      if (history.length === 0) {
        return null;
      }

      const { totalTokens: originalTokenCount } =
        await this.getContentGenerator().countTokens({
          model: this.model,
          contents: history,
        });

      // If not forced, check if we should compress based on context size.
      if (!force) {
        if (originalTokenCount === undefined) {
          // If token count is undefined, we can't determine if we need to compress.
          console.warn(
            `Could not determine token count for model ${this.model}. Skipping compression check.`,
          );
          return null;
        }
        const tokenCount = originalTokenCount; // Now guaranteed to be a number

        const limit = tokenLimit(this.model);
        if (!limit) {
          // If no limit is defined for the model, we can't compress.
          console.warn(
            `No token limit defined for model ${this.model}. Skipping compression check.`,
          );
          return null;
        }

        if (tokenCount < 0.95 * limit) {
          return null;
        }
      }

      const summarizationRequestMessage = {
        text: 'Summarize our conversation up to this point. The summary should be a concise yet comprehensive overview of all key topics, questions, answers, and important details discussed. This summary will replace the current chat history to conserve tokens, so it must capture everything essential to understand the context and continue our conversation effectively as if no information was lost.',
      };
      const response = await this.getChat().sendMessage({
        message: summarizationRequestMessage,
      });
      const newHistory = [
        {
          role: 'user',
          parts: [summarizationRequestMessage],
        },
        {
          role: 'model',
          parts: [{ text: response.text }],
        },
      ];
      this.chat = (await this.startChat(newHistory)) as GeminiChat;
      const newTokenCount = (
        await this.getContentGenerator().countTokens({
          model: this.model,
          contents: newHistory,
        })
      ).totalTokens;

      return originalTokenCount && newTokenCount
        ? {
            originalTokenCount,
            newTokenCount,
          }
        : null;
    });
  }

  /**
   * Handles fallback to Flash model when persistent 429 errors occur for OAuth users.
   * Uses a fallback handler if provided by the config, otherwise returns null.
   */
  private async handleFlashFallback(authType?: string): Promise<string | null> {
    // Only handle fallback for OAuth users
    if (authType !== AuthType.LOGIN_WITH_GOOGLE_PERSONAL) {
      return null;
    }

    const currentModel = this.model;
    const fallbackModel = DEFAULT_GEMINI_FLASH_MODEL;

    // Don't fallback if already using Flash model
    if (currentModel === fallbackModel) {
      return null;
    }

    // Check if config has a fallback handler (set by CLI package)
    const fallbackHandler = this.config.flashFallbackHandler;
    if (typeof fallbackHandler === 'function') {
      try {
        const accepted = await fallbackHandler(currentModel, fallbackModel);
        if (accepted) {
          this.config.setModel(fallbackModel);
          this.model = fallbackModel;
          return fallbackModel;
        }
      } catch (error) {
        console.warn('Flash fallback handler failed:', error);
      }
    }

    return null;
  }
}
