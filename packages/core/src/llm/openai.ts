/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { LlmClient } from './index.js';
import { Config } from '../config/config.js';
import { LangfuseService } from './langfuse.js';
import {
  Content,
  GenerateContentResponse,
  Part,
  GenerateContentParameters,
  SchemaUnion,
  GenerateContentConfig,
} from '@google/genai';
import OpenAI from 'openai';
import { ChatCompletionMessageParam } from 'openai/resources/chat/completions';
import { ServerGeminiStreamEvent, GeminiEventType } from '../core/turn.js';
import { Turn } from '../core/turn.js';

export class OpenAIClient implements LlmClient {
  private openai: OpenAI;
  private langfuseService: LangfuseService;
  private history: Content[] = [];

  constructor(private config: Config) {
    this.openai = new OpenAI({
      apiKey: config.getOpenaiApiKey(),
      baseURL: config.getOpenaiBaseUrl(),
    });
    this.langfuseService = new LangfuseService(config);
  }

  async initialize(): Promise<void> {
    // No initialization required for OpenAIClient
  }

  addHistory(content: Content): void {
    this.history.push(content);
  }

  async getHistory(): Promise<Content[]> {
    return this.history;
  }

  async setHistory(history: Content[]): Promise<void> {
    this.history = history;
  }

  async resetChat(): Promise<void> {
    this.history = [];
  }

  async *sendMessageStream(
    request: Part[],
    signal: AbortSignal,
  ): AsyncGenerator<ServerGeminiStreamEvent, Turn> {
    const messages: ChatCompletionMessageParam[] = this.history.flatMap(
      (content) =>
        content.parts.map((part) => ({
          role: content.role === 'user' ? 'user' : 'assistant',
          content: part.text!,
        })),
    );

    messages.push({
      role: 'user',
      content: request[0].text!,
    });

    const stream = await this.openai.chat.completions.create({
      model: this.config.getModel(),
      messages,
      stream: true,
    });

    let fullText = '';
    for await (const chunk of stream) {
      if (signal.aborted) {
        break;
      }
      const text = chunk.choices[0]?.delta?.content || '';
      fullText += text;
      yield { type: GeminiEventType.Chunk, value: text };
    }

    const turn = new Turn(this);
    turn.addResponseChunk({ text: fullText });
    return turn;
  }

  async generateJson(
    contents: Content[],
    schema: SchemaUnion,
    abortSignal: AbortSignal,
    model?: string,
    _config?: GenerateContentConfig,
  ): Promise<Record<string, unknown>> {
    return this.langfuseService.trace('generateJson', async () => {
      const messages: ChatCompletionMessageParam[] = contents.flatMap(
        (content) =>
          content.parts.map((part) => ({
            role: content.role === 'user' ? 'user' : 'assistant',
            content: part.text!,
          })),
      );

      const response = await this.openai.chat.completions.create({
        model: model || this.config.getModel(),
        messages,
        response_format: { type: 'json_object' },
        // TODO: Map GenerateContentConfig to OpenAI parameters if needed
      });

      const text = response.choices[0].message.content;
      if (!text) {
        throw new Error(
          'OpenAI API returned an empty response for generateJson.',
        );
      }

      try {
        return JSON.parse(text);
      } catch (parseError) {
        throw new Error(
          `Failed to parse JSON response from OpenAI: ${parseError}`,
        );
      }
    });
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    return this.langfuseService.trace('generateContent', async () => {
      const response = await this.openai.chat.completions.create({
        model: this.config.getModel(),
        messages: request.contents.flatMap((content) =>
          content.parts.map((part) => ({
            role: 'user', // Assuming all parts are from the user for now
            content: part.text!,
          })),
        ),
      });

      // This is a hack to make the response compatible with the GenerateContentResponse type.
      // You will need to implement a proper mapping between the OpenAI response and the GenerateContentResponse type.
      return {
        response: {
          candidates: [
            {
              content: {
                parts: [
                  {
                    text: response.choices[0].message.content!,
                  },
                ],
                role: 'model',
              },
              index: 0,
              finishReason: 'SUCCESS',
              citationMetadata: {
                citationSources: [],
              },
              safetyRatings: [],
            },
          ],
          usageMetadata: {
            promptTokenCount: 0,
            candidatesTokenCount: 0,
            totalTokenCount: 0,
          },
        },
      };
    });
  }

  async generateEmbedding(texts: string[]): Promise<number[][]> {
    return this.langfuseService.trace('generateEmbedding', async () => {
      if (!texts || texts.length === 0) {
        return [];
      }
      const response = await this.openai.embeddings.create({
        model: 'text-embedding-ada-002', // OpenAI's embedding model
        input: texts,
      });

      if (!response.data || response.data.length === 0) {
        throw new Error('No embeddings found in OpenAI API response.');
      }

      return response.data.map((embedding, index) => {
        const values = embedding.embedding;
        if (!values || values.length === 0) {
          throw new Error(
            `OpenAI API returned an empty embedding for input text at index ${index}: "${texts[index]}"`,
          );
        }
        return values;
      });
    });
  }

  tryCompressChat(_force?: boolean): Promise<unknown | null> {
    // OpenAI does not have a direct equivalent for chat compression.
    // This method is primarily for Gemini's token limits.
    return Promise.resolve(null);
  }

  async startChat(history: Content[]): Promise<LlmClient> {
    this.history = history;
    return this;
  }
}
