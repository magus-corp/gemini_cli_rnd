/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Config } from '../config/config.js';
import { GoogleClient } from './google.js';
import { OpenAIClient } from './openai.js';
import {
  Content,
  GenerateContentResponse,
  Part,
  GenerateContentParameters,
  SchemaUnion,
  GenerateContentConfig,
} from '@google/genai';
import { ServerGeminiStreamEvent } from './turn.js';
import { Turn } from './turn.js';

export interface LlmClient {
  addHistory(content: Content): void;
  getHistory(): Promise<Content[]>;
  setHistory(history: Content[]): Promise<void>;
  resetChat(): Promise<void>;
  sendMessageStream(
    request: Part[],
    signal: AbortSignal,
  ): AsyncGenerator<ServerGeminiStreamEvent, Turn>;
  generateJson(
    contents: Content[],
    schema: SchemaUnion,
    abortSignal: AbortSignal,
    model?: string,
    config?: GenerateContentConfig,
  ): Promise<Record<string, unknown>>;
  generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse>;
  generateEmbedding(texts: string[]): Promise<number[][]>;
  tryCompressChat(force?: boolean): Promise<unknown | null>;
  startChat(history: Content[]): Promise<LlmClient>;
  initialize(): Promise<void>;
}

export async function getLlmClient(config: Config): Promise<LlmClient> {
  const client =
    config.getOpenaiApiKey() && config.getOpenaiBaseUrl()
      ? new OpenAIClient(config)
      : new GoogleClient(config);
  await client.initialize();
  return client;
}
