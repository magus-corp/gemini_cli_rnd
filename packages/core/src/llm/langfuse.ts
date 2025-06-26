/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Langfuse } from 'langfuse';
import { Config } from '../config/config.js';

export class LangfuseService {
  private langfuse: Langfuse;

  constructor(config: Config) {
    this.langfuse = new Langfuse({
      publicKey: config.getLangfusePublicKey(),
      secretKey: config.getLangfuseSecretKey(),
      baseUrl: config.getLangfuseHost(),
    });
  }

  async trace<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const trace = this.langfuse.trace({ name });
    try {
      const result = await fn();
      trace.update({ output: result });
      return result;
    } catch (error) {
      trace.update({ level: 'ERROR', statusMessage: (error as Error).message });
      throw error;
    } finally {
      await this.langfuse.shutdownAsync();
    }
  }
}
