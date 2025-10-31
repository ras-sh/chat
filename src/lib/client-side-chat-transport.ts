import { builtInAI } from "@built-in-ai/core";
import {
  type ChatRequestOptions,
  type ChatTransport,
  convertToModelMessages,
  createUIMessageStream,
  streamObject,
  type UIMessageChunk,
} from "ai";
import { z } from "zod";
import type { ExtendedBuiltInAIUIMessage } from "~/types/ui-message";

const SYSTEM_PROMPT =
  "You are a helpful AI assistant. Be concise and brief in your responses. Keep answers short and to the point.";

const responseSchema = z.object({
  response: z.string().describe("The text response to the user's message"),
  suggestions: z
    .array(z.string())
    .optional()
    .describe(
      "3-4 relevant follow-up questions the USER could ask next (from the user's perspective, not the assistant's)"
    ),
});

/**
 * Client-side chat transport implementation that handles AI model communication
 * using Chrome's built-in Prompt API.
 *
 * Best practices implemented:
 * - Proper availability state management
 * - Progress tracking for model downloads
 *
 * @implements {ChatTransport<ExtendedBuiltInAIUIMessage>}
 */
export class ClientSideChatTransport
  implements ChatTransport<ExtendedBuiltInAIUIMessage>
{
  /**
   * Initialize and return the base model.
   */
  private createModel() {
    const model = builtInAI();
    return model;
  }

  /**
   * Write download progress update to the stream.
   */
  private writeDownloadProgress({
    writer,
    id,
    status,
    progress,
    message,
  }: {
    writer: any;
    id: string;
    status: "downloading" | "complete";
    progress: number;
    message: string;
  }): void {
    writer.write({
      type: "data-modelDownloadProgress",
      id,
      data: { status, progress, message },
      transient: true,
    });
  }

  /**
   * Stream response chunks from the model to the writer.
   * Handles text deltas from structured output and sends suggestions when complete.
   */
  private async streamResponse({
    model,
    prompt,
    writer,
    abortSignal,
  }: {
    model: ReturnType<typeof builtInAI>;
    prompt: ReturnType<typeof convertToModelMessages>;
    writer: any;
    abortSignal?: AbortSignal;
  }): Promise<void> {
    const result = streamObject({
      model,
      system: SYSTEM_PROMPT,
      messages: prompt,
      schema: responseSchema,
      abortSignal,
    });

    let textId: string | undefined;
    let previousText = "";

    for await (const partialObject of result.partialObjectStream) {
      if (partialObject.response && partialObject.response !== previousText) {
        if (!textId) {
          textId = `text-${Date.now()}`;
          writer.write({
            type: "text-start",
            id: textId,
          });
        }
        const delta = partialObject.response.slice(previousText.length);
        if (delta) {
          writer.write({
            type: "text-delta",
            id: textId,
            delta,
          });
        }
        previousText = partialObject.response;
      }
    }

    // End the text stream
    if (textId) {
      writer.write({
        type: "text-end",
        id: textId,
      });
    }

    // Send suggestions after the response is complete
    const finalObject = await result.object;
    if (finalObject.suggestions && finalObject.suggestions.length > 0) {
      writer.write({
        type: "data-suggestions",
        id: `suggestions-${Date.now()}`,
        data: finalObject.suggestions,
      });
    }
  }

  async sendMessages(
    options: {
      chatId: string;
      messages: ExtendedBuiltInAIUIMessage[];
      abortSignal: AbortSignal | undefined;
    } & {
      trigger: "submit-message" | "submit-tool-result" | "regenerate-message";
      messageId: string | undefined;
    } & ChatRequestOptions
  ): Promise<ReadableStream<UIMessageChunk>> {
    const { messages, abortSignal } = options;

    const prompt = convertToModelMessages(messages);

    // Use Chrome's built-in Prompt API for fast, efficient inference
    const model = this.createModel();

    // Best practice: Check availability state before operations
    // States: "unavailable" | "downloadable" | "available"
    const availability = await model.availability();
    if (availability === "available") {
      return createUIMessageStream<ExtendedBuiltInAIUIMessage>({
        execute: async ({ writer }) => {
          await this.streamResponse({ model, prompt, writer, abortSignal });
        },
      });
    }

    // Best practice: Handle model download with progress tracking
    return createUIMessageStream<ExtendedBuiltInAIUIMessage>({
      execute: async ({ writer }) => {
        try {
          let downloadProgressId: string | undefined;

          // Best practice: Use createSessionWithProgress for monitoring initialization
          // Critical for UX, especially with model downloads
          await model.createSessionWithProgress((progress: number) => {
            const percent = Math.round(progress * 100);

            if (progress >= 1) {
              // Download complete - ready for inference
              if (downloadProgressId) {
                this.writeDownloadProgress({
                  writer,
                  id: downloadProgressId,
                  status: "complete",
                  progress: 100,
                  message:
                    "Model finished downloading! Getting ready for inference...",
                });
              }
              return;
            }

            // First progress update - initialize tracking
            if (!downloadProgressId) {
              downloadProgressId = `download-${Date.now()}`;
              this.writeDownloadProgress({
                writer,
                id: downloadProgressId,
                status: "downloading",
                progress: percent,
                message: "Downloading browser AI model...",
              });
              return;
            }

            // Ongoing progress updates - track download state
            this.writeDownloadProgress({
              writer,
              id: downloadProgressId,
              status: "downloading",
              progress: percent,
              message: "Downloading browser AI model...",
            });
          });

          // Clear progress message before streaming response
          if (downloadProgressId) {
            this.writeDownloadProgress({
              writer,
              id: downloadProgressId,
              status: "complete",
              progress: 100,
              message: "",
            });
          }

          // Stream the actual text response after model is ready
          await this.streamResponse({ model, prompt, writer, abortSignal });
        } catch (error) {
          writer.write({
            type: "data-notification",
            data: {
              message: `Error: ${error instanceof Error ? error.message : "Unknown error"}`,
              level: "error",
            },
            transient: true,
          });
          throw error;
        }
      },
    });
  }

  async reconnectToStream(): Promise<ReadableStream<UIMessageChunk> | null> {
    // Client-side AI doesn't support stream reconnection
    return await Promise.resolve(null);
  }
}
