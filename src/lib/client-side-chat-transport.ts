import { type BuiltInAIUIMessage, builtInAI } from "@built-in-ai/core";
import {
  type ChatRequestOptions,
  type ChatTransport,
  convertToModelMessages,
  createUIMessageStream,
  streamObject,
  type UIMessageChunk,
} from "ai";
import { z } from "zod";
import { useSuggestionsStore } from "~/stores/suggestions-store";

const SYSTEM_PROMPT = "You are a helpful AI assistant.";

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
 * @implements {ChatTransport<BuiltInAIUIMessage>}
 */
export class ClientSideChatTransport
  implements ChatTransport<BuiltInAIUIMessage>
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
    let textId: string | undefined;
    let previousText = "";
    let aborted = false;

    // Set up abort listener to force end the stream
    const abortHandler = () => {
      aborted = true;
      // End the text stream immediately
      if (textId) {
        writer.write({
          type: "text-end",
          id: textId,
        });
      }
    };

    abortSignal?.addEventListener("abort", abortHandler);

    const result = streamObject({
      model,
      system: SYSTEM_PROMPT,
      messages: prompt,
      schema: responseSchema,
      abortSignal,
    });

    try {
      for await (const partialObject of result.partialObjectStream) {
        // Check if aborted
        if (aborted || abortSignal?.aborted) {
          break;
        }

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

      // End the text stream (only if not already aborted)
      if (textId && !aborted) {
        writer.write({
          type: "text-end",
          id: textId,
        });
      }

      // Send suggestions after the response is complete (skip if aborted)
      if (!aborted) {
        const finalObject = await result.object;
        if (finalObject.suggestions && finalObject.suggestions.length > 0) {
          // Update Zustand store with suggestions
          useSuggestionsStore
            .getState()
            .setSuggestions(finalObject.suggestions);
        }
      }
    } catch (error) {
      // If aborted, end the text stream gracefully (only if not already ended)
      if (textId && !aborted) {
        writer.write({
          type: "text-end",
          id: textId,
        });
      }
      // Re-throw if it's not an abort-related error and not already aborted
      if (!aborted) {
        const isAbortError =
          error instanceof Error &&
          (error.name === "AbortError" ||
            error.message.includes("aborted") ||
            error.message.includes("abort"));

        if (!isAbortError) {
          throw error;
        }
      }
      // Abort errors are silently handled (stream already ended above)
    } finally {
      // Clean up abort listener
      if (abortSignal) {
        abortSignal.removeEventListener("abort", abortHandler);
      }
    }
  }

  async sendMessages(
    options: {
      chatId: string;
      messages: BuiltInAIUIMessage[];
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
      return createUIMessageStream<BuiltInAIUIMessage>({
        execute: async ({ writer }) => {
          await this.streamResponse({ model, prompt, writer, abortSignal });
        },
      });
    }

    // Best practice: Handle model download with progress tracking
    return createUIMessageStream<BuiltInAIUIMessage>({
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
