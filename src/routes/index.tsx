"use client";

import { useChat } from "@ai-sdk/react";
import type { BuiltInAIUIMessage } from "@built-in-ai/core";
import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { toast } from "sonner";
import { ChatEmptyState } from "~/components/chat-empty-state";
import { ChatInput } from "~/components/chat-input";
import { ChatMessages } from "~/components/chat-messages";
import { Footer } from "~/components/footer";
import { Header } from "~/components/header";
import { ModelDownloadBanner } from "~/components/model-download-banner";
import { ClientSideChatTransport } from "~/lib/client-side-chat-transport";
import { cn } from "~/lib/utils";
import { useSuggestionsStore } from "~/stores/suggestions-store";

export const Route = createFileRoute("/")({
  component: Home,
});

export default function Home() {
  const [input, setInput] = useState("");
  const [files, setFiles] = useState<FileList | undefined>(undefined);
  const suggestions = useSuggestionsStore((state) => state.suggestions);
  const [modelDownload, setModelDownload] = useState<{
    status: "downloading" | "complete" | "error";
    progress: number;
    message: string;
  } | null>(null);

  const {
    error,
    status,
    sendMessage,
    messages,
    regenerate,
    stop,
    setMessages,
  } = useChat<BuiltInAIUIMessage>({
    transport: new ClientSideChatTransport(),
    onError(error) {
      toast.error(error.message);
    },
    onData: (dataPart) => {
      // Handle model download progress
      if (dataPart.type === "data-modelDownloadProgress") {
        setModelDownload({
          status: dataPart.data.status,
          progress: dataPart.data.progress ?? 0,
          message: dataPart.data.message,
        });
        // Clear the download banner when complete
        if (dataPart.data.status === "complete") {
          setTimeout(() => setModelDownload(null), 500);
        }
        return;
      }
      // Handle transient notifications
      if (dataPart.type === "data-notification") {
        if (dataPart.data.level === "error") {
          toast.error(dataPart.data.message);
        } else if (dataPart.data.level === "warning") {
          toast.warning(dataPart.data.message);
        } else {
          toast.info(dataPart.data.message);
        }
      }
    },
  });

  const isLoading = status !== "ready";

  // Send a message
  const handleSubmit = () => {
    // Allow submission when ready, or after an error/stop (not during submit/stream)
    const canSubmit =
      (input.trim() || files) &&
      status !== "submitted" &&
      status !== "streaming";

    if (canSubmit) {
      // Track message submission
      window.umami?.track("message_submitted");

      sendMessage({
        text: input,
        files,
      });
      setInput("");
      setFiles(undefined);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    // Allow submission when not actively submitting or streaming
    if (status !== "submitted" && status !== "streaming") {
      sendMessage({
        text: suggestion,
        files,
      });
      setInput("");
      setFiles(undefined);
    }
  };

  const handleClearConversation = () => {
    // Stop any ongoing generation
    if (status === "submitted" || status === "streaming") {
      stop();
      setTimeout(() => setMessages([]), 100);
    } else {
      setMessages([]);
    }
  };

  return (
    <div className="relative flex h-screen w-full flex-col overflow-hidden">
      <Header />

      {/* Model Download Banner - Absolutely positioned */}
      {modelDownload && (
        <ModelDownloadBanner
          message={modelDownload.message}
          progress={modelDownload.progress}
          status={modelDownload.status}
        />
      )}

      {/* Main Content */}
      <main
        className={cn(
          "relative mx-auto flex min-h-0 w-full max-w-3xl flex-1 flex-col",
          messages.length === 0 && "items-center justify-center"
        )}
      >
        {messages.length === 0 ? (
          <ChatEmptyState onSuggestionClick={handleSuggestionClick} />
        ) : (
          <ChatMessages
            error={error}
            messages={messages}
            onRegenerate={regenerate}
            status={status}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="sticky bottom-0 z-20">
        <div className="mx-auto w-full max-w-3xl space-y-4 p-4">
          <ChatInput
            files={files}
            hasMessages={messages.length > 0}
            input={input}
            isLoading={isLoading}
            onClearConversation={handleClearConversation}
            onFilesChange={setFiles}
            onInputChange={setInput}
            onSubmit={handleSubmit}
            onSuggestionClick={handleSuggestionClick}
            showSuggestions={messages.length > 0}
            status={status}
            stop={stop}
            suggestions={suggestions}
          />

          <Footer />
        </div>
      </footer>
    </div>
  );
}
