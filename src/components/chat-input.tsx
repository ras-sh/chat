import type { UseChatHelpers } from "@ai-sdk/react";
import type { BuiltInAIUIMessage } from "@built-in-ai/core";
import { Paperclip } from "lucide-react";
import { useRef } from "react";
import {
  PromptInput,
  PromptInputBody,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
} from "~/components/ai-elements/prompt-input";
import { Suggestion, Suggestions } from "~/components/ai-elements/suggestion";
import { FileUpload, type FileUploadRef } from "~/components/file-upload";
import { Button } from "~/components/ui/button";
import { useBrowserAISupport } from "~/hooks/use-browser-ai-support";

type ChatInputProps = {
  input: string;
  onInputChange: (value: string) => void;
  files: FileList | undefined;
  onFilesChange: (files: FileList | undefined) => void;
  onSubmit: () => void;
  isLoading: boolean;
  status: UseChatHelpers<BuiltInAIUIMessage>["status"];
  stop: UseChatHelpers<BuiltInAIUIMessage>["stop"];
  suggestions: string[];
  onSuggestionClick: (suggestion: string) => void;
  showSuggestions: boolean;
};

export function ChatInput({
  input,
  onInputChange,
  files,
  onFilesChange,
  onSubmit,
  isLoading,
  status,
  stop,
  suggestions,
  onSuggestionClick,
  showSuggestions,
}: ChatInputProps) {
  const fileUploadRef = useRef<FileUploadRef>(null);
  const browserSupportsModel = useBrowserAISupport();
  const isDisabled = browserSupportsModel === false;
  // Only disable inputs when submitting (not when streaming), so users can type next message
  const isInputDisabled = isDisabled || status === "submitted";

  return (
    <div className="space-y-3">
      {/* Suggestions from model */}
      {showSuggestions &&
        suggestions.length > 0 &&
        !isLoading &&
        browserSupportsModel !== false && (
          <Suggestions>
            {suggestions.map((suggestion, index) => (
              <Suggestion
                key={index}
                onClick={onSuggestionClick}
                suggestion={suggestion}
              />
            ))}
          </Suggestions>
        )}

      <FileUpload
        disabled={isInputDisabled}
        files={files}
        onFilesChange={onFilesChange}
        ref={fileUploadRef}
      />

      <PromptInput onSubmit={onSubmit}>
        <PromptInputBody>
          <PromptInputTextarea
            autoFocus
            className="text-base!"
            disabled={isInputDisabled}
            onChange={(e) => onInputChange(e.target.value)}
            placeholder="Ask anything..."
            value={input}
          />
        </PromptInputBody>
        <PromptInputFooter>
          <Button
            disabled={isInputDisabled}
            onClick={() => fileUploadRef.current?.openFileDialog()}
            size="icon"
            type="button"
            variant="ghost"
          >
            <Paperclip className="h-4 w-4" />
          </Button>
          <PromptInputSubmit
            disabled={
              status === "submitted" || status === "streaming"
                ? false
                : isDisabled ||
                  (!input.trim() && (!files || files.length === 0))
            }
            onClick={
              status === "submitted" || status === "streaming"
                ? stop
                : undefined
            }
            status={status}
            type={
              status === "submitted" || status === "streaming"
                ? "button"
                : "submit"
            }
          />
        </PromptInputFooter>
      </PromptInput>
    </div>
  );
}
