import type { UseChatHelpers } from "@ai-sdk/react";
import type { BuiltInAIUIMessage } from "@built-in-ai/core";
import { Paperclip, Trash2 } from "lucide-react";
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
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "~/components/ui/alert-dialog";
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
  onClearConversation?: () => void;
  hasMessages?: boolean;
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
  onClearConversation,
  hasMessages,
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
                data-umami-event="suggestion_clicked"
                data-umami-event-index={index}
                data-umami-event-source="chat_input"
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
      >
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
            <div className="flex items-center gap-1">
              <Button
                data-umami-event="file_upload_opened"
                disabled={isInputDisabled}
                onClick={() => fileUploadRef.current?.openFileDialog()}
                size="icon"
                type="button"
                variant="ghost"
              >
                <Paperclip className="h-4 w-4" />
              </Button>
              {hasMessages && onClearConversation && (
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button
                      data-umami-event="clear_conversation_button_clicked"
                      disabled={isInputDisabled}
                      size="icon"
                      type="button"
                      variant="ghost"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>Clear conversation?</AlertDialogTitle>
                      <AlertDialogDescription>
                        This will delete all messages in the current
                        conversation. This action cannot be undone.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>Cancel</AlertDialogCancel>
                      <AlertDialogAction
                        data-umami-event="conversation_cleared"
                        onClick={onClearConversation}
                      >
                        Clear
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              )}
            </div>
            <PromptInputSubmit
              data-umami-event={
                status === "submitted" || status === "streaming"
                  ? "message_stopped"
                  : undefined
              }
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
      </FileUpload>
    </div>
  );
}
