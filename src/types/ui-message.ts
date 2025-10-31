import type { UIMessage } from "ai";

/**
 * Extended UI message type that includes built-in AI data parts
 */
export type ExtendedBuiltInAIUIMessage = UIMessage<
  never,
  {
    modelDownloadProgress: {
      status: "downloading" | "complete" | "error";
      progress?: number;
      message: string;
    };
    notification: {
      message: string;
      level: "info" | "warning" | "error";
    };
  }
>;
