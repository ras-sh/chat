# chat.ras.sh

💬 AI chat that runs entirely in your browser using Transformers.js. No servers, no tracking, complete privacy.

## Overview

A privacy-first chat interface powered by Transformers.js and SmolLM2. All inference happens locally on your device - your conversations never leave your browser.

**Supported browsers:**
- Chrome 113+ (WebGPU support)
- Edge 113+ (WebGPU support)
- Safari 18+ (WebGPU support)

## Features

- **100% Local** - All AI inference runs directly in your browser
- **Complete Privacy** - Zero external API calls, no data collection
- **Smart Suggestions** - Get contextual follow-up questions after each response
- **Streaming Responses** - Real-time AI responses as they generate
- **Modern UI** - Clean, responsive interface with dark mode
- **Model Download Progress** - Visual indicators when downloading models

## Tech Stack

- [TanStack Start](https://tanstack.com/start) - Full-stack React framework
- [@built-in-ai/transformers-js](https://github.com/jakobhoeg/built-in-ai/tree/main/packages/transformers-js) - Transformers.js integration for AI SDK
- [AI SDK](https://ai-sdk.dev/) - Streaming and message handling
- [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) - Lightweight language model
- React 19, TypeScript, Tailwind CSS

## Quick Start

```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev
```

Open http://localhost:3000 in any WebGPU-compatible browser.

## How It Works

This app uses Transformers.js to run the SmolLM2-360M language model entirely in your browser via WebGPU. On first use, the model will be downloaded and cached locally.

All processing happens on your device using WebGPU for hardware acceleration. The model runs in a dedicated Web Worker to avoid blocking the UI thread, ensuring smooth performance. No data is sent to external servers. The SmolLM2-360M model provides a perfect balance between performance and capability for browser-based inference.

## Project Structure

```
src/
├── routes/
│   ├── __root.tsx              # Root layout
│   └── index.tsx               # Main chat interface
├── components/
│   ├── chat-empty-state.tsx    # Empty state with suggestions
│   ├── chat-input.tsx          # Message input component
│   ├── chat-messages.tsx       # Message list display
│   ├── header.tsx              # App header
│   └── footer.tsx              # App footer
├── lib/
│   ├── client-side-chat-transport.ts  # Browser AI integration
│   └── transformers-worker.ts  # Web Worker for Transformers.js
├── hooks/
│   └── use-browser-ai-support.ts      # Browser compatibility check
└── types/
    └── ui-message.ts           # Message type definitions
```

## Scripts

| Command | Description |
|---------|-------------|
| `pnpm dev` | Start development server |
| `pnpm build` | Build for production |
| `pnpm preview` | Preview production build |
| `pnpm check-types` | TypeScript type checking |
| `pnpm check` | Lint code with Biome |
| `pnpm fix` | Auto-fix linting issues |

## Browser Requirements

This app requires a browser with WebGPU support:

- **Chrome 113+** - Full WebGPU support
- **Edge 113+** - Full WebGPU support
- **Safari 18+** - WebGPU support (macOS Sonoma+)

If your browser doesn't support WebGPU, you'll see a warning message on the chat interface. Check [WebGPU browser compatibility](https://caniuse.com/webgpu) for the latest support status.

## Privacy & Data

- **No external servers** - All processing happens in your browser
- **No data collection** - Your conversations are never logged or sent anywhere
- **No persistence** - Messages are kept in memory only (refresh to clear)
- **Local model caching** - Downloaded models are cached by the browser

## Development

Built with modern web technologies:

- **React 19** for UI components
- **TanStack Start** for full-stack React
- **AI SDK** for streaming message handling
- **Tailwind CSS** for styling
- **Biome** for linting and formatting

## License

MIT License - see [LICENSE](LICENSE) for details.
