## This is WIP

We use shadcdn/ui [Shadcn UI](https://ui.shadcn.com/) for the UI components.

## Getting Started

First, install dependencies:

```bash
npm install
```

Then, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:8322](http://localhost:8322) with your browser to see the result.

## Model filtering

The Chat Playground shows all available LLM models by default. To limit the
model dropdown, set `NEXT_PUBLIC_OGX_UI_ALLOWED_MODELS` to a comma-separated
list of model IDs:

```bash
NEXT_PUBLIC_OGX_UI_ALLOWED_MODELS=openai/gpt-4.1-mini,anthropic/claude-sonnet-4-6
```

When configured, only matching models are shown in the configured order, and
the first matching model is selected by default. If the variable is unset, the
first model in the existing sorted list is selected by default.
