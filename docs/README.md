# OGX Documentation

Here's a collection of comprehensive guides, examples, and resources for building AI applications with OGX. For the complete documentation, visit our [Github page](https://ogx-ai.github.io/getting_started/quickstart).

## Render locally

From the ogx `docs/` directory, run the following commands to render the docs locally:

```bash
npm install
npm run gen-api-docs all
npm run build
npm run serve
```

You can open up the docs in your browser at <http://localhost:3000>

## File Import System

This documentation uses `remark-code-import` to import files directly from the repository, eliminating copy-paste maintenance. Files are automatically embedded during build time.

### Importing Code Files

To import Python code (or any code files) with syntax highlighting, use this syntax in `.mdx` files:

```markdown
```python file=./demo_script.py title="demo_script.py"
```

```text

This automatically imports the file content and displays it as a formatted code block with Python syntax highlighting.

**Note:** Paths are relative to the current `.mdx` file location, not the repository root.

### Importing Markdown Files as Content

For importing and rendering markdown files (like CONTRIBUTING.md), use the raw-loader approach:

```jsx
import Contributing from '!!raw-loader!../../../CONTRIBUTING.md';
import ReactMarkdown from 'react-markdown';

<ReactMarkdown>{Contributing}</ReactMarkdown>
```

**Requirements:**

- Install dependencies: `npm install --save-dev raw-loader react-markdown`

**Path Resolution:**

- For `remark-code-import`: Paths are relative to the current `.mdx` file location
- For `raw-loader`: Paths are relative to the current `.mdx` file location
- Use `../` to navigate up directories as needed

## Content

Try out OGX's capabilities through our detailed Jupyter notebooks:

- [Building AI Applications Notebook](./getting_started.ipynb) - A comprehensive guide to building production-ready AI applications using OGX
- [Zero-to-Hero Guide](./zero_to_hero_guide) - Step-by-step guide for getting started with OGX
- [Migrating from Agent objects to Responses](./notebooks/responses-api-agent-migration.ipynb) - A guide to migrating from legacy OGX Agent APIs to the OpenAI-compatible Responses API
