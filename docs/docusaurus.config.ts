// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

import type * as Preset from "@docusaurus/preset-classic";
import type { Config } from "@docusaurus/types";
import type * as Plugin from "@docusaurus/types/src/plugin";
import type * as OpenApiPlugin from "docusaurus-plugin-openapi-docs";

const config: Config = {
  title: 'OGX',
  tagline: 'The open-source framework for building generative AI applications',
  url: 'https://ogx-ai.github.io',
  baseUrl: '/',
  onBrokenLinks: "warn",
  favicon: "img/favicon.ico",

  // Enhanced favicon and meta configuration
  headTags: [
    {
      tagName: 'link',
      attributes: {
        rel: 'icon',
        type: 'image/png',
        sizes: '32x32',
        href: '/img/favicon-32x32.png',
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'icon',
        type: 'image/png',
        sizes: '16x16',
        href: '/img/favicon-16x16.png',
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'apple-touch-icon',
        sizes: '180x180',
        href: '/img/ogx-logo.png',
      },
    },
    {
      tagName: 'meta',
      attributes: {
        name: 'theme-color',
        content: '#0d7377', // Teal from logo gradient
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'manifest',
        href: '/site.webmanifest',
      },
    },
  ],

  // GitHub pages deployment config.
  organizationName: 'reluctantfuturist',
  projectName: 'ogx',
  trailingSlash: false,

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: require.resolve("./sidebars.ts"),
          // disableVersioning: true,
          docItemComponent: "@theme/ApiItem", // Derived from docusaurus-theme-openapi
          editUrl: 'https://github.com/ogx-ai/ogx/edit/main/docs/',
          showLastUpdateTime: true,
          showLastUpdateAuthor: false,
          remarkPlugins: [
            [require('remark-code-import'), {
              rootDir: require('path').join(__dirname, '..') // Repository root
            }]
          ],
        },
        blog: {
          onUntruncatedBlogPosts: 'ignore',
          blogSidebarCount: 'ALL',
          showReadingTime: true,
          postsPerPage: 10,
          exclude: ['**/building-agentic-flows/**'],
          readingTime: ({content, frontMatter, defaultReadingTime}) =>
            defaultReadingTime({content, options: {wordsPerMinute: 300}}),
          feedOptions: {
            type: 'all',
            title: 'OGX Blog',
            description: 'Blog posts about OGX',
            copyright: `Copyright © ${new Date().getFullYear()} The OGX Contributors`,
            language: 'en',
          },
        },
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/ogx.png',
    navbar: {
      hideOnScroll: true,
      title: 'OGX',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          type: 'dropdown',
          label: 'API Reference',
          position: 'left',
          to: '/docs/api-overview',
          items: [
            {
              type: 'docSidebar',
              sidebarId: 'stableApiSidebar',
              label: 'Stable',
            },
            {
              type: 'docSidebar',
              sidebarId: 'experimentalApiSidebar',
              label: 'Experimental',
            },
            {
              type: 'docSidebar',
              sidebarId: 'deprecatedApiSidebar',
              label: 'Deprecated',
            },
          ],
        },
        {
          to: '/blog',
          label: 'Blogs',
          position: 'left',
        },
        {
          href: 'https://github.com/ogx-ai/ogx',
          label: 'GitHub',
          position: 'right',
        },
        {
          type: 'docsVersionDropdown',
          position: 'right',
          dropdownActiveClassDisabled: true,
          dropdownItemsAfter: [
            { to: '/versions', label: 'All versions' },
          ],
        },
      ],
    },
    footer: {
      style: 'light',
      links: [
        {
          title: 'Getting Started',
          items: [
            {
              label: 'Quickstart',
              to: '/docs/getting_started/quickstart',
            },
            {
              label: 'Concepts',
              to: '/docs/concepts',
            },
            {
              label: 'Distributions',
              to: '/docs/distributions/building_distro',
            },
            {
              label: 'Providers',
              to: '/docs/providers',
            },
          ],
        },
        {
          title: 'API',
          items: [
            {
              label: 'API Reference',
              to: '/docs/api-overview',
            },
            {
              label: 'OpenAI',
              to: '/docs/api-openai',
            },
            {
              label: 'Anthropic Messages',
              to: '/docs/api-anthropic-messages',
            },
            {
              label: 'Google Interactions',
              to: '/docs/api-google-interactions',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Discord',
              href: 'https://discord.gg/ZAFjsrcw',
            },
            {
              label: 'Issues',
              href: 'https://github.com/ogx-ai/ogx/issues',
            },
            {
              label: 'Contributing',
              to: '/docs/contributing',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Brand',
              to: '/brand',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/ogx-ai/ogx',
            },
            {
              label: 'PyPI',
              href: 'https://pypi.org/project/ogx/',
            },
            {
              label: 'Releases',
              href: 'https://github.com/ogx-ai/ogx/releases',
            },
            {
              label: 'Docker Hub',
              href: 'https://hub.docker.com/u/llamastack',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} The OGX Contributors`,
    },
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: false,
    },
    prism: {
      theme: require('prism-react-renderer').themes.oneDark,
      darkTheme: require('prism-react-renderer').themes.oneDark,
      additionalLanguages: [
        'ruby',
        'csharp',
        'php',
        'java',
        'powershell',
        'json',
        'bash',
        'python',
        'yaml',
      ],
    },
    docs: {
      sidebar: {
        hideable: false,
        autoCollapseCategories: true,
      },
    },
    // Language tabs for API documentation
    languageTabs: [
      {
        highlight: "python",
        language: "python",
        logoClass: "python",
      },
      {
        highlight: "bash",
        language: "curl",
        logoClass: "curl",
      },
    ],
  } satisfies Preset.ThemeConfig,

  plugins: [
    function webpackFallbackPlugin() {
      return {
        name: 'webpack-fallback',
        configureWebpack() {
          return {
            resolve: {
              fallback: {
                path: false,
                fs: false,
                os: false,
                crypto: false,
              },
            },
          };
        },
      };
    },
    [
      "docusaurus-plugin-openapi-docs",
      {
        id: "openapi",
        docsPluginId: "classic",
        config: {
          stable: {
            specPath: "static/ogx-spec.yaml",
            outputDir: "docs/api",
            downloadUrl: "https://raw.githubusercontent.com/ogx-ai/ogx/main/docs/static/ogx-spec.yaml",
            sidebarOptions: {
              groupPathsBy: "tag",
              categoryLinkSource: "tag",
            },
          } satisfies OpenApiPlugin.Options,
          experimental: {
            specPath: "static/experimental-ogx-spec.yaml",
            outputDir: "docs/api-experimental",
            downloadUrl: "https://raw.githubusercontent.com/ogx-ai/ogx/main/docs/static/experimental-ogx-spec.yaml",
            sidebarOptions: {
              groupPathsBy: "tag",
              categoryLinkSource: "tag",
            },
          } satisfies OpenApiPlugin.Options,
          deprecated: {
            specPath: "static/deprecated-ogx-spec.yaml",
            outputDir: "docs/api-deprecated",
            downloadUrl: "https://raw.githubusercontent.com/ogx-ai/ogx/main/docs/static/deprecated-ogx-spec.yaml",
            sidebarOptions: {
              groupPathsBy: "tag",
              categoryLinkSource: "tag",
            },
          } satisfies OpenApiPlugin.Options,
        } satisfies Plugin.PluginOptions,
      },
    ],
  ],

  themes: [
    "docusaurus-theme-openapi-docs",
    "@docusaurus/theme-mermaid",
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        // Optimization for production
        hashed: true,

        // Language settings
        language: ["en"],

        // Content indexing settings
        indexDocs: true,
        indexBlog: true, // Enable blog indexing
        indexPages: true,

        // Route configuration
        docsRouteBasePath: '/docs',

        // Search behavior optimization for technical docs
        searchResultLimits: 12,
        searchResultContextMaxLength: 80,
        explicitSearchResultPath: true,

        // User experience enhancements
        searchBarShortcut: true,
        searchBarShortcutHint: true,
        searchBarPosition: "right",

        // Performance optimizations
        ignoreFiles: [
          "node_modules/**/*",
        ],
      },
    ],
  ],

  markdown: {
    mermaid: true,
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
};

export default config;
