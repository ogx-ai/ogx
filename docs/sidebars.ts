import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'index',
    {
      type: 'category',
      label: 'Getting Started',
      className: 'sidebar-getting-started',
      collapsed: false,
      items: [
        'getting_started/quickstart',
        'getting_started/migrate_existing_app',
        'getting_started/detailed_tutorial',
        'getting_started/libraries',
      ],
    },
    {
      type: 'category',
      label: 'Concepts',
      className: 'sidebar-concepts',
      collapsed: false,
      items: [
        'concepts/index',
        'concepts/architecture',
        {
          type: 'category',
          label: 'APIs',
          collapsed: false,
          items: [
            'concepts/apis/index',
            'concepts/apis/api_providers',
            {
              type: 'category',
              label: 'OpenAI',
              collapsed: false,
              link: { type: 'doc', id: 'api-openai/index' },
              items: [
                'api-openai/responses-flow',
                'api-openai/conformance',
                'api-openai/provider_matrix',
              ],
            },
            {
              type: 'category',
              label: 'Anthropic Messages',
              collapsed: false,
              link: { type: 'doc', id: 'api-anthropic-messages/index' },
              items: [
                'api-openai/anthropic_messages',
                'api-anthropic-messages/conformance',
              ],
            },
            {
              type: 'category',
              label: 'Google Interactions',
              collapsed: false,
              link: { type: 'doc', id: 'api-google-interactions/index' },
              items: [
                'api-google-interactions/conformance',
              ],
            },
            'concepts/apis/external',
            'concepts/apis/api_leveling',
          ],
        },
        {
          type: 'category',
          label: 'Vector Stores',
          collapsed: false,
          items: [
            'concepts/file_operations_vector_stores',
            'concepts/vector_stores_configuration',
          ],
        },
        'concepts/distributions',
        'concepts/resources',
      ],
    },
    {
      type: 'category',
      label: 'Distributions',
      className: 'sidebar-distributions',
      collapsed: false,
      items: [
        'distributions/index',
        'distributions/list_of_distributions',
        'distributions/building_distro',
        'distributions/customizing_run_yaml',
        'distributions/importing_as_library',
        'distributions/configuration',
        'distributions/starting_ogx_server',
        'distributions/ogx_ui',
        {
          type: 'category',
          label: 'Self-Hosted Distributions',
          collapsed: false,
          items: [
            'distributions/self_hosted_distro/starter',
            'distributions/self_hosted_distro/nvidia',
            'distributions/self_hosted_distro/passthrough',
          ],
        },
        {
          type: 'category',
          label: 'Remote-Hosted Distributions',
          collapsed: false,
          items: [
            'distributions/remote_hosted_distro/index',
            'distributions/remote_hosted_distro/watsonx',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Providers',
      className: 'sidebar-providers',
      collapsed: false,
      items: [
        'providers/index',
        'providers/external/index',
        'providers/external/external-providers-guide',
        'providers/external/external-providers-list',
        'providers/openai',
        {
          type: 'category',
          label: 'Batches',
          collapsed: false,
          link: { type: 'doc', id: 'providers/batches/index' },
          items: [
            'providers/batches/inline_reference'
          ],
        },
        {
          type: 'category',
          label: 'DatasetIO',
          collapsed: false,
          link: { type: 'doc', id: 'providers/datasetio/index' },
          items: [
            'providers/datasetio/inline_localfs',
            'providers/datasetio/remote_huggingface',
            'providers/datasetio/remote_nvidia'
          ],
        },
        {
          type: 'category',
          label: 'Eval',
          collapsed: false,
          link: { type: 'doc', id: 'providers/eval/index' },
          items: [
            'providers/eval/inline_builtin',
            'providers/eval/remote_nvidia'
          ],
        },
        {
          type: 'category',
          label: 'File Processors',
          collapsed: false,
          link: { type: 'doc', id: 'providers/file_processors/index' },
          items: [
            'providers/file_processors/inline_docling',
            'providers/file_processors/inline_pypdf'
          ],
        },
        {
          type: 'category',
          label: 'Files',
          collapsed: false,
          link: { type: 'doc', id: 'providers/files/index' },
          items: [
            'providers/files/inline_localfs',
            'providers/files/remote_s3',
            'providers/files/remote_openai'
          ],
        },
        {
          type: 'category',
          label: 'Inference',
          collapsed: false,
          link: { type: 'doc', id: 'providers/inference/index' },
          items: [
            'providers/inference/inline_sentence-transformers',
            'providers/inference/inline_transformers',
            'providers/inference/remote_anthropic',
            'providers/inference/remote_azure',
            'providers/inference/remote_bedrock',
            'providers/inference/remote_cerebras',
            'providers/inference/remote_databricks',
            'providers/inference/remote_fireworks',
            'providers/inference/remote_gemini',
            'providers/inference/remote_groq',
            'providers/inference/remote_llama-cpp-server',
            'providers/inference/remote_llama-openai-compat',
            'providers/inference/remote_nvidia',
            'providers/inference/remote_oci',
            'providers/inference/remote_ollama',
            'providers/inference/remote_openai',
            'providers/inference/remote_passthrough',
            'providers/inference/remote_runpod',
            'providers/inference/remote_sambanova',
            'providers/inference/remote_together',
            'providers/inference/remote_vertexai',
            'providers/inference/remote_vllm',
            'providers/inference/remote_watsonx',
          ],
        },
        {
          type: 'category',
          label: 'Interactions',
          collapsed: false,
          link: { type: 'doc', id: 'providers/interactions/index' },
          items: [
            'providers/interactions/inline_builtin'
          ],
        },
        {
          type: 'category',
          label: 'Messages',
          collapsed: false,
          link: { type: 'doc', id: 'providers/messages/index' },
          items: [
            'providers/messages/inline_builtin'
          ],
        },
        {
          type: 'category',
          label: 'Responses',
          collapsed: false,
          link: { type: 'doc', id: 'providers/responses/index' },
          items: [
            'providers/responses/inline_builtin'
          ],
        },
        {
          type: 'category',
          label: 'Safety',
          collapsed: false,
          link: { type: 'doc', id: 'providers/safety/index' },
          items: [
            'providers/safety/inline_code-scanner',
            'providers/safety/inline_llama-guard',
            'providers/safety/inline_prompt-guard',
            'providers/safety/remote_bedrock',
            'providers/safety/remote_nvidia',
            'providers/safety/remote_passthrough',
            'providers/safety/remote_sambanova',
          ],
        },
        {
          type: 'category',
          label: 'Tool Runtime',
          collapsed: false,
          link: { type: 'doc', id: 'providers/tool_runtime/index' },
          items: [
            'providers/tool_runtime/inline_file-search',
            'providers/tool_runtime/remote_bing-search',
            'providers/tool_runtime/remote_brave-search',
            'providers/tool_runtime/remote_model-context-protocol',
            'providers/tool_runtime/remote_tavily-search',
            'providers/tool_runtime/remote_wolfram-alpha'
          ],
        },
        {
          type: 'category',
          label: 'Vector IO',
          collapsed: false,
          link: { type: 'doc', id: 'providers/vector_io/index' },
          items: [
            'providers/vector_io/inline_builtin',
            'providers/vector_io/inline_chromadb',
            'providers/vector_io/inline_faiss',
            'providers/vector_io/inline_milvus',
            'providers/vector_io/inline_qdrant',
            'providers/vector_io/inline_sqlite-vec',
            'providers/vector_io/remote_chromadb',
            'providers/vector_io/remote_elasticsearch',
            'providers/vector_io/remote_infinispan',
            'providers/vector_io/remote_milvus',
            'providers/vector_io/remote_oci',
            'providers/vector_io/remote_pgvector',
            'providers/vector_io/remote_qdrant',
            'providers/vector_io/remote_weaviate',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Building Applications',
      className: 'sidebar-building-apps',
      collapsed: false,
      items: [
        'building_applications/index',
        'building_applications/rag',
        'building_applications/rag_benchmarks',
        'building_applications/agent',
        'building_applications/agent_execution_loop',
        'building_applications/responses_vs_agents',
        'building_applications/tools',
        'building_applications/evals',
        'building_applications/telemetry',
        'building_applications/safety',
        'building_applications/native_responses',
        'building_applications/playground',
        'building_applications/claude_code_integration',
        'building_applications/codex_cli_integration',
      ],
    },
    {
      type: 'category',
      label: 'Deploying',
      className: 'sidebar-deploying',
      collapsed: false,
      items: [
        'deploying/index',
        'deploying/kubernetes_deployment',
        'deploying/aws_eks_deployment',
      ],
    },
    {
      type: 'category',
      label: 'Contributing',
      className: 'sidebar-contributing',
      collapsed: false,
      items: [
        'contributing/index',
        'contributing/new_api_provider',
        'contributing/new_vector_database',
        'contributing/testing/record-replay',
      ],
    },
    {
      type: 'category',
      label: 'References',
      className: 'sidebar-references',
      collapsed: false,
      items: [
        'references/index',
        'references/ogx_cli_reference/index',
        'references/ogx_client_cli_reference',
        'references/python_sdk_reference/index',
      ],
    },
  ],

  // API Reference sidebars - use plugin-generated sidebars
  stableApiSidebar: require('./docs/api/sidebar.ts').default,
  experimentalApiSidebar: require('./docs/api-experimental/sidebar.ts').default,
  deprecatedApiSidebar: require('./docs/api-deprecated/sidebar.ts').default,

};

export default sidebars;
