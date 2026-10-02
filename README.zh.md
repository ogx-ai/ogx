<h1 align="center">OGX</h1>

<p align="center">
  <a href="https://pypi.org/project/ogx/"><img src="https://img.shields.io/pypi/v/ogx?logo=pypi" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/ogx/"><img src="https://img.shields.io/pypi/dm/ogx" alt="PyPI Downloads"></a>
  <a href="https://hub.docker.com/u/ogxai"><img src="https://img.shields.io/docker/pulls/ogxai/distribution-starter?logo=docker" alt="Docker Hub Pulls"></a>
  <a href="https://github.com/ogx-ai/ogx/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/ogx.svg" alt="License"></a>
  <a href="https://discord.gg/bUYRqEvK6"><img src="https://img.shields.io/discord/1257833999603335178?color=5865F2&logo=discord&logoColor=ffffff" alt="Discord"></a>
  <a href="https://github.com/ogx-ai/ogx/actions/workflows/unit-tests.yml?query=branch%3Amain"><img src="https://github.com/ogx-ai/ogx/actions/workflows/unit-tests.yml/badge.svg?branch=main" alt="Unit Tests"></a>
  <a href="https://github.com/ogx-ai/ogx/actions/workflows/integration-tests.yml?query=branch%3Amain"><img src="https://github.com/ogx-ai/ogx/actions/workflows/integration-tests.yml/badge.svg?branch=main" alt="Integration Tests"></a>
  <a href="https://ogx-ai.github.io/docs/api-openai/conformance"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fogx-ai%2Fogx%2Fmain%2Fdocs%2Fstatic%2Fopenai-coverage.json&query=%24.summary.conformance.score&suffix=%25&label=OpenResponses%20Conformance&color=brightgreen" alt="OpenResponses Conformance"></a>
  <a href="https://deepwiki.com/ogx-ai/ogx"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

[**快速上手**](https://ogx-ai.github.io/docs/getting_started/quickstart) | [**官方文档**](https://ogx-ai.github.io/docs) | [**OpenAI API 兼容性矩阵**](https://ogx-ai.github.io/docs/api-openai) | [**Discord 社区**](https://discord.gg/E8M7xraH8)

> [!IMPORTANT]
> **Llama Stack 现已升级并更名为 OGX。** 随着名称的变更，我们的使命也全面拓展——模型无关（Model-Agnostic）、跨多 SDK 支持、开箱即用的生产级底座。[阅读官方发布公告 →](https://ogx-ai.github.io/blog/from-llama-stack-to-ogx)

**面向 AI 应用构建的开源智能体（Agentic）API 服务端。全面兼容 OpenAI 接口规范。支持任意大模型与任意计算基础设施。**

<p align="center">
  <img src="docs/static/img/architecture-animated.svg" alt="OGX 架构图" width="100%">
</p>

OGX 是 OpenAI API 的即插即用替代方案，你可以将其部署运行在任何环境——个人笔记本电脑、本地数据中心或各大公有云。你可以直接使用现有的任何 OpenAI 兼容客户端或 Agent 框架。在 Llama、GPT、Gemini、Mistral 或任何模型之间自由切换，完全无需改动上层业务应用代码。

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")
response = client.chat.completions.create(
    model="llama-3.3-70b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## 核心功能与特性

- **聊天补全与向量嵌入 (Chat Completions & Embeddings)** — 标准 `/v1/chat/completions`、`/v1/completions` 和 `/v1/embeddings` 端点，与任何 OpenAI 官方及第三方客户端无缝兼容。
- **Responses API (智能体编排)** — 服务端原生 Agentic 编排，支持在单次 API 调用中完成工具调用（Tool Calling）、MCP 协议服务器集成及内置文档检索增强（RAG）([了解更多](https://ogx-ai.github.io/docs/api-openai))。
- **向量存储与文件管理 (Vector Stores & Files)** — 提供 `/v1/vector_stores` 与 `/v1/files` 端点，实现受管的文档存储与高效相似度检索。
- **批处理接口 (Batches)** — 提供 `/v1/batches` 支持离线大规模批处理任务。
- **技能拓展体系 (Skills)** — 提供 `/v1alpha/skills` 管理带版本控制的技能包（包含 `SKILL.md` 描述清单的 zip 归档包），方便智能体随时动态加载调用。
- **完全符合 [Open Responses](https://www.openresponses.org/) 标准** — Responses API 实现已全面通过 Open Responses 官方标准合规性测试套件。
- **原生多 SDK 支持** — 原生支持并行使用 [Anthropic SDK](https://docs.anthropic.com/en/api/messages)（`/v1/messages`）或 [Google GenAI SDK](https://ai.google.dev/gemini-api/docs/interactions)（`/v1alpha/interactions`），与 OpenAI API 齐头并进。

## 支持任意大模型，适配任意计算基础设施

OGX 拥有高度可插拔的 Provider（服务提供商）架构。在本地开发时可使用 Ollama，上线生产时可切换为 vLLM，亦可直接对接各类云端托管模型服务——上层对外暴露的 API 接口完全保持一致。

查阅完整 Provider 列表请见 [Provider 文档](https://ogx-ai.github.io/docs/providers)。

## 快速上手

安装并运行 OGX 服务端：

```bash
# 一键式安装脚本
curl -LsSf https://github.com/ogx-ai/ogx/raw/main/scripts/install.sh | bash

# 或者通过 uv 进行安装
uv pip install ogx

# 启动服务（自动根据当前环境监测并加载对应的 Providers）
uv run ogx go
```

随后即可使用任意 OpenAI、Anthropic 或 Google GenAI 客户端建立连接——包括 [Python](https://github.com/openai/openai-python)、[TypeScript](https://github.com/openai/openai-node)、[curl](https://platform.openai.com/docs/api-reference) 或任何支持这些协议的框架。

关于详细环境配置与选项，请参阅 [快速上手指南 (Quick Start)](https://ogx-ai.github.io/docs/getting_started/quickstart)。

## 资源索引

- [官方文档 (Documentation)](https://ogx-ai.github.io/docs) — 完整技术参考
- [OpenAI API 兼容性矩阵 (OpenAI API Compatibility)](https://ogx-ai.github.io/docs/api-openai) — API 端点覆盖范围与 Provider 矩阵
- [入门实践 Notebook (Getting Started Notebook)](./docs/getting_started.ipynb) — 文本与多模态视觉推理端到端教程
- [贡献指南 (Contributing)](CONTRIBUTING.md) — 如何参与开源贡献

**官方客户端 SDK：**

OGX 提供针对 Python 和 TypeScript 的官方客户端 SDK：

| 编程语言 | SDK 仓库 | 发布包 |
| :----: | :----: | :----: |
| Python | [ogx-client-python](https://github.com/ogx-ai/ogx-client-python) | [![PyPI version](https://img.shields.io/pypi/v/ogx_client.svg)](https://pypi.org/project/ogx_client/) |
| TypeScript | [ogx-client-typescript](https://github.com/ogx-ai/ogx-client-typescript) | [![NPM version](https://img.shields.io/npm/v/ogx-client.svg)](https://npmjs.org/package/ogx-client) |

**替代性 Python SDK：**

对于倾向于使用基于 OpenAPI Generator 构建 SDK 的用户，我们提供了替代的 Python 客户端：

- **[ogx-open-client](https://pypi.org/project/ogx-open-client/)** — 基于 OpenAPI Specification 自动生成，通过不同的生成范式提供类似的功能
- **[使用示例 (Usage Examples)](client-sdks/openapi/USAGE_EXAMPLES.md)** — 涵盖所有核心特性的端到端代码实战范例
- **[技术演进与选型策略 (Strategy & Rationale)](client-sdks/openapi/STRATEGY.md)** — 双 SDK 架构的设计初衷、选型建议与长期路线图

绝大多数使用场景下推荐使用官方 `ogx_client` SDK。对于有特定 OpenAPI 工具链需求的技术团队，`ogx_open_client` 则提供了更切合的替代选择。

## 生态社区

我们于每周四上午 09:00 PST（北京时间周五凌晨 01:00 / 02:00）定期举行社区交流会（Community Calls）——详情请查看 [Discord 社区活动通知](https://discord.gg/bUYRqEvK6)。

[![Star 增长历史趋势](https://api.star-history.com/svg?repos=ogx-ai/ogx&type=Date)](https://www.star-history.com/#ogx-ai/ogx&Date)

由衷感谢所有为 OGX 做出卓越贡献的开发者们！

<a href="https://github.com/ogx-ai/ogx/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ogx-ai/ogx" alt="OGX 贡献者列表" />
</a>

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月13日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
