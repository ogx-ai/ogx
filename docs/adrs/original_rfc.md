# The OGX API

<!-- markdownlint-disable-next-line MD036 -->
*Originally authored Jul 23, 2024*

**Authors:**

* Meta: @raghotham, @ashwinb, @hjshah, @jspisak

## Summary

As part of the Llama 3.1 release, Meta is releasing an RFC for ‘OGX’, a comprehensive set of interfaces / API for ML developers building on top of Llama foundation models. We are looking for feedback on where the API can be improved, any corner cases we may have missed and your general thoughts on how useful this will be. Ultimately, our hope is to create a standard for working with Llama models in order to simplify the developer experience and foster innovation across the Llama ecosystem.

## Motivation

Llama models were always intended to work as part of an overall system that can orchestrate several components, including calling external tools. Our vision is to go beyond the foundation models and give developers access to a broader system that gives them the flexibility to design and create custom offerings that align with their vision. This thinking started last year when we first introduced a system-level safety model. Meta has continued to release new components for orchestration at the system level and, most recently in Llama 3.1, we’ve introduced the Llama Guard 3 safety model that is multilingual, a prompt injection filter, Prompt Guard and refreshed v3 of our CyberSec Evals. We are also releasing a reference implementation of an agentic system to demonstrate how all the pieces fit together.

While building the reference implementation, we realized that having a clean and consistent way to interface between components could be valuable not only for us but for anyone leveraging Llama models and other components as part of their system. We’ve also heard from the community as they face a similar challenge as components exist with overlapping functionality and there are incompatible interfaces and yet don't cover the end-to-end model life cycle.

With these motivations, we engaged folks in industry, startups, and the broader developer community to help better define the interfaces of these components. We’re releasing this OGX RFC as a set of standardized and opinionated interfaces for how to surface canonical toolchain components (like inference, fine-tuning, evals, synthetic data generation) and agentic applications to ML developers. Our hope is to have these become well adopted across the ecosystem, which should help with easier interoperability. We would like for builders of multiple components to provide implementations to these standard APIs so that there can be vertically integrated “distributions” of the OGX that can work out of the box easily.

We welcome feedback and ways to improve the proposal. We’re excited to grow the ecosystem around Llama and lower barriers for both developers and platform providers.

## Design decisions

Meta releases weights of both the pretrained and instruction fine-tuned Llama models to support several use cases. These weights can be improved  -  fine tuned and aligned - with curated datasets to then be deployed for inference to support specific applications. The curated datasets can be produced manually by humans or synthetically by other models or by leveraging human feedback by collecting usage data of the application itself. This results in a continuous improvement cycle where the model gets better over time. This is the model life cycle.

### Model Lifecycle

![Figure 1: Model Life Cycle](resources/model-lifecycle.png)

For each of the operations that need to be performed (e.g. fine tuning, inference, evals etc) during the model life cycle, we identified the capabilities as toolchain APIs that are needed. Some of these capabilities are primitive operations like inference while other capabilities like synthetic data generation are composed of other capabilities. The list of APIs we have identified to support the lifecycle of Llama models is below:

* /datasets - to support creating training and evaluation data sets
* /post_training - to support creating and managing supervised finetuning (SFT) or preference optimization jobs
* /evaluations - to support creating and managing evaluations for capabilities like question answering, summarization, or text - generation
* /synthetic_data_generation - to support generating synthetic data using data generation model and a reward model
* /reward_scoring - to support synthetic data generation
* /inference - to support serving the models for applications

### Agentic System

![Figure 2: Agentic System](resources/agentic-system.png)

In addition to the model lifecycle, we considered the different components involved in an agentic system. Specifically around tool calling and shields. Since the model may decide to call tools, a single model inference call is not enough. What’s needed is an agentic loop consisting of tool calls and inference. The model provides separate tokens representing end-of-message and end-of-turn. A message represents a possible stopping point for execution where the model can inform the execution environment that a tool call needs to be made. The execution environment, upon execution, adds back the result to the context window and makes another inference call. This process can get repeated until an end-of-turn token is generated.
Note that as of today, in the OSS world, such a “loop” is often coded explicitly via elaborate prompt engineering using a ReAct pattern (typically) or preconstructed execution graph. Llama 3.1 (and future Llamas) attempts to absorb this multi-step reasoning loop inside the main model itself.

**Let's consider an example:**

1. The user asks the system "Who played the NBA finals last year?"
1. The model "understands" that this question needs to be answered using web search. It answers this abstractly with a message of the form "Please call the search tool for me with the query: 'List finalist teams for NBA in the last year' ". Note that the model by itself does not call the tool (of course!)
1. The executor consults the set of tool implementations which have been configured by the developer to find an implementation for the "search tool". If it does not find it, it returns an error to the model. Otherwise, it executes this tool and returns the result of this tool back to the model.
1. The model reasons once again (using all the messages above) and decides to send a final response "In 2023, Denver Nuggets played against the Miami Heat in the NBA finals." to the executor
1. The executor returns the response directly to the user (since there is no tool call to be executed.)

The sequence diagram that details the steps is in the [agentic system docs](https://github.com/meta-llama/llama-agentic-system/blob/main/docs/sequence-diagram.md).

* /memory_banks - to support creating multiple repositories of data that can be available for agentic systems
* /agentic_system - to support creating and running agentic systems. The sub-APIs support the creation and management of the steps, turns, and sessions within agentic applications.
  * /step - there can be inference, memory retrieval, tool call, or shield call steps
  * /turn - each turn begins with a user message and results in a loop consisting of multiple steps, followed by a response back to the user
  * /session - each session consists of multiple turns that the model is reasoning over
  * /memory_bank - a memory bank allows for the agentic system to perform retrieval augmented generation

## OGX API/CLI

We define the OGX as a layer cake shown below.

![Figure 3: OGX](resources/ogx.png)

The API is defined in the [YAML](_static/ogx-spec.yaml) and [HTML](_static/ogx-spec.html) files.

## Sample implementations

To prove out the API, we implemented a handful of use cases to make things more concrete. The [ogx-apps](https://github.com/ogx-ai/ogx-apps) repository contains [6 different examples](https://github.com/ogx-ai/ogx-apps/tree/main/examples) ranging from very basic to a multi turn agent.

There is also a sample inference endpoint implementation in the [ogx](https://github.com/ogx-ai/ogx/blob/main/ogx.core/server/server.py) repository.

## Limitations

The reference implementation for OGX APIs to date only includes sample implementations using the inference API. We are planning to flesh out the design of OGX Distributions (distros) by combining capabilities from different providers into a single vertically integrated stack. We plan to implement other APIs and, of course, we’d love contributions!!

Thank you in advance for your feedback, support and contributions to make this a better API.

Cheers!
