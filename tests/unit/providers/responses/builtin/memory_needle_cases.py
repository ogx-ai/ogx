# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from dataclasses import dataclass

from ogx_api import (
    ConversationItem,
    OpenAIAssistantMessageParam,
    OpenAIMessageParam,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAISystemMessageParam,
    OpenAIUserMessageParam,
)


@dataclass(frozen=True)
class StoredResponsesConversation:
    conversation_id: str
    response_id: str
    items: list[ConversationItem]
    messages_for_summary: list[OpenAIMessageParam]


@dataclass(frozen=True)
class DerivedMemoryArtifact:
    summary: str
    omitted_detail: str


@dataclass(frozen=True)
class MemoryNeedleCase:
    retrieval_prompt: str
    conversation: StoredResponsesConversation
    memory_artifact: DerivedMemoryArtifact


ConversationSpec = tuple[str, str, str, str, str, str, str, str]


_MEMORY_NEEDLE_SPECS: list[ConversationSpec] = [
    (
        "cutover-window-myrtle-1407",
        "What was the cutover window code for the identity rollout migration?",
        "The user is planning an identity rollout migration. They care about staged rollout, audit logging, "
        "and tenant-scoped rollback safety.",
        "You help capture durable launch decisions for identity infrastructure work.",
        "For the identity rollout migration, keep the cutover window code `cutover-window-myrtle-1407`.",
        "Recorded the identity rollout migration cutover window code and staged rollout safety.",
        "Also remember that tenant admins need audit log parity before we enable the second cohort.",
        "I will preserve audit log parity as a rollout requirement.",
    ),
    (
        "export-bucket-harbor-5821",
        "Which export bucket did we pick for the billing dashboard export?",
        "The billing dashboard export discussion settled on CSV first, delayed XLSX support, "
        "and finance-only access while the dashboard is in beta.",
        "You retain product analytics decisions for finance workflows.",
        "For billing dashboard export, the export bucket is `export-bucket-harbor-5821`.",
        "Noted the billing export bucket and beta access constraint.",
        "Keep XLSX out of scope until finance signs off on the CSV field order.",
        "I will remember CSV field order approval before XLSX support.",
    ),
    (
        "reservation-code-orchid-9033",
        "What reservation code was chosen during GPU capacity planning?",
        "GPU capacity planning focused on batch inference isolation, evening demand spikes, "
        "and avoiding preemption for premium workloads.",
        "You track infrastructure planning decisions for compute capacity.",
        "During GPU capacity planning, the reservation code is `reservation-code-orchid-9033`.",
        "I have the GPU reservation context and premium workload constraint.",
        "The important tradeoff is preserving batch isolation even if utilization dips.",
        "I will keep batch isolation as the planning preference.",
    ),
    (
        "retro-doc-copper-7142",
        "Which retro doc anchor did we use for the incident review workflow?",
        "The incident review workflow should prioritize action owners, timeline clarity, "
        "and follow-up tracking over blame-oriented language.",
        "You preserve process decisions from reliability reviews.",
        "For the incident review workflow, the retro doc anchor is `retro-doc-copper-7142`.",
        "I will remember the incident review workflow and retro documentation anchor.",
        "Make sure action owners matter more than narrative polish.",
        "Action ownership will be treated as durable context.",
    ),
    (
        "changelog-sentinel-2680",
        "What sentinel did we pick for release note automation?",
        "Release note automation should group changes by user-visible impact, keep internal chores out, "
        "and leave final wording editable by maintainers.",
        "You retain release process decisions for developer tooling.",
        "For release note automation, remember changelog sentinel `changelog-sentinel-2680`.",
        "I have the release note automation sentinel.",
        "Do not include dependency churn unless it affects users directly.",
        "I will preserve that release note exclusion rule.",
    ),
    (
        "partition-salt-maple-4419",
        "Which partition salt did we settle on for warehouse partition tuning?",
        "Warehouse partition tuning centered on reducing skew for customer events, keeping backfills bounded, "
        "and preserving dashboard latency.",
        "You remember data warehouse tuning decisions.",
        "For warehouse partition tuning, keep partition salt `partition-salt-maple-4419`.",
        "I will associate that partition salt with skew reduction.",
        "Backfills should be capped at one week per job until we confirm planner behavior.",
        "I will keep the one-week backfill cap with the tuning notes.",
    ),
    (
        "cohort-pin-cedar-1095",
        "What cohort pin did we use for the mobile onboarding experiment?",
        "The mobile onboarding experiment should compare checklist-first onboarding against guided setup, "
        "with guardrails around notification prompts.",
        "You retain mobile product experiment context.",
        "For the mobile onboarding experiment, the cohort pin is `cohort-pin-cedar-1095`.",
        "I will remember the onboarding experiment cohort pin.",
        "The first measurement should be activation by day two, not day seven retention.",
        "Activation by day two will remain the primary measurement.",
    ),
    (
        "compat-header-lilac-7730",
        "Which compatibility header did we decide for partner API versioning?",
        "Partner API versioning should support a compatibility window, explicit migration dates, "
        "and partner-specific rollout communications.",
        "You capture partner platform API decisions.",
        "For partner API versioning, remember compatibility header `compat-header-lilac-7730`.",
        "I will retain the partner API compatibility header.",
        "The beta partners need migration dates before we update the public quickstart.",
        "I will preserve that sequencing constraint.",
    ),
    (
        "judgment-set-onyx-6154",
        "What judgment set did we choose for search ranking evaluation?",
        "Search ranking evaluation should blend editorial judgments with click data, "
        "but avoid optimizing solely for navigational queries.",
        "You retain search quality evaluation decisions.",
        "For search ranking evaluation, the judgment set is `judgment-set-onyx-6154`.",
        "I will remember the search ranking judgment set.",
        "Treat navigational queries as a separate slice so they do not dominate the score.",
        "I will keep navigational queries separated in evaluation.",
    ),
    (
        "sample-rate-pebble-3376",
        "What sample-rate key did we discuss for observability sampling?",
        "Observability sampling policy should retain all errors, sample successful traces by tier, "
        "and keep audit events complete.",
        "You capture observability and telemetry policy decisions.",
        "For observability sampling policy, remember sample-rate key `sample-rate-pebble-3376`.",
        "I will retain the sampling key and error-retention requirement.",
        "Never sample audit events, even in development environments.",
        "Audit events will stay complete in the stored policy context.",
    ),
    (
        "retention-ticket-iris-8840",
        "Which ticket id tracked dataset retention cleanup?",
        "Dataset retention cleanup should start with expired experiment data, preserve legal holds, "
        "and publish a dry-run report before deletion.",
        "You retain data governance cleanup decisions.",
        "For dataset retention cleanup, the tracking ticket is `retention-ticket-iris-8840`.",
        "I will remember the retention cleanup ticket.",
        "Legal holds override every automated cleanup rule.",
        "I will preserve legal hold precedence.",
    ),
    (
        "schema-freeze-basil-5208",
        "What schema freeze marker came up in the compatibility audit?",
        "Schema compatibility audit work should check additive changes, nullable migrations, "
        "and generated client behavior before release.",
        "You remember API schema governance choices.",
        "For the schema compatibility audit, remember freeze marker `schema-freeze-basil-5208`.",
        "I will retain the schema freeze marker.",
        "Generated clients need a smoke test before we tag the release.",
        "I will keep generated client smoke tests in the audit context.",
    ),
    (
        "dns-drain-cobalt-2469",
        "What DNS drain code did we pick for the regional failover exercise?",
        "Regional failover exercise planning should test DNS drain timing, customer notification paths, "
        "and data-plane recovery metrics.",
        "You retain disaster recovery exercise context.",
        "For the regional failover exercise, remember DNS drain code `dns-drain-cobalt-2469`.",
        "I will remember the DNS drain code for failover planning.",
        "Customer notifications should be rehearsed but not sent during the tabletop.",
        "I will keep tabletop notification rehearsal scoped correctly.",
    ),
    (
        "rollback-key-heliotrope-7421",
        "What rollback key did we choose for authentication migration rollback?",
        "Authentication migration rollback planning should preserve session continuity, "
        "stage the old provider fallback, and keep support escalation ready.",
        "You retain authentication migration and rollback decisions.",
        "For authentication migration rollback, remember rollback key `rollback-key-heliotrope-7421`.",
        "I will retain the authentication rollback key.",
        "Support needs a canned response before we start the second migration batch.",
        "I will keep support readiness tied to the second batch.",
    ),
    (
        "flag-owner-grove-6712",
        "Which flag owner token did we decide for feature flag governance?",
        "Feature flag governance should assign explicit owners, expiration dates, and cleanup checks "
        "for flags that affect checkout behavior.",
        "You remember release governance and flag ownership decisions.",
        "For feature flag governance, remember flag owner token `flag-owner-grove-6712`.",
        "I will keep the feature flag owner token in memory.",
        "Checkout-impacting flags should have a two-week review even if the launch is delayed.",
        "I will retain the checkout flag review window.",
    ),
    (
        "quota-ledger-ember-4571",
        "What quota ledger id was chosen for the quota enforcement rollout?",
        "Quota enforcement rollout should begin in report-only mode, give account teams previews, "
        "and avoid blocking existing enterprise contracts.",
        "You capture rollout decisions for billing and quota systems.",
        "For quota enforcement rollout, remember quota ledger id `quota-ledger-ember-4571`.",
        "I will remember the quota ledger id and report-only start.",
        "Enterprise contracts should get warnings first, not hard blocks.",
        "I will preserve the warning-before-blocking constraint.",
    ),
    (
        "sandbox-image-amber-3320",
        "Which image tag did we pick for sandbox provisioning?",
        "Sandbox provisioning should use pre-warmed images, rotate credentials per workspace, "
        "and expose cleanup status in admin views.",
        "You retain developer sandbox provisioning decisions.",
        "For sandbox provisioning flow, remember image tag `sandbox-image-amber-3320`.",
        "I will retain the sandbox image tag.",
        "Credential rotation must happen before the workspace is marked ready.",
        "I will keep credential rotation as a readiness gate.",
    ),
    (
        "pager-lane-frost-9186",
        "What pager lane did we use for support escalation triage?",
        "Support escalation triage should separate billing, authentication, and infrastructure incidents, "
        "with fast handoff to engineering only for confirmed platform regressions.",
        "You remember support process decisions.",
        "For support escalation triage, remember pager lane `pager-lane-frost-9186`.",
        "I will remember the support escalation pager lane.",
        "Only confirmed platform regressions should bypass the support lead.",
        "I will retain that escalation threshold.",
    ),
    (
        "routing-snapshot-opal-8893",
        "Which routing snapshot was used for model routing calibration?",
        "Model routing calibration should compare latency, tool-call behavior, and refusal rates "
        "before changing the default route.",
        "You retain model-routing evaluation decisions.",
        "For model routing calibration, remember routing snapshot `routing-snapshot-opal-8893`.",
        "I will keep the model routing snapshot available.",
        "Do not promote a route if tool-call behavior regresses, even when latency improves.",
        "I will preserve tool-call behavior as a promotion gate.",
    ),
    (
        "publish-token-jade-4736",
        "What publish token did we discuss for the docs publishing checklist?",
        "Docs publishing checklist work should validate generated examples, preview links, "
        "and redirects before the docs release goes live.",
        "You remember documentation release process details.",
        "For docs publishing checklist, remember publish token `publish-token-jade-4736`.",
        "I will remember the docs publish token and preview requirement.",
        "Generated examples should be checked before redirects are merged.",
        "I will keep generated example validation before redirect merging.",
    ),
]


def build_memory_needle_cases() -> list[MemoryNeedleCase]:
    return [_memory_needle_case(index, *spec) for index, spec in enumerate(_MEMORY_NEEDLE_SPECS)]


def conversation_item_text(item: OpenAIResponseMessage) -> str:
    if isinstance(item.content, str):
        return item.content

    text_segments: list[str] = []
    for content_part in item.content:
        text = getattr(content_part, "text", None)
        if isinstance(text, str) and text:
            text_segments.append(text)
    return "\n".join(text_segments)


def _memory_needle_case(
    index: int,
    omitted_detail: str,
    retrieval_prompt: str,
    summary: str,
    system_message: str,
    first_user_message: str,
    first_assistant_message: str,
    second_user_message: str,
    second_assistant_message: str,
) -> MemoryNeedleCase:
    items = [
        _conversation_message("system", system_message),
        _conversation_message("user", first_user_message),
        _conversation_message("assistant", first_assistant_message),
        _conversation_message("user", second_user_message),
        _conversation_message("assistant", second_assistant_message),
    ]
    return MemoryNeedleCase(
        retrieval_prompt=retrieval_prompt,
        conversation=StoredResponsesConversation(
            conversation_id=f"conv_memory_{index:02d}",
            response_id=f"resp_memory_{index:02d}",
            items=items,
            messages_for_summary=_chat_messages_from_conversation_items(items),
        ),
        memory_artifact=DerivedMemoryArtifact(
            summary=summary,
            omitted_detail=omitted_detail,
        ),
    )


def _conversation_message(role: str, text: str) -> OpenAIResponseMessage:
    if role == "assistant":
        content = [OpenAIResponseOutputMessageContentOutputText(text=text)]
    else:
        content = [OpenAIResponseInputMessageContentText(text=text)]
    return OpenAIResponseMessage(role=role, content=content, status="completed")


def _chat_messages_from_conversation_items(items: list[ConversationItem]) -> list[OpenAIMessageParam]:
    messages: list[OpenAIMessageParam] = []
    for item in items:
        if not isinstance(item, OpenAIResponseMessage):
            continue
        content = conversation_item_text(item)
        if item.role == "system":
            messages.append(OpenAISystemMessageParam(content=content))
        elif item.role == "user":
            messages.append(OpenAIUserMessageParam(content=content))
        elif item.role == "assistant":
            messages.append(OpenAIAssistantMessageParam(content=content))
    return messages
