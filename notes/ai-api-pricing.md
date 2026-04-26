# 💸 AI Model API Pricing: A Reference

A practical guide to reading and understanding LLM API costs — tokens, caching, batching, and provider comparisons.

---

## 🔢 What is a Token?

A *token* is the atomic unit models process — not a word, not a character, but something in between. Roughly:

- **~4 characters** of English text ≈ 1 token
- **~0.75 words** ≈ 1 token
- Common short words (`the`, `is`, `a`) → 1 token each
- Rare or long words may be split across 2–3 tokens
- Numbers and punctuation each often count as 1 token

> [!EXAMPLE] Rough sense of scale
> "The quick brown fox jumps over the lazy dog." ≈ **11 tokens**
> A 1,000-word essay ≈ **~1,300 tokens**
> This entire note ≈ **~2,000 tokens**

The tokenization scheme varies by model family (Anthropic uses its own tokenizer; OpenAI uses tiktoken; Opus 4.7 uses a newer tokenizer that may consume up to 35% more tokens for the same text than earlier Claude models).

---

## 💰 Pricing Per Million Tokens

Individual token costs are tiny fractions of a cent, so APIs denominate in **MTok (million tokens)**. The conversion:

$$
\text{cost} = \frac{\text{tokens}}{10^6} \times \text{price per MTok}
$$

At Sonnet 4.6's rate of **$3/MTok input**:

| Tokens | Cost |
|--------|------|
| 1,000 (a short request) | $0.000003 = 0.03¢ |
| 100,000 (large document) | $0.30 |
| 1,000,000 (1M tokens) | $3.00 |

This is why pricing feels "cheap per call" but adds up at scale — 10,000 API calls with a 1,000-token prompt costs $0.03 total in input, but 10,000 calls with a 50,000-token system prompt costs $1,500.

---

## ↕️ Input vs Output Tokens

Every API call has two cost components:

**Input tokens** — everything you *send* to the model:
- System prompt
- Conversation history (all prior turns)
- The current user message
- Tool definitions (JSON schemas for function calling)
- Any documents or context you inject

**Output tokens** — everything the model *generates*:
- The assistant's response text
- Any tool calls/function invocations it emits

> [!INFO] Why output costs more
> Input tokens are processed in one parallel forward pass — the attention mechanism attends over the entire prompt simultaneously. Output tokens are generated *autoregressively*: each new token requires a full forward pass through the model. More compute per output token → consistently higher price.

Output is typically **5x the input price** on Claude models (e.g. $3 input / $15 output for Sonnet 4.6).

---

## 🗄️ Prompt Caching

The single highest-leverage cost optimization for repeated or long contexts.

### The Problem Without Caching

If your system prompt is 10,000 tokens and you make 1,000 API calls, you pay for **10,000 × 1,000 = 10M input tokens** just to re-send the same prompt over and over. At $3/MTok, that's $30 wasted.

### How Caching Works

Instead of reprocessing the same prefix on every request, the server stores the *KV cache* (the internal attention state) from a prior call and reuses it. You must opt in by marking content with `cache_control`.

There are two durations:

| Operation | Multiplier | What you pay | Duration |
|-----------|-----------|-------------|----------|
| **5-min cache write** | 1.25× base input | Slightly more than uncached, to store | 5 minutes |
| **1-hour cache write** | 2.0× base input | Double, to store for longer | 1 hour |
| **Cache read (hit)** | 0.1× base input | 10% of standard — the big saving | Same as write duration |

For Sonnet 4.6 ($3/MTok base input):

| Operation | Price |
|-----------|-------|
| Uncached input | $3.00/MTok |
| 5-min cache write | $3.75/MTok |
| 1-hour cache write | $6.00/MTok |
| Cache hit (read) | $0.30/MTok |

> [!TIP] Break-even analysis
> **5-minute cache:** You write at 1.25× and read at 0.1×. After 1 cache hit the net is (1.25 + 0.1) = 1.35× vs 2× for two uncached calls — you break even after **just 1 hit**.
>
> **1-hour cache:** Write at 2×, read at 0.1×. You need **at least 2 hits** to beat repeated uncached calls (2 + 0.1 + 0.1 = 2.2× vs 3× for three uncached calls). Very high ROI if you're making many calls within the hour.

> [!WARNING] Cache misses
> The cache is keyed on the *exact* prefix. Any change to the beginning of your prompt (including system prompt edits) invalidates the cache. Place stable, large content (system prompts, reference documents) early in the prompt; dynamic content (user message, current date) at the end.

### Automatic vs Explicit Caching

- **Automatic (`cache_control` at request level):** Anthropic manages cache breakpoints as the conversation grows. Simplest to use.
- **Explicit (per content block):** You tag specific blocks for caching — fine-grained control, better for complex multi-document prompts.

---

## ⚡ Batch API: 50% Off for Async

If you don't need a real-time response (e.g. bulk processing, overnight jobs), the Batch API halves your costs:

| Model | Realtime input | Batch input | Realtime output | Batch output |
|-------|---------------|-------------|----------------|--------------|
| Haiku 4.5 | $1.00/MTok | $0.50/MTok | $5.00/MTok | $2.50/MTok |
| Sonnet 4.6 | $3.00/MTok | $1.50/MTok | $15.00/MTok | $7.50/MTok |
| Opus 4.7 | $5.00/MTok | $2.50/MTok | $25.00/MTok | $12.50/MTok |

Batches complete within 24 hours. Prompt caching discounts stack on top of the batch discount.

---

## 📊 Full Anthropic Pricing Table (Direct API)

*All prices in USD per MTok. Source: [platform.claude.com/docs](https://platform.claude.com/docs/en/about-claude/pricing), as of April 2026.*

| Model | Input | 5-min write | 1-hr write | Cache read | Output |
|-------|-------|------------|-----------|-----------|--------|
| Haiku 3 | $0.25 | $0.30 | $0.50 | $0.03 | $1.25 |
| Haiku 3.5 | $0.80 | $1.00 | $1.60 | $0.08 | $4.00 |
| Haiku 4.5 | $1.00 | $1.25 | $2.00 | $0.10 | $5.00 |
| Sonnet 4 / 4.5 / 4.6 | $3.00 | $3.75 | $6.00 | $0.30 | $15.00 |
| Opus 4.5 / 4.6 / 4.7 | $5.00 | $6.25 | $10.00 | $0.50 | $25.00 |
| Opus 4 / 4.1 *(legacy)* | $15.00 | $18.75 | $30.00 | $1.50 | $75.00 |

> [!NOTE] Output-to-input ratio
> Across all current models the output rate is exactly **5× the input rate**. This is a stable design choice — plan budgets accordingly.

---

## 🌐 Third-Party Providers: DeepInfra

DeepInfra hosts both open-source models and API-resold proprietary models (Claude, Gemini). For proprietary models they apply a markup over Anthropic's direct pricing.

*Source: [deepinfra.com/pricing](https://deepinfra.com/pricing), as of April 2026.*

| Model | Input | Cached input | Output | vs Anthropic |
|-------|-------|------------|-------|-------------|
| claude-4-sonnet | $3.30 | — | $16.50 | +10% markup |
| claude-3-7-sonnet | $3.30 | $0.33 | $16.50 | +10% markup |
| claude-4-opus | $16.50 | — | $82.50 | priced as legacy Opus + markup |
| DeepSeek-V3.2 | $0.26 | $0.13 | $0.38 | open-source |
| DeepSeek-R1-0528 | $0.50 | $0.35 | $2.15 | open-source reasoning |
| Llama 3.1 8B | $0.02 | — | $0.05 | open-source, tiny |
| Gemini 2.5 Pro | $1.25 | — | $10.00 | via DeepInfra |
| Gemini 2.5 Flash | $0.30 | — | $2.50 | via DeepInfra |

> [!WARNING] Provider markup
> DeepInfra's Claude pricing is consistently ~10% above Anthropic's direct rates. For prompt caching, they only surface cache reads (not both write tiers), and coverage varies by model. If you're already an Anthropic API customer, direct access is cheaper; DeepInfra is useful for unified multi-provider billing or for accessing open-source models alongside Claude.

> [!INFO] Why open-source models are so cheap
> Open-source models like Llama and DeepSeek are hosted on shared GPU clusters at commodity rates. DeepInfra charges for GPU-hours, not model-level margins — hence Llama 3.1 8B at $0.02/MTok vs Sonnet 4.6 at $3.00/MTok, a **150× difference**.

---

## 🔑 Key Rules of Thumb

| Situation | Guideline |
|-----------|-----------|
| Estimating cost | tokens ÷ 1,000,000 × price; rough estimate: 1 token ≈ ¾ of a word |
| Input vs output budget | Output tokens cost 5× input; if your response is long, output dominates |
| Should I cache? | Yes, if the same prefix appears in >2 consecutive calls |
| Which cache tier? | 5-min if calls cluster in short bursts; 1-hr for long sessions or pipelines |
| Real-time vs batch | Use Batch API for any workload that tolerates ~1 hour latency — 50% savings |
| Direct vs third-party | Direct Anthropic API is cheaper for Claude; third-party useful for multi-model access |

---

## 📚 References

| Reference | Brief Summary | Link |
|-----------|--------------|------|
| Anthropic Pricing Docs | Full pricing tables for all Claude models, prompt caching, batch, tool use, and managed agents | [platform.claude.com/docs/en/about-claude/pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| DeepInfra Pricing | Per-token pricing for open-source and proprietary models hosted on DeepInfra | [deepinfra.com/pricing](https://deepinfra.com/pricing) |
| Anthropic Prompt Caching Guide | Implementation details, code examples, and supported models for prompt caching | [platform.claude.com/docs/en/build-with-claude/prompt-caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) |
