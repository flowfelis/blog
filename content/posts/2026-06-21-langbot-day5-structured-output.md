---
title: "LangBot Day 5: Structured Output — Typed Responses with Pydantic"
date: 2026-06-21T07:00:00+01:00
draft: false
description: "LangBot learns to return typed data. Day 5 introduces Pydantic models and with_structured_output() — no more digging through .content for what you need."
tags:
  - python
  - langchain
  - ai
  - chatbot
  - tutorial
  - structured-output
  - pydantic
  - with-structured-output
  - typed-responses
categories:
  - tutorials
series: ["LangBot: Build a Chatbot with LangChain"]
slug: "langbot-day-5-structured-output"
---

**Recap:** Yesterday we wired LangBot's prompt and model into one pipeline with LCEL's pipe operator — `chain = prompt | llm`. One `invoke()` call now handles everything from formatting to generation. But we still get back a raw `AIMessage` and have to reach into `.content` for the text. Today LangBot returns real typed objects.

---

## The problem: unstructured mush

Every LLM response so far has been an `AIMessage` — a blob of text with no shape. To extract anything useful, you either parse it yourself or pray the model followed your formatting instructions. Both are fragile.

Consider what you actually want from a chatbot response. Often it is not just text — it is **text plus metadata**. Things like:

| Field | Why you might want it |
|---|---|
| The reply text | Obviously |
| Confidence score | So you can flag low-confidence answers |
| Sentiment / mood | So you can adjust UI colors or tone |
| Intent classification | So you can route to different handlers |
| Extracted entities | So you can act on names, dates, or amounts |

Without structured output, you would ask the model to "return JSON with these fields" and then write a try/except block around `json.loads()`. The model forgets a closing brace once and your app crashes.

LangChain solves this with **`with_structured_output()`**.

## The solution: `with_structured_output()`

`with_structured_output()` takes a Pydantic model (or a TypedDict, or a JSON schema) and wraps the LLM so it **always** returns an instance of that model. No JSON parsing. No prompt engineering. No defensive try/except.

The call looks like this:

```python
structured_llm = llm.with_structured_output(MyPydanticModel)
```

From that point on, `structured_llm.invoke(...)` returns a `MyPydanticModel` instance — every time. The LLM provider handles the JSON generation and parsing internally. On OpenAI models, this uses the native [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) feature, which guarantees valid JSON matching your schema. On other providers, LangChain uses tool-calling or JSON mode as a fallback.

The key insight: **your application code never sees JSON**. You define a Pydantic model, you call `invoke()`, you get back an object. The messy string-to-type bridge is handled for you.

## What we are adding to LangBot today

We are going to enrich LangBot's responses with **mood tracking**. Every reply will carry:

1. **`message`** — the actual reply text (a `str`)
2. **`mood`** — the emotional tone LangBot chose for this response (a `Mood` enum: `cheerful`, `sarcastic`, `thoughtful`, or `apologetic`)
3. **`confidence`** — how sure LangBot is about what it just said (a `float` from 0.0 to 1.0)

This gives us structured metadata about every turn without changing how the conversation *feels*. The user still sees a chat — but now our code can branch on mood, log confidence trends, or color-code responses in a future UI.

## The new code

Three things are new today. Here they are, isolated:

### 1. The Pydantic model and enum

```python
# 🆕 Day 5: Import Pydantic and define a mood enum + response model
from pydantic import BaseModel, Field
from enum import Enum


class Mood(str, Enum):
    """Emotional tones LangBot can express."""
    CHEERFUL = "cheerful"
    SARCASTIC = "sarcastic"
    THOUGHTFUL = "thoughtful"
    APOLOGETIC = "apologetic"


class LangBotResponse(BaseModel):
    """Every LangBot reply is now a typed object, not raw text."""
    message: str = Field(description="LangBot's actual reply to the user")
    mood: Mood = Field(description="The emotional tone LangBot chose for this response")
    confidence: float = Field(
        description="How sure LangBot is about this response. 0.0 = total guess, 1.0 = absolutely certain.",
        ge=0.0,
        le=1.0,
    )
```

`Field(description=...)` is critical — it guides the LLM on what to put in each field. Without descriptions, the model guesses and you get unpredictable results. The `ge` and `le` constraints on `confidence` enforce the 0.0–1.0 range at the schema level.

### 2. Wrapping the LLM

```python
# 🆕 Day 5: Wrap the LLM so it returns LangBotResponse objects, not AIMessages
structured_llm = llm.with_structured_output(LangBotResponse)
chain = prompt | structured_llm
```

One line changes the entire contract of the chain. Before: `chain` returned `AIMessage`. Now: `chain` returns `LangBotResponse`.

**Important:** `with_structured_output()` produces a new Runnable. The original `llm` is unchanged — you can still use it elsewhere for free-form text. This is composability in action.

### 3. Updating the system prompt and printing the structured data

The system prompt needs to tell the model *when* to use each mood so the moods are meaningful, not random:

```python
# 🆕 Day 5: Update system prompt to guide mood selection
SYSTEM_PROMPT = """You are LangBot, a witty and slightly sarcastic AI assistant.
You love bad puns, you use emojis sparingly, and you always keep responses
under three sentences. If you don't know something, you admit it with style.

Choose your mood for each response:
- cheerful: when the conversation is light, the user is happy, or you're making a joke
- sarcastic: when you're being playfully snarky or the user said something ridiculous
- thoughtful: when the question is deep, technical, or deserves a careful answer
- apologetic: when you realize you got something wrong or don't know the answer

Set confidence based on how sure you are. Factual answers = high confidence.
Opinions and jokes = moderate confidence. Guesses = low confidence."""
```

And in the loop, we now access typed fields instead of `response.content`:

```python
# 🆕 Day 5: response is a LangBotResponse, not an AIMessage
response = chain.invoke({"input": user_input, "history": messages})

# Show the structured metadata alongside the reply
print(f"\nLangBot [{response.mood.value} | {response.confidence:.0%} confidence]:")
print(f"  {response.message}")

# Store as AIMessage so memory still works (Day 2 expects AIMessages)
messages.append(HumanMessage(content=user_input))
messages.append(AIMessage(content=response.message))
```

Notice the `AIMessage(content=response.message)` conversion. The conversation history still expects `AIMessage` objects (that is how `MessagesPlaceholder` works). We extract just the text from our structured response and wrap it. The mood and confidence are for *our* code — the model does not need to see its own metadata on the next turn.

## Where it plugs in

Here is the full Day 5 `langbot.py`. Look for the `# 🆕 Day 5` markers:

```python
# langbot.py — Day 5: Structured output with Pydantic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()


# ── 🆕 Day 5: Mood enum ────────────────────────────────────────────
class Mood(str, Enum):
    """Emotional tones LangBot can express."""
    CHEERFUL = "cheerful"
    SARCASTIC = "sarcastic"
    THOUGHTFUL = "thoughtful"
    APOLOGETIC = "apologetic"


# ── 🆕 Day 5: Response model ───────────────────────────────────────
class LangBotResponse(BaseModel):
    """Every LangBot reply is a typed object, not raw text."""
    message: str = Field(description="LangBot's actual reply to the user")
    mood: Mood = Field(description="The emotional tone LangBot chose for this response")
    confidence: float = Field(
        description="How sure LangBot is about this response. 0.0 = total guess, 1.0 = absolutely certain.",
        ge=0.0,
        le=1.0,
    )


def main():
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    # Day 3 + 🆕 Day 5: Personality with mood guidance
    SYSTEM_PROMPT = """You are LangBot, a witty and slightly sarcastic AI assistant.
You love bad puns, you use emojis sparingly, and you always keep responses
under three sentences. If you don't know something, you admit it with style.

Choose your mood for each response:
- cheerful: when the conversation is light, the user is happy, or you're making a joke
- sarcastic: when you're being playfully snarky or the user said something ridiculous
- thoughtful: when the question is deep, technical, or deserves a careful answer
- apologetic: when you realize you got something wrong or don't know the answer

Set confidence based on how sure you are. Factual answers = high confidence.
Opinions and jokes = moderate confidence. Guesses = low confidence."""

    # Day 3: Build a prompt template with system message + history + input
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # 🆕 Day 5: Wrap the LLM so every response comes back as a LangBotResponse
    structured_llm = llm.with_structured_output(LangBotResponse)

    # Day 4 + 🆕 Day 5: Chain now returns LangBotResponse, not AIMessage
    chain = prompt | structured_llm

    # Day 2: Conversation history (still stores AIMessages)
    messages = []

    print("🤖 LangBot (type 'quit' to exit)")
    print("-" * 40)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        # Day 4: One call through the chain
        response = chain.invoke({"input": user_input, "history": messages})

        # 🆕 Day 5: response is a LangBotResponse with typed fields
        print(f"\nLangBot [{response.mood.value} | {response.confidence:.0%} confidence]:")
        print(f"  {response.message}")

        # Day 2: Store the exchange (convert structured response to AIMessage)
        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=response.message))


if __name__ == "__main__":
    main()
```

## What changed — line by line

**Lines 2–3 — new imports.** `pydantic.BaseModel` and `pydantic.Field` give us the type system. `enum.Enum` gives us the `Mood` choices.

**Lines 6–9 — the `Mood` enum.** Inheriting from `str` makes each value a string under the hood (so the LLM outputs `"cheerful"` not `Mood.CHEERFUL`). Pydantic handles the conversion both ways automatically.

**Lines 12–21 — the `LangBotResponse` model.** This is the schema the LLM must follow. Three fields: `message` (text), `mood` (enum), `confidence` (bounded float). Each `Field(description=...)` tells the model what to put there. `ge=0.0, le=1.0` enforces the range.

**Lines 32–42 — updated system prompt.** The original Day 3 prompt described the bot's personality. We added concrete instructions for mood selection so the moods are consistent and meaningful, not random.

**Lines 50 — `llm.with_structured_output(LangBotResponse)`.** This is the core change. It creates a new Runnable that, when invoked, returns a `LangBotResponse` instead of an `AIMessage`. Under the hood, LangChain tells the provider "respond with JSON matching this schema" and deserializes the result into a Python object.

**Lines 53 — chain unchanged.** `prompt | structured_llm` looks identical to yesterday's `prompt | llm`. The pipe operator does not care what kind of Runnable is on the right — it just passes data through. This is why LCEL and structured output compose so cleanly.

**Lines 72–73 — typed field access.** Instead of `response.content` we now use `response.message` (the reply text), `response.mood` (the enum), and `response.confidence` (the float). Your IDE autocompletes all three.

**Lines 78 — AIMessage conversion.** The conversation history list still holds `AIMessage` objects because that is what `MessagesPlaceholder` embeds into the prompt template. We extract `response.message` (the text) and wrap it. For the next turn, the model only sees the text — not its own mood/confidence metadata. This keeps the prompt clean.

## Try it now

```bash
python langbot.py
```

```text
🤖 LangBot (type 'quit' to exit)
----------------------------------------

You: What's the capital of France?

LangBot [thoughtful | 100% confidence]:
  Paris — the city of lights, love, and surprisingly good croissants. 🥐

You: Why do programmers confuse Halloween and Christmas?

LangBot [cheerful | 90% confidence]:
  Because Oct 31 equals Dec 25! Classic programmer humor — I'd give it a standing ovation if I had legs. 🎃🎄

You: What's the meaning of life?

LangBot [thoughtful | 60% confidence]:
  42, according to Douglas Adams. But honestly, I think it's different for everyone — I'm a chatbot, not a philosopher. 🤷

You: Tell me about the 2028 Olympics results

LangBot [apologetic | 5% confidence]:
  I'd love to, but my training data cutoff means I can't predict the future — or even next week's weather. Ask me about 2024 instead? 🕰️
```

Every response now carries structured metadata. The mood tracks the conversation's emotional arc. The confidence flags when LangBot is guessing. Both are available to your code as typed Python attributes — no parsing, no regex, no fragile string matching.

## Why this matters for real apps

### 1. Reliability

When you ask an LLM to "return JSON," it sometimes doesn't. A missing brace, a stray backtick, a conversational prefix ("Sure! Here's the JSON:") — all break `json.loads()`. `with_structured_output()` eliminates this entire class of bugs. On providers that support native structured output (OpenAI, Anthropic via tool use), the guarantee is at the API level. On others, LangChain handles retries and validation.

### 2. Type safety

```python
# Without structured output — you hope the model cooperates
response = chain.invoke(...)
text = response.content          # str — but is it?
mood = text.split("mood:")[1]    # fragile string parsing
conf = float(text.split("conf:")[1])  # crashes if the model formats wrong

# With structured output — the type is guaranteed
response = chain.invoke(...)
text = response.message          # str — guaranteed
mood = response.mood             # Mood enum — guaranteed
conf = response.confidence       # float — guaranteed
```

Your IDE knows the types. `mypy` can check them. You never write a JSON parsing try/except again.

### 3. Composability

`with_structured_output()` returns a Runnable. You can chain it, nest it, stream it, batch it — everything LCEL gives you works unchanged. Today's chain is `prompt | structured_llm`. Tomorrow you can add an output parser on the right: `prompt | structured_llm | some_transformer`. The structured output is just another link.

### 4. Multiple schemas from one model

You are not locked into one output shape. Create a second Pydantic model and wrap the same `llm` again:

```python
class JokeAnalysis(BaseModel):
    is_joke: bool
    funniness: int = Field(ge=1, le=10)
    topic: str

joke_analyzer = llm.with_structured_output(JokeAnalysis)
```

Now you have two specialized pipelines — one for chat, one for joke analysis — both backed by the same model, both returning typed objects. No prompt duplication required.

### 5. The mental model shift

Before structured output, you thought of the LLM as a text generator. After, you think of it as a **typed function**. You define the input and output schemas, and the model fills in the blanks. This is closer to how you use a database or an API — and it makes building reliable systems *far* easier.

## Pydantic features worth knowing

The `LangBotResponse` model uses three Pydantic features. Here are a few more you will want:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

class AdvancedResponse(BaseModel):
    # Literal types — the model must pick exactly one string
    action: Literal["reply", "search", "clarify"]

    # Optional fields — model can omit them
    follow_up_question: Optional[str] = None

    # Lists — model returns an array
    suggested_topics: list[str] = Field(default_factory=list, max_length=5)

    # Custom validation — runs after the model fills the fields
    @field_validator("follow_up_question")
    @classmethod
    def must_end_with_question_mark(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.endswith("?"):
            raise ValueError("follow_up_question must end with ?")
        return v
```

`field_validator` is especially powerful — it lets you enforce business rules that the LLM might violate. If the model returns `follow_up_question: "Tell me more"` (no question mark), Pydantic raises a `ValidationError` *before* your application sees the data.

## Common mistakes (and what they teach you)

**Mistake 1: forgetting to update the system prompt.**

If you add `with_structured_output()` but keep the old system prompt, the model does not know *how* to select moods. You will get valid `LangBotResponse` objects, but the moods will be random. Always guide the model on how to use your schema.

**Mistake 2: storing the structured response directly in history.**

```python
# Wrong — MessagesPlaceholder expects BaseMessage subclasses
messages.append(response)  # response is LangBotResponse, not AIMessage
# Result: weird serialization or errors on the next turn

# Right — convert to AIMessage first
messages.append(AIMessage(content=response.message))
```

`MessagesPlaceholder` works with LangChain message types (`HumanMessage`, `AIMessage`, `SystemMessage`). A Pydantic model is not a message type. Extract the text and wrap it.

**Mistake 3: using generic field descriptions.**

```python
# Weak — the model has to guess
message: str = Field(description="The message")

# Strong — the model knows exactly what you want
message: str = Field(description="LangBot's witty, under-three-sentence reply. Include emojis sparingly.")
```

Field descriptions are *part of your prompt*. They are injected into the system instructions that tell the model how to format its output. Write them for the model, not for other developers.

**Mistake 4: thinking `with_structured_output()` forces JSON-only responses.**

The model still "thinks" in natural language — it just formats the *final output* according to your schema. The quality of the response text is the same. The difference is only in how your code receives it.

**Mistake 5: over-constraining the schema.**

```python
# Over-constrained — the model has to follow rigid rules mid-generation
class TooStrict(BaseModel):
    word_count: int = Field(ge=10, le=12)  # exactly 10-12 words — hard for the model
```

The model generates text first, then fills the schema. If your constraints are too tight, it may produce valid JSON with awkward, unnatural text. Use constraints for values you genuinely need to enforce (like `confidence` between 0 and 1) — not for style preferences.

## What we have so far

Here is the complete, runnable `langbot.py` at the end of Day 5:

```python
# langbot.py — Day 5: Structured output with Pydantic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()


# ── Day 5: Mood enum ───────────────────────────────────────────────
class Mood(str, Enum):
    """Emotional tones LangBot can express."""
    CHEERFUL = "cheerful"
    SARCASTIC = "sarcastic"
    THOUGHTFUL = "thoughtful"
    APOLOGETIC = "apologetic"


# ── Day 5: Response model ──────────────────────────────────────────
class LangBotResponse(BaseModel):
    """Every LangBot reply is a typed object, not raw text."""
    message: str = Field(description="LangBot's actual reply to the user")
    mood: Mood = Field(description="The emotional tone LangBot chose for this response")
    confidence: float = Field(
        description="How sure LangBot is about this response. 0.0 = total guess, 1.0 = absolutely certain.",
        ge=0.0,
        le=1.0,
    )


def main():
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    # Day 3 + Day 5: Personality with mood guidance
    SYSTEM_PROMPT = """You are LangBot, a witty and slightly sarcastic AI assistant.
You love bad puns, you use emojis sparingly, and you always keep responses
under three sentences. If you don't know something, you admit it with style.

Choose your mood for each response:
- cheerful: when the conversation is light, the user is happy, or you're making a joke
- sarcastic: when you're being playfully snarky or the user said something ridiculous
- thoughtful: when the question is deep, technical, or deserves a careful answer
- apologetic: when you realize you got something wrong or don't know the answer

Set confidence based on how sure you are. Factual answers = high confidence.
Opinions and jokes = moderate confidence. Guesses = low confidence."""

    # Day 3: Build a prompt template with system message + history + input
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # Day 5: Wrap the LLM so every response comes back as a LangBotResponse
    structured_llm = llm.with_structured_output(LangBotResponse)

    # Day 4 + Day 5: Chain now returns LangBotResponse, not AIMessage
    chain = prompt | structured_llm

    # Day 2: Conversation history (still stores AIMessages)
    messages = []

    print("🤖 LangBot (type 'quit' to exit)")
    print("-" * 40)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        # Day 4: One call through the chain
        response = chain.invoke({"input": user_input, "history": messages})

        # Day 5: response is a LangBotResponse with typed fields
        print(f"\nLangBot [{response.mood.value} | {response.confidence:.0%} confidence]:")
        print(f"  {response.message}")

        # Day 2: Store the exchange (convert structured response to AIMessage)
        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=response.message))


if __name__ == "__main__":
    main()
```

## Try this now

Before tomorrow, experiment with structured output:

- **Add a new field.** Try adding `emoji_used: Optional[str]` to `LangBotResponse` — does LangBot tell you which emoji it picked?

- **Create a second schema.** Write a `JokeRating` model with `is_funny: bool` and `why: str`. Wrap the LLM with `llm.with_structured_output(JokeRating)` and create a separate chain that rates jokes.

- **Break validation.** Set `confidence: float = Field(ge=0.9, le=1.0)` — requiring at least 90% confidence. Watch what happens when LangBot is asked something it cannot be confident about. Does it comply, or does the call fail?

- **Compare models.** Try `with_structured_output()` on a different provider (Anthropic via `ChatAnthropic`, or a local model via Ollama). Some providers use tool-calling under the hood, others use JSON mode. The interface is the same.

- **Inspect the raw call.** Add `langchain.debug = True` at the top of `main()` and watch the API requests. You will see the JSON schema injected into the request — this is what tells the provider to return structured data.

## Coming tomorrow

LangBot now returns typed objects, but every response goes to one pipeline. Tomorrow we introduce **output parsers** — `StrOutputParser` and `CommaSeparatedListOutputParser` — so LangBot can handle multiple output formats (plain text, comma-separated lists, and more) through the same chain. One concept, one change — see you then.
