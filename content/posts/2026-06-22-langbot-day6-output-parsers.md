---
title: "LangBot Day 6: Output Parsers — Transforming Raw Responses into Usable Data"
date: 2026-06-22T07:00:00+01:00
draft: false
description: "LangBot learns output parsers. Day 6 introduces StrOutputParser and CommaSeparatedListOutputParser — lightweight LCEL components that turn raw AIMessages into clean strings and parsed lists."
tags:
  - python
  - langchain
  - ai
  - chatbot
  - tutorial
  - output-parsers
  - stroutputparser
  - commaseparatedlistoutputparser
  - lcel
categories:
  - tutorials
series: ["LangBot: Build a Chatbot with LangChain"]
slug: "langbot-day-6-output-parsers"
---

**Recap:** On Day 5, LangBot started returning typed objects — every response is now a `LangBotResponse` with `.message`, `.mood`, and `.confidence`. That is great when you need structured metadata. But structured output is not always the right tool. Sometimes you just want a clean string. Sometimes you want a parsed list. Today we add output parsers — lightweight LCEL components that transform raw `AIMessage` output into exactly the shape you need.

---

## The problem: not every answer needs a typed object

Day 5's `with_structured_output()` is powerful, but it comes with tradeoffs:

| Scenario | Structured output | What you actually want |
|---|---|---|
| User says "hello" | Full `LangBotResponse` with mood + confidence | Just the text |
| User asks "list 5 sci-fi books" | One `LangBotResponse` containing a comma-separated string | A `list[str]` you can iterate |
| User asks "what is Python?" | `LangBotResponse` with typed fields | A clean string for a UI |

`with_structured_output()` bakes the output shape into the LLM itself — the model *must* generate JSON matching your schema. That adds token overhead and latency. For simple cases, it is overkill.

Output parsers solve this from the other direction: let the model respond naturally, then *parse* whatever it returns into the shape you want. They sit at the end of an LCEL chain and transform the `AIMessage` — no model-side schema enforcement needed.

---

## The solution: output parsers as chain endpoints

An output parser is a **Runnable** that takes an `AIMessage` (or a string) and returns something else. Because it is a Runnable, you pipe it onto the end of any LCEL chain:

```python
chain = prompt | llm | parser
```

Two parsers are built into LangChain and cover 80% of cases:

### `StrOutputParser` — extract the text

The simplest parser in existence. Takes an `AIMessage` and returns `message.content` as a plain `str`. That is it.

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"input": "Hello"})
# result: "Hey there! What can I help you with today? 👋"
#         ^-- plain str, no .content needed
```

Before: `response.content` everywhere. After: the chain itself returns a string. Your loop code goes from:

```python
response = chain.invoke(...)
print(f"LangBot: {response.content}")   # AIMessage attribute access
```

to:

```python
response = chain.invoke(...)
print(f"LangBot: {response}")           # already a string
```

`StrOutputParser` also handles the case where the chain's penultimate step returns a string instead of an `AIMessage` — it passes strings through unchanged. This makes it a safe default when you are not sure what the upstream component returns.

### `CommaSeparatedListOutputParser` — get back a real list

When you ask an LLM to "list things," it returns a comma-separated string. You then write `response.content.split(",")` and `strip()` each item and hope the model did not use semicolons or bullet points. Fragile.

`CommaSeparatedListOutputParser` handles this reliably:

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
chain = prompt | llm | parser

result = chain.invoke({"input": "List 5 programming languages"})
# result: ["Python", "JavaScript", "Rust", "Go", "TypeScript"]
#         ^-- actual list[str], not a string you have to split
```

The parser:
1. Extracts the text from the `AIMessage`
2. Splits on commas
3. Strips whitespace from each item
4. Removes empty entries
5. Returns a `list[str]`

And it has one more trick: `parser.get_format_instructions()` returns a string you can inject into your prompt to tell the model how to format its response:

```python
print(parser.get_format_instructions())
# Your response should be a list of comma separated values, eg: `foo, bar, baz`
```

This is the parser's equivalent of a schema — but instead of enforcing structure at the API level, it guides the model with prompt instructions and then cleans up whatever comes back.

---

## What we are adding to LangBot today

We are going to give LangBot **two output modes**:

1. **Chat mode** (existing): uses the Day 5 structured-output chain for normal conversation — returns `LangBotResponse` with mood and confidence.

2. **List mode** (new): uses a second chain with `CommaSeparatedListOutputParser`. When the user asks for a list, suggestions, or recommendations, LangBot switches to this chain and returns a clean `list[str]` the code can iterate over.

We are also adding `StrOutputParser` as a third chain, demonstrating how to get a plain string when you do not need any metadata at all.

The routing logic: if the user's input contains list-signaling keywords (`list`, `suggest`, `recommend`, `top`, `options`, `ideas`, `rank`), we use the list chain. Otherwise, we use the chat chain.

---

## The new code

Four things are new today:

### 1. New imports

```python
# 🆕 Day 6: Output parsers
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
```

### 2. The list-intent detector

```python
# 🆕 Day 6: Keywords that signal the user wants a list
LIST_KEYWORDS = {"list", "suggest", "recommend", "top", "options", "ideas", "rank"}

def wants_list(user_input: str) -> bool:
    """Return True if the user is asking for a list or suggestions."""
    words = set(user_input.lower().split())
    return bool(words & LIST_KEYWORDS)
```

Simple keyword matching. Not an LLM call, not a classifier — just a fast check that determines which chain to use.

### 3. The list-specific prompt

```python
# 🆕 Day 6: Prompt that tells the model to output a comma-separated list
list_parser = CommaSeparatedListOutputParser()

LIST_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + list_parser.get_format_instructions()
```

`list_parser.get_format_instructions()` appends `"Your response should be a list of comma separated values, eg: \`foo, bar, baz\`"` to the system prompt. The model now knows to format its output as a comma-separated list. The parser on the other end of the chain will split it into a real Python list.

### 4. The list chain and the routing logic

```python
# 🆕 Day 6: Build a list-specific chain
list_prompt = ChatPromptTemplate.from_messages([
    ("system", LIST_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
list_chain = list_prompt | llm | list_parser   # returns list[str]

# 🆕 Day 6: Also showcase StrOutputParser for plain-string output
string_chain = prompt | llm | StrOutputParser   # returns str (no .content needed)
```

Then in the loop, we route:

```python
# 🆕 Day 6: Route to the right chain based on intent
if wants_list(user_input):
    items = list_chain.invoke({"input": user_input, "history": messages})
    print(f"\nLangBot's suggestions ({len(items)} items):")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")
    # Store as a regular text reply for history
    reply_text = f"Here are some suggestions: {', '.join(items)}"
else:
    response = chat_chain.invoke({"input": user_input, "history": messages})
    print(f"\nLangBot [{response.mood.value} | {response.confidence:.0%} confidence]:")
    print(f"  {response.message}")
    reply_text = response.message

messages.append(HumanMessage(content=user_input))
messages.append(AIMessage(content=reply_text))
```

When the list chain fires, the response is a `list[str]`. We iterate over it with `enumerate()` for numbered output. For history, we reconstruct a text reply from the items — the model on the next turn does not need to see the parsed list, just a plain-text summary.

---

## Where it plugs in

Here is the full Day 6 `langbot.py`. Look for the `# 🆕 Day 6` markers:

```python
# langbot.py — Day 6: Output parsers
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser

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


# 🆕 Day 6: Keywords that signal the user wants a list, not a chat
LIST_KEYWORDS = {"list", "suggest", "recommend", "top", "options", "ideas", "rank"}


def wants_list(user_input: str) -> bool:
    """Return True if the user is asking for a list or suggestions."""
    words = set(user_input.lower().split())
    return bool(words & LIST_KEYWORDS)


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

    # Day 3: Chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # Day 5: Structured output chain — returns LangBotResponse objects
    structured_llm = llm.with_structured_output(LangBotResponse)
    chat_chain = prompt | structured_llm

    # 🆕 Day 6: Build the list-mode output parser
    list_parser = CommaSeparatedListOutputParser()

    # 🆕 Day 6: List-specific prompt — injects format instructions for the model
    LIST_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + list_parser.get_format_instructions()

    list_prompt = ChatPromptTemplate.from_messages([
        ("system", LIST_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # 🆕 Day 6: List chain — prompt → raw LLM → comma-split parser → list[str]
    list_chain = list_prompt | llm | list_parser

    # 🆕 Day 6: String chain — prompt → raw LLM → StrOutputParser → plain str
    string_chain = prompt | llm | StrOutputParser()

    # Day 2: Conversation history
    messages = []

    print("🤖 LangBot (type 'quit' to exit)")
    print("─" * 40)
    print("  Try: 'list 5 programming books' or 'suggest weekend activities'")
    print("─" * 40)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        # 🆕 Day 6: Route to the right chain based on intent
        if wants_list(user_input):
            # List mode — returns list[str]
            items = list_chain.invoke({"input": user_input, "history": messages})

            print(f"\nLangBot's suggestions ({len(items)} items):")
            for i, item in enumerate(items, 1):
                print(f"  {i}. {item}")

            # Reconstruct a text reply for history (model doesn't need the parsed list)
            reply_text = f"Here are some suggestions: {', '.join(items)}"
        else:
            # Chat mode — returns LangBotResponse (Day 5)
            response = chat_chain.invoke({"input": user_input, "history": messages})

            print(f"\nLangBot [{response.mood.value} | {response.confidence:.0%} confidence]:")
            print(f"  {response.message}")

            reply_text = response.message

        # Day 2: Store the exchange in history
        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=reply_text))


if __name__ == "__main__":
    main()
```

---

## What changed — line by line

**Line 8 — new import.** `StrOutputParser` and `CommaSeparatedListOutputParser` come from `langchain_core.output_parsers`. Both are Runnables, so they compose with `|`.

**Lines 44–49 — `LIST_KEYWORDS` and `wants_list()`.** The detector is deliberately simple — a set intersection of lowercase words. No LLM call, no classifier, no regex. Fast and predictable. If you need smarter routing later (Day 10 covers agents that can decide for themselves), you will swap this out for a tool-calling model. For now, keyword matching keeps the concept focused on output parsers.

**Lines 81–90 — the list prompt and list chain.** Three steps:
1. `list_parser.get_format_instructions()` appends formatting guidance to the system prompt so the model knows to output comma-separated values.
2. `list_prompt` is a new `ChatPromptTemplate` using the augmented prompt. This is a separate template from the chat prompt — the list prompt tells the model to output lists; the chat prompt tells it to output structured JSON. Different jobs, different instructions.
3. `list_chain = list_prompt | llm | list_parser` chains the prompt, the raw LLM (not `structured_llm`!), and the parser. The raw `llm` returns an `AIMessage`; `list_parser` extracts and splits its `.content` into a `list[str]`.

**Line 93 — the string chain.** `prompt | llm | StrOutputParser()` is a third chain that returns a plain `str` — no object, no list, no `.content` access. We define it to show the pattern; the loop still uses the chat chain for normal conversation so the mood/confidence display still works.

**Lines 118–130 — the routing block.** The `if wants_list(user_input)` branch uses `list_chain`, iterates the returned list with `enumerate()`, and reconstructs a text summary for history. The `else` branch uses the Day 5 `chat_chain` — unchanged from yesterday.

**Lines 132–133 — unified history append.** Both branches now set `reply_text`, so the history append at the bottom is shared. Dryer than yesterday's approach.

---

## Try it now

```bash
python langbot.py
```

```text
🤖 LangBot (type 'quit' to exit)
────────────────────────────────────────
  Try: 'list 5 programming books' or 'suggest weekend activities'
────────────────────────────────────────

You: What is Python?

LangBot [thoughtful | 100% confidence]:
  Python is a high-level programming language known for its readability — kind of like English, but with more indentations and fewer Oxford commas. 🐍

You: list 5 must-read sci-fi books

LangBot's suggestions (5 items):
  1. Dune by Frank Herbert
  2. Neuromancer by William Gibson
  3. The Left Hand of Darkness by Ursula K. Le Guin
  4. Foundation by Isaac Asimov
  5. Snow Crash by Neal Stephenson

You: suggest weekend activities for a programmer

LangBot's suggestions (5 items):
  1. Build a tiny side project in a language you don't know
  2. Refactor something you wrote 6 months ago and cringe
  3. Go outside and touch grass (then come back and write a script to water it)
  4. Read the documentation of a library you use daily but never actually read
  5. Attend a local hackathon or join an online game jam

You: What's the best book from that sci-fi list?

LangBot [thoughtful | 85% confidence]:
  Dune — it's the one every sci-fi fan eventually circle back to. Frank Herbert built a universe so detailed you'll start checking your water consumption. 🏜️

You: quit

👋 Goodbye!
```

Notice: the conversation flows naturally between list requests and regular chat. LangBot remembers the sci-fi list and can reason about it on the next turn — because we stored the list items as a text summary in history.

---

## Why output parsers matter for real apps

### 1. They are lightweight

`with_structured_output()` forces the model to generate valid JSON matching a schema. That adds tokens to every request (the schema itself) and every response (the JSON wrapper). For simple string output, `StrOutputParser` adds zero overhead — it just accesses `.content`. For lists, `CommaSeparatedListOutputParser` adds one prompt line (the format instructions) and the model responds with a comma-separated string instead of JSON.

| Parser | Extra tokens in request | Extra tokens in response |
|---|---|---|
| `with_structured_output()` | ~200–500 (schema) | ~20–50 (JSON wrapper) |
| `StrOutputParser` | 0 | 0 |
| `CommaSeparatedListOutputParser` | ~15 (format instruction) | 0 (comma-separated is shorter than JSON array) |

For high-throughput applications, that difference adds up.

### 2. They work with any model

`with_structured_output()` relies on provider-specific features — OpenAI's structured outputs, Anthropic's tool use, or a fallback JSON mode. If you switch to a model that supports none of these (a local Ollama model, an older API), `with_structured_output()` may fail or degrade.

Output parsers work with *any* model that returns text. The model does not even know the parser exists — it just responds naturally, and the parser cleans up the result. This makes output parsers the most portable option.

### 3. They compose with everything

Because output parsers are Runnables, they compose with every other LCEL component:

```python
# Chain with a retriever (Day 9 preview)
rag_chain = retriever | prompt | llm | StrOutputParser()

# Chain with multiple parsers (one for each branch)
main_chain = prompt | llm
text_branch = main_chain | StrOutputParser()
list_branch = main_chain | CommaSeparatedListOutputParser()

# Custom parser as a lambda
weird_chain = prompt | llm | (lambda msg: msg.content[::-1])  # reverses the text
```

The pattern is always the same: pipe it on the right. The chain does not care what the parser does internally.

### 4. They separate concerns

Without output parsers, your loop code mixes two responsibilities:

```python
response = chain.invoke(...)
text = response.content           # parsing responsibility
text = text.strip()               # cleanup responsibility
if "," in text:                   # format-detection responsibility
    items = [x.strip() for x in text.split(",")]
print(items)                      # display responsibility
```

With output parsers, the parsing and cleanup live in the chain definition. Your loop code only displays:

```python
items = list_chain.invoke(...)    # parsing + cleanup handled by the chain
for i, item in enumerate(items, 1):
    print(f"  {i}. {item}")       # display is the only responsibility left
```

This is the same separation-of-concerns principle that prompted templates brought to system instructions (Day 3) and LCEL brought to pipeline wiring (Day 4). Move the mechanism into the chain; keep the intent in your loop.

### 5. They are the foundation for custom parsers

Once you understand `StrOutputParser` and `CommaSeparatedListOutputParser`, writing your own is trivial. A custom output parser is any callable that takes an `AIMessage` (or `str`) and returns whatever you want:

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import List

class NumberedListOutputParser(BaseOutputParser[List[str]]):
    """Parse '1. foo\n2. bar' into ['foo', 'bar']."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return [line.split(". ", 1)[1] for line in lines if ". " in line]

# Use it just like the built-in ones
chain = prompt | llm | NumberedListOutputParser()
```

LangChain's built-in parsers cover the common cases. For everything else, you subclass `BaseOutputParser` and implement `parse()`. Ten lines of code.

---

## When to use which

| You want... | Use... | Because... |
|---|---|---|
| A plain string | `StrOutputParser` | Zero overhead, works everywhere |
| A list of items | `CommaSeparatedListOutputParser` | Handles splitting + stripping + empties |
| Typed fields with validation | `with_structured_output()` (Day 5) | Guarantees schema compliance at the API level |
| A custom format (numbered list, JSON with a specific shape, etc.) | Your own `BaseOutputParser` subclass | You control the parsing logic |
| Structured data + portability | `JsonOutputParser` (a built-in you can explore) | Parses JSON from the model's text output — less reliable than `with_structured_output()` but works with any model |

For LangBot, we now use **both**: structured output for normal chat (typed responses with mood + confidence) and `CommaSeparatedListOutputParser` for list requests (clean `list[str]` output). The right tool for each job.

---

## Common mistakes (and what they teach you)

**Mistake 1: using an output parser after `with_structured_output()`.**

```python
# Wrong — structured_llm returns LangBotResponse, not AIMessage
chain = prompt | structured_llm | StrOutputParser()
# Error: StrOutputParser expects an AIMessage or str, got LangBotResponse
```

`with_structured_output()` already transforms the output into a Pydantic model. An output parser expects raw `AIMessage` input. You use one or the other — not both on the same chain. If you want structured output, access the fields directly (`response.message`). If you want a parser, use the raw `llm` (not `structured_llm`).

**Mistake 2: forgetting format instructions in the prompt.**

```python
# Wrong — the model has no idea it should output comma-separated values
list_chain = prompt | llm | CommaSeparatedListOutputParser()

# Result: the model returns full sentences. The parser splits on commas and
# you get garbage like ["I'd recommend Dune", " which is a classic", ...]
```

Always inject `parser.get_format_instructions()` into your prompt. The parser splits on commas; the model needs to know to *use* commas. Without the instruction, the model's natural prose style will produce nonsensical splits.

**Mistake 3: not handling the parsed output in history.**

```python
# Wrong — storing the list directly in the message list
messages.append(AIMessage(content=items))  # items is a list, not a string
```

`AIMessage(content=...)` expects a string. If you pass a list, LangChain may serialize it awkwardly or raise an error. Always reconstruct a text summary for history:

```python
reply_text = f"Here are some suggestions: {', '.join(items)}"
messages.append(AIMessage(content=reply_text))
```

**Mistake 4: confusing `StrOutputParser` with `.content` access.**

```python
# These do the same thing:
chain_a = prompt | llm | StrOutputParser()
chain_b = prompt | llm

text_a = chain_a.invoke(...)          # str
text_b = chain_b.invoke(...).content  # str (manual .content access)
```

They return the same value. The difference is where the extraction happens — in the chain definition vs. in your loop code. LCEL philosophy: put it in the chain. It makes the loop cleaner and makes the chain reusable in other contexts (a FastAPI endpoint, a Streamlit UI) where `.content` access might be boilerplate you forget.

**Mistake 5: over-engineering the routing.**

```python
# Over-engineered: using an LLM call to decide which chain to use
intent = llm.invoke(f"Is this a list request? {user_input}")
if "yes" in intent.content.lower():
    ...
```

Simple keyword matching works fine for a CLI tutorial. You will add agent-based routing on Day 10 with tool calling. Until then, keep the routing logic proportional to the problem.

---

## What we have so far

Here is the complete, runnable `langbot.py` at the end of Day 6:

```python
# langbot.py — Day 6: Output parsers
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser

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


# Day 6: Keywords that signal the user wants a list, not a chat
LIST_KEYWORDS = {"list", "suggest", "recommend", "top", "options", "ideas", "rank"}


def wants_list(user_input: str) -> bool:
    """Return True if the user is asking for a list or suggestions."""
    words = set(user_input.lower().split())
    return bool(words & LIST_KEYWORDS)


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

    # Day 3: Chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # Day 5: Structured output chain — returns LangBotResponse objects
    structured_llm = llm.with_structured_output(LangBotResponse)
    chat_chain = prompt | structured_llm

    # Day 6: Build the list-mode output parser
    list_parser = CommaSeparatedListOutputParser()

    # Day 6: List-specific prompt — injects format instructions for the model
    LIST_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + list_parser.get_format_instructions()

    list_prompt = ChatPromptTemplate.from_messages([
        ("system", LIST_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # Day 6: List chain — prompt → raw LLM → comma-split parser → list[str]
    list_chain = list_prompt | llm | list_parser

    # Day 6: String chain — prompt → raw LLM → StrOutputParser → plain str
    string_chain = prompt | llm | StrOutputParser()

    # Day 2: Conversation history
    messages = []

    print("🤖 LangBot (type 'quit' to exit)")
    print("─" * 40)
    print("  Try: 'list 5 programming books' or 'suggest weekend activities'")
    print("─" * 40)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        # Day 6: Route to the right chain based on intent
        if wants_list(user_input):
            # List mode — returns list[str]
            items = list_chain.invoke({"input": user_input, "history": messages})

            print(f"\nLangBot's suggestions ({len(items)} items):")
            for i, item in enumerate(items, 1):
                print(f"  {i}. {item}")

            # Reconstruct a text reply for history (model doesn't need the parsed list)
            reply_text = f"Here are some suggestions: {', '.join(items)}"
        else:
            # Chat mode — returns LangBotResponse (Day 5)
            response = chat_chain.invoke({"input": user_input, "history": messages})

            print(f"\nLangBot [{response.mood.value} | {response.confidence:.0%} confidence]:")
            print(f"  {response.message}")

            reply_text = response.message

        # Day 2: Store the exchange in history
        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=reply_text))


if __name__ == "__main__":
    main()
```

---

## Try this now

Before tomorrow, experiment with output parsers:

- **Add more list keywords.** Expand `LIST_KEYWORDS` with words like `"name"`, `"give me"`, `"choices"`. Does the routing still feel right, or do false positives start appearing? This is why agents (Day 10) eventually replace keyword routing — they understand intent, not just words.

- **Write a custom parser.** Subclass `BaseOutputParser` and build a `BulletPointOutputParser` that splits on `\n- ` or `\n* `. Test it with a prompt that asks the model for bullet-pointed answers.

- **Chain parsers.** Try `chain = prompt | llm | StrOutputParser() | (lambda s: s.upper())`. The lambda receives a string (thanks to `StrOutputParser`) and returns an uppercase version. This is the power of LCEL — every link is independent and composable.

- **Compare token usage.** Add `langchain.debug = True` at the top of `main()` and compare the API requests for the chat chain vs. the list chain. The list chain's request is noticeably smaller — no JSON schema in the payload.

- **Try `JsonOutputParser`.** LangChain also includes `JsonOutputParser` for when you want JSON output from any model without using `with_structured_output()`. It is less reliable (the model can still produce invalid JSON) but more portable. Compare it to Day 5's approach.

---

## Coming tomorrow

LangBot now responds in two modes — chat and list — but every response arrives all at once. Tomorrow we introduce **streaming output**: `.stream()` for token-by-token responses, so LangBot can type its replies character by character like a real chat app. One concept, one change — see you then.
