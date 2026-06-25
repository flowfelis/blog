---
title: "LangBot Day 7: Streaming Output — Token-by-Token Responses"
date: 2026-06-23T07:00:00+01:00
draft: false
description: "LangBot learns to stream. Day 7 introduces .stream() — watch LangBot type its replies character by character, just like ChatGPT. Half the perceived wait, twice the delight."
tags:
  - python
  - langchain
  - ai
  - chatbot
  - tutorial
  - streaming
  - stream
  - astream
  - lcel
  - stroutputparser
categories:
  - tutorials
series: ["LangBot: Build a Chatbot with LangChain"]
slug: "langbot-day-7-streaming"
---

**Recap:** On Day 6, LangBot gained output parsers and now responds in two modes — chat mode returns typed `LangBotResponse` objects with mood and confidence, and list mode returns clean `list[str]` via `CommaSeparatedListOutputParser`. Both modes use `.invoke()`, which means the entire response arrives at once after the model finishes generating. Today we fix that — LangBot learns to type its replies one token at a time.

---

## The problem: waiting for the whole answer

Every interaction so far has the same rhythm:

```
You: What's the best way to learn Rust?
[1–3 second pause while the LLM generates the ENTIRE response]
LangBot [thoughtful | 90% confidence]:
  Start with The Book — it's the official Rust tutorial and it's genuinely good...
```

That pause is the model generating all its tokens, packaging them into one response, and sending them back. For a three-sentence answer it is fine. For a long explanation, a generated poem, or a code block, the silence stretches to 5, 10, even 20 seconds. The user stares at a blinking cursor with no feedback. Did it crash? Is it thinking? Should they tab away?

Real chat apps — ChatGPT, Claude, the ChatGPT API playground — do not work this way. They **stream**: as soon as the first token is ready, it appears on screen. Then the next. Then the next. The user sees text accumulating in real time, which makes the wait feel shorter and confirms that something is happening.

LangChain supports this out of the box with `.stream()`.

---

## The solution: `.stream()`

`.stream()` is a method on every Runnable. Instead of waiting for the full response and returning it as one object, `.stream()` returns an **iterator** that yields chunks as they arrive from the model:

```python
# Before (Day 6): wait for everything
response = chain.invoke({"input": "Hello"})
print(response)                        # all at once

# After (Day 7): get chunks as they arrive
for chunk in chain.stream({"input": "Hello"}):
    print(chunk, end="", flush=True)   # one piece at a time
```

The `end=""` keeps the cursor on the same line. `flush=True` forces output immediately instead of waiting for the buffer to fill. Together they create the typing effect.

### What chunks look like

The shape of each chunk depends on what is in your chain:

```python
# Chain: prompt | llm | StrOutputParser()
# Chunk type: str
for chunk in chain.stream(...):
    # chunk = "The" then " best" then " way" then " to" then " learn" ...
```

```python
# Chain: prompt | llm   (no output parser)
# Chunk type: AIMessageChunk
for chunk in chain.stream(...):
    # chunk.content = "The" then " best" then " way" ...
    # You access .content manually each iteration
```

```python
# Chain: prompt | structured_llm   (with_structured_output)
# Chunk type: partial JSON strings or incremental model objects
# (Provider-dependent. OpenAI streams JSON fragments; LangChain
#  tries to assemble them into complete objects but the behavior
#  varies. More on this below.)
```

`StrOutputParser` is the streaming sweet spot: it normalizes whatever the upstream Runnable emits into plain strings, so every chunk you receive is immediately printable.

---

## What we are adding to LangBot today

One change: **the chat branch now streams**. Instead of calling `.invoke()` and printing the whole response at once, we call `.stream()` and print each chunk as it arrives.

The list branch stays on `.invoke()` — streaming a `CommaSeparatedListOutputParser` would yield individual tokens like "Dune", ",", " Neu", "romancer", which is nonsense to display. The list makes sense as a complete unit, so we keep it synchronous.

The streaming chain we add:

```python
stream_chain = prompt | llm | StrOutputParser()
```

This is the same `prompt` and `llm` we already have, piped into `StrOutputParser()` so every chunk is a plain string. We define it alongside the existing chains — no new dependencies, no new concepts beyond `.stream()`.

---

## The new code

Three things change today:

### 1. The streaming chain

```python
# 🆕 Day 7: Streaming chain — prompt → LLM → StrOutputParser → streaming str chunks
stream_chain = prompt | llm | StrOutputParser()
```

This is identical to the Day 6 `string_chain`, but we are giving it a distinct name (`stream_chain`) to signal its role: this is the chain we stream, not the one we invoke. In practice, you can call `.stream()` on `string_chain` too — they are the same object. The name is a convention for readability.

### 2. The streaming loop

The chat branch goes from:

```python
# Day 6: invoke — wait for everything
response = chat_chain.invoke({"input": user_input, "history": messages})

print(f"\nLangBot [{response.mood.value} | {response.confidence:.0%} confidence]:")
print(f"  {response.message}")

reply_text = response.message
```

To:

```python
# 🆕 Day 7: stream — print tokens as they arrive
print(f"\nLangBot: ", end="", flush=True)
full_response = ""
for chunk in stream_chain.stream({"input": user_input, "history": messages}):
    print(chunk, end="", flush=True)
    full_response += chunk
print()  # final newline after streaming finishes

reply_text = full_response
```

Four lines of streaming code replace the invoke-and-print pattern. The `full_response` accumulator collects chunks so we can store the complete message in history — the model on the next turn needs the full text, not a stream iterator.

### 3. The mood and confidence summary (optional)

Without `structured_llm`, we lose the mood/confidence display. If you want it back, you can make a second, fast `.invoke()` call after streaming finishes:

```python
# Optional: get structured metadata after streaming
meta = chat_chain.invoke({"input": user_input, "history": messages})
print(f"  [{meta.mood.value} | {meta.confidence:.0%} confidence]")
```

I am including this as a commented-out option in the final `app.py` — it doubles your API calls but keeps the metadata display. In production, you would pick one: stream for UX, or structured output for metadata. Combining both is possible but costs extra. The tutorial shows the streaming path as the default because the UX improvement is what this day is about.

---

## Where it plugs in

Here is the full Day 7 `langbot.py`. Look for the `# 🆕 Day 7` markers:

```python
# langbot.py — Day 7: Streaming output
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

    # 🆕 Day 7: Streaming chain — prompt → LLM → StrOutputParser → str chunks
    #           Replace the Day 6 string_chain with an explicitly named streaming
    #           chain. Same composition, new role: the chain we stream.
    stream_chain = prompt | llm | StrOutputParser()

    # Day 2: Conversation history
    messages = []

    print("🤖 LangBot (type 'quit' to exit)")
    print("─" * 40)
    print("  Try: 'list 5 programming books' or 'suggest weekend activities'")
    print("  Now streaming — watch LangBot type! ⚡")
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
            # List mode — invoke for a complete list (streaming makes no sense here)
            items = list_chain.invoke({"input": user_input, "history": messages})

            print(f"\nLangBot's suggestions ({len(items)} items):")
            for i, item in enumerate(items, 1):
                print(f"  {i}. {item}")

            # Reconstruct a text reply for history
            reply_text = f"Here are some suggestions: {', '.join(items)}"
        else:
            # 🆕 Day 7: Chat mode — stream the response token by token
            print(f"\nLangBot: ", end="", flush=True)
            full_response = ""
            for chunk in stream_chain.stream({"input": user_input, "history": messages}):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()  # final newline after streaming finishes

            # 🆕 Day 7: reply_text is the accumulated full response
            reply_text = full_response

            # Optional: uncomment the next three lines to also get mood/confidence
            # meta = chat_chain.invoke({"input": user_input, "history": messages})
            # print(f"  [{meta.mood.value} | {meta.confidence:.0%} confidence]")
            # reply_text = meta.message   # use the structured version for history

        # Day 2: Store the exchange in history
        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=reply_text))


if __name__ == "__main__":
    main()
```

---

## What changed — line by line

**`stream_chain = prompt | llm | StrOutputParser()` (new line 107–109).** This is the chain we stream. It uses the raw `llm` (not `structured_llm`) because streaming is simplest with plain text — `StrOutputParser` extracts `.content` from each `AIMessageChunk` and passes the string downstream. Every chunk in the loop is a `str` you can print directly.

**The stream loop (new lines 149–154).** Five lines that replace the old `.invoke()` + print pattern:

1. `print(f"\nLangBot: ", end="", flush=True)` — prints the prefix and stays on the same line
2. `full_response = ""` — accumulator for the complete text
3. `for chunk in stream_chain.stream(...)` — iterates over tokens as they arrive
4. `print(chunk, end="", flush=True)` — prints each token immediately, no newline
5. `full_response += chunk` — builds the complete response for history storage

**The `print()` after the loop (line 155).** After streaming finishes, we print an empty newline so the next `input("\nYou: ")` prompt starts on its own line. Without this, the first prompt of the next turn would appear on the same line as the last streamed token.

**`reply_text = full_response` (line 158).** The accumulated full response becomes the history entry. This is critical: the model on the next turn needs the complete message, not individual tokens. Always reconstruct the full text from streamed chunks before storing in history.

---

## Try it now

```bash
python langbot.py
```

```text
🤖 LangBot (type 'quit' to exit)
────────────────────────────────────────
  Try: 'list 5 programming books' or 'suggest weekend activities'
  Now streaming — watch LangBot type! ⚡
────────────────────────────────────────

You: What is the difference between .stream() and .invoke()?

LangBot: .invoke() waits for the entire response to finish — like ordering a pizza and
staring at the door until it arrives. .stream() is the tracker app that shows "your pizza
is being made... now it's in the oven... now it's on the way." Same pizza, less anxiety. 🍕

You: Explain quantum computing in one sentence

LangBot: Quantum computing uses qubits that can be both 0 and 1 simultaneously thanks to
superposition — which is either incredibly powerful or a great excuse when your code doesn't
work, depending on how you measure it. ⚛️

You: list 3 programming languages for beginners

LangBot's suggestions (3 items):
  1. Python
  2. JavaScript
  3. Scratch

You: quit

👋 Goodbye!
```

Notice the chat responses now appear **character by character** instead of arriving all at once. The list responses still appear instantly (`.invoke()`), which is the right UX — numbered lists are more readable as a complete block.

---

## How streaming actually works under the hood

When you call `.stream()`, LangChain does not change *how* the model generates tokens. It changes *when* your code receives them.

### Without streaming (`.invoke()`)

```
Your code                     LangChain                     OpenAI API
    |                            |                              |
    |--- chain.invoke() -------->|                              |
    |                            |--- POST /chat/completions -->|
    |                            |   (stream=false)             |
    |                            |                              |--- begin generation
    |                            |                              |--- token "The"
    |                            |                              |--- token " best"
    |                            |                              |--- token " way"
    |                            |                              |--- ... all tokens ...
    |                            |                              |--- generation complete
    |                            |<-- full response ------------|
    |                            |   (all tokens bundled)       |
    |<-- full result ------------|                              |
    |                            |                              |
    |--- print(response) ------->|                              |
```

One round trip. Your code blocks until the model is completely done.

### With streaming (`.stream()`)

```
Your code                     LangChain                     OpenAI API
    |                            |                              |
    |--- chain.stream() -------->|                              |
    |                            |--- POST /chat/completions -->|
    |                            |   (stream=true)              |
    |                            |                              |--- begin generation
    |                            |<-- data: {"token": "The"} ---|
    |<-- yield "The" ------------|                              |
    |--- print("The")            |                              |
    |                            |<-- data: {"token": " best"} -|
    |<-- yield " best" ----------|                              |
    |--- print(" best")          |                              |
    |                            |<-- data: {"token": " way"} --|
    |<-- yield " way" -----------|                              |
    |--- print(" way")           |                              |
    |                            |                              |--- ... more tokens ...
    |                            |<-- data: [DONE] -------------|
    |<-- StopIteration ----------|                              |
```

Your code's `for` loop receives each token as soon as the API sends it. LangChain wraps the SSE (Server-Sent Events) stream from the API into a Python iterator. Your code never touches SSE — it just iterates.

### Stream chunks: raw vs. parsed

What you actually receive in your `for chunk in ...` loop depends on the chain:

| Chain | What `.stream()` yields | Printable directly? |
|---|---|---|
| `prompt \| llm \| StrOutputParser()` | `str` chunks | Yes |
| `prompt \| llm` | `AIMessageChunk` objects | No — access `.content` |
| `prompt \| structured_llm` | Provider-dependent. OpenAI: partial JSON strings that accumulate into a valid object. LangChain tries to coalesce them into complete `LangBotResponse` instances, but behavior varies by provider and model. | Sometimes |

`StrOutputParser` is the reliable choice for streaming: it normalizes everything to strings. If you chain `prompt | llm` without a parser and try to print `chunk`, you will see `<AIMessageChunk content='The'>` instead of `The`. Always put `StrOutputParser()` at the end when streaming for display.

---

## Streaming and structured output: the tradeoff

Structured output (`with_structured_output()`) and streaming pull in opposite directions:

- **Structured output wants the complete JSON** before it can validate fields. `confidence` must be a float between 0.0 and 1.0 — you cannot validate a partial number.
- **Streaming wants to show partial text immediately.** Every token that arrives should appear on screen.

LangChain bridges this gap but the result is imperfect. When you call `.stream()` on a chain with `with_structured_output()`:

1. The API streams partial JSON fragments (e.g., `{"mess`, then `age":`, then ` "Hello"}`)
2. LangChain buffers these fragments
3. When a complete JSON object is parseable, it yields the parsed Pydantic model
4. If parsing fails (incomplete JSON), it either yields nothing or yields a partial update — provider-dependent

In practice, for OpenAI models, streaming `with_structured_output()` typically yields **one chunk** when the full JSON is ready — which defeats the point of streaming. The alternative is to stream with `StrOutputParser()` and parse the accumulated result afterward, or make a second `.invoke()` call for metadata.

For LangBot, we chose the pragmatic path: **stream for UX, structured output for metadata — pick the one that matters for each interaction.** The chat branch streams with `StrOutputParser()`. The commented-out metadata block shows how to get structured data when you need it.

---

## `.astream()` — streaming with asyncio

If your app uses `async`/`await` (FastAPI, Discord bots, etc.), LangChain provides `.astream()` — the async equivalent of `.stream()`:

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

async def stream_chat(user_input: str):
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant.\n\n{input}")
    chain = prompt | llm | StrOutputParser()

    full_response = ""
    async for chunk in chain.astream({"input": user_input}):
        print(chunk, end="", flush=True)
        full_response += chunk
    print()
    return full_response

# Run it
asyncio.run(stream_chat("Explain async/await in one sentence"))
```

The pattern is identical — `async for` instead of `for`. Everything else (the accumulator, the print, the history append) stays the same.

---

## Common mistakes (and what they teach you)

**Mistake 1: forgetting `flush=True`.**

```python
# Wrong — output is buffered and may not appear until the loop finishes
for chunk in chain.stream(...):
    print(chunk, end="")   # no flush
```

Without `flush=True`, Python's output buffering may hold all the chunks until the program ends or the buffer fills. The user sees nothing, then everything at once — the exact opposite of streaming. Always `flush=True` when printing incrementally.

**Mistake 2: printing a newline on every chunk.**

```python
# Wrong — each token on its own line
for chunk in chain.stream(...):
    print(chunk)   # default end="\n"
```

Output:
```
The

best

way

to

learn
```

Use `end=""` to keep tokens on the same line. Add one `print()` after the loop to move to the next line.

**Mistake 3: storing stream chunks directly in history.**

```python
# Wrong — storing the iterator or individual chunks
for chunk in chain.stream(...):
    messages.append(AIMessage(content=chunk))   # one message per token!
```

History explodes from one message per turn to dozens of one-token messages. Always accumulate chunks into a single string first, then store once.

**Mistake 4: streaming lists.**

```python
# Wrong — streaming a CommaSeparatedListOutputParser
for chunk in list_chain.stream({"input": "list 5 movies"}):
    print(chunk, end="", flush=True)
# Output: "TheGodfather, PulpFiction, TheDarkKnight, ..." — unreadable
```

Lists are meant to be consumed as a complete unit. Stream text responses; invoke for structured/lists.

**Mistake 5: not starting with `end=""`.**

```python
# Wrong — "LangBot:" and then tokens start on the next line
print("LangBot:")          # ends with \n
for chunk in chain.stream(...):
    print(chunk, end="", flush=True)
```

The label and the response end up on separate lines. Use `print("LangBot: ", end="", flush=True)` to keep them together.

---

## Why streaming matters for real apps

### 1. Perceived performance trumps actual performance

A response that arrives all at once after 3 seconds feels slower than a response that starts appearing after 0.3 seconds and finishes after 3 seconds — even though both take exactly 3 seconds. Streaming front-loads the feedback, which research consistently shows improves user satisfaction more than actually making things faster.

### 2. Users can start reading immediately

With streaming, the user reads the first sentence while the model generates the third. By the time the full response is done, the user has already processed the beginning. The total interaction time (reading + generation) overlaps instead of stacking sequentially.

### 3. Early termination

If the user sees the first sentence and realizes the model misunderstood, they can interrupt (Ctrl+C in the CLI, or a stop button in a UI). With `.invoke()`, they have to wait for the full response before realizing it is wrong. Streaming lets them bail early.

### 4. It is the expected UX

Every major chat interface streams. ChatGPT, Claude, Gemini, Perplexity — all stream. If your chatbot does not stream, it feels outdated before the user finishes their first prompt.

### 5. It composes with everything

`.stream()` works on any Runnable chain — prompts, retrievers, output parsers, custom components. Once you add retriever-augmented generation (Day 9), the entire RAG pipeline streams: retrieve documents → stream the generated answer. No special wiring needed.

---

## What we have so far

Here is the complete, runnable `langbot.py` at the end of Day 7:

```python
# langbot.py — Day 7: Streaming output
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

    # 🆕 Day 7: Streaming chain — prompt → LLM → StrOutputParser → str chunks
    stream_chain = prompt | llm | StrOutputParser()

    # Day 2: Conversation history
    messages = []

    print("🤖 LangBot (type 'quit' to exit)")
    print("─" * 40)
    print("  Try: 'list 5 programming books' or 'suggest weekend activities'")
    print("  Now streaming — watch LangBot type! ⚡")
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
            # List mode — invoke for a complete list
            items = list_chain.invoke({"input": user_input, "history": messages})

            print(f"\nLangBot's suggestions ({len(items)} items):")
            for i, item in enumerate(items, 1):
                print(f"  {i}. {item}")

            # Reconstruct a text reply for history
            reply_text = f"Here are some suggestions: {', '.join(items)}"
        else:
            # 🆕 Day 7: Chat mode — stream the response token by token
            print(f"\nLangBot: ", end="", flush=True)
            full_response = ""
            for chunk in stream_chain.stream({"input": user_input, "history": messages}):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()  # final newline

            reply_text = full_response

            # Optional: uncomment for mood/confidence metadata
            # meta = chat_chain.invoke({"input": user_input, "history": messages})
            # print(f"  [{meta.mood.value} | {meta.confidence:.0%} confidence]")
            # reply_text = meta.message

        # Day 2: Store the exchange in history
        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=reply_text))


if __name__ == "__main__":
    main()
```

---

## Try this now

Before tomorrow, experiment with streaming:

- **Stream the list chain.** Call `list_chain.stream()` instead of `.invoke()`. What do the chunks look like? Individual list items? Tokens? This will show you why streaming a list parser is a bad idea — and teach you to think about what your parser emits.

- **Add `time.sleep(0.02)` between chunks.** Insert a tiny delay in the stream loop to simulate a slower connection. How does the experience change? At what delay does streaming still feel responsive? At what point does it become annoying?

- **Stream without `StrOutputParser`.** Remove the parser from the chain: `raw_chain = prompt | llm`. Stream it. What are the chunks? (They are `AIMessageChunk` objects.) Access `.content` on each one. This teaches you what `StrOutputParser` actually does — it is a convenience, not magic.

- **Try `.astream()`.** Convert the chat loop to async and use `astream()`. This is forward practice for Day 16 (FastAPI deployment), where async streaming is the standard pattern for server-side chatbots.

- **Stream a long response.** Ask LangBot to "write a haiku about debugging, then explain the haiku line by line." With streaming, you see the poem appear word by word. With `.invoke()`, you wait 5+ seconds for everything. The difference is visceral.

---

## Coming tomorrow

LangBot now types its responses in real time — a real chat experience. But ask it a factual question and it relies entirely on its training data, frozen at some past cutoff date. Tomorrow we introduce **retrieval**: vector stores, embeddings, and Pinecone — the first half of RAG that lets LangBot search external knowledge for answers it does not already know. One concept, one change — see you then.
