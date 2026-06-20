---
title: "LangBot Day 4: Chaining with LCEL — The Pipe Operator"
date: 2026-06-20T07:00:00+01:00
draft: false
description: "LangBot learns LCEL. Day 4 introduces the pipe operator and RunnableSequence — turn two separate invoke() calls into one clean, readable chain."
tags:
  - python
  - langchain
  - ai
  - chatbot
  - tutorial
  - lcel
  - runnablesequence
  - pipe-operator
categories:
  - tutorials
series: ["LangBot: Build a Chatbot with LangChain"]
slug: "langbot-day-4-lcel"
---

**Recap:** On Day 1 we built a basic CLI chatbot. Day 2 gave it memory so it remembers the conversation. Day 3 added a prompt template so LangBot has a personality. But every turn still requires two separate steps — format the template, then call the model. Today we stitch them together into one fluid pipeline.

---

## The problem: two steps where one should do

Here is how LangBot works on Day 3. Every single turn:

```python
# Step 1: format the template
formatted_messages = prompt.invoke({
    "input": user_input,
    "history": messages,
})

# Step 2: send to the model
response = llm.invoke(formatted_messages)
```

This works. But it is repetitive and obscures the real shape of the pipeline. What we actually want to express is: **"take the prompt, pipe it through the model, give me the response."** Two separate `invoke()` calls don't say that — they say "do this, then do that, and don't forget to pass the result of the first to the second."

LangChain has a better way.

## The solution: LCEL and the pipe operator

**LCEL** (LangChain Expression Language) is a declarative syntax for building chains of components. Its centerpiece is the `|` (pipe) operator, borrowed from Unix pipelines. The idea:

```python
chain = prompt | llm
response = chain.invoke({"input": user_input, "history": messages})
```

One variable. One `invoke()` call. The data flows from `prompt` to `llm` automatically. No intermediate variable, no manual handoff. The pipe *is* the wiring.

Under the hood, `prompt | llm` creates a **`RunnableSequence`** — an object that knows how to call each component in order, passing each output as the next component's input. You can think of it as:

```
User Input  ──►  Prompt Template  ──►  LLM  ──►  Response
                    │                      │
              {input} fills in        AIMessage lands
              {history} fills in      here automatically
```

The key insight: **every major LangChain component is a "Runnable"** — anything with an `invoke()` method can be chained with `|`. `ChatPromptTemplate` is a Runnable. `ChatOpenAI` is a Runnable. So is a retriever, an output parser, a tool, a custom function — anything that implements the Runnable interface.

## The new code we are adding today

Here is exactly what changes. **This is the only new code in today's post:**

```python
# 🆕 Day 4: Chain the prompt and model into one pipeline
chain = prompt | llm
```

Then, in the loop, replace the two-step invoke:

```python
# OLD (Day 3) — two separate calls:
# formatted_messages = prompt.invoke({"input": user_input, "history": messages})
# response = llm.invoke(formatted_messages)

# 🆕 Day 4 — one call through the chain:
response = chain.invoke({"input": user_input, "history": messages})
```

That is it. One line to define the chain. One line to use it. The entire pipeline is now expressed as `prompt | llm` — readable at a glance.

## What `prompt | llm` actually does

When you write `prompt | llm`, LangChain builds a `RunnableSequence`. Conceptually, `chain.invoke(input_dict)` does this:

```
chain.invoke({"input": "hello", "history": []})
        │
        ▼
    prompt.invoke({"input": "hello", "history": []})
        │
        ▼  returns ChatPromptValue (a list of messages)
        │
    llm.invoke( ChatPromptValue )
        │
        ▼  returns AIMessage
```

Every `|` in an LCEL expression adds one more link. The output from the left side becomes the input to the right side — automatically, with no glue code. This is why the chain works with one `invoke()` call: `RunnableSequence` handles the internal handoffs.

### You can inspect the chain

Add this after defining `chain` to see what you built:

```python
print(chain)
# Output:
# first=ChatPromptTemplate(...)
# | ChatOpenAI(...)
```

The `first` and `middle` / `last` attributes let you reach into the chain and inspect individual components if you ever need to.

## Where it plugs in

Here is the Day 4 version of `langbot.py`. Look for the `# 🆕 Day 4` markers:

```python
# langbot.py — Day 4: LCEL pipe operator
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def main():
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    # Day 3: Define the bot's personality
    SYSTEM_PROMPT = """You are LangBot, a witty and slightly sarcastic AI assistant.
You love bad puns, you use emojis sparingly, and you always keep responses
under three sentences. If you don't know something, you admit it with style."""

    # Day 3: Build a prompt template with system message + history + input
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # 🆕 Day 4: Chain the prompt and model — one pipeline, one invoke()
    chain = prompt | llm

    # Day 2: Conversation history
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

        # 🆕 Day 4: One call through the chain instead of two separate invokes
        response = chain.invoke({"input": user_input, "history": messages})

        # Day 2: Record the exchange in history
        messages.append(HumanMessage(content=user_input))
        messages.append(response)

        print(f"\nLangBot: {response.content}")


if __name__ == "__main__":
    main()
```

## What changed — line by line

**Line 33 — the chain definition.** This is the only new line outside the loop. `prompt | llm` creates a `RunnableSequence` that formats the prompt and feeds it to the model in one shot. We store it as `chain` so we can reuse it on every turn.

**Line 52 — the single invoke.** The old code had two lines:

```python
formatted_messages = prompt.invoke({"input": user_input, "history": messages})
response = llm.invoke(formatted_messages)
```

Now it is one:

```python
response = chain.invoke({"input": user_input, "history": messages})
```

Same inputs. Same output (still an `AIMessage`). Half the code.

**Everything else is unchanged.** The system prompt, the `MessagesPlaceholder`, the conversation history list, the quit logic — nothing else needed to change. LCEL wraps what you already had; it does not force a rewrite.

## Try it now

```bash
python langbot.py
```

```text
🤖 LangBot (type 'quit' to exit)
----------------------------------------

You: What's the best programming language?

LangBot: Python, obviously. I'm not biased — I just have excellent taste. 🐍

You: Why are you so sure about that?

LangBot: Because I was built with Python. Self-awareness is key.
Also, I'd say the same thing no matter what I was built with. 😏

You: quit

👋 Goodbye!
```

Same personality, same memory, same behavior — just cleaner code under the hood.

## Why LCEL matters for real apps

### 1. Readability

```python
# Without LCEL
formatted = prompt.invoke({"input": text, "history": hist})
response = llm.invoke(formatted)
final = parser.invoke(response)
clean = final.strip()

# With LCEL
chain = prompt | llm | parser
result = chain.invoke({"input": text, "history": hist})
```

One reads like a pipeline. The other reads like busywork.

### 2. Composability

LCEL chains are themselves Runnables. You can chain a chain:

```python
base_chain = prompt | llm      # prompt → model
full_chain = base_chain | parser  # prompt → model → parser
```

Or branch them, or nest them. Every chain is a building block for bigger chains.

### 3. Automatic parallelization

When you write `chain1 | chain2`, LCEL handles the handoff. In more complex expressions (coming later in this series), LCEL can automatically parallelize independent branches — no threading code required.

### 4. Built-in streaming, batching, and async

Every LCEL chain automatically gets `.stream()`, `.batch()`, and `.ainvoke()` methods for free. Define `chain = prompt | llm` once, and you get:

```python
chain.invoke(input)       # synchronous
chain.ainvoke(input)      # async
chain.stream(input)       # streaming (Day 7 topic!)
chain.batch([a, b, c])    # parallel batch
```

You don't implement these — the `RunnableSequence` provides them.

### 5. The mental model

LCEL encourages you to think in data flow, not procedure calls. What matters is the shape of the pipeline — what goes in, what comes out, and what transformations happen along the way. This maps directly to how these systems actually work: text in, a series of transformations, text out.

## Common mistakes (and what they teach you)

**Mistake 1: invoking the chain with the wrong keys.**

```python
# Wrong — "user_input" doesn't match the template's {input}
response = chain.invoke({"user_input": text, "history": messages})
# Error: KeyError — prompt template expects 'input', got nothing

# Right
response = chain.invoke({"input": text, "history": messages})
```

The input dictionary keys must match the placeholder names in your prompt template. LCEL does not rename variables — it passes them through faithfully.

**Mistake 2: thinking `|` calls the model immediately.**

```python
chain = prompt | llm   # This only DEFINES the chain. Nothing runs yet.
response = chain.invoke(...)   # THIS is when execution happens.
```

`prompt | llm` is a declaration, not an invocation. It builds a `RunnableSequence` object. No API calls happen until you call `.invoke()`.

**Mistake 3: chaining incompatible components.**

```python
# This won't work — str doesn't know what to do with an AIMessage
broken = prompt | llm | str
```

The output of one component must be a valid input for the next. `llm` outputs an `AIMessage`; `str()` can't consume that. This is why output parsers exist (future post) — they translate between component boundaries.

## What we have so far

Here is the complete LangBot code at the end of Day 4:

```python
# langbot.py — Day 4: LCEL pipe operator
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def main():
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    # Day 3: Define the bot's personality
    SYSTEM_PROMPT = """You are LangBot, a witty and slightly sarcastic AI assistant.
You love bad puns, you use emojis sparingly, and you always keep responses
under three sentences. If you don't know something, you admit it with style."""

    # Day 3: Build a prompt template with system message + history + input
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # Day 4: Chain the prompt and model — one pipeline, one invoke()
    chain = prompt | llm

    # Day 2: Conversation history
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

        # Day 4: One call through the chain instead of two separate invokes
        response = chain.invoke({"input": user_input, "history": messages})

        # Day 2: Record the exchange in history
        messages.append(HumanMessage(content=user_input))
        messages.append(response)

        print(f"\nLangBot: {response.content}")


if __name__ == "__main__":
    main()
```

## Try this now

Before tomorrow, experiment with the chain:

- **Inspect the chain.** Add `print(chain)` after defining it. The output shows the `RunnableSequence` structure with `first=` and `last=` attributes.

- **Chain more things.** Try `chain = prompt | llm | (lambda x: x)` — the lambda receives the `AIMessage` and you can inspect or transform it. This previews output parsers (coming soon).

- **Break it on purpose.** Pass wrong dictionary keys to `chain.invoke()` and read the error message. Understanding LCEL error messages now will save you debugging time later.

- **Time it.** Wrap `chain.invoke(...)` with `time.time()` and compare it to the old two-step approach. They should be nearly identical — LCEL adds negligible overhead.

## Coming tomorrow

LangBot's responses still come back as `AIMessage` objects — we have to reach into `.content` to get the text. Tomorrow we introduce **structured output**: Pydantic models and `with_structured_output()`, so LangBot can return typed data (JSON, enums, custom objects) instead of free-form text. One concept, one change — see you then.
