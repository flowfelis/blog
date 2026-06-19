---
title: "LangBot Day 3: Giving Your Chatbot a Personality with Prompt Templates"
date: 2026-06-19T07:00:00+01:00
draft: false
description: "LangBot gets a personality. Day 3 introduces ChatPromptTemplate and SystemMessage — separating what the bot says from how it behaves."
tags:
  - python
  - langchain
  - ai
  - chatbot
  - tutorial
  - prompt-templates
  - chatprompttemplate
  - system-message
categories:
  - tutorials
series: ["LangBot: Build a Chatbot with LangChain"]
slug: "langbot-day-3-prompt-templates"
---

**Recap:** On Day 1 we built a CLI chatbot with `ChatOpenAI`. On Day 2 we gave it memory so it remembers the conversation. But LangBot still has no personality — every session starts with a blank slate, and the model defaults to its generic "helpful assistant" tone. Today we change that.

---

## The problem: a chatbot with no identity

Right now, LangBot's response to *any* first message is unpredictable. Ask "What do you do?" and you get whatever the base model decides. There is no way to control:

- **Tone** — friendly? formal? sarcastic?
- **Behavior** — concise or verbose? does it use emojis? does it admit when it doesn't know something?
- **Constraints** — should it refuse certain topics? speak only in haikus?

Every real chatbot has a personality defined by a **system prompt** — a hidden message injected at the top of every conversation that tells the model *how* to behave. LangChain models this with `ChatPromptTemplate` and `SystemMessage`.

## The solution: ChatPromptTemplate + SystemMessage

LangChain provides three message types for building prompt templates:

| Class | Purpose |
|---|---|
| `SystemMessage` | Sets the model's behavior, tone, and constraints. Injected once at the top. |
| `HumanMessage` | The user's input. Can include variable placeholders like `{input}`. |
| `AIMessage` | The model's previous responses. Used for conversation history. |

A `ChatPromptTemplate` combines these into a reusable template. Instead of manually building a message list every time, you define the structure once and fill it with variables at runtime.

Here is the new code we are adding today:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define the bot's personality
SYSTEM_PROMPT = """You are LangBot, a witty and slightly sarcastic AI assistant.
You love bad puns, you use emojis sparingly, and you always keep responses
under three sentences. If you don't know something, you admit it with style."""

# Build a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Use it: fill the template, then send to the model
formatted_messages = prompt.invoke({
    "input": user_input,
    "history": messages,
})
response = llm.invoke(formatted_messages)
```

Three things happening:

1. **The template defines the structure** — system message first, then conversation history, then the user's latest input.
2. **`prompt.invoke()` fills the variables** — replaces `{input}` with the user's text and `{history}` with the message list. Returns a `ChatPromptValue` (which behaves like a list of messages).
3. **`llm.invoke()` processes the result** — same `llm.invoke()` we have used since Day 1. The template produces the message list, the model produces the response.

## Where it plugs in

Here is the Day 3 version of `langbot.py`. Look for the `# 🆕 Day 3` markers:

```python
# langbot.py — Day 3: Prompt templates and system message
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

def main():
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    # 🆕 Day 3: Define the bot's personality
    SYSTEM_PROMPT = """You are LangBot, a witty and slightly sarcastic AI assistant.
You love bad puns, you use emojis sparingly, and you always keep responses
under three sentences. If you don't know something, you admit it with style."""

    # 🆕 Day 3: Build a prompt template with system message + history + input
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

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

        # 🆕 Day 3: Format the prompt template, then send to model
        formatted_messages = prompt.invoke({
            "input": user_input,
            "history": messages,
        })
        response = llm.invoke(formatted_messages)

        # Day 2: Record the exchange in history
        messages.append(HumanMessage(content=user_input))
        messages.append(response)

        print(f"\nLangBot: {response.content}")


if __name__ == "__main__":
    main()
```

## What changed — line by line

**Lines 4 and 10-15 — the prompt template.** We import `ChatPromptTemplate` and `MessagesPlaceholder`, then define a template with three slots:

1. `("system", SYSTEM_PROMPT)` — a static `SystemMessage` that sets the bot's personality. This is always the first thing the model sees.
2. `MessagesPlaceholder(variable_name="history")` — a dynamic slot where the conversation transcript gets injected at runtime. This is how Day 2's `messages` list connects to the template.
3. `("human", "{input}")` — a `HumanMessage` template with a `{input}` placeholder that gets replaced by the user's actual text.

**Lines 39-42 — formatting and invoking.** Instead of calling `llm.invoke(messages)` directly like on Day 2, we first run the user input and history through the prompt template:

```python
formatted_messages = prompt.invoke({
    "input": user_input,
    "history": messages,
})
response = llm.invoke(formatted_messages)
```

The dictionary keys (`"input"` and `"history"`) must match the placeholder names in the template. `prompt.invoke()` produces a `ChatPromptValue` — which acts like a list of messages and can be passed directly to `llm.invoke()`.

**Lines 45-46 — history recording stays the same.** As on Day 2, we push `HumanMessage` and `AIMessage` onto the `messages` list after the call, so they feed into the next turn.

## Try it now

```bash
python langbot.py
```

```text
🤖 LangBot (type 'quit' to exit)
----------------------------------------

You: What's 2 + 2?

LangBot: Four. Groundbreaking math, I know. 🎉

You: Tell me about quantum physics.

LangBot: I'd need a PhD and a bigger context window to do it justice.
Let's just say Schrödinger's cat is both alive and tired of being a metaphor.

You: What's my name?

LangBot: You haven't told me — my memory works, not my psychic powers. 😏
```

The difference is immediate. LangBot now has a distinct voice: witty, slightly sarcastic, emoji-aware, and self-deprecating about its limits.

## Why ChatPromptTemplate matters for real apps

In production, you almost never use a raw model without a prompt template. Here is why:

### 1. Separation of concerns

The system prompt lives in one place. If your product manager says "make the tone friendlier," you change one string — not every place the model is called.

### 2. Version control for prompts

Prompts are code. With `ChatPromptTemplate`, your system prompt lives in source control alongside the rest of your app. You can diff it, review it in PRs, and roll it back if a personality change goes wrong.

### 3. Multi-turn context with zero overhead

The `MessagesPlaceholder` slot is empty on turn 1 (an empty list) and grows with every turn. You don't need conditional logic — the template handles both the cold-start case and the 50-turn-deep case identically.

### 4. Provider portability

The `("system", "...")` tuple syntax is provider-agnostic. If you swap `ChatOpenAI` for `ChatAnthropic` tomorrow, the prompt template works unchanged — LangChain translates it to the right format for each provider.

### 5. Dynamic prompts at runtime

Templates support variables. You can inject the current date, the user's name, or any runtime data:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant. Today is {today}. The user's name is {name}."),
    ("human", "{input}"),
])
formatted = prompt.invoke({
    "input": "What day is it?",
    "today": "Thursday",
    "name": "Alex",
})
response = llm.invoke(formatted)
```

## What we have so far

Here is the complete LangBot code at the end of Day 3:

```python
# langbot.py — Day 3: Prompt templates and system message
from dotenv import load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

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

        # Day 3: Format the prompt template, then send to model
        formatted_messages = prompt.invoke({
            "input": user_input,
            "history": messages,
        })
        response = llm.invoke(formatted_messages)

        # Day 2: Record the exchange in history
        messages.append(HumanMessage(content=user_input))
        messages.append(response)

        print(f"\nLangBot: {response.content}")


if __name__ == "__main__":
    main()
```

## Try this now

Before tomorrow, experiment with the system prompt:

- **Change the personality.** Make LangBot a formal Victorian butler, a pirate, or a therapist. Observe how the same user input produces wildly different responses.
- **Add constraints.** Try `"Never use the word 'the'"` or `"Always respond in exactly two sentences."` — see how well the model follows the rules.
- **Inject runtime data.** Add a `{username}` variable to the system prompt and pass a name when calling `prompt.invoke()`.
- **Watch the history.** Add `print(f"[HISTORY] {len(messages)} messages")` before `prompt.invoke()` to see the transcript grow.

## Coming tomorrow

LangBot has a personality now, but calling `prompt.invoke()` then `llm.invoke()` as two separate steps is slightly verbose. Tomorrow we introduce **LCEL (LangChain Expression Language)** — the `|` syntax that lets you write `prompt | llm` and turn two steps into one clean chain.

See you then.
