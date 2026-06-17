---
title: "LangBot Day 1: Building a Chatbot with LangChain from Scratch"
date: 2026-06-17T07:00:00+01:00
draft: false
description: "Kick off a hands-on series where we build a real chatbot app called LangBot using LangChain. Day 1: project setup, ChatOpenAI, and a working CLI."
tags:
  - python
  - langchain
  - ai
  - chatbot
  - tutorial
  - openai
  - chatopenai
  - cli
categories:
  - tutorials
series: ["LangBot: Build a Chatbot with LangChain"]
slug: "langbot-day-1-project-setup-basic-chatbot"
---

Welcome to **LangBot** — a daily tutorial series where we build one chatbot app from scratch using [LangChain](https://www.langchain.com/). Every post adds one new concept to the same codebase, so by the end you will have a production-ready chatbot and a solid mental model of how LangChain works.

No magic. No abandoned side projects. One app, built in public, one concept at a time.

## What we are building

LangBot is a conversational chatbot that runs in your terminal. Today it will be simple: you type a message, it replies. But every day this week we will add memory, prompt templates, structured output, retrieval, agents, streaming, and more — all in the same codebase.

Here is what Day 1 covers:

- Setting up a Python project with `uv`
- Installing LangChain, the OpenAI integration, and `python-dotenv` using `uv add`
- Managing API keys securely using a `.env` file
- Writing a minimal chatbot with `ChatOpenAI`
- Wrapping it in a clean CLI loop
- Running your first conversation

Let's go.

## Step 1: Create the project

Open a terminal and scaffold the project. We will use [uv](https://github.com/astral-sh/uv), a fast Python package and workflow manager, to initialize our project:

```bash
uv init langbot
cd langbot
```

This creates a new project directory with a `pyproject.toml` file to manage dependencies and a default `hello.py` file.

## Step 2: Install dependencies

We need three packages to start. We will add them to our project configuration using `uv add`:

```bash
uv add langchain langchain-openai python-dotenv
```

- `langchain` gives us chains, prompts, and the LCEL syntax.
- `langchain-openai` wraps the OpenAI chat models.
- `python-dotenv` allows us to load environment variables from a local `.env` file.

This command automatically creates a virtual environment in `.venv/` and pins your dependencies in `pyproject.toml`.

Check what we installed by viewing the `pyproject.toml` file:

```bash
cat pyproject.toml
```

You should see the packages listed under the `[project]` or `[dependency-groups]` section.

## Step 3: Set your API key

LangBot uses OpenAI under the hood, so you need an API key. Instead of exporting it manually in the terminal every session, create a `.env` file in the root of your project:

```env
OPENAI_API_KEY="sk-..."
```

Since we don't want to commit our secret API keys to version control, make sure to add it to a `.gitignore` file:

```bash
echo ".env" > .gitignore
```

Our Python script will load this file automatically at startup.

> **Tip:** Never commit your `.env` file to public repositories. If you do, your API key will be compromised and automatically revoked by OpenAI.

## Step 4: Write the chatbot

Create `langbot.py`:

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables from .env
load_dotenv()


def main():
    # 1. Create the model
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    # 2. Greet the user
    print("🤖 LangBot (type 'quit' to exit)")
    print("-" * 40)

    # 3. Conversation loop
    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        # 4. Call the model
        response = llm.invoke([HumanMessage(content=user_input)])

        # 5. Print the response
        print(f"\nLangBot: {response.content}")


if __name__ == "__main__":
    main()
```

That is the entire app. Let's walk through each piece.

### What each part does

**Lines 1-3 — imports.** We pull in `load_dotenv` (to load our `.env` file), `ChatOpenAI` (the chat model), and `HumanMessage` / `AIMessage` (LangChain's typed message objects). We will use `AIMessage` later when we add conversation history.

**Line 6 — loading configuration.** `load_dotenv()` reads the `.env` file and populates your environment variables. This ensures `ChatOpenAI` can find the `OPENAI_API_KEY` automatically.

**Line 11 — the model.** `ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)` creates a client pointed at OpenAI's fast, cheap model. `temperature=0.7` gives it a conversational tone without being too random.

**Lines 16-29 — the CLI loop.** A standard `while True` loop that reads from `input()`, quits on `exit`/`quit`/`q`, and calls the model on every non-empty line.

**Line 27 — `.invoke()`.** This is the core LangChain pattern. You pass a list of messages to `.invoke()` and get an `AIMessage` back. The `content` attribute holds the model's reply as a string.

## Step 5: Run it

You can run the script using `uv run`, which automatically executes the script in the context of the project's virtual environment:

```bash
uv run langbot.py
```

Alternatively, if you prefer a traditional workflow, you can activate the environment and run python directly:

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python langbot.py
```

Try a conversation:

```
🤖 LangBot (type 'quit' to exit)
----------------------------------------

You: Hello!

LangBot: Hi there! How can I help you today?

You: What is LangChain?

LangBot: LangChain is an open-source framework for building applications
powered by large language models. It provides abstractions for chaining
calls to LLMs, integrating with external tools and data sources, and
managing conversation state. Think of it as the "wiring" that connects
your app logic to AI models.

You: quit
👋 Goodbye!
```

That is a working chatbot in 30 lines of Python. No web framework, no frontend — just a model and a loop.

## Why ChatOpenAI instead of raw OpenAI?

You might be wondering: why not just use `openai.OpenAI().chat.completions.create(...)` directly?

Two reasons:

1. **LangChain gives you a consistent interface.** Later we will swap in Anthropic, Ollama, or any other provider with a one-line change. The `.invoke()` call stays the same.
2. **LangChain messages are composable.** `HumanMessage`, `AIMessage`, and `SystemMessage` can be combined, reordered, truncated, and fed into prompt templates. Raw dicts don't give you that.

## What we have so far

At the end of Day 1, LangBot is a single file:

```python
# langbot.py — Day 1: Basic conversational chatbot
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables from .env
load_dotenv()


def main():
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    print("🤖 LangBot (type 'quit' to exit)")
    print("-" * 40)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        response = llm.invoke([HumanMessage(content=user_input)])

        print(f"\nLangBot: {response.content}")


if __name__ == "__main__":
    main()
```

## Try this now

Before tomorrow, experiment with the code:

- Change `temperature` to `0.0` and `1.5` — notice how the tone shifts.
- Swap `gpt-4.1-mini` for `gpt-4.1` — see the difference in response quality and speed.
- Send a multi-line message by pressing Enter without typing `quit` — the bot handles it fine.
- Try asking a follow-up question that references something you said earlier. It won't remember — and that is exactly the problem we will solve tomorrow.

## Coming tomorrow

Right now LangBot has amnesia. Every message is treated as a brand-new conversation. Tomorrow we will add **conversation memory** using LangChain's `ConversationBufferMemory`, so LangBot can remember what you said earlier in the chat and carry on a real conversation.

See you then.
