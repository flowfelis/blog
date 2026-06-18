---
title: "LangBot Day 2: Giving Your Chatbot a Memory with Conversation History"
date: 2026-06-18T07:00:00+01:00
draft: false
description: "LangBot gets a memory. Day 2 adds conversation history so the bot remembers what you said — and replies like a real conversation partner."
tags:
  - python
  - langchain
  - ai
  - chatbot
  - tutorial
  - memory
  - conversation-history
categories:
  - tutorials
series: ["LangBot: Build a Chatbot with LangChain"]
slug: "langbot-day-2-conversation-memory"
---

**Recap:** Yesterday we built LangBot — a 27-line CLI chatbot powered by `ChatOpenAI`. It works, but every message starts from scratch. Ask "What's my name?" and it invents one. Remind it "My name is Alex" and ask again — it still guesses. Today we fix that.

---

## The problem: amnesia mode

Try this in Day 1's LangBot:

```
You: My favorite color is blue.

LangBot: That's a great choice! Blue is calming and versatile.

You: What did I just say my favorite color was?

LangBot: I'm not sure — you haven't mentioned a favorite color in this conversation.
```

Every call to `llm.invoke()` sends a single `HumanMessage`. The model has zero context from previous turns. For a real chatbot, this is a dealbreaker.

## The solution: message history

The fix is straightforward: instead of sending *just* the latest message, send *all* previous messages along with it. Every `HumanMessage` you typed and every `AIMessage` the model returned — the full transcript.

Here is how that looks in code. We replace one line and add two.

### Before (Day 1)

```python
response = llm.invoke([HumanMessage(content=user_input)])
```

We create a brand-new list with a single message. The model sees nothing else.

### After (Day 2)

```python
# Store conversation history as a list of messages
messages = []

# Inside the loop:
messages.append(HumanMessage(content=user_input))
response = llm.invoke(messages)
messages.append(response)
```

Three changes:

1. **Line 1 — declare history.** An empty `messages` list before the loop starts.
2. **Line 4 — append the user's message.** Before calling the model, we push a `HumanMessage` onto the end of the list.
3. **Lines 5-6 — invoke with full history, then record the reply.** We pass the entire list — every message so far — to `.invoke()`. Then we save the `AIMessage` response so it is included next time.

That is it. Three lines of code turn an amnesiac chatbot into one that remembers the whole conversation.

## Where it plugs in

Here is the Day 2 version of `langbot.py` with the changes highlighted:

```python
# langbot.py — Day 2: Conversation memory added
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


def main():
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    # 🆕 Day 2: Initialize conversation history
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

        # 🆕 Day 2: Build history, then call the model
        messages.append(HumanMessage(content=user_input))
        response = llm.invoke(messages)
        messages.append(response)

        print(f"\nLangBot: {response.content}")


if __name__ == "__main__":
    main()
```

The `AIMessage` import — which we included on Day 1 but never used — finally earns its keep. Every response the model returns is an `AIMessage` object, and we stash it in `messages` right after receiving it.

## Try it now

```bash
python langbot.py
```

```
🤖 LangBot (type 'quit' to exit)
----------------------------------------

You: My name is Alex and I live in Berlin.

LangBot: Nice to meet you, Alex! Berlin is a fantastic city — what do you
love most about living there?

You: The coffee scene is incredible.

LangBot: It really is! Have you been to The Barn or Five Elephant? Both are
legendary among Berlin coffee people.

You: What's my name and where do I live?

LangBot: Your name is Alex and you live in Berlin! 🎉
```

The bot remembered. No external database, no vector store — just a Python list that grows with every turn.

## How this works under the hood

OpenAI's chat models are stateless. They do not store anything between API calls. The model only "remembers" because we feed the entire conversation history back to it on each request.

Behind the scenes, `llm.invoke(messages)` sends this payload to the API:

```json
{
  "model": "gpt-4.1-mini",
  "messages": [
    {"role": "user", "content": "My name is Alex and I live in Berlin."},
    {"role": "assistant", "content": "Nice to meet you, Alex! Berlin is..."},
    {"role": "user", "content": "The coffee scene is incredible."},
    {"role": "assistant", "content": "It really is! Have you been to..."},
    {"role": "user", "content": "What's my name and where do I live?"}
  ]
}
```

LangChain's `HumanMessage` becomes `"role": "user"` and `AIMessage` becomes `"role": "assistant"`. The model sees the full transcript and responds accordingly.

## A word on ConversationBufferMemory

If you browse LangChain's documentation, you will encounter `ConversationBufferMemory` — a class that does exactly what we just built:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
```

It is a wrapper around a message list with a few extra features (token counting, pruning, key mapping). We will explore those later. But for now, the raw list approach matters because it makes the mechanism *visible*. You can `print(messages)`, inspect it in a debugger, and see exactly what the model sees.

Once you understand the list, the higher-level memory classes will feel like conveniences — not magic.

## One gotcha: token limits

Every message you add makes the payload larger. GPT-4.1-mini has a 128k token context window, so a CLI chat session will not hit it in practice. But if you built a long-running service that holds thousands of turns, you would eventually overflow the limit.

LangChain handles this with **conversation summarization** and **token-buffer memory** — topics for a future post.

## What we have so far

Here is the complete LangBot code at the end of Day 2:

```python
# langbot.py — Day 2: Conversation memory added
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


def main():
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    # Day 2: Conversation history — a running transcript of every turn
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

        messages.append(HumanMessage(content=user_input))
        response = llm.invoke(messages)
        messages.append(response)

        print(f"\nLangBot: {response.content}")


if __name__ == "__main__":
    main()
```

## Try this now

Before tomorrow, experiment:

- **Clear history mid-session.** Add a command like `if user_input == "/clear": messages = []; print("Memory cleared."); continue` — reset the bot without restarting.
- **Inspect the message list.** Add `print(f"[DEBUG] {len(messages)} messages in history")` after `.invoke()` to see the list grow.
- **Simulate a long conversation.** Paste a paragraph, then ask about it 20 turns later. Watch the model retrieve details from the transcript.

## Coming tomorrow

LangBot remembers the conversation now, but it has no personality. Every session starts the same way — no system prompt, no tone, no behavior rules. Tomorrow we add **prompt templates with `ChatPromptTemplate`** and give LangBot a distinct voice: a witty, sarcastic assistant, a formal butler, or whatever persona you choose.

See you then.
