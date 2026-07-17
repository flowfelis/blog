---
title: "LangBot Day 10: Agents and Tool Calling — Let the LLM Decide What to Do"
date: 2026-06-27T07:00:00+01:00
draft: false
description: "LangBot stops following a script. Day 10 replaces hardcoded if/elif routing with an LLM-powered agent that decides — autonomously — whether to search documents, suggest a list, or just chat. One tool at a time."
tags:
  - python
  - langchain
  - ai
  - chatbot
  - tutorial
  - agents
  - tool-calling
  - function-calling
  - agent-executor
  - rag
  - lcel
categories:
  - tutorials
series: ["LangBot: Build a Chatbot with LangChain"]
slug: "langbot-day-10-agents"
---

**Recap:** On Day 9, LangBot became a full RAG pipeline — every chat message retrieves relevant documents and the LLM answers from them. The `/search` command, list-mode routing, and regular chat all coexist. But there is a problem hiding in plain sight: the routing logic is hand-coded in `if`/`elif` blocks. LangBot doesn't *decide* anything — it follows a script. Today we replace that script with an **agent**: an LLM that reasons about which tool to use, calls it, and responds — autonomously.

---

## The problem: hardcoded routing doesn't scale

Here is the routing logic from Day 9:

```python
if user_input.startswith("/search"):
    # explicit search
    ...
elif wants_list(user_input):
    # list mode
    ...
else:
    docs = search_docs(vector_store, user_input, k=3)
    if docs:
        # RAG
        ...
    else:
        # regular chat
        ...
```

This works for three capabilities. But what happens when we add more?

```python
Day 12: chat history persistence → new code path
Day 13: caching → new code path
Day 14: guardrails → new code path, possibly before everything else
Day 15: multi-modal input → new code path
Day 18: hybrid search → new code path that replaces RAG routing
```

Every new capability means another `if`/`elif` branch. The branching logic multiplies. The ordering matters (guardrails must run first, but caching might short-circuit everything). The `wants_list()` keyword heuristic was always fragile — what if the user says *"Don't list all the policies, just tell me about the PTO one"*? The heuristic sees "list" and routes to list mode, producing a bullet-point list when the user explicitly asked not to.

The fundamental issue: **we are making routing decisions with Python code, but the LLM is better at understanding the user's intent than any `if` statement we can write.**

---

## The solution: agents that choose their own tools

An **agent** is an LLM that decides what to do next. Instead of hardcoding the decision tree, you give the LLM a set of **tools** — Python functions it can call — and let it pick the right one for each user message:

```
                   ┌─────────────────────────────────┐
                   │          User Message            │
                   │  "How much PTO do I get?"        │
                   └───────────────┬─────────────────┘
                                   │
                                   ▼
                   ┌─────────────────────────────────┐
                   │        Agent (LLM decides)       │
                   │  "This is a document question.   │
                   │   I'll call search_docs."        │
                   └───────────────┬─────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │search_docs│ │list_items│  │  chat    │
              │  (tool)   │ │  (tool)  │  │ (no tool)│
              └─────┬─────┘ └─────┬────┘  └─────┬────┘
                    │              │              │
                    ▼              ▼              ▼
              ┌─────────────────────────────────────┐
              │   Tool output → back to agent →     │
              │   final answer                      │
              └─────────────────────────────────────┘
```

The agent's decision is not based on keywords or heuristics — it is based on the LLM's understanding of the user's intent. *"How much PTO do I get?"* → agent reasons *"This sounds like a question about company policy. I should search the documents."* It calls `search_docs`, gets the results, and synthesizes an answer.

---

## What is a tool?

A tool in LangChain is just a Python function with a description. The description tells the LLM what the tool does, when to use it, and what parameters it needs. LangChain serializes the tool metadata (name, description, parameter schema) and sends it alongside the chat prompt so the model knows what is available.

```python
from langchain.tools import tool

@tool
def search_docs_tool(query: str) -> str:
    """Search the company document library for policies, benefits, and procedures.
    Use this when the user asks about company policies, benefits, HR questions,
    remote work rules, or anything likely to be in internal documents."""
    docs = search_docs(vector_store, query, k=3)
    if not docs:
        return "No relevant documents found."
    return format_context(docs)
```

That is it. The `@tool` decorator reads the function's signature (`query: str`), docstring, and return type, and generates a schema the LLM can understand. The docstring — especially the "Use this when…" guidance — is critical: it is how the agent decides whether to call this tool.

---

## Step 1: Install the dependency

One new package today: `langchain-community` provides the `@tool` decorator and `create_tool_calling_agent`. If you have been following since Day 8, you already have it. If not:

```bash
uv add langchain-community
```

No new model, no new API key. The agent reuses the same `ChatOpenAI` LLM you have been using since Day 1.

---

## Step 2: Wrap LangBot's capabilities as tools

We have three capabilities to convert into tools:

### Tool 1: Search documents

```python
from langchain.tools import tool

@tool
def search_docs_tool(query: str) -> str:
    """Search the company document library for policies, benefits, and procedures.

    Use this when the user asks about company policies, benefits, HR questions,
    remote work rules, expenses, time off, or anything likely to be in internal
    documents. Do NOT use this for general knowledge questions (capital cities,
    math, coding help) — those don't need document retrieval."""
    docs = search_docs(vector_store, query, k=3)
    if not docs:
        return "No relevant documents found."
    return format_context(docs)
```

This wraps the existing `search_docs()` + `format_context()` from Days 8–9. The agent sees a function called `search_docs_tool` that takes a `query` string and returns formatted document text. The docstring's "Use this when… / Do NOT use this when…" instructions are the agent's decision-making guide.

### Tool 2: List suggestions

```python
@tool
def list_suggestions_tool(topic: str, count: int = 5) -> str:
    """Generate a list of suggestions or recommendations on a topic.

    Use this when the user explicitly asks for a list, suggestions, recommendations,
    top N items, or ideas. Examples: 'list 5 programming books', 'suggest weekend
    activities', 'recommend sci-fi movies'. Do NOT use this for factual questions
    or document searches."""
    items = list_chain.invoke({"input": f"List {count} {topic}", "history": messages})
    if not items:
        return "Could not generate suggestions."
    lines = [f"  {i}. {item}" for i, item in enumerate(items, 1)]
    return "\n".join([f"Here are {len(items)} {topic} suggestions:"] + lines)
```

The `list_chain` from Day 6 (LLM → `CommaSeparatedListOutputParser`) now becomes a tool the agent can call. No more keyword heuristic — the agent decides, from the semantics of the user's message, whether this is a list request.

### Tool 3: General chat (no tool)

There is no tool for general chat. If the agent decides none of the tools are appropriate — the user is saying hello, telling a joke, asking a general knowledge question — it simply responds directly. This is the agent's *fallback* behavior: no tool call, just a normal chat response.

---

## Step 3: Create the agent

LangChain's modern agent API has two parts: `create_tool_calling_agent` (which builds the prompt + chat model pipeline) and `AgentExecutor` (which runs the agent in a loop, handling tool calls and responses).

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 🆕 Day 10: The agent prompt — tells the model HOW to behave as an agent
AGENT_SYSTEM_PROMPT = """You are LangBot, a witty and slightly sarcastic AI assistant.
You love bad puns, you use emojis sparingly, and you always keep responses
under three sentences. If you don't know something, you admit it with style.

You have access to tools. Use them when they are relevant:
- search_docs_tool: for questions about company policies, benefits, or internal docs
- list_suggestions_tool: when the user asks for a list, suggestions, or recommendations

If neither tool is relevant, just respond directly — no tool call needed.

When you use search_docs_tool and find relevant documents:
- Answer from the provided context
- Cite which document(s) you used
- If the context doesn't contain the answer, say so honestly

Be decisive. If a tool would help, use it immediately. Don't ask the user
whether they want you to search — just search and answer."""

# 🆕 Day 10: Assemble the agent prompt
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", AGENT_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 🆕 Day 10: Create the agent — LLM + tools + prompt
tools = [search_docs_tool, list_suggestions_tool]
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# 🆕 Day 10: AgentExecutor — runs the agent loop
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,          # Set True during debugging to see agent reasoning
    handle_parsing_errors=True,
    max_iterations=3,       # Prevent infinite loops
)
```

The `agent_scratchpad` is an internal `MessagesPlaceholder` where LangChain stores the agent's intermediate reasoning — tool call requests, tool outputs, and the agent's reflection on those outputs. You never set this yourself; `AgentExecutor` populates it automatically during the agent loop.

Here is what happens inside `agent_executor.invoke()` for a single user message:

```
1. Agent receives: system prompt + history + user message
2. Agent outputs EITHER:
   a. ToolCall(name="search_docs_tool", args={"query": "PTO policy"})
      → AgentExecutor runs the tool, gets the output
      → Agent receives tool output, reasons about it
      → Agent outputs final answer
   b. AIMessage("You get 2 days of PTO per month...")
      → AgentExecutor returns this as the final answer
```

The loop can run up to `max_iterations` times — the agent could call one tool, reflect, call another, reflect, and then answer. In practice, for the simple tools we have, one iteration is sufficient.

---

## Step 4: Update the main loop

The main loop simplifies dramatically. Gone are the `if`/`elif` branches for list-mode routing and RAG fallback. One call handles everything:

```python
while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ("quit", "exit", "q"):
        print("👋 Goodbye!")
        break

    if not user_input:
        continue

    # ── Day 8: /search command (kept for explicit searches) ──
    if user_input.startswith("/search"):
        query = user_input.removeprefix("/search").strip()
        if not query:
            print("\n⚠ Usage: /search <your query>")
            continue
        if vector_store is None:
            print("\n⚠ Vector store not available.")
            continue

        results = search_docs(vector_store, query, k=3)
        if not results:
            print(f"\n📭 No relevant documents found for: \"{query}\"")
        else:
            print(f"\n📖 Top {len(results)} results for: \"{query}\"")
            print("─" * 40)
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get("source", "unknown")
                preview = doc.page_content[:200].replace("\n", " ")
                print(f"\n  [{i}] 📄 {source}")
                print(f"      {preview}...")
            print(f"\n─" * 40)

        messages.append(HumanMessage(content=user_input))
        summary = f"Searched for '{query}': found {len(results)} document(s)."
        messages.append(AIMessage(content=summary))
        continue

    # ═══════════════════════════════════════════════════════════
    # 🆕 Day 10: Agent handles everything — chat, RAG, lists
    # ═══════════════════════════════════════════════════════════

    print(f"\nLangBot: ", end="", flush=True)

    full_response = ""
    for chunk in agent_executor.stream({
        "input": user_input,
        "history": messages,
    }):
        # agent_executor.stream() yields dicts with "output" key
        if "output" in chunk:
            text = chunk["output"]
            print(text, end="", flush=True)
            full_response += text

    print()

    # Day 2: Store the exchange in history
    messages.append(HumanMessage(content=user_input))
    messages.append(AIMessage(content=full_response))
```

That is the entire main loop. No `wants_list()` heuristic. No `if docs: … else: …` RAG routing. The agent decides everything. The `agent_executor.stream()` call yields chunks as the agent generates its final answer — preserving the streaming experience from Day 7.

---

## Where it plugs in

Here is the full Day 10 `langbot.py`. Look for the `# 🆕 Day 10` markers:

```python
# langbot.py — Day 10: Agents and tool calling
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from enum import Enum
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool

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


# ═══════════════════════════════════════════════════════════════════
# Day 8: Document retrieval infrastructure (unchanged)
# ═══════════════════════════════════════════════════════════════════

def load_documents(docs_dir: str = "docs") -> list:
    """Load all .txt and .md files from a directory. Returns a list of Documents."""
    documents = []
    for pattern in ["**/*.txt", "**/*.md"]:
        try:
            loader = DirectoryLoader(docs_dir, glob=pattern, loader_cls=TextLoader,
                                     silent_errors=True)
            documents.extend(loader.load())
        except Exception:
            pass
    return documents


def chunk_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """Split long documents into smaller, overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def setup_vector_store(chunks: list, index_name: str = "langbot-docs"):
    """Embed all chunks and store them in Pinecone. Returns the vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    return vector_store


def search_docs(vector_store, query: str, k: int = 3) -> list:
    """Search the vector store for the k most relevant document chunks."""
    if vector_store is None:
        return []
    return vector_store.similarity_search(query, k=k)


# ═══════════════════════════════════════════════════════════════════
# Day 9: RAG context formatting (unchanged)
# ═══════════════════════════════════════════════════════════════════

def format_context(docs: list) -> str:
    """Format retrieved documents into a context string for the prompt."""
    if not docs:
        return "No relevant documents found."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Document {i} — {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def main():
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    # Day 3: Personality (used by list_chain, agent has its own prompt)
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

    # Day 3: Chat prompt template (still used by list_chain)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # Day 5: Structured output chain
    structured_llm = llm.with_structured_output(LangBotResponse)
    chat_chain = prompt | structured_llm

    # Day 6: List-mode output parser
    list_parser = CommaSeparatedListOutputParser()

    # Day 6: List-specific prompt
    LIST_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + list_parser.get_format_instructions()

    list_prompt = ChatPromptTemplate.from_messages([
        ("system", LIST_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # Day 6: List chain — prompt → raw LLM → comma-split parser → list[str]
    list_chain = list_prompt | llm | list_parser

    # ═══════════════════════════════════════════════════════════
    # Day 8: Set up document retrieval at startup
    # ═══════════════════════════════════════════════════════════
    vector_store = None
    chunk_count = 0

    print("📚 Loading documents from docs/ ...")
    try:
        documents = load_documents()
        if documents:
            doc_count = len(documents)
            chunks = chunk_documents(documents)
            chunk_count = len(chunks)
            print(f"   Loaded {doc_count} document(s) → split into {chunk_count} chunks")

            print("🔢 Embedding chunks and storing in Pinecone ...")
            vector_store = setup_vector_store(chunks)
            print(f"   ✓ Vector store ready ({chunk_count} vectors indexed)")
        else:
            print("   (No documents found in docs/ — search will be unavailable)")
            print("   (Create docs/*.txt or docs/*.md files and restart)")
    except Exception as e:
        print(f"   ⚠ Vector store setup skipped: {e}")
        print("   (Make sure PINECONE_API_KEY is set and the index exists)")

    # ═══════════════════════════════════════════════════════════
    # 🆕 Day 10: Define tools for the agent
    # ═══════════════════════════════════════════════════════════

    @tool
    def search_docs_tool(query: str) -> str:
        """Search the company document library for policies, benefits, and procedures.

        Use this when the user asks about company policies, benefits, HR questions,
        remote work rules, expenses, time off, or anything likely to be in internal
        documents. Do NOT use this for general knowledge questions (capital cities,
        math, coding help) — those don't need document retrieval."""
        docs = search_docs(vector_store, query, k=3)
        if not docs:
            return "No relevant documents found."
        return format_context(docs)

    @tool
    def list_suggestions_tool(topic: str, count: int = 5) -> str:
        """Generate a list of suggestions or recommendations on a topic.

        Use this when the user explicitly asks for a list, suggestions,
        recommendations, top N items, or ideas. Examples: 'list 5 programming
        books', 'suggest weekend activities', 'recommend sci-fi movies'.
        Do NOT use this for factual questions or document searches."""
        items = list_chain.invoke({
            "input": f"List {count} {topic}",
            "history": messages,
        })
        if not items:
            return "Could not generate suggestions."
        lines = [f"  {i}. {item}" for i, item in enumerate(items, 1)]
        return "\n".join([f"Here are {len(items)} {topic} suggestions:"] + lines)

    # ═══════════════════════════════════════════════════════════
    # 🆕 Day 10: Build the agent
    # ═══════════════════════════════════════════════════════════

    AGENT_SYSTEM_PROMPT = """You are LangBot, a witty and slightly sarcastic AI assistant.
You love bad puns, you use emojis sparingly, and you always keep responses
under three sentences. If you don't know something, you admit it with style.

You have access to tools. Use them when they are relevant:
- search_docs_tool: for questions about company policies, benefits, or internal docs
- list_suggestions_tool: when the user asks for a list, suggestions, or recommendations

If neither tool is relevant, just respond directly — no tool call needed.

When you use search_docs_tool and find relevant documents:
- Answer from the provided context
- Cite which document(s) you used
- If the context doesn't contain the answer, say so honestly

Be decisive. If a tool would help, use it immediately. Don't ask the user
whether they want you to search — just search and answer."""

    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    tools = [search_docs_tool, list_suggestions_tool]
    agent = create_tool_calling_agent(llm, tools, agent_prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3,
    )

    # Day 2: Conversation history
    messages = []

    print("🤖 LangBot (type 'quit' to exit)")
    print("─" * 40)
    print("  Agent mode — LangBot decides whether to search, list, or chat")
    print("  Just ask naturally — the agent picks the right tool 🤖🔧")
    print("─" * 40)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        # ── Day 8: /search command (kept for explicit searches) ──
        if user_input.startswith("/search"):
            query = user_input.removeprefix("/search").strip()
            if not query:
                print("\n⚠ Usage: /search <your query>")
                continue
            if vector_store is None:
                print("\n⚠ Vector store not available.")
                continue

            results = search_docs(vector_store, query, k=3)
            if not results:
                print(f"\n📭 No relevant documents found for: \"{query}\"")
            else:
                print(f"\n📖 Top {len(results)} results for: \"{query}\"")
                print("─" * 40)
                for i, doc in enumerate(results, 1):
                    source = doc.metadata.get("source", "unknown")
                    preview = doc.page_content[:200].replace("\n", " ")
                    print(f"\n  [{i}] 📄 {source}")
                    print(f"      {preview}...")
                print(f"\n─" * 40)

            messages.append(HumanMessage(content=user_input))
            summary = f"Searched for '{query}': found {len(results)} document(s)."
            messages.append(AIMessage(content=summary))
            continue

        # ═══════════════════════════════════════════════════════
        # 🆕 Day 10: Agent handles everything — chat, RAG, lists
        # ═══════════════════════════════════════════════════════

        print(f"\nLangBot: ", end="", flush=True)

        full_response = ""
        for chunk in agent_executor.stream({
            "input": user_input,
            "history": messages,
        }):
            if "output" in chunk:
                text = chunk["output"]
                print(text, end="", flush=True)
                full_response += text

        print()

        # Day 2: Store the exchange in history
        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=full_response))


if __name__ == "__main__":
    main()
```

---

## Try it now

```bash
python langbot.py
```

```text
📚 Loading documents from docs/ ...
   Loaded 3 document(s) → split into 14 chunks
🔢 Embedding chunks and storing in Pinecone ...
   ✓ Vector store ready (14 vectors indexed)
🤖 LangBot (type 'quit' to exit)
────────────────────────────────────────
  Agent mode — LangBot decides whether to search, list, or chat
  Just ask naturally — the agent picks the right tool 🤖🔧
────────────────────────────────────────

You: How much PTO do I get?

LangBot: According to the remote work policy, you get 2 days of PTO per month,
up to a maximum of 24 days per year. 🏖️

You: Suggest 5 weekend activities

LangBot: Here are 5 weekend activities suggestions:
  1. Go for a hike or nature walk
  2. Visit a local museum or art gallery
  3. Try a new recipe or bake something
  4. Have a movie marathon with friends
  5. Start a DIY project or craft
Weekend, sorted! 🎨🥾

You: What's the capital of France?

LangBot: Paris — and no, I didn't need to search any documents for that one. 🇫🇷

You: What health insurance plans does the company offer?

LangBot: The company offers three plans — HMO, PPO, and HDHP with HSA — per the
benefits document. The company covers 80% of your premium. 🏥

You: Tell me a joke

LangBot: Why do programmers prefer dark mode? Because light attracts bugs! 🐛💡

You: /search remote work

📖 Top 3 results for: "remote work"
────────────────────────────────────────

  [1] 📄 docs/remote-work-policy.txt
      Remote Work Policy — Section 1: Eligibility All full-time employees...

  [2] 📄 docs/benefits.txt
      ...home office stipend of $500 per year for eligible remote employees...

  [3] 📄 docs/engineering.txt
      ...remote pairing sessions are encouraged but not mandatory...

────────────────────────────────────────

You: quit
👋 Goodbye!
```

Notice what the agent does on each turn:

| User says | Agent's decision | Why |
|---|---|---|
| "How much PTO?" | Calls `search_docs_tool` | It sounds like a policy question |
| "Suggest 5 weekend activities" | Calls `list_suggestions_tool` | Explicit list/suggest request |
| "What's the capital of France?" | Responds directly | General knowledge, no tool needed |
| "What health insurance plans?" | Calls `search_docs_tool` | Benefits are in the documents |
| "Tell me a joke" | Responds directly | Neither tool is appropriate |
| "/search remote work" | Explicit command | Still works — bypasses the agent |

No `if`/`elif`. No keyword matching. No hardcoded decision tree. The LLM understood intent and chose the right tool every time.

---

## What changed — line by line

**Lines 15–16 — new imports.** `create_tool_calling_agent` and `AgentExecutor` come from `langchain.agents`. `tool` comes from `langchain.tools`. These are the only new dependencies.

**Lines 184–208 — tool definitions.** Two `@tool`-decorated functions. Each has a name, a typed signature (`query: str`, `topic: str, count: int = 5`), and a docstring that doubles as the agent's instruction manual. The docstring's "Use this when… / Do NOT use this when…" pattern is not just documentation — it is the primary mechanism by which the agent decides to call the tool.

**Lines 211–236 — agent assembly.** Four pieces come together:

1. `AGENT_SYSTEM_PROMPT` — instructs the LLM to act as an agent: use tools when relevant, respond directly when not, and be decisive.
2. `agent_prompt` — a `ChatPromptTemplate` with `{agent_scratchpad}` for the agent's internal reasoning loop.
3. `create_tool_calling_agent(llm, tools, agent_prompt)` — builds the agent from the LLM, tools, and prompt. Under the hood, this serializes each tool's name, description, and parameter schema into the system message so the model knows what is available.
4. `AgentExecutor` — wraps the agent in an execution loop. When the agent outputs a tool call, `AgentExecutor` runs the tool, feeds the output back to the agent, and repeats until the agent produces a final answer.

**Lines 271–283 — the new main loop.** The agent call replaces three code paths (list mode, RAG chat, regular chat) with one:

```python
for chunk in agent_executor.stream({"input": user_input, "history": messages}):
    if "output" in chunk:
        text = chunk["output"]
        print(text, end="", flush=True)
        full_response += text
```

`agent_executor.stream()` yields a sequence of dicts. When the agent is calling a tool, you get intermediate chunks with keys like `"actions"` and `"steps"`. When the agent produces its final answer, you get chunks with `"output"`. Filtering for `"output"` gives you the streaming text the user should see.

---

## Why this pattern matters for real apps

### 1. Separation of concerns: reasoning vs. execution

The agent architecture separates two responsibilities that were tangled together in Days 1–9:

| Role | Who does it | Example |
|---|---|---|
| **Reasoning** | The agent (LLM) | "This is a policy question → I should search" |
| **Execution** | The tools (Python functions) | `search_docs(vector_store, "PTO policy", k=3)` |

Before Day 10, the reasoning was hardcoded in `if`/`elif` branches. Now the LLM does the reasoning — and the LLM is better at it than any hand-written condition. It understands semantics, not just keywords. It can handle edge cases ("Don't list all the policies, just tell me about the PTO one") without special-casing.

### 2. Adding a new capability is adding a new tool

Tomorrow (Day 11), we will add custom tools. The change is two lines:

```python
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Use for arithmetic, unit conversions, percentages."""
    ...

tools = [search_docs_tool, list_suggestions_tool, calculator]  # +1 tool
```

No new routing logic. No `elif` branch. No ordering concerns. The agent automatically considers the new tool alongside the existing ones and decides when to use it. The complexity stays linear — O(n) in the number of tools — instead of combinatorial.

### 3. The agent prompt is the control surface

Everything the agent knows about when and how to use tools comes from two places:

1. **The system prompt** — high-level instructions: "Be decisive. Use tools when relevant."
2. **The tool docstrings** — per-tool guidance: "Use this when… / Do NOT use this when…"

Both are plain English. You tune agent behavior by editing text, not code. Want the agent to be more conservative about document searches? Change the docstring: *"Only use this when the user explicitly references company documents or policies."* Want it to always try retrieval first? *"Always search the document library before answering — even for general questions."*

---

## Common mistakes (and what they teach you)

**Mistake 1: vague tool descriptions.**

```python
# Wrong — too vague, the agent won't know when to call this
@tool
def search_docs_tool(query: str) -> str:
    """Search documents."""
    ...

# Right — specific guidance about when (and when not) to use it
@tool
def search_docs_tool(query: str) -> str:
    """Search the company document library for policies, benefits, and procedures.
    Use this when the user asks about company policies, benefits, HR questions,
    remote work rules, expenses, time off, or anything likely to be in internal
    documents. Do NOT use for general knowledge questions."""
    ...
```

The tool description is the agent's only information about what the tool does and when to call it. A vague description produces vague decisions. Be specific, give examples, and include negative guidance ("Do NOT use for…").

**Mistake 2: forgetting `agent_scratchpad` in the prompt template.**

```python
# Wrong — missing agent_scratchpad
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", AGENT_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    # Missing: MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Error: KeyError: 'agent_scratchpad' when running the agent
```

`agent_scratchpad` is where the agent loop stores its intermediate messages — tool call requests and tool outputs. Without this placeholder, `AgentExecutor` cannot inject the agent's reasoning into the prompt and raises a `KeyError` at runtime.

**Mistake 3: expecting the agent to stream tool output to the user.**

```python
# Wrong — printing ALL chunks, including tool calls
for chunk in agent_executor.stream({"input": user_input, "history": messages}):
    print(chunk, end="", flush=True)
    # User sees: "{'actions': [ToolAgentAction(...)], 'messages': [...]}"
```

`agent_executor.stream()` yields intermediate chunks during tool execution — not just the final answer. Filter for `"output"` to show only the user-facing text. Set `verbose=True` on `AgentExecutor` during development to see the full agent trace in your terminal — it prints the tool calls and intermediate reasoning.

**Mistake 4: infinite loops.**

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=100,   # Danger — a confused agent can loop forever
)
```

If the agent gets stuck — calls a tool, doesn't like the output, calls it again with a slightly different query, repeats — you want a circuit breaker. `max_iterations=3` means the agent can call at most 3 tools per turn before being forced to answer. For simple chatbots, 3 is plenty. For complex multi-step reasoning, you might go to 5 or 10 — but always have a limit.

---

## What we have so far

At the end of Day 10, `langbot.py` is an agent-powered chatbot with:

| Feature | Day added | What it does |
|---|---|---|
| Basic chat loop | Day 1 | Takes input, calls LLM, prints response |
| Conversation memory | Day 2 | Stores history in `messages` list |
| Prompt templates | Day 3 | Separates system prompt from loop logic |
| LCEL chains | Day 4 | Composes components with `\|` operator |
| Structured output | Day 5 | Returns typed `LangBotResponse` objects |
| Output parsers | Day 6 | Two modes: chat (typed) and list (`list[str]`) |
| Streaming | Day 7 | Token-by-token output |
| Document retrieval | Day 8 | Vector store, embeddings, `/search` command |
| RAG pipeline | Day 9 | Automatic retrieval + generation from documents |
| **Agents and tools** | **Day 10** | **LLM decides which tool to use — no hardcoded routing** |

One file. ~300 lines. The agent makes decisions autonomously — retrieving documents, generating lists, or chatting — based on the user's intent, not a keyword heuristic.

---

## Try this now

Before tomorrow, solidify your understanding of agents:

- **Ask ambiguous questions.** *"Can you help me with office stuff?"* — the agent must decide: is this a document search or general chat? Watch how it handles ambiguity.

- **Ask questions that could go either way.** *"Suggest some good employee benefits"* — the word "suggest" signals list mode, but "employee benefits" signals document search. Which does the agent pick? (It should search, because the list tool generates generic suggestions while the search tool finds your actual benefits.)

- **Set `verbose=True` on `AgentExecutor`** and watch the agent's internal reasoning in the terminal. You will see the exact tool call JSON, the tool output, and the agent's final synthesis. This is invaluable for debugging agent behavior.

- **Add a bad tool description** — change `search_docs_tool`'s docstring to `"""Search things."""` — and see how the agent's tool selection degrades. Then restore the detailed description. The difference in agent quality is dramatic and makes the lesson concrete.

- **Comment out `search_docs_tool`** from the `tools` list and ask a policy question. The agent has no document retrieval tool, so it falls back to general chat — just like Day 7. The graceful degradation from "agent with tools" to "just chat" is a natural property of the architecture.

---

## Coming tomorrow

LangBot now decides what to do — but its tools are generic (document search, list generation). Tomorrow we give LangBot **custom tools**: a calculator, a weather lookup, maybe a joke fetcher. You will learn to define tools that call external APIs, transform data, and give the agent capabilities that go beyond document retrieval. One tool, one concept — see you then.
