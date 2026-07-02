---
title: "LangBot Day 9: The RAG Pipeline — Retriever Meets Generator"
date: 2026-06-26T07:00:00+01:00
draft: false
description: "LangBot closes the loop. Day 9 wires the vector store retriever into the chat prompt — no more /search commands, just ask naturally and get answers from your documents."
tags:
  - python
  - langchain
  - ai
  - chatbot
  - tutorial
  - rag
  - retrieval-augmented-generation
  - retriever
  - pinecone
  - chroma
  - embeddings
  - lcel
categories:
  - tutorials
series: ["LangBot: Build a Chatbot with LangChain"]
slug: "langbot-day-9-rag-pipeline"
---

**Recap:** On Day 8, LangBot gained a long-term memory — a vector store (Pinecone) filled with document embeddings. You can ask `/search home office stipend` and get relevant chunks back. But that is half the picture. You still have to type `/search`, read raw chunks, and synthesize the answer yourself. Today we close the loop: the retriever feeds into the chatbot automatically, so you ask *"How much is the home office stipend?"* and LangBot answers from your documents — no slash command, no manual lookup.

---

## The problem: retrieval without generation is a library, not a librarian

Day 8's LangBot has two disconnected capabilities:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Day 8 LangBot                             │
│                                                                  │
│  You: "How much PTO do I get?"                                   │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────┐     ┌──────────────┐                               │
│  │ Chat LLM │ ──► │ "I don't     │   Model has no access         │
│  │  (GPT)   │     │  know your   │   to documents during chat    │
│  └──────────┘     │  PTO policy" │                               │
│                   └──────────────┘                               │
│                                                                  │
│  You: "/search PTO"                                              │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────┐     ┌──────────────┐                               │
│  │  Vector  │ ──► │ "Employees    │   Raw chunks, not answers    │
│  │  Store   │     │  accrue 2     │                               │
│  └──────────┘     │  days of PTO  │                               │
│                   │  per month…"  │                               │
│                   └──────────────┘                               │
│                                                                  │
│  These two pipelines don't talk to each other.                   │
└─────────────────────────────────────────────────────────────────┘
```

**RAG (Retrieval-Augmented Generation)** connects them: every chat message first retrieves relevant documents, then the LLM generates an answer *from* those documents. One unified pipeline where retrieval feeds generation:

```
┌──────────────────────────────────────────────────────────────────┐
│                        Day 9 LangBot                              │
│                                                                   │
│  You: "How much PTO do I get?"                                    │
│       │                                                           │
│       ▼                                                           │
│  ┌──────────────┐     ┌─────────────────┐     ┌───────────────┐  │
│  │   Vector     │ ──► │    RAG Prompt   │ ──► │   Chat LLM    │  │
│  │   Store      │     │  Context: "...  │     │   (GPT)       │  │
│  │  similarity  │     │  accrue 2 days  │     │               │  │
│  │  search      │     │  per month..."  │     │               │  │
│  └──────────────┘     │                 │     │               │  │
│                       │  Question:      │     │               │  │
│                       │  "How much      │     │               │  │
│                       │   PTO…?"        │     │               │  │
│                       └─────────────────┘     └───────┬───────┘  │
│                                                       │          │
│                                                       ▼          │
│                                              ┌──────────────┐   │
│                                              │ "You get 2   │   │
│                                              │  days of PTO │   │
│                                              │  per month,   │   │
│                                              │  up to 24 per │   │
│                                              │  year."       │   │
│                                              └──────────────┘   │
│                                                                   │
│  One pipeline. One answer. No slash commands.                     │
└──────────────────────────────────────────────────────────────────┘
```

This is the full RAG pipeline — retrieval + generation, wired together. The user experience changes from *"I have to remember to search"* to *"it just knows."*

---

## The solution: inject retrieved context into the prompt

The change is conceptually small. We already have:

1. **Day 8**: `search_docs(vector_store, query, k=3)` → returns relevant `Document` chunks
2. **Day 3**: a chat prompt template with `{input}` and `{history}`

Day 9 adds one thing: a **third slot** in the prompt — `{context}` — that we fill with retrieved documents before calling the LLM:

```python
# Day 8 prompt (chat only)
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 🆕 Day 9 prompt (RAG)
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Context:\n{context}\n\nQuestion: {input}"),
])
```

The `{context}` variable is a string — the concatenated text of the top-K retrieved chunks. The model sees the relevant passages right there in the prompt and can answer from them. No API changes. No new models. Just a prompt that includes evidence alongside the question.

---

## Step 1: Install the new dependency (none!)

Day 9 requires zero new packages. Everything we need — `ChatOpenAI`, `OpenAIEmbeddings`, `PineconeVectorStore`, `StrOutputParser`, `RecursiveCharacterTextSplitter` — was installed on Day 8. If you skipped Day 8, you need:

```bash
uv add langchain-openai langchain-pinecone langchain-text-splitters langchain-community python-dotenv
```

But if you have been following along, you are ready to go with no `uv add` step today.

---

## The new code

Three things are new today:

### 1. The context formatter

```python
# 🆕 Day 9: Format retrieved documents into a single context string
def format_context(docs: list) -> str:
    """Join retrieved document chunks into a context block for the prompt."""
    if not docs:
        return "No relevant documents found."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Document {i} — {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
```

This function takes the list of `Document` objects from `similarity_search()` and turns them into a single string the prompt can consume. It preserves the source filename so the model can cite its sources. The `---` separators make it easy for the model to distinguish between documents.

### 2. The RAG prompt and chain

```python
# 🆕 Day 9: RAG-specific system prompt — tells the model HOW to use context
RAG_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You have access to relevant documents shown in the Context section below.
Use them to answer the user's question accurately. Rules:
- Answer ONLY from the provided context unless the context doesn't contain the answer.
- If the context doesn't have the answer, say so honestly and offer a general response.
- Cite which document(s) you used when the answer comes from context.
- Keep your tone consistent with your personality (witty, concise, under three sentences)."""

# 🆕 Day 9: RAG prompt — adds {context} to the human message
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Context:\n{context}\n\nQuestion: {input}"),
])

# 🆕 Day 9: RAG stream chain — prompt → LLM → StrOutputParser
rag_stream_chain = rag_prompt | llm | StrOutputParser()
```

The RAG prompt has the same shape as the chat prompt — system message, history, human message — but the human message now includes `{context}`. The LLM sees the retrieved documents inline and answers from them.

The RAG chain (`rag_stream_chain`) is identical to the chat chain (`stream_chain`) except it uses the RAG prompt. Same LLM, same `StrOutputParser()`. The only difference is what the prompt template expects.

### 3. The routing logic

In the main loop, before the chat branch, we now attempt retrieval:

```python
# 🆕 Day 9: Try RAG first — if we have documents, use them
docs = search_docs(vector_store, user_input, k=3) if vector_store else []

if docs:
    context = format_context(docs)
    print(f"\nLangBot [📚 using {len(docs)} document(s)]: ", end="", flush=True)
    full_response = ""
    rag_input = {"input": user_input, "history": messages, "context": context}
    for chunk in rag_stream_chain.stream(rag_input):
        print(chunk, end="", flush=True)
        full_response += chunk
    print()
    reply_text = full_response
else:
    # Day 7: Regular chat — stream without context
    print(f"\nLangBot: ", end="", flush=True)
    full_response = ""
    for chunk in stream_chain.stream({"input": user_input, "history": messages}):
        print(chunk, end="", flush=True)
        full_response += chunk
    print()
    reply_text = full_response
```

Every message now attempts retrieval *first*. If documents are found, the RAG chain gets the context and answers from it. If no documents are found (or `vector_store` is `None`), we fall back to the regular chat chain — exactly as it worked on Day 7. The `/search` command still exists for explicit searches, but it is now secondary to the automatic RAG pipeline.

---

## Where it plugs in

Here is the full Day 9 `langbot.py`. Look for the `# 🆕 Day 9` markers:

```python
# langbot.py — Day 9: RAG pipeline (retriever + generation)
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from enum import Enum
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

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


# ═══════════════════════════════════════════════════════════════════
# Day 8: Document retrieval infrastructure (unchanged from Day 8)
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
# 🆕 Day 9: RAG pipeline — connect retrieval to generation
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

    # Day 7: Streaming chain — prompt → LLM → StrOutputParser → str chunks
    stream_chain = prompt | llm | StrOutputParser()

    # ═══════════════════════════════════════════════════════════
    # 🆕 Day 9: RAG prompt and chain
    # ═══════════════════════════════════════════════════════════

    RAG_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You have access to relevant documents shown in the Context section below.
Use them to answer the user's question accurately. Rules:
- Answer ONLY from the provided context unless the context doesn't contain the answer.
- If the context doesn't have the answer, say so honestly and offer a general response.
- Cite which document(s) you used when the answer comes from context.
- Keep your tone consistent with your personality (witty, concise, under three sentences)."""

    # 🆕 Day 9: RAG prompt — adds {context} to the human message
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Context:\n{context}\n\nQuestion: {input}"),
    ])

    # 🆕 Day 9: RAG stream chain — same LLM, different prompt
    rag_stream_chain = rag_prompt | llm | StrOutputParser()

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
            print("   (No documents found in docs/ — RAG will be unavailable)")
            print("   (Create docs/*.txt or docs/*.md files and restart)")
    except Exception as e:
        print(f"   ⚠ Vector store setup skipped: {e}")
        print("   (Make sure PINECONE_API_KEY is set and the index exists)")

    # Day 2: Conversation history
    messages = []

    print("🤖 LangBot (type 'quit' to exit)")
    print("─" * 40)
    print("  Chat: just ask — RAG answers from your docs automatically")
    print("  List: 'list 5 programming books' or 'suggest weekend activities'")
    print("  Search: '/search remote work policy' (explicit search)")
    print("  Now with RAG — ask about your documents naturally! 📚⚡")
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
                print("\n⚠ Vector store not available. Add documents to docs/ and restart.")
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

            # Store the search command and results summary in history
            messages.append(HumanMessage(content=user_input))
            summary = f"Searched for '{query}': found {len(results)} document(s)."
            messages.append(AIMessage(content=summary))
            continue

        # Day 6: Route list-mode requests
        if wants_list(user_input):
            # List mode — invoke for a complete list
            items = list_chain.invoke({"input": user_input, "history": messages})

            print(f"\nLangBot's suggestions ({len(items)} items):")
            for i, item in enumerate(items, 1):
                print(f"  {i}. {item}")

            # Reconstruct a text reply for history
            reply_text = f"Here are some suggestions: {', '.join(items)}"
        else:
            # ═══════════════════════════════════════════════════════
            # 🆕 Day 9: RAG chat — retrieve docs and answer from them
            # ═══════════════════════════════════════════════════════
            docs = search_docs(vector_store, user_input, k=3) if vector_store else []

            if docs:
                # We have relevant documents — use the RAG chain
                context = format_context(docs)
                print(f"\nLangBot [📚 {len(docs)} doc(s)]: ", end="", flush=True)
                full_response = ""
                for chunk in rag_stream_chain.stream({
                    "input": user_input,
                    "history": messages,
                    "context": context,
                }):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()
                reply_text = full_response
            else:
                # No relevant docs — fall back to regular chat
                print(f"\nLangBot: ", end="", flush=True)
                full_response = ""
                for chunk in stream_chain.stream({
                    "input": user_input,
                    "history": messages,
                }):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()
                reply_text = full_response

        # Day 2: Store the exchange in history
        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=reply_text))


if __name__ == "__main__":
    main()
```

---

## What changed — line by line

**Lines 99–107 — `format_context()`.** A new function that takes the list of `Document` objects from `similarity_search()` and converts them into a single string. Each document gets a header (`[Document 1 — docs/policy.txt]`) followed by its text, with `---` separators between documents. This is the bridge between the vector store's `Document` objects and the prompt's `{context}` string.

**Lines 114–126 — the RAG system prompt.** `RAG_SYSTEM_PROMPT` extends `SYSTEM_PROMPT` with four instructions specific to RAG: answer from context, be honest when context is insufficient, cite sources, and maintain personality. This prompt augmentation is the only thing telling the model *how* to use the context — without it, the model might ignore the context or blend it with its own knowledge unpredictably.

**Lines 128–132 — the RAG prompt template.** `rag_prompt` is a new `ChatPromptTemplate` identical to the chat `prompt` except the human message includes `{context}`. The template expects three variables: `{input}`, `{history}`, and `{context}`. When we call `rag_stream_chain.stream(...)`, we now pass all three.

**Lines 134–135 — the RAG stream chain.** `rag_stream_chain = rag_prompt | llm | StrOutputParser()` — same LLM, same parser, different prompt. The chain composition is identical to `stream_chain`. This is the beauty of LCEL: adding RAG support is a *prompt change*, not a code architecture change.

**Lines 229–257 — the new routing logic.** The chat branch now has two sub-branches:

1. **RAG path** (lines 232–245): `search_docs()` runs on every message. If documents are found, we format them into context, pass them into `rag_stream_chain`, and stream the response. The `[📚 N doc(s)]` indicator shows when RAG is active.

2. **Fallback path** (lines 246–257): if no documents are found (or `vector_store` is `None`), we use the regular `stream_chain` — the exact same behavior as Day 7.

This means every chat message now goes through retrieval. For general conversation ("hello", "tell me a joke"), `similarity_search()` will return low-relevance results — but we still have the `if docs:` check, so those results still trigger the RAG path. In practice, the model will see irrelevant context and follow the instruction *"If the context doesn't have the answer, say so honestly and offer a general response"* — effectively falling back to normal chat behavior.

A more sophisticated approach (coming later in this series) would use a relevance threshold or a classifier to decide whether to use RAG. For Day 9, the simple "always retrieve, trust the model to ignore irrelevant context" approach keeps the concept focused.

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
  Chat: just ask — RAG answers from your docs automatically
  List: 'list 5 programming books' or 'suggest weekend activities'
  Search: '/search remote work policy' (explicit search)
  Now with RAG — ask about your documents naturally! 📚⚡
────────────────────────────────────────

You: How much is the home office stipend?

LangBot [📚 3 doc(s)]: The company provides a $500 annual home office stipend for
equipment and furniture — according to the remote work policy. 🪑💰

You: What health insurance plans are available?

LangBot [📚 3 doc(s)]: Three plans: HMO, PPO, and HDHP with HSA. Enrollment is in
November, and the company covers 80% of your premium (50% for dependents) — per
the benefits doc. 🏥

You: When are production deployments allowed?

LangBot [📚 3 doc(s)]: Tuesdays and Thursdays only, with hotfixes allowed any day
if the on-call engineer approves — per the engineering team practices. All deploys
go through staging first with a smoke test. 🚀

You: What's the capital of France?

LangBot [📚 3 doc(s)]: The provided documents don't mention world capitals — they
cover company policies, not geography. But as your friendly AI with knowledge of the
outside world: it's Paris. 🇫🇷 (The context didn't help here, so I'm going off my
own training.)

You: Tell me a joke

LangBot [📚 3 doc(s)]: The documents don't seem to contain any jokes — shocking,
I know. But here's one from my own repertoire: Why do programmers prefer dark mode?
Because light attracts bugs! 🐛💡

You: quit
👋 Goodbye!
```

Notice how the conversation flows naturally between document-based questions and general chat. LangBot now answers from your documents *automatically* — no `/search` required. When the context is irrelevant (the France question, the joke request), the model acknowledges it and falls back to its own knowledge.

The `/search` command still works for explicit searches, but you will rarely need it — the RAG pipeline handles retrieval transparently on every message.

---

## Why this pattern matters for real apps

### 1. The user shouldn't know about retrieval

Users don't want to type `/search` — they want to ask questions. A chatbot that requires a slash command to access documents is a search engine with a chat interface. A chatbot that *uses* documents transparently is an assistant.

The RAG pipeline hides the retrieval mechanism behind the same chat interface the user already knows. There is no new command to learn, no mode to switch into. The system does the work of finding relevant information and synthesizing an answer.

### 2. Retrieval + generation is more than the sum of its parts

Day 8's `/search` gave you document chunks. Day 7's chat gave you LLM answers. Neither alone could answer *"How much PTO do I get?"* from your documents.

Together, they form a pipeline where each component does what it does best:

| Component | Strength | Weakness |
|---|---|---|
| **Vector store** | Finds relevant information fast | Returns raw text, not answers |
| **LLM** | Synthesizes natural language | Doesn't know your documents |

The RAG pipeline combines them: the vector store finds the evidence, the LLM turns it into an answer. Neither component changes — they are the same `PineconeVectorStore` and `ChatOpenAI` from Day 8 and Day 1. The only new thing is the *connection* between them.

### 3. The prompt is the integration point

Notice that the entire RAG pipeline is implemented as a **prompt change**, not a code change. The chain composition is identical:

```python
# Chat chain
stream_chain = prompt | llm | StrOutputParser()

# RAG chain — same pattern, different prompt
rag_stream_chain = rag_prompt | llm | StrOutputParser()
```

This is a deep lesson about LLM application architecture: the prompt is the API between retrieval and generation. The retriever's output (`context`) flows into the prompt, and the LLM's output flows out. The prompt template is the contract that both sides agree on.

This means you can evolve retrieval (better embeddings, hybrid search, reranking) or generation (different models, higher temperature, structured output) independently — as long as the prompt template's variable names stay the same, the pipeline holds together.

### 4. Transparency builds trust

The `[📚 N doc(s)]` indicator tells the user when LangBot is answering from documents vs. its own knowledge. This is a small UI detail that builds trust: users know whether the answer came from their data or from the model's training.

In a production app, you would take this further — adding inline citations, showing which document each claim came from, and letting users click through to the source. Day 9's `[Document N — source]` headers in the context give the model enough information to cite sources, and the RAG system prompt instructs it to do so. The foundation for citations is already in place.

---

## The RAG pipeline, visualized

Here is the exact data flow for one RAG turn:

```
User: "How much PTO do I get?"
  │
  ▼
search_docs(vector_store, "How much PTO do I get?", k=3)
  │
  │  Embed query → [0.011, -0.031, ...]
  │  Cosine similarity against 14 stored vectors
  │  Return top 3 Documents
  │
  ▼
[
  Document(page_content="Vacation policy: Employees accrue 2 days of PTO
    per month, up to a maximum of 24 days per year...",
    metadata={"source": "docs/policy.txt"}),
  Document(page_content="Health insurance: The company offers three plans...",
    metadata={"source": "docs/benefits.txt"}),
  Document(page_content="On-call: Engineers rotate weekly...",
    metadata={"source": "docs/engineering.txt"}),
]
  │
  ▼
format_context(docs)
  │
  ▼
"[Document 1 — docs/policy.txt]
Vacation policy: Employees accrue 2 days of PTO per month, up to a maximum
of 24 days per year. Unused PTO rolls over up to 5 days into the next
calendar year.

---

[Document 2 — docs/benefits.txt]
Health insurance: The company offers three plans — HMO, PPO, and HDHP with HSA...

---

[Document 3 — docs/engineering.txt]
On-call: Engineers rotate weekly..."
  │
  ▼
rag_stream_chain.stream({
    "input": "How much PTO do I get?",
    "history": [...],
    "context": <the formatted string above>,
})
  │
  ▼
OpenAI API receives:
  System: "You are LangBot... You have access to relevant documents..."
  History: [...previous messages...]
  Human: "Context:
           [Document 1 — docs/policy.txt]
           Vacation policy: Employees accrue 2 days...
           ---
           [Document 2 — docs/benefits.txt]
           ...
           ---
           Question: How much PTO do I get?"
  │
  ▼
OpenAI API streams back:
  "You" → " get" → " 2" → " days" → " of" → " PTO" → " per" → " month" → ...
  │
  ▼
LangBot [📚 3 doc(s)]: You get 2 days of PTO per month, up to 24 days per year,
with up to 5 unused days rolling over — per the remote work policy. 🏖️
```

Six steps. Two network calls (embedding API → Pinecone, then chat API → OpenAI). ~500ms total. The user just typed a question and got an answer from their documents.

---

## When the RAG path doesn't fire: no documents available

If `docs/` is empty, `vector_store` is `None`, and the RAG branch silently skips. Every message routes through the regular `stream_chain`. LangBot behaves exactly like Day 7 — no errors, no degraded experience.

```text
📚 Loading documents from docs/ ...
   (No documents found in docs/ — RAG will be unavailable)
   (Create docs/*.txt or docs/*.md files and restart)
🤖 LangBot (type 'quit' to exit)
────────────────────────────────────────
...

You: How much PTO do I get?

LangBot: I don't have access to your company's PTO policy — my knowledge cuts off
at my training date, and your HR department probably updated things since then. 😅
Try adding your policy docs to the docs/ folder and restarting me!
```

The graceful fallback is important: RAG is an *enhancement*, not a requirement. Your chatbot works with or without documents.

---

## Common mistakes (and what they teach you)

**Mistake 1: passing the wrong variable names to `rag_stream_chain`.**

```python
# Wrong — the prompt expects "context", "input", "history"
# but we only pass "input" and "history"
for chunk in rag_stream_chain.stream({"input": user_input, "history": messages}):
    ...
# KeyError: 'context'
```

The RAG prompt template declares three variables: `{context}`, `{input}`, and `{history}`. You must pass all three in the dict. Forgetting `context` raises a `KeyError` — LangChain validates this at `.stream()` time, not at chain definition time.

**Mistake 2: sending raw `Document` objects instead of formatting them.**

```python
# Wrong — passing a list of Document objects as context
docs = search_docs(vector_store, user_input)
rag_input = {"input": user_input, "history": messages, "context": docs}
# The prompt receives: "context=[Document(...), Document(...)]"
# The model sees Python reprs instead of document text
```

`ChatPromptTemplate` calls `str()` on whatever you pass as `{context}`. A list of `Document` objects stringifies to `[Document(page_content='...'), ...]` — ugly and full of Python noise. Always format documents into a clean text string before passing them to the prompt. `format_context()` exists for exactly this reason.

**Mistake 3: stuffing too many documents into the context.**

```python
# Wrong — k=20 floods the prompt with noise
docs = search_docs(vector_store, user_input, k=20)
```

Every document you include consumes context window tokens. `gpt-4.1-mini` has a 128K token context, so 20 chunks of 500 characters (~125 tokens each = 2,500 tokens total) is fine. But when you later add more documents, larger chunks, or longer conversations, the context adds up. Start with `k=3` — it covers most questions and leaves room for conversation history.

**Mistake 4: letting irrelevant context confuse the model.**

If `search_docs()` returns documents about benefits when the user asks about deployment schedules, the model sees irrelevant text and must decide what to trust. The RAG system prompt handles this with *"If the context doesn't have the answer, say so honestly"* — but not all models follow instructions equally well. gpt-4.1-mini handles it gracefully. A smaller model might get confused and blend irrelevant context into its answer.

The fix (coming Day 19: Reranking) is to add a relevance filter between retrieval and the prompt — keep only the chunks that are actually relevant to the question. For now, `k=3` and a well-crafted system prompt are sufficient.

---

## What we have so far

At the end of Day 9, `langbot.py` is a fully functional RAG chatbot with:

| Feature | Day added | What it does |
|---|---|---|
| Basic chat loop | Day 1 | Takes input, calls LLM, prints response |
| Conversation memory | Day 2 | Stores history in `messages` list |
| Prompt templates | Day 3 | Separates system prompt from loop logic |
| LCEL chains | Day 4 | Composes components with `\|` operator |
| Structured output | Day 5 | Returns typed `LangBotResponse` objects |
| Output parsers | Day 6 | Two modes: chat (typed) and list (`list[str]`) |
| Streaming | Day 7 | Token-by-token output with `StrOutputParser` |
| Document retrieval | Day 8 | Vector store, embeddings, `/search` command |
| **RAG pipeline** | **Day 9** | **Automatic retrieval + generation from documents** |

One file. ~270 lines. Every feature composes with every other feature because they all use the same LCEL pattern.

---

## Try this now

Before tomorrow, solidify your understanding of RAG:

- **Ask questions that span multiple documents.** *"What benefits does the company offer for remote workers?"* — this requires combining information from the remote work policy and the benefits document. Good RAG should synthesize across chunks.

- **Ask questions the documents don't cover.** *"What's the company's policy on pet insurance?"* — the model should acknowledge the gap and not hallucinate. This tests the RAG system prompt's honesty instruction.

- **Add a new document mid-conversation** (stop LangBot, add `docs/onboarding.txt`, restart). Ask a question about the new document. The vector store re-indexes at startup, so the new document is available immediately.

- **Vary `k` in `search_docs()`.** Change `k=3` to `k=1` and then `k=5`. With `k=1`, you risk missing relevant context if the top hit isn't the right chunk. With `k=5`, you get more context but more noise. `k=3` is the sweet spot for these 500-character chunks.

- **Comment out the `format_context()` source headers** (`[Document N — source]`) and ask a question that spans documents. Without source headers, the model might blend documents without attribution. With them, it can cite sources. Small formatting decisions have big effects on output quality.

---

## Coming tomorrow

LangBot now retrieves and generates — but it follows a script. It always retrieves, always generates, always in the same order. Tomorrow we break that script: **Agents and tool calling**. LangBot will *decide for itself* whether to search, chat, list, or do something else entirely — using the LLM as a reasoning engine that picks the right tool for each turn. One concept, one change — see you then.
