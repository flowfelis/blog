---
title: "LangBot Day 8: Retrieval — Vector Stores, Embeddings, and Pinecone"
date: 2026-06-25T07:00:00+01:00
draft: false
description: "LangBot gets a long-term memory. Day 8 introduces embeddings, vector stores, and Pinecone — the infrastructure that lets LangBot search external documents for answers it doesn't already know."
tags:
  - python
  - langchain
  - ai
  - chatbot
  - tutorial
  - retrieval
  - embeddings
  - vector-store
  - pinecone
  - chroma
  - openai-embeddings
categories:
  - tutorials
series: ["LangBot: Build a Chatbot with LangChain"]
slug: "langbot-day-8-retrieval-vector-stores"
---

**Recap:** On Day 7, LangBot learned to stream — token-by-token responses that feel like a real chat app. It remembers the conversation within a session (Day 2 memory), it has a personality (Day 3 prompt templates), it returns typed data (Day 5 structured output), it handles lists (Day 6 output parsers), and it streams those replies in real time (Day 7). But ask it anything outside its training data and it either guesses, hallucinates, or apologizes. Today we fix that by giving LangBot the ability to search external documents — the retrieval half of RAG.

---

## The problem: frozen knowledge

Try this in Day 7's LangBot:

```text
You: What does the company's remote work policy say about home office stipends?

LangBot: I don't have access to your company's internal documents, so I can't answer
that specifically. I'd recommend checking your employee handbook or HR portal.
```

The model is honest — but not useful. It has no access to your documents. Its knowledge is frozen at training time. Every question about your own data — company policies, project documentation, personal notes, research papers — gets the same shrug.

The solution is **retrieval-augmented generation (RAG)**. RAG works in two stages:

1. **Retrieval** (today): load your documents, split them into chunks, convert each chunk to a vector embedding, and store those vectors in a database optimized for similarity search.
2. **Generation** (tomorrow, Day 9): take the user's question, find the most relevant chunks, and feed them into the prompt alongside the question so the model can answer from provided context.

Today we build stage 1 — the retrieval pipeline. By the end of this post, LangBot will have a `/search` command that finds relevant passages from your documents in milliseconds.

---

## The solution: embeddings and vector stores

Here is the data flow we are building:

```
Your documents          Embedded vectors        Vector database
──────────────         ────────────────        ───────────────
docs/                  OpenAI Embeddings       Pinecone
  ├── policy.txt  ──►  [0.012, -0.034, ...] ──► Index
  ├── faq.md      ──►  [0.008,  0.041, ...] ──► Index
  └── guide.txt   ──►  [0.019, -0.027, ...] ──► Index

User query: "What's the remote work policy?"
  ──► [0.011, -0.031, ...] ──► similarity_search() ──► top 3 chunks
```

Three concepts to understand before we write code:

### 1. Embeddings

An embedding is a list of floats — typically 1,536 numbers for OpenAI's `text-embedding-3-small` — that represents the *meaning* of a piece of text. Texts with similar meanings have similar vectors. "The cat sat on the mat" and "A feline rested on a rug" produce embeddings that are close together in vector space. "The stock market crashed" produces an embedding far from both.

LangChain provides `OpenAIEmbeddings` (and many alternatives: HuggingFace, Cohere, Voyage AI) — a class that wraps the embedding API and returns a vector for any text you give it.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("What is retrieval-augmented generation?")
# vector: [0.012, -0.034, 0.008, ...]  (1,536 floats)
```

### 2. Document chunking

You cannot embed an entire 50-page PDF as one vector. The embedding would be a blurry average of every topic in the document. Instead, you split documents into **chunks** — overlapping segments of a few hundred characters each. The overlap ensures a sentence that spans a chunk boundary appears in both chunks.

LangChain's `RecursiveCharacterTextSplitter` handles this. It tries to split on paragraph boundaries (`\n\n`), then newlines (`\n`), then sentences (`. `), then spaces, then individual characters — preserving as much semantic coherence as possible at each level.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # ~500 characters per chunk
    chunk_overlap=50,      # 50 characters of overlap between chunks
)
chunks = splitter.split_documents(documents)
```

### 3. Vector stores

A vector store is a database that indexes embeddings and can answer "which stored vectors are closest to this query vector?" — a **similarity search**. You upload your chunks + their embeddings to the store at indexing time. At query time, you embed the user's question and ask the store for the top-K closest chunks.

Today we use **Pinecone** — a hosted vector database with a generous free tier (one index, up to 100K vectors). If you prefer an all-local option, I will also show the **Chroma** alternative, which runs entirely on your machine with zero setup.

---

## Step 0: Set up Pinecone (one-time)

Before writing code, create a Pinecone account and index.

### 1. Sign up and get an API key

Go to **[pinecone.io](https://www.pinecone.io/)** and sign up. The free tier requires no credit card.

After signing in, go to **API Keys** in the left sidebar and copy your key. It starts with `pcsk_...`. Add it to your `.env` file:

```bash
# .env
OPENAI_API_KEY="sk-..."
PINECONE_API_KEY="pcsk_..."
```

### 2. Create an index

Go to **Indexes** in the Pinecone console and click **Create Index**. Configure it like this:

| Setting | Value | Why |
|---|---|---|
| **Name** | `langbot-docs` | Matches the index name in our code |
| **Dimensions** | `1536` | Matches `text-embedding-3-small` output size |
| **Metric** | `cosine` | Cosine similarity — standard for text embeddings |
| **Cloud provider** | AWS (or GCP — free tier works on both) | |
| **Region** | Pick the one closest to you | Lower latency |

Click **Create Index**. It takes 10–30 seconds to provision. While it spins up, let's write the code.

> **Alternative: Chroma (zero setup).** If you do not want to create a Pinecone account, replace the Pinecone code with Chroma — an in-memory or on-disk vector store that runs locally. Install it with `uv add langchain-chroma` and swap `PineconeVectorStore` for `Chroma`. I will show both at the end. The rest of the code is identical.

---

## Step 1: Install new dependencies

```bash
uv add langchain-pinecone langchain-text-splitters langchain-community
```

| Package | What it gives us |
|---|---|
| `langchain-pinecone` | `PineconeVectorStore` — the LangChain wrapper for Pinecone |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` — the chunking logic |
| `langchain-community` | `DirectoryLoader`, `TextLoader` — document loading utilities |

We already have `langchain-openai` (for `ChatOpenAI` and `OpenAIEmbeddings`) from Day 1.

---

## Step 2: Create some test documents

LangBot expects documents in a `docs/` directory relative to where you run it. Create a few:

```bash
mkdir docs
```

**`docs/policy.txt`:**

```text
Remote Work Policy

Employees may work remotely up to 4 days per week. One day per week in the office
is required for team collaboration. The company provides a $500 annual home office
stipend for equipment and furniture. Remote employees must be available during core
hours (10am–3pm in their local timezone). All remote meetings require video on.

Vacation policy: Employees accrue 2 days of PTO per month, up to a maximum of 24
days per year. Unused PTO rolls over up to 5 days into the next calendar year.
```

**`docs/benefits.txt`:**

```text
Employee Benefits Overview

Health insurance: The company offers three plans — HMO, PPO, and HDHP with HSA.
Enrollment opens in November for the following calendar year. The company covers
80% of premiums for employees and 50% for dependents.

Retirement: 401(k) with 4% company match, fully vested after 2 years of employment.
Employees can contribute pre-tax or Roth. The plan is administered through Fidelity.

Education: $2,000 annual tuition reimbursement for job-related courses and certifications.
Must be pre-approved by your manager 30 days before the course start date.
```

**`docs/engineering.txt`:**

```text
Engineering Team Practices

Code review: All pull requests require at least one approval from a teammate before
merging. CI must pass (tests + linting) before review is requested. We use GitHub
Actions for CI/CD.

Deployments: We deploy to production on Tuesdays and Thursdays only. Hotfixes can
be deployed any day with approval from the on-call engineer. All deploys go through
staging first and require a passing smoke test.

On-call: Engineers rotate weekly. The on-call engineer carries a pager and is
expected to respond within 15 minutes during business hours and 30 minutes outside
business hours. On-call duty comes with a $150/day stipend.
```

These are simple text files. LangChain's `TextLoader` handles `.txt` and `.md` files. For PDFs, you would use `PyPDFLoader` (from `langchain-community`). For web pages, `WebBaseLoader`. The pattern is the same regardless of format.

---

## The new code

Four functions are new today. Here they are, isolated:

### 1. Loading documents

```python
# 🆕 Day 8: Load all .txt files from the docs/ directory
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_documents(docs_dir: str = "docs") -> list:
    """Load all .txt and .md files from a directory. Returns a list of Documents."""
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",        # match all .txt files recursively
        loader_cls=TextLoader,  # use TextLoader for each file
    )
    # Also try loading .md files
    md_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
    )
    documents = loader.load()
    try:
        documents.extend(md_loader.load())
    except Exception:
        pass  # no .md files — that's fine
    return documents
```

`DirectoryLoader` scans a directory, finds files matching a glob pattern, and passes each one to `TextLoader`. Each loaded document is a `Document` object with `page_content` (the text) and `metadata` (the filename, source path).

### 2. Chunking documents

```python
# 🆕 Day 8: Split documents into overlapping chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """Split long documents into smaller, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
```

The `separators` list defines the splitting priority: try to break on double-newlines first (paragraphs), then single newlines (lines), then sentence boundaries, then spaces, then characters. This produces chunks that are as semantically intact as possible.

`chunk_size=500` gives us roughly 125 tokens per chunk with typical English text — small enough for precise retrieval, large enough to contain full thoughts.

### 3. Creating the vector store

```python
# 🆕 Day 8: Embed documents and store in Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

def setup_vector_store(chunks: list, index_name: str = "langbot-docs"):
    """Embed all chunks and store them in Pinecone. Returns the vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    return vector_store
```

`PineconeVectorStore.from_documents()` does three things automatically:

1. Embeds every chunk using `OpenAIEmbeddings`
2. Creates or connects to the named Pinecone index
3. Uploads all vectors to the index

On subsequent runs, calling this again will add duplicate vectors. In production, you would check if the index already has data and skip or clear it. For this tutorial, the function is called once at startup and the vector store lives in memory for the session.

### 4. Searching the vector store

```python
# 🆕 Day 8: Query the vector store for relevant documents
def search_docs(vector_store, query: str, k: int = 3) -> list:
    """Search for the k most relevant document chunks. Returns a list of Documents."""
    return vector_store.similarity_search(query, k=k)
```

`ssearch_docs("remote work policy", k=3)` does this internally:

1. Embed `"remote work policy"` → `[0.011, -0.031, ...]`
2. Find the 3 vectors in the Pinecone index closest to that embedding (using cosine similarity)
3. Return the corresponding `Document` objects (with `page_content` and `metadata`)

The entire search takes ~200ms — embedding generation + network round trip to Pinecone.

---

## Where it plugs in

Here is the full Day 8 `langbot.py`. Look for the `# 🆕 Day 8` markers:

```python
# langbot.py — Day 8: Retrieval with vector stores and embeddings
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
# 🆕 Day 8: Document retrieval infrastructure
# ═══════════════════════════════════════════════════════════════════

def load_documents(docs_dir: str = "docs") -> list:
    """Load all .txt and .md files from a directory. Returns a list of Documents."""
    documents = []
    for pattern, ext in [("**/*.txt", ".txt"), ("**/*.md", ".md")]:
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

    # ═══════════════════════════════════════════════════════════════
    # 🆕 Day 8: Set up document retrieval at startup
    # ═══════════════════════════════════════════════════════════════
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
            print("   (No documents found in docs/ — /search will be unavailable)")
            print("   (Create docs/*.txt or docs/*.md files and restart)")
    except Exception as e:
        print(f"   ⚠ Vector store setup skipped: {e}")
        print("   (Make sure PINECONE_API_KEY is set and the index exists)")

    # Day 2: Conversation history
    messages = []

    print("🤖 LangBot (type 'quit' to exit)")
    print("─" * 40)
    print("  Chat: just type a message")
    print("  List: 'list 5 programming books' or 'suggest weekend activities'")
    print("  Search: '/search remote work policy'")
    print("  Now streaming — watch LangBot type! ⚡")
    print("─" * 40)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        # ── 🆕 Day 8: /search command ─────────────────────────────
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
                print(f"  (These chunks can feed into the chatbot — coming Day 9)")

            # Store the search command and results summary in history
            messages.append(HumanMessage(content=user_input))
            summary = f"Searched for '{query}': found {len(results)} document(s)."
            messages.append(AIMessage(content=summary))
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
            # Day 7: Chat mode — stream the response token by token
            print(f"\nLangBot: ", end="", flush=True)
            full_response = ""
            for chunk in stream_chain.stream({"input": user_input, "history": messages}):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()  # final newline

            reply_text = full_response

        # Day 2: Store the exchange in history
        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=reply_text))


if __name__ == "__main__":
    main()
```

---

## What changed — line by line

**Lines 2–5 — new imports.** `OpenAIEmbeddings` (Day 8) joins `ChatOpenAI` (Day 1) from `langchain-openai`. `DirectoryLoader` and `TextLoader` come from `langchain-community`. `RecursiveCharacterTextSplitter` from `langchain-text-splitters`. `PineconeVectorStore` from `langchain-pinecone`. Seven new imports — all for the retrieval pipeline.

**Lines 53–84 — the four retrieval functions.** `load_documents()`, `chunk_documents()`, `setup_vector_store()`, and `search_docs()`. These are the entire retrieval stack in four functions. Each is independent and testable on its own.

**Lines 137–153 — startup ingestion.** At the top of `main()`, we load documents, chunk them, embed them, and store them in Pinecone. This runs once per session. In production you would separate ingestion from serving (run a script that uploads documents once, then a separate process that queries them). For a CLI tutorial, doing it at startup is the simplest path.

**Lines 173–197 — the `/search` command.** A new branch in the main loop. When the user types `/search <query>`, we call `search_docs()` and display the top 3 results with source filenames and content previews. This is the retrieval half of RAG in isolation — no generation, just "here is what the documents say." Tomorrow we will pipe these results into the chat prompt.

**Lines 200–202 — storing search history.** We record the search command in conversation history so LangBot remembers what you searched for. The summary is factual ("found 3 documents") rather than the search result content — the model does not need to see raw document chunks in history.

**Everything else is unchanged.** The chat chain, list chain, streaming, history, system prompt — all work exactly as they did on Day 7. The retrieval pipeline is additive. It does not modify any existing behavior.

---

## Try it now

```bash
# Make sure your .env has both keys
# OPENAI_API_KEY="sk-..."
# PINECONE_API_KEY="pcsk_..."

python langbot.py
```

```text
📚 Loading documents from docs/ ...
   Loaded 3 document(s) → split into 14 chunks
🔢 Embedding chunks and storing in Pinecone ...
   ✓ Vector store ready (14 vectors indexed)
🤖 LangBot (type 'quit' to exit)
────────────────────────────────────────
  Chat: just type a message
  List: 'list 5 programming books' or 'suggest weekend activities'
  Search: '/search remote work policy'
  Now streaming — watch LangBot type! ⚡
────────────────────────────────────────

You: /search home office stipend

📖 Top 3 results for: "home office stipend"
────────────────────────────────────────

  [1] 📄 docs/policy.txt
      Remote Work Policy Employees may work remotely up to 4 days per week. One day per week in the office is required for team collaboration. The company provides a $500 annual home office stipend for equipment and furniture. Remote...

  [2] 📄 docs/benefits.txt
      Employee Benefits Overview Health insurance: The company offers three plans — HMO, PPO, and HDHP with HSA. Enrollment opens in November for the following calendar year. The company covers 80% of premiums for employees and 50% for dep...

  [3] 📄 docs/engineering.txt
      Engineering Team Practices Code review: All pull requests require at least one approval from a teammate before merging. CI must pass (tests + linting) before review is requested. We use GitHub Actions for CI/CD. Deployments: We deploy...

────────────────────────────────────────
  (These chunks can feed into the chatbot — coming Day 9)

You: How much PTO do employees get?

LangBot: I actually don't know your company's specific PTO policy — but I did just
search it for you when you looked up the home office stuff. Try /search vacation
or PTO and I'll pull up the relevant section! 🔍

You: /search vacation PTO

📖 Top 3 results for: "vacation PTO"
────────────────────────────────────────

  [1] 📄 docs/policy.txt
      Vacation policy: Employees accrue 2 days of PTO per month, up to a maximum of 24 days per year. Unused PTO rolls over up to 5 days into the next calendar year....

  [2] 📄 docs/benefits.txt
      Employee Benefits Overview Health insurance: The company offers three plans...

  [3] 📄 docs/engineering.txt
      On-call: Engineers rotate weekly. The on-call engineer carries a pager and is expected to respond within 15 minutes...

────────────────────────────────────────
  (These chunks can feed into the chatbot — coming Day 9)

You: quit
👋 Goodbye!
```

The search works. The first result for "vacation PTO" is exactly the paragraph about vacation policy. The retrieval pipeline is functional. Tomorrow we will wire these results into the chat prompt so LangBot can answer "How much PTO do I get?" directly — using the retrieved context.

---

## Chroma: the zero-setup alternative

If Pinecone registration feels like overhead for a tutorial, Chroma is a local vector store that works out of the box. Install it:

```bash
uv add langchain-chroma
```

Then swap two lines in `langbot.py`:

```python
# Replace the Pinecone import
# from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma

# Replace the vector store setup
def setup_vector_store(chunks: list, index_name: str = "langbot-docs"):
    """Embed all chunks and store in Chroma (local vector store)."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",   # saves vectors to disk
    )
    return vector_store
```

That is it. No API key. No cloud index. No network calls at query time (embeddings still call OpenAI, but the vector search is local). Chroma stores everything in a `./chroma_db/` directory — delete it to reset the index.

For the rest of this series, I will use Pinecone in the main code and note the Chroma swap where it matters. The concepts — loading, chunking, embedding, similarity search — are identical across every vector store. LangChain's abstraction ensures you can switch providers without changing retrieval logic.

---

## How similarity search actually works

When you call `vector_store.similarity_search("remote work policy", k=3)`, here is what happens:

### 1. Embed the query

```
"remote work policy" ──► OpenAI text-embedding-3-small ──► [0.011, -0.031, 0.008, ...]
                                                               1536 dimensions
```

The embedding model converts the query string into a point in 1,536-dimensional space. This vector captures the *semantic meaning* of the query — not just the words, but the intent.

### 2. Compare to every stored vector

The vector store compares this query vector to every stored document chunk vector using **cosine similarity**:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

This returns a score from -1 to 1:

| Score | Meaning |
|---|---|
| **1.0** | Vectors point in exactly the same direction — identical meaning |
| **0.0** | Vectors are orthogonal — unrelated topics |
| **-1.0** | Vectors point in opposite directions — opposite meaning (rare in practice) |

For "remote work policy", the chunk containing "Employees may work remotely up to 4 days per week" scores ~0.89. The chunk about 401(k) contributions scores ~0.12. The chunk about code review practices scores ~0.05.

### 3. Return the top K

Pinecone sorts all chunks by similarity score and returns the top K. This is an approximate nearest-neighbor search — Pinecone uses an indexing algorithm (HNSW or IVF) that finds the closest vectors in logarithmic time rather than scanning every vector. For 14 vectors this is instant. For 14 million vectors, it is still milliseconds.

### Why this beats keyword search

A keyword search for "remote work policy" would find the policy document — but it would miss the benefits document's reference to "work from home," the engineering document's "offsite days," or any document that uses different words for the same concept. Embedding-based search captures meaning, not just string matching. "Home office," "telecommuting," "WFH," and "remote work" all map to similar vectors even though they share zero keywords with each other.

---

## Why retrieval matters for real apps

### 1. Your own data

General-purpose models know about Wikipedia and GitHub but nothing about your company's internal docs, your project's codebase, or your personal notes. Retrieval bridges that gap without fine-tuning — you keep your documents up to date, and the model answers from them. No training required.

### 2. Freshness

A model's training cutoff is fixed. If your company changes the PTO policy on Monday and you upload the new document, retrieval finds it immediately. The model sees the updated text in the prompt without any retraining.

### 3. Attribution

When the model answers from retrieved documents, you know *exactly* which document and which chunk informed the answer. This is the foundation for citations ("according to policy.txt, section 3...") — a feature we will add later in this series.

### 4. Reduced hallucination

Models are less likely to invent facts when they have relevant context in the prompt. Retrieval gives the model something concrete to work with — the chunk text is right there in the prompt, so the model paraphrases it rather than guessing. This does not eliminate hallucination, but it dramatically reduces it for factual questions.

### 5. Separation of concerns

The retrieval pipeline is independent of the generation pipeline. You can swap the vector store (Pinecone → Chroma → Weaviate) without touching the chat logic. You can upgrade the embedding model (text-embedding-3-small → text-embedding-3-large) without changing the vector store. You can add new document formats (PDF, web pages, code files) without changing the search code. Each component has a single responsibility.

---

## Common mistakes (and what they teach you)

**Mistake 1: using a different embedding model than what the index expects.**

```python
# Wrong — the index was created with 1536-dim vectors (text-embedding-3-small),
# but now you're querying with 3072-dim vectors (text-embedding-3-large)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
results = vector_store.similarity_search("query", k=3)
# Error: dimension mismatch — 3072 != 1536
```

The embedding model you use for queries must match the one you used for indexing. Vector dimensions are fixed per model. If you change models, you must re-index all documents. Write the model name in a comment near your vector store setup so future-you does not accidentally break it.

**Mistake 2: chunking before understanding the data.**

```python
# Too small — chunks are sentence fragments with no context
chunks = split_docs(docs, chunk_size=100, chunk_overlap=0)

# Too large — each chunk is a full page, retrieval is imprecise
chunks = split_docs(docs, chunk_size=5000, chunk_overlap=0)
```

`chunk_size=500` is a good default for prose documents. For code, you might go smaller (200–300) to capture individual functions. For legal documents with long paragraphs, you might go larger (800–1000). The right size depends on your documents' structure. When in doubt, test: does a retrieved chunk contain enough context to answer a question without also containing three unrelated topics?

**Mistake 3: skipping overlap.**

```python
# chunk_overlap=0 — a sentence that spans a chunk boundary is broken
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
```

Without overlap, the sentence "The company provides a $500 annual home office stipend for equipment and furniture" might be split as "The company provides a $" in chunk 4 and "500 annual home office stipend for equipment and furniture" in chunk 5. A search for "home office stipend" might only match chunk 5, losing the "$500" context. Overlap of 50–100 characters prevents this.

**Mistake 4: re-indexing on every startup without checking.**

The Day 8 code calls `setup_vector_store()` every time you run the app. If you restart 10 times, Pinecone has 10 copies of every document. For a tutorial this is harmless (duplicates don't break search, they just waste space). In production, add a check: load documents, hash their contents, compare to a stored hash, and only re-index if something changed.

**Mistake 5: treating `/search` results as answers.**

The `/search` command returns raw document chunks — not generated answers. "2 days of PTO per month, up to 24 days per year" is a chunk, not an answer. Tomorrow (Day 9), we will feed these chunks into the prompt and have the LLM synthesize a natural-language answer: "You get 2 days of PTO per month, maxing out at 24 days per year, with up to 5 days rolling over." The retrieval finds the data; the LLM turns it into an answer. Both pieces are necessary — neither is sufficient alone.

---

## What we have so far

Here is the complete, runnable `langbot.py` at the end of Day 8:

```python
# langbot.py — Day 8: Retrieval with vector stores and embeddings
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
# Day 8: Document retrieval infrastructure
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
            print("   (No documents found in docs/ — /search will be unavailable)")
            print("   (Create docs/*.txt or docs/*.md files and restart)")
    except Exception as e:
        print(f"   ⚠ Vector store setup skipped: {e}")
        print("   (Make sure PINECONE_API_KEY is set and the index exists)")

    # Day 2: Conversation history
    messages = []

    print("🤖 LangBot (type 'quit' to exit)")
    print("─" * 40)
    print("  Chat: just type a message")
    print("  List: 'list 5 programming books' or 'suggest weekend activities'")
    print("  Search: '/search remote work policy'")
    print("  Now streaming — watch LangBot type! ⚡")
    print("─" * 40)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        # ── Day 8: /search command ─────────────────────────────
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
                print(f"  (These chunks can feed into the chatbot — coming Day 9)")

            # Store the search command and results summary in history
            messages.append(HumanMessage(content=user_input))
            summary = f"Searched for '{query}': found {len(results)} document(s)."
            messages.append(AIMessage(content=summary))
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
            # Day 7: Chat mode — stream the response token by token
            print(f"\nLangBot: ", end="", flush=True)
            full_response = ""
            for chunk in stream_chain.stream({"input": user_input, "history": messages}):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()  # final newline

            reply_text = full_response

        # Day 2: Store the exchange in history
        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=reply_text))


if __name__ == "__main__":
    main()
```

---

## Try this now

Before tomorrow, experiment with the retrieval pipeline:

- **Add your own documents.** Drop a real `.txt` or `.md` file into `docs/` — your project's README, meeting notes, or a blog post draft. Restart LangBot and search it. The retrieval quality on your own data is the true test of the pipeline.

- **Vary `chunk_size` and `chunk_overlap`.** Change `chunk_size` to `200` and then `1000`. Re-index (delete and recreate the Pinecone index, or delete `./chroma_db` if using Chroma) and search the same query. How do the results differ? Smaller chunks are more precise but lose context. Larger chunks keep context but might pull in irrelevant information.

- **Try a query that is semantically similar but lexically different.** Search for "time off" (instead of "PTO" or "vacation"). The embedding model should still find the vacation policy chunk — because "time off" and "PTO" are semantically close even though they share no words. This is the power of embeddings over keyword search.

- **Add a PDF.** Install `pypdf` (`uv add pypdf`), then change the loader to `from langchain_community.document_loaders import PyPDFLoader`. Point `DirectoryLoader` at a directory with PDFs. The chunking, embedding, and search logic stays identical — only the loader changes.

- **Compare Pinecone and Chroma.** If you set up Pinecone, also try the Chroma alternative (the two-line swap shown above). The retrieval quality is identical. The difference is operational: Pinecone is hosted (no local storage, works across machines), Chroma is local (no network latency, no API key). Both are production-ready.

- **Watch the embeddings getting created.** Add `print(f"Embedding: {doc.metadata.get('source', '?')[:30]}...")` inside a loop before `PineconeVectorStore.from_documents()`. Each embedding API call costs ~$0.00002 with `text-embedding-3-small`. For 14 chunks, that is $0.00028. Understanding embedding costs now will help you budget for larger document collections later.

---

## Coming tomorrow

LangBot can now search documents with `/search` — but it cannot use those results in conversation. Ask "How much PTO do I get?" and it still shrugs. Tomorrow we close the loop: **RAG pipeline** — a retriever that feeds relevant document chunks into the chat prompt automatically, so LangBot answers factual questions from your documents without manual `/search` commands. One concept, one change — see you then.
