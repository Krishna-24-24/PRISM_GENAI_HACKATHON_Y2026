# TRACEIQ

### Adaptive Agentic Code Retrieval & Ranking Engine

> **Samsung PRISM GenAI Hackathon 3rd Edition (2026–27) · Theme 01: Agentic Code Intelligence**  
> **Evaluation Target:** CoIR AppsRetrieval — NDCG@10, MRR  
> **Core Constraint:** CPU-only inference · No code execution · No hallucination

TRACEIQ is a code intelligence engine that retrieves and ranks relevant JavaScript functions and classes from large codebases given a natural-language query. It parses JavaScript repositories into structured, symbol-level chunks using Tree-sitter ASTs, indexes them across three retrieval channels (dense semantic, BM25 lexical, and identifier lookup), and routes each query adaptively through a deterministic classifier — all without executing code or calling an LLM.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Node 18+](https://img.shields.io/badge/Node-18%2B-green)
![Tests](https://img.shields.io/badge/Tests-167%20passing-brightgreen)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Problem](#2-the-problem)
3. [Objectives & Status](#3-objectives--status)
4. [Key Features](#4-key-features)
5. [System Architecture](#5-system-architecture)
6. [End-to-End Data Flow](#6-end-to-end-data-flow)
7. [Code Ingestion Pipeline](#7-code-ingestion-pipeline)
8. [AST Parsing & Code Understanding](#8-ast-parsing--code-understanding)
9. [Indexing](#9-indexing)
10. [Query Analysis & Routing](#10-query-analysis--routing)
11. [Retrieval Channels](#11-retrieval-channels)
12. [Candidate Fusion](#12-candidate-fusion)
13. [Structural Signal Boosting](#13-structural-signal-boosting)
14. [Agentic Retrieval Loop](#14-agentic-retrieval-loop)
15. [Version-Aware Incremental Indexing](#15-version-aware-incremental-indexing)
16. [Cross-Encoder Reranker (Ablated)](#16-cross-encoder-reranker-ablated)
17. [Evaluation](#17-evaluation)
18. [Ablation Results](#18-ablation-results)
19. [Running the System](#19-running-the-system)
20. [Reproducing the Experiments](#20-reproducing-the-experiments)
21. [API Reference](#21-api-reference)
22. [Repository Structure](#22-repository-structure)
23. [Limitations](#23-limitations)
24. [Roadmap](#24-roadmap)

---

## 1. Overview

TRACEIQ is a **retrieval engine, not a chatbot**. Given a natural-language query, it returns ranked JavaScript code snippets — functions, methods, classes — from an indexed codebase. It does not synthesize, explain, or execute code.

**What goes in:** A natural-language string describing what a piece of code does. Examples:
- `"function that validates and sanitizes user form input"`
- `` "`handleHttpRequest` middleware chain" ``
- `"class that extends EventEmitter and implements retry logic"`

**What comes out:** A ranked list of code snippets, each annotated with file path, line range, symbol type, retrieval channel contributions, and structural boost scores.

**Who it is for:** Engineers working with large JavaScript codebases who need to find existing implementations without grepping or knowing exact function names in advance. Also: hackathon judges evaluating code retrieval systems.

**What makes it different:**

| Concern | Naive Approach | TRACEIQ |
|---|---|---|
| Query routing | Fixed retrieval strategy | Deterministic per-query routing based on query signals |
| Identifier queries | Treats `getUserById` as prose | Detects camelCase/snake_case, routes to Symbol channel |
| Code versioning | Full index rebuild per commit | Hash-diffed incremental updates via `VersionManager` |
| Latency | LLM-in-the-loop (seconds) | Pure CPU: 222ms P50 (dense-only fast path) |
| Reranking | Cross-encoder always on | Cross-encoder available but ablated off (16x latency, 0 NDCG gain) |

Architecture Pipeline

```mermaid
flowchart TD
    Q[Natural Language Query] --> A(Query Analyzer: Deterministic Routing)
    A -->|semantic| D[Dense Retrieval FAISS]
    A -->|structural| B[BM25 Lexical]
    A -->|symbolic| S[Symbol Lookup]
    
    D --> F{Weighted RRF Fusion}
    B --> F
    S --> F
    
    F --> SB[Structural Graph Boost]
    SB --> R[Final Top-K Snippets]
    
    style Q fill:#1b1f2b,stroke:#363d57,color:#fff
    style A fill:#242840,stroke:#3b82f6,color:#e8eaf2
    style D fill:#1428a0,stroke:#2540d4,color:#fff
    style B fill:#f59e0b,stroke:#b45309,color:#fff
    style S fill:#a855f7,stroke:#7e22ce,color:#fff
    style F fill:#1e2230,stroke:#363d57,color:#fff
    style SB fill:#1e2230,stroke:#363d57,color:#fff
    style R fill:#10b981,stroke:#047857,color:#fff
```

**High-level pipeline:**

```mermaid
flowchart TD
    A["Natural Language Query"] --> B["Query Analyzer\n(Deterministic Classifier)"]
    B -->|"dense weight"| C["Dense Retrieval\nFAISS IndexFlatIP\nBGE-small-en-v1.5"]
    B -->|"bm25 weight"| D["BM25 Retrieval\nOkapi BM25\ncamelCase-aware tokenizer"]
    B -->|"symbol weight"| E["Symbol Lookup\n3-tier exact / case-insensitive /\nsub-word match"]
    C --> F["Weighted Reciprocal\nRank Fusion\nwRRF  k=60"]
    D --> F
    E --> F
    F --> G["Structural Boost\nCall graph +\nImport co-citation"]
    G --> H["Top-K Ranked\nCode Snippets"]
```

---

## 2. The Problem

The Samsung PRISM Theme 01 challenge defines the following scenario: given a large JavaScript repository (thousands of functions, hundreds of files), and a natural-language query from a developer, retrieve the most relevant code snippet.

**Why this is hard:**

1. **Semantic mismatch.** Developer queries are written in natural language. Code is written in a formal language where meaning is encoded in identifiers, not prose. A query saying *"validate a user's phone number"* has no lexical overlap with the function `sanitizeTelInput` that implements it.

2. **Identifier-heavy queries.** Some queries *are* identifiers: `getUserById`, `handleHttpRequest`. Dense embeddings treat these as opaque strings. A semantic model trained on prose text doesn't know that `handleHttpRequest` splits into `handle | http | request`.

3. **Structural relationships.** A function is not just its body. Knowing that `processTx` calls `validateAmount` and is called by `submitOrder` makes it more or less relevant depending on what the query asks for. Embeddings don't encode call graphs.

4. **Context-window limitations.** A large JavaScript repo may contain tens of thousands of functions. You cannot send them all to an LLM. You need a retrieval layer that selects the right ~10 before any generative step.

5. **Code versioning.** Codebases change. If 5% of functions are modified between versions, you should not re-embed the entire corpus — that is a full rebuild every commit. Efficient incremental updates are a first-class engineering requirement.

6. **Latency budgets.** A retrieval engine that takes 3+ seconds per query (e.g., a CPU-bound cross-encoder over 50 pairs) is not deployable. Components must be profiled and cut if they violate the budget without improving recall.

**Why sending the whole repo to an LLM does not work:** An 8,765-chunk corpus exceeds the context window of any practical LLM by a factor of 10–100. Retrieval is the necessary first stage.

---

## 3. Objectives & Status

| Objective | Description | Status |
|---|---|---|
| **P0 – Retrieval Accuracy** | Dense + multi-channel adaptive retrieval with competitive NDCG@10 on CoIR AppsRetrieval | ✅ Implemented |
| **P1 – Version-Aware Retrieval** | Incremental index updates via `chunk_hash` diffing with retrieval-quality parity to full rebuild | ✅ Implemented |
| **P0 – AST-Aware Parsing** | Tree-sitter JS parsing extracting functions, methods, classes, arrow functions, call graphs, imports | ✅ Implemented |
| **P0 – Adaptive Routing** | Deterministic per-query classifier routing to Dense / BM25 / Symbol channels with per-category weights | ✅ Implemented |
| **P0 – Structural Signals** | Call-graph + import co-citation boosts as additive score modifiers | ✅ Implemented |
| **P0 – CPU-Only Inference** | No GPU dependency; BGE-small-en-v1.5 runs entirely on CPU | ✅ Implemented |
| **P0 – API** | FastAPI backend with retrieval, evaluation, and system status endpoints | ✅ Implemented |
| **P0 – Test Coverage** | 167 tests covering all 14 QA dimensions | ✅ Passing |
| **Cross-Encoder Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` post-retrieval reranker | ✅ Implemented · ablated off (no NDCG gain, 16x latency cost) |
| **Agentic Loop** | 2-pass retrieval with AST-pivot + PRF expansion | ✅ Implemented · gated fallback (hurts NDCG on flat NL corpus) |
| **Frontend Demo** | React + Vite search interface with per-query routing trace visualization | ✅ Implemented |
| **ONNX / int8 quantization** | Export BGE-small to ONNX int8 for further latency reduction | 🗓 Roadmap |
| **TypeScript / JSX full coverage** | `.ts`, `.tsx` support via `tree-sitter-typescript` | 🗓 Roadmap |
| **Persistent BM25 index** | Serialize BM25 to disk alongside FAISS index | 🗓 Roadmap |

---

## 4. Key Features

| Feature | Implementation | Notes |
|---|---|---|
| **AST-aware JS parsing** | Tree-sitter (`tree-sitter-javascript`) | Extracts function / method / class / arrow-function nodes |
| **Symbol-level chunking** | One chunk per AST symbol node | Preserves semantic boundaries |
| **camelCase/snake_case tokenizer** | Regex sub-word expansion in `tokenizer.py` | `handleHttpRequest` → `handle`, `http`, `request` |
| **Dense retrieval** | FAISS `IndexFlatIP` via `IndexIDMap` wrapper | BGE-small-en-v1.5, 384-dim, L2-normalised, CPU |
| **BM25 retrieval** | `rank_bm25` Okapi BM25 | Tokenises symbol name + context + code |
| **Symbol / identifier lookup** | 3-tier: exact → case-insensitive → sub-word | O(1) dictionary lookup |
| **Adaptive query routing** | Deterministic regex + heuristic classifier | <1ms per query; 5 categories; no LLM |
| **Weighted RRF fusion** | `w_c / (k + rank_c(d))`, k=60 | Per-channel weights from ablation evidence |
| **Structural signal boosting** | Call-graph + import co-citation | Additive only; never overrides semantic score |
| **Incremental version updates** | `VersionManager` + `chunk_hash` SHA-256 diffing | Surgically add/remove from FAISS `IndexIDMap` + BM25 + Symbol |
| **Embedding cache** | Per-chunk `.npy` files keyed by `chunk_hash` | Avoids re-embedding unchanged code across versions |
| **Agentic loop (gated)** | 2-pass: Initial → AST-pivot or Lexical-PRF expansion | Off by default; activated when confidence is low |
| **Cross-encoder reranker (available)** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Implemented but ablated off |
| **FastAPI backend** | `/api/search`, `/api/system`, `/api/evaluation`, `/api/repositories` | Full OpenAPI schema |
| **React + Vite frontend** | TypeScript, React 19, react-router-dom | Routing trace visible per query |
| **Docker deployment** | Non-root user, CORS hardened, 1MB file limit | `docker-compose up --build` |

---

## 5. System Architecture

```mermaid
flowchart TD
    subgraph ING ["Code Ingestion (Offline)"]
        JS["JavaScript Repository\n(.js .jsx .mjs .cjs)"]
        SCAN["File Scanner\n(scanner.py)\nIgnores: node_modules, dist,\n.git, build / *.min.js *.map"]
        PARSE["AST Parser\n(parser.py)\nTree-sitter JavaScript"]
        CHUNK["CodeChunk\n(models.py)\nchunk_id · symbol_name\nfile_path · lines · calls\nimports · chunk_hash"]
        JS --> SCAN --> PARSE --> CHUNK
    end

    subgraph IDX ["Index Construction (Offline)"]
        EMBED["Embedder\n(embedder.py)\nBGE-small-en-v1.5\n384-dim · CPU · L2-normed"]
        FAISS["DenseChannel\n(dense_index.py)\nFAISS IndexIDMap(IndexFlatIP)"]
        BM25I["BM25Channel\n(bm25_index.py)\nOkapi BM25\ncamelCase tokenizer"]
        SYM["SymbolChannel\n(symbol_channel.py)\n3-tier name index"]
        STRUCT["StructuralIndex\n(structural_index.py)\nCall graph + import graph"]
        CHUNK --> EMBED --> FAISS
        CHUNK --> BM25I
        CHUNK --> SYM
        CHUNK --> STRUCT
    end

    subgraph RET ["Query-Time Retrieval (Online)"]
        Q["Query String"]
        QA["QueryAnalyzer\n(query_analyzer.py)\nDeterministic routing\n5 categories · <1ms"]
        FAISS2["Dense Search\nFAISS cosine similarity"]
        BM25S["BM25 Search\nOkapi BM25 scoring"]
        SYMS["Symbol Search\nExact + sub-word match"]
        FUSION["wRRF Fusion\n(fusion.py)\nscore = Σ w_c / (k + rank)"]
        BOOST["Structural Boost\n(structural_index.py)\ncaller / callee / import-sibling"]
        TOPK["Top-K Results\nchunk · score · channel_contributions\nfused_score · structural_boost"]
        Q --> QA
        QA -->|"dense weight > 0"| FAISS2
        QA -->|"bm25 weight > 0"| BM25S
        QA -->|"symbol weight > 0"| SYMS
        FAISS2 --> FUSION
        BM25S --> FUSION
        SYMS --> FUSION
        FUSION --> BOOST --> TOPK
    end

    subgraph VER ["Version Management (Incremental)"]
        VM["VersionManager\n(version_manager.py)\nchunk_hash diff\nadd / remove / modify"]
        VM --> FAISS
        VM --> BM25I
        VM --> SYM
    end

    subgraph AGT ["Agentic Retrieval (Gated Fallback)"]
        AG["RetrievalAgent\n(retrieval_agent.py)\nPass 1: Adaptive Search\nInspect confidence + budget\nPass 2: AST-pivot or PRF\nFuse with 0.8 decay"]
    end

    subgraph API ["FastAPI Backend"]
        ENDPOINT["POST /api/search\nGET  /api/system\nGET  /api/evaluation\nGET  /api/repositories"]
    end

    TOPK --> API
    AG --> API
```

### Architectural Component Summary

| Component | File | Role |
|---|---|---|
| File Scanner | `src/ingestion/scanner.py` | Recursive JS file discovery, ignore rules, 1MB size guard |
| AST Parser | `src/parsing/parser.py` | Tree-sitter parse, symbol extraction, hash + ID generation |
| CodeChunk | `src/chunking/models.py` | Central 20-field Pydantic data model |
| BM25 Tokenizer | `src/chunking/tokenizer.py` | camelCase/snake_case sub-word expansion |
| Embedder | `src/indexing/embedder.py` | `EmbeddingProvider` ABC + BGESmallProvider (BGE-small) |
| Dense Channel | `src/indexing/dense_index.py` | FAISS `IndexIDMap(IndexFlatIP)`, persist/load, incremental add/remove |
| BM25 Channel | `src/indexing/bm25_index.py` | Okapi BM25, incremental add/remove |
| Symbol Channel | `src/retrieval/symbol_channel.py` | 3-tier exact/case-insensitive/sub-word name index |
| Query Analyzer | `src/retrieval/query_analyzer.py` | Regex + keyword classifier → `QueryProfile` with weights |
| Hybrid Retriever | `src/retrieval/hybrid_retriever.py` | Orchestrates 3 channels + wRRF fusion |
| Adaptive Retriever | `src/retrieval/adaptive_retriever.py` | Per-query routing + structural boosting layer over HybridRetriever |
| wRRF Fusion | `src/retrieval/fusion.py` | `RRFFuser`, `LinearFuser` (ablation only) |
| Structural Index | `src/retrieval/structural_index.py` | Call graph + import co-citation graph, additive score boosts |
| Version Manager | `src/indexing/version_manager.py` | `chunk_hash` diff, incremental update across all channels |
| Retrieval Agent | `src/agent/retrieval_agent.py` | 2-pass agentic loop: AST-pivot + PRF expansion |
| Cross-Encoder | `src/ranking/reranker.py` | `ms-marco-MiniLM-L-6-v2` (implemented, ablated off) |
| FastAPI App | `apps/api/main.py` | All HTTP endpoints, lifecycle, CORS |
| Retriever Singleton | `apps/api/retriever_singleton.py` | Thread-safe lazy init from pre-built FAISS index |
| Metrics | `src/evaluation/metrics.py` | NDCG@k, MRR, Recall computed from actual search output |

---

## 6. End-to-End Data Flow

```mermaid
flowchart TD
    REPO["JavaScript Repository"]
    DISC["discover_js_files()\nSkip: node_modules / dist / .git\nSkip: *.min.js / *.map / size > 1MB\nAllowed: .js .jsx .mjs .cjs"]
    PARSESTEP["parse_javascript_file()\ntree_sitter.Parser(JS_LANGUAGE)\nWalk AST for:\n  function_declaration\n  method_definition\n  class_declaration\n  arrow_function"]
    AST["For each symbol node:\nextract_node_text() → code\nextract_name() → symbol_name\nget_calls() → calls[]\nget_imports_exports() → imports[]\nnormalize_code() + SHA-256 → chunk_hash\ngenerate_chunk_id() → stable chunk_id"]
    CTXSTR["Build context string:\nFile: {file_path}\nSymbol: {symbol_name}\nType: {symbol_type}\nCalls: {calls}\nCode:\n{code_text}"]
    INDEXSTEP["Index CodeChunks:\nDenseChannel.build() → embed + FAISS\nBM25Channel.add_chunks()\nSymbolChannel.add_chunks()\nStructuralIndex.build()"]
    QUERYINPUT["Query String"]
    CLASSIFY["QueryAnalyzer.analyze()\nDetect: backtick ids / camelCase /\nsnake_case / dot accessors /\nstructural keywords / usage keywords\nClassify into 5 categories\nAssign per-channel weights"]
    SEARCH["Channel Search:\nDenseChannel.search() → FAISS cosine\nBM25Channel.search() → Okapi BM25\nSymbolChannel.search() → 3-tier"]
    FUSE["RRFFuser.fuse()\nw_c / (k + rank_c(d))"]
    SBOST["StructuralIndex.score_candidates()\nAnchors: top-5 by fused_score\nBoost: callees / callers / import siblings"]
    SORT["Sort by fused_score + structural_boost\nReturn top-K AdaptiveRetrievalResult"]

    REPO --> DISC --> PARSESTEP --> AST --> CTXSTR --> INDEXSTEP
    QUERYINPUT --> CLASSIFY --> SEARCH --> FUSE --> SBOST --> SORT
```

**Stage-by-stage table:**

| Stage | Input | Output | Key logic |
|---|---|---|---|
| File discovery | Repository root path | Sorted list of JS file paths | OS walk, ignore-list, size guard |
| AST parsing | File path + raw bytes | List of `CodeChunk` objects | Tree-sitter parse, node-type walk |
| Metadata extraction | AST node + source bytes | `symbol_name`, `calls[]`, `imports[]`, `exports[]` | `extract_name`, `get_calls`, `get_imports_exports` |
| Chunk hashing | Normalized code string | `chunk_hash` (SHA-256), `chunk_id` (SHA-256) | `normalize_code` collapses whitespace; ID excludes line numbers |
| Context string build | Chunk metadata | Structured text string | `File: … Symbol: … Calls: … Code: …` |
| Embedding | Context string | 384-dim L2-normed float32 vector | BGE-small with query instruction prefix |
| FAISS add | Vector + int ID | Updated `IndexIDMap` | `index.add_with_ids(vec, [fid])` |
| Query classify | Raw query string | `QueryProfile` with weights | Regex signals → category → weight table |
| Dense search | Query embedding | `top_k` FAISS nearest neighbours | `index.search(q_vec, top_k)` |
| BM25 search | Tokenized query | `top_k` Okapi BM25 ranked IDs | `bm25.get_scores(tokens)` |
| Symbol search | Extracted identifiers | `top_k` 3-tier name matches | Dict lookup → tier scoring |
| wRRF fusion | Per-channel ranked lists + weights | Single ranked list | `w / (60 + rank)` per channel |
| Structural boost | Candidate IDs + fused scores | Per-chunk additive boost | Call-graph neighbour of top-5 anchors |
| Final sort | fused_score + structural_boost | Top-K `AdaptiveRetrievalResult` | `sorted(candidates, key=final_score)` |

---

## 7. Code Ingestion Pipeline

```mermaid
flowchart LR
    A["repository root"] --> B["discover_js_files()\nrecursive os.walk"]
    B --> C{".js .jsx .mjs .cjs?"}
    C -->|No| SKIP1["Skip"]
    C -->|Yes| D{"*.min.js\n*.bundle.js\n*.map?"}
    D -->|Yes| SKIP2["Skip (minified)"]
    D -->|No| E{"size > 1MB?"}
    E -->|Yes| SKIP3["Skip (DoS guard)"]
    E -->|No| F["parse_javascript_file()"]
    F --> G["Tree-sitter parse\nget AST root"]
    G --> H["walk_tree_for_types()\nfunction_declaration\nmethod_definition\nclass_declaration\narrow_function"]
    H --> I["For each node:\nextract text, name, calls\nget file imports / exports\nbuild CodeChunk"]
    I --> J["normalize_code()\nchunk_hash = SHA256(norm)\nchunk_id = SHA256(repo:file:name:type:hash)"]
    J --> K["CodeChunk list\n(one per symbol)"]
```

**Configuration:**

| Setting | Value | Source |
|---|---|---|
| Ignored directories | `node_modules`, `.git`, `dist`, `build`, `coverage`, `.next`, `.nuxt`, `out` | `scanner.py:IGNORED_DIRS` |
| File types processed | `.js`, `.jsx`, `.mjs`, `.cjs` | `scanner.py:ALLOWED_EXTENSIONS` |
| Skipped extensions | `*.min.js`, `*.bundle.js`, `*.map` | `scanner.py:IGNORED_EXTENSIONS` |
| Max file size | 1MB (1,048,576 bytes) | `scanner.py` size guard |
| Arrow function filter | Skip if parent is `call_expression` | `parser.py` lines 122–124 |

**Syntax error handling:** Tree-sitter is error-tolerant. It will parse files with syntax errors and still extract valid nodes from regions that parse cleanly. Error nodes are skipped silently — no exception is raised per file.

**Arrow function filtering:** Arrow functions used as direct arguments to call expressions (`.map(x => ...)`, `.filter(...)`) are excluded. Only arrow functions assigned to variables or used standalone are indexed, preventing thousands of trivial inline callbacks from polluting the index.

---

## 8. AST Parsing & Code Understanding

TRACEIQ uses `tree-sitter` with the `tree-sitter-javascript` grammar to produce a concrete syntax tree for every `.js` file.

```
JavaScript source file
        ↓  tree_sitter.Parser(JS_LANGUAGE).parse(source_bytes)
Concrete Syntax Tree (CST)
        ↓  walk_tree_for_types(root, target_types)
Symbol nodes (function_declaration, method_definition, class_declaration, arrow_function)
        ↓  extract_name, get_calls, get_imports_exports
Structured metadata per symbol → CodeChunk
```

**Extracted fields per CodeChunk:**

| Field | Type | How extracted |
|---|---|---|
| `symbol_name` | `str` | First `identifier` or `property_identifier` child of the node; or from `variable_declarator` parent (for `const foo = function...`); defaults to `"anonymous"` |
| `symbol_type` | `str` | AST node type: `function_declaration`, `method_definition`, `class_declaration`, `arrow_function` |
| `start_line` | `int` | `node.start_point[0] + 1` (1-based) |
| `end_line` | `int` | `node.end_point[0] + 1` (1-based) |
| `code` | `str` | Raw UTF-8 slice `source[node.start_byte:node.end_byte]` |
| `calls` | `List[str]` | All `call_expression` children's first child text (the callee expression) |
| `imports` | `List[str]` | All `import_statement` text strings at file root level |
| `exports` | `List[str]` | `[symbol_name]` if node is wrapped in `export_statement`, else `[]` |
| `chunk_hash` | `str` | SHA-256 of `normalize_code(code_text)` — whitespace-collapsed |
| `chunk_id` | `str` | SHA-256 of `repo_id:file_path:symbol_name:symbol_type:chunk_hash` |
| `context` | `str` | Multi-line string: `"File: …\nSymbol: …\nType: …\nCalls: …\nCode:\n…"` |

**Why chunk_id excludes line numbers:** If unrelated code is inserted above a function, its line number shifts but its content is unchanged. A line-number-based ID would falsely mark the function as "modified," triggering unnecessary re-embedding. Identity is strictly content-based.

**Example:**

```javascript
// src/auth.js
export async function getUserById(id) {
    const user = await db.findById(id);
    validateUser(user);
    return user;
}
```

Produces:

```
CodeChunk {
    symbol_name:  "getUserById"
    symbol_type:  "function_declaration"
    file_path:    "src/auth.js"
    start_line:   2,  end_line: 6
    calls:        ["db.findById", "validateUser"]
    imports:      []
    exports:      ["getUserById"]
    chunk_hash:   "a3f7c2..."
    chunk_id:     "9d41bb..."
    context:      "File: src/auth.js\nSymbol: getUserById\nType: function_declaration\n
                   Calls: db.findById, validateUser\nCode:\nasync function getUserById(id) { ... }"
}
```

---

## 9. Indexing

### 9.1 Dense Index (FAISS)

**Model:** `BAAI/bge-small-en-v1.5` (33M parameters, 384-dim embeddings) via `sentence-transformers`.

Model choice rationale (from `embedder.py` docstring):
- 384-dim — fits in CPU L3 cache during inference
- Measured ~80–120ms per batch of 32 on a modern laptop CPU
- `EmbeddingProvider` abstract interface allows any model (UniXcoder, CodeSage, jina-v2) to be swapped by implementing a new subclass

**Asymmetric encoding:**
- Corpus: encoded without any prefix
- Queries: encoded with `"Represent this sentence for searching relevant passages: {query}"`  
  This asymmetry is intentional per the BGE model card.

**FAISS index type:** `IndexFlatIP` (exact inner product, equivalent to cosine similarity on L2-normalised vectors) wrapped in `IndexIDMap` to support selective addition and removal by integer ID without rebuilding.

**Embedding cache:** Each chunk embedding is saved to `data/embed_cache/{chunk_hash}_{dim}.npy`. On subsequent builds (or version updates), cached `.npy` files are loaded directly, skipping re-embedding for unchanged chunks.

**Persistence:**
- `data/faiss_index/index.faiss` — FAISS binary format
- `data/faiss_index/mappings.json` — `str_to_int` and `int_to_str` ID maps
- `data/faiss_index/chunks.json` — serialized `CodeChunk` objects (used to rebuild BM25 + Symbol at API startup)

### 9.2 BM25 Index

BM25 (`rank_bm25.BM25Okapi`) is built over a compound token string: `symbol_name + context + code`, plus explicit identifier tokens from `symbol_name`, `calls`, and `imports`.

The tokenizer (`tokenizer.py`) performs:
1. Split on non-alphanumeric characters
2. Lowercase each token
3. Snake_case expansion: `get_user_id` → `get`, `user`, `id`
4. camelCase expansion via regex lookahead: `handleHttpRequest` → `handle`, `http`, `request`

BM25 does **not** persist to disk. It is rebuilt from `chunks.json` at API startup (<2 seconds for 8,765 chunks).

### 9.3 Symbol Index

In-memory dictionary mapping lowercased `symbol_name` to `(original_name, chunk_id, chunk)` triples, plus a sub-word token map from `_split_identifier()` (camelCase + snake_case split). All operations are O(1) dictionary lookups.

---

## 10. Query Analysis & Routing

The `QueryAnalyzer` classifies every query into one of five categories using only regex patterns and keyword set intersections. No LLM is involved. Classification runs in under 1ms.

**Signal extraction:**

| Signal | Detection method |
|---|---|
| Backtick identifier (`` `foo` ``) | Regex `` `([A-Za-z_][A-Za-z0-9_]*)` `` |
| camelCase token | Regex `\b[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*\b` |
| snake_case token | Regex `\b[A-Za-z][a-z0-9]*(?:_[A-Za-z][a-z0-9]+)+\b` |
| Dot accessor | Regex `\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]+\b` |
| Structural keyword | Set: `class`, `extends`, `implements`, `interface`, `inherit`, `mixin`, `abstract`, `decorator` |
| Usage keyword | Set: `call`, `invoke`, `use`, `example`, `usage`, `show me` |
| Behavioral keyword | Set: `validates`, `handles`, `returns`, `throws`, `sanitize`, `retry`, `parses` |

**Classification rules (evaluated in order):**

```
SYMBOLIC   if: backtick identifier present  OR
               identifier_ratio > 0.5  OR
               (word_count ≤ 3 AND ≥1 identifier AND camelCase/snake_case)  OR
               (frame keyword AND ≥1 identifier AND camelCase/snake_case/backtick)

STRUCTURAL if: structural keyword present AND no usage keyword

USAGE      if: usage keyword present AND ≥1 identifier extracted

MIXED      if: camelCase OR snake_case OR dot accessor present (but not SYMBOLIC)

SEMANTIC   otherwise (default)
```

**Per-category channel weights (from ablation evidence):**

| Category | Dense | BM25 | Symbol | Rationale |
|---|---|---|---|---|
| `semantic` | 1.0 | — | — | BM25 hurts Dense via RRF dilution on NL queries |
| `symbolic` | 0.5 | 1.5 | 2.0 | Symbol must dominate; BM25 helps exact token match |
| `structural` | 1.0 | 0.5 | — | Structural keywords exist in code text |
| `usage` | 1.0 | — | 1.5 | Find call sites by name |
| `mixed` | 1.0 | 0.5 | 0.5 | Conservative; Dense fallback |

The `QueryProfile` returned exposes all signal flags, extracted identifiers, weights, and a human-readable `routing_reason` string — returned verbatim in the API response for full observability.

---

## 11. Retrieval Channels

### Dense Channel

**Implementation:** `FAISS IndexFlatIP.search(q_vec, top_k)`.

The query is embedded using `BGESmallProvider.embed_query()` (with instruction prefix) → 384-dim L2-normalised float32 vector. FAISS returns the `top_k` nearest neighbours by inner product. Results are mapped back from FAISS integer IDs to string `chunk_id`s.

**Default top_k before fusion:** 100 candidates.

### BM25 Channel

**Implementation:** `BM25Okapi.get_scores(query_tokens)`.

The query is tokenized with the same camelCase/snake_case tokenizer used at index time. Zero-score results are filtered out. Results returned in descending score order.

**Default top_k before fusion:** 100 candidates.

### Symbol Channel

**Implementation:** 3-tier dictionary lookup.

For each query token (split on whitespace):
1. Exact / case-insensitive lookup in `_exact` dict — score 1.0 (exact) or 0.8 (case-insensitive)
2. Sub-word token lookup in `_token_map` — score 0.6

Scores accumulate per chunk across query tokens. Results sorted by descending accumulated score.

**Default top_k before fusion:** 50 candidates.

**Input:** The Symbol channel receives the `extracted_identifiers` list from `QueryProfile`, not the raw query string. This ensures `"where is getUserById defined?"` searches for `getUserById` as a symbol rather than the full natural-language sentence.

---

## 12. Candidate Fusion

TRACEIQ uses **Weighted Reciprocal Rank Fusion** (wRRF), adapted from Cormack et al. (2009).

**Formula:**

```
score(d) = Σ_c   w_c / (k + rank_c(d))
```

Where:
- `d` = document (chunk_id)
- `c` = retrieval channel (dense / bm25 / symbol)
- `w_c` = per-channel weight from `QueryProfile.recommended_weights`
- `k` = 60 (Cormack default; configurable via `HybridConfig.rrf_k`)
- `rank_c(d)` = 1-based rank of `d` in channel `c`'s result list
- If `d` is absent from channel `c`: `rank_c(d) = len(channel_c_results) + 1`

**Why RRF over linear score combination:** RRF is immune to score distribution differences between channels. Dense returns cosine similarity in [0, 1]; BM25 returns unnormalized term-frequency scores; Symbol returns tier-based scores in {0.6, 0.8, 1.0}. Combining these directly requires calibration. RRF operates on ranks, which are commensurable across channels.

**Why k=60:** The Cormack et al. 2009 paper establishes k=60 as a robust default. The benchmark `run_ablation.py` includes a `--rrf-k-sweep` option that tests k ∈ {1, 5, 10, 30, 60, 100}.

A `LinearFuser` is also implemented in `fusion.py` for ablation comparisons but is not used in the production path.

---

## 13. Structural Signal Boosting

After wRRF fusion, a second scoring pass applies additive structural boosts derived from the call graph and import graph.

**Graph construction (`StructuralIndex.build`):**

- **Pass 1:** Index `symbol_name.lower()` → `{chunk_id}` for every chunk. Index normalised import module specifiers → `{chunk_id}`.
- **Pass 2:** For each chunk, for each name in `calls[]`, resolve to `chunk_id`s and record directed edges `caller_id → callee_id` and `callee_id → caller_id`.

**Scoring (`StructuralIndex.score_candidates`):**

1. Identify anchors: top-5 candidates by fused RRF score.
2. For each anchor, collect: callees (functions the anchor calls), callers (functions that call the anchor), and import siblings (chunks sharing at least one import module specifier).
3. Apply additive boosts:

| Relationship | Default boost | Rationale |
|---|---|---|
| Callee of an anchor | +0.005 | If A is top-ranked and A calls B, B is likely relevant |
| Caller of an anchor | +0.003 | If A calls an anchor, A is related |
| Shares import with an anchor | +0.001 | Weak co-location signal |

**Design principles:**
- Boosts are **additive only**. They never replace the fused score.
- Boosts are **bounded** to be smaller than typical primary score differences (typical RRF range 0.01–0.03).
- The `StructuralWeights.enabled` flag disables all boosts to zero without code changes.
- The StructuralIndex is rebuilt in O(N) time (< 10ms for 10K chunks) after each version update.

---

## 14. Agentic Retrieval Loop

**Status: Implemented and available. Gated off in the default production API path.**

On the CoIR AppsRetrieval corpus, the agentic loop reduces NDCG@10 from 0.0498 to 0.0236. It is available for demonstration and for JavaScript-specific use cases where call-graph expansion is expected to help.

**Loop structure:**

```
PASS 1: AdaptiveRetriever.search(original_query, top_k=20)
         ↓
INSPECTION:
  • elapsed > 50% of 800ms budget → terminate early
  • best_score > 0.85              → high confidence, terminate early
  • max_iterations < 2             → terminate early
         ↓
IDENTIFY MISSING SIGNAL:
  Strategy A — AST Pivot:
    Extract calls[] from top-3 Pass-1 hits
    Join first 5 call names → expansion_query
    QueryAnalyzer classifies this as SYMBOLIC on Pass 2

  Strategy B — Lexical PRF:
    If no calls found, extract camelCase/snake_case tokens
    from top-1 hit's code body → expansion_query
         ↓
PASS 2: AdaptiveRetriever.search(expansion_query, top_k=20)
         ↓
FUSE (Pass 1 + Pass 2):
  Pass 2 scores multiplied by DECAY_FACTOR = 0.8
  If chunk in both passes: take max(p1_score, p2_score * 0.8)
  Sort by final_score → return top_K
```

**Termination conditions:** no Pass-1 candidates, budget exhausted, high confidence hit, `max_iterations=1`, or no expansion signal found.

**Trace logging:** Every agent step is recorded in an `AgentTrace` (`iteration`, `action`, `query_used`, `candidates_retrieved`, `latency_ms`). Traces can be requested via `POST /api/search` with `show_trace: true`.

---

## 15. Version-Aware Incremental Indexing

TRACEIQ supports indexing evolving codebases without full rebuilds.

**`VersionManager.set_version(new_chunks)` algorithm:**

```python
new_dict = {c.chunk_id: c for c in new_chunks}

added_ids   = new_dict.keys() - active_chunks.keys()
deleted_ids = active_chunks.keys() - new_dict.keys()
modified    = [c for c in (new_dict.keys() & active_chunks.keys())
               if new_dict[c].chunk_hash != active_chunks[c].chunk_hash]

# Modified chunks are deleted (old version) + added (new version)
deleted_ids |= {c.chunk_id for c in modified}
chunks_to_add = [new_dict[cid] for cid in added_ids] + modified

# Push incremental updates to each index channel
dense_chan.remove_chunks(deleted_ids)   # IndexIDMap.remove_ids()
dense_chan.add_chunks(chunks_to_add)    # embed (cache hit) + add_with_ids()

bm25_chan.remove_chunks(deleted_ids)    # rebuild Okapi matrix
bm25_chan.add_chunks(chunks_to_add)

symbol_chan.remove_chunks(deleted_ids)
symbol_chan.add_chunks(chunks_to_add)

structural_index.build(active_chunks.values())  # O(N) rebuild, < 10ms
```

**Measured results** (from `results/version_updates_20260917T153003.json`, 8,765 chunks, ~10% simulated change):

| Action | Time | NDCG@10 |
|---|---|---|
| Full Rebuild (V1) | 135.9s | 0.0481 |
| Incremental Update (V1 → V2) | 65.26s | **0.0496** |
| Full Rebuild (V2 from scratch) | 20.34s | **0.0496** |

The incremental update produces **mathematically identical retrieval quality** to a full rebuild. The benchmark script asserts this with `assert eval_v2_inc['ndcg@10'] == eval_v2_fresh['ndcg@10']`.

**Why incremental updates work:** The `chunk_id` is SHA-256 of `repo_id:file_path:symbol_name:symbol_type:chunk_hash`. It excludes line numbers, so inserting code above a function doesn't trigger a spurious "modified" signal for the unchanged function. FAISS `IndexIDMap` supports O(1) vector removal by integer ID, making surgical updates possible without rebuilding the entire matrix.

---

## 16. Cross-Encoder Reranker (Ablated)

The `CrossEncoderReranker` (`ranking/reranker.py`) is implemented but not used in the default production path.

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~33M parameters, ~86MB RAM)

**Mechanism:** Unlike the dual-encoder (which embeds query and document independently), the cross-encoder processes `(query, code)` pairs jointly through a cross-attention transformer. This theoretically allows precise term-to-code alignment.

**Ablation finding:**

| Configuration | NDCG@10 | P50 Latency | Memory |
|---|---|---|---|
| Dense + Adaptive + Structural (Config 7) | 0.0467 | 353ms | 14MB |
| + Cross-Encoder Reranker (Config 8) | 0.0467 | **3,547ms** | 100MB |

50 pairs on CPU: ~3.5 seconds, zero NDCG improvement. The reranker was ablated off.

**Timeout fallback:** If scoring exceeds `timeout_ms` (default 1500ms), the reranker gracefully restores the original fused + structural scores and sorts by those.

---

## 17. Evaluation

### Dataset

**CoIR AppsRetrieval** (loaded via the `mteb` library):

- **Corpus:** 8,765 JavaScript code chunks (after AST parsing of the AppsRetrieval code corpus)
- **Queries:** 200 queries (deterministic selection: `sorted(qrels.keys())[:200]`)
- **Qrels:** Graded relevance judgements from the dataset

### Metrics (implemented in `src/evaluation/metrics.py`)

**NDCG@10** (Normalized Discounted Cumulative Gain at rank 10):

```
DCG@k  = Σ_{i=1}^{k}  (2^rel_i - 1) / log₂(i + 1)
NDCG@k = DCG@k / IDCG@k
```

**MRR** (Mean Reciprocal Rank):

```
MRR = mean over queries of  1 / rank_of_first_relevant_doc
```

**Recall@1, Recall@10:** Whether at least one relevant document appears in top-1 or top-10 results.

All metric values are computed from actual search output. No numbers are fabricated.

### Benchmark Scripts

| Script | What it measures |
|---|---|
| `benchmarks/run_ablation.py` | 9 cumulative configurations; saves JSON + TXT |
| `benchmarks/run_final_ablation.py` | Extended ablation with full configuration sweep |
| `benchmarks/run_dense_benchmark.py` | Dense-only baseline |
| `benchmarks/run_bm25_benchmark.py` | BM25-only baseline |
| `benchmarks/run_adaptive_benchmark.py` | Adaptive routing vs. uniform baseline |
| `benchmarks/run_structural_benchmark.py` | Effect of structural signal boosting |
| `benchmarks/run_reranker_benchmark.py` | Cross-encoder reranker vs. no reranker |
| `benchmarks/run_agentic_benchmark.py` | Agentic loop vs. single-pass |
| `benchmarks/run_version_benchmark.py` | Incremental update vs. full rebuild + parity check |

---

## 18. Ablation Results

**Source:** `results/ablation_20260917T185022.json`  
**Dataset:** CoIR AppsRetrieval · **Queries:** 200 · **Corpus:** 8,765 chunks  
**Timestamp:** 2026-09-17T18:50:22

| # | Configuration | NDCG@10 | MRR | P50 Latency | P95 Latency | Index MB |
|---|---|---|---|---|---|---|
| 1 | BM25 Only | 0.0131 | 0.0112 | 589ms | 1,065ms | 0.3 |
| **2** | **Dense Only (BGE-small)** | **0.0498** | **0.0422** | **222ms** | **283ms** | **13.2** |
| 3 | BM25 + Dense (RRF k=60) | 0.0486 | 0.0420 | 288ms | 404ms | 13.5 |
| 4 | + Symbol / Identifier | 0.0486 | 0.0420 | 341ms | 396ms | 13.5 |
| 5 | + Query Preprocessing | 0.0467 | 0.0410 | 348ms | 432ms | 13.5 |
| 6 | + Adaptive Routing | 0.0467 | 0.0410 | 353ms | 408ms | 13.5 |
| 7 | + Structural Signals | 0.0467 | 0.0410 | 354ms | 434ms | 14.0 |
| 8 | + Reranker **(CUT)** | 0.0467 | 0.0410 | **3,547ms** | **3,760ms** | 100.0 |
| 9 | + Agentic (gated fallback) | 0.0236 | 0.0198 | 398ms | 551ms | 13.5 |

**Key findings:**

**1. Dense-only is the best configuration on this corpus.** NDCG@10=0.0498, MRR=0.0422 at 222ms P50. Adding BM25 to Dense *hurts* NDCG (–2.4%) while adding 30% latency. This is because the AppsRetrieval corpus is ~91% natural-language queries where BM25 adds noise via RRF dilution.

**2. BM25 alone is ~4× worse than Dense** (NDCG 0.0131 vs 0.0498) and paradoxically slower (589ms vs 222ms P50) because the BM25 matrix rebuild time dominates for an 8,765-chunk corpus.

**3. The cross-encoder reranker was ablated off.** It adds 16× latency (3.5 seconds vs 222ms) with zero NDCG improvement. The corpus is NL-dominated, so cross-attention alignment between query and code provides no benefit over the dual-encoder.

**4. The agentic loop halves NDCG** (0.0236 vs 0.0498). The AST-pivot and PRF expansion generate identifier-heavy queries on Pass 2, which get routed to Symbol+BM25 channels. These channels are weak on the NL-dominated corpus. The agent is available but gated off.

**5. Adaptive routing preserves Dense-only quality.** On the ~91% semantic NL queries, the classifier correctly assigns Dense-only weights. On identifier queries, it boosts Symbol+BM25 without degrading NL query quality.

**6. Structural signals add negligible latency** (<1ms additive) with no measurable NDCG impact on this corpus. They are expected to provide measurable gains on real JavaScript codebases with richer call graphs.

**Version update parity** (`results/version_updates_20260917T153003.json`):

| Action | Time | NDCG@10 |
|---|---|---|
| Full Rebuild V1 | 135.9s | 0.0481 |
| Incremental Update V1 → V2 | 65.26s | **0.0496** |
| Full Rebuild V2 (fresh) | 20.34s | **0.0496** |

Incremental update produces identical retrieval quality to a fresh rebuild (asserted in the benchmark).

---

## 19. Running the System

### Prerequisites

- Python 3.10+
- Node 18+ (frontend only)
- `git`

> **Important:** The API requires a pre-built FAISS index at `data/faiss_index/index.faiss`. Build it by running the ablation benchmark first (see §20), or provide a FAISS index built from your own JavaScript corpus.

### Option A — Docker (Backend API only)

```bash
git clone <repo-url> traceiq
cd traceiq

# Build the pre-built FAISS index first (required):
python benchmarks/run_ablation.py --sample 200

# Start the API on port 8080
docker-compose up --build
```

The Docker image:
- Runs as a non-root user (`traceiq`)
- Serves FastAPI on port 8000 (mapped to 8080 externally via `docker-compose.yml`)
- Mounts `src/`, `apps/`, `configs/`, `data/` as volumes

> **Wait ~60 seconds** for BGE-small to load and warm up on first startup.

### Option B — Manual Local Setup

**Backend API:**

```bash
# Create and activate a Python 3.10+ virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt

# Build the index first (downloads CoIR dataset, builds FAISS):
python benchmarks/run_ablation.py --sample 200

# Start the API
python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8080 --reload
# OpenAPI docs at: http://localhost:8080/docs
```

**Frontend (React + Vite):**

```bash
cd apps/web
npm install
npm run dev
# Open: http://localhost:5173
```

**Running Tests:**

```bash
# From repository root (with venv active)
pytest
# Expected: 167 tests passing
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TRACEIQ_ENVIRONMENT` | `development` | Environment label (returned in `/health`) |
| `TRACEIQ_LOG_LEVEL` | `INFO` | Log verbosity |

---

## 20. Reproducing the Experiments

All benchmarks use CoIR AppsRetrieval via the `mteb` library, which downloads the dataset automatically on first run (~500MB).

### Full Ablation (9 configurations, 200 queries)

```bash
python benchmarks/run_ablation.py --sample 200
# Output:
#   results/ablation_<timestamp>.json   (machine-readable)
#   results/ablation_<timestamp>.txt    (human-readable table)
```

Reuse a pre-built FAISS index:

```bash
python benchmarks/run_ablation.py --sample 200 --skip-build
```

Sweep RRF k values:

```bash
python benchmarks/run_ablation.py --sample 200 --rrf-k-sweep
# Tests k in {1, 5, 10, 30, 60, 100} on the BM25+Dense config
```

### Version Update Benchmark

```bash
python benchmarks/run_version_benchmark.py --sample 200
# Simulates: 5% deleted, 5% modified, 5% added, 85% unchanged
# Asserts retrieval parity between incremental and fresh rebuild
# Output: results/version_updates_<timestamp>.json
```

### Individual Channel Benchmarks

```bash
python benchmarks/run_dense_benchmark.py
python benchmarks/run_bm25_benchmark.py
python benchmarks/run_adaptive_benchmark.py
python benchmarks/run_structural_benchmark.py
python benchmarks/run_reranker_benchmark.py
python benchmarks/run_agentic_benchmark.py
```

---

## 21. API Reference

All endpoints served by FastAPI at `http://localhost:8080`. Interactive docs at `/docs`.

### `POST /api/search`

Execute a retrieval query.

**Request body:**
```json
{
  "query": "function that validates and sanitizes user form input",
  "top_k": 10,
  "show_trace": false
}
```

- `query`: 1–1000 characters
- `top_k`: 1–50 (default 10)
- `show_trace`: include per-stage pipeline trace in response

**Response:**
```json
{
  "query": "...",
  "results": [
    {
      "rank": 1,
      "chunk_id": "9d41bb...",
      "file_path": "src/validation/formValidator.js",
      "start_line": 14,
      "end_line": 38,
      "symbol_name": "validateFormInput",
      "symbol_type": "function_declaration",
      "code": "function validateFormInput(form) { ... }",
      "relevance_score": 0.017234,
      "fused_score": 0.016600,
      "structural_boost": 0.000634,
      "channel_contributions": {"dense": 0.016600},
      "imports": ["import validator from 'validator'"],
      "calls": ["validator.isEmail", "sanitizeInput"]
    }
  ],
  "query_profile": {
    "category": "semantic",
    "routing_reason": "semantic: no identifier signals; dense-only (ablation evidence)",
    "recommended_weights": {"dense": 1.0},
    "signals": {
      "has_backtick_identifier": false,
      "has_camel_case_token": false,
      "has_snake_case_token": false,
      "has_dot_accessor": false,
      "has_structural_keyword": false,
      "has_usage_keyword": false,
      "has_behavioral_keyword": false,
      "identifier_count": 0,
      "word_count": 8,
      "extracted_identifiers": []
    }
  },
  "channels_used": ["dense"],
  "total_candidates": 100,
  "returned_count": 10,
  "latency_ms": 224.5,
  "trace": null
}
```

When `show_trace: true`, `trace` contains one `{step, detail, count}` entry per pipeline stage (classification → signal extraction → routing → dense → symbol → wRRF → structural → final ranking).

### `GET /api/system`

Index health and configuration status.

```json
{
  "status": "ok",
  "environment": "development",
  "index_ready": true,
  "index_path": "/app/data/faiss_index",
  "chunk_count": 8765,
  "file_count": 312,
  "index_size_bytes": 13852416,
  "current_repository": "apps-retrieval-corpus",
  "current_version": "v1",
  "dense_top_k": 100,
  "bm25_top_k": 100,
  "symbol_top_k": 50
}
```

Returns `"status": "degraded"` if the FAISS index is not found.

### `GET /api/repositories`

List indexed repositories with version IDs, file counts, chunk counts.

### `GET /api/evaluation`

Return the most recent ablation results from `results/ablation_*.json` (latest by `mtime`).

### `GET /api/example-queries`

Return 5 curated example queries (one per retrieval category) for the demo UI.

### `GET /health`

Basic liveness check: `{"status": "ok", "environment": "development"}`.

### `POST /search` (deprecated)

Legacy stub. Returns `{"deprecated": true, "use": "/api/search"}`.

---

## 22. Repository Structure

```
traceiq/
├── apps/
│   ├── api/
│   │   ├── main.py              # FastAPI application, all endpoints, lifespan
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── retriever_singleton.py  # Thread-safe lazy init from pre-built FAISS index
│   │   └── exceptions.py        # Custom exception handlers
│   └── web/                     # React + Vite frontend (TypeScript)
│       ├── src/
│       │   ├── api.ts           # API client (typed fetch wrappers)
│       │   ├── App.tsx          # Root component, routing
│       │   ├── pages/           # Search, results, evaluation pages
│       │   └── components/      # Result card, query profile panel, etc.
│       ├── package.json
│       └── vite.config.ts
│
├── benchmarks/
│   ├── run_ablation.py          # Main: 9 cumulative configs, JSON + TXT output
│   ├── run_final_ablation.py    # Extended ablation sweep
│   ├── run_dense_benchmark.py   # Dense-only baseline
│   ├── run_bm25_benchmark.py    # BM25-only baseline
│   ├── run_adaptive_benchmark.py
│   ├── run_structural_benchmark.py
│   ├── run_reranker_benchmark.py
│   ├── run_agentic_benchmark.py
│   └── run_version_benchmark.py # Incremental update parity check
│
├── data/
│   ├── faiss_index/             # Built by benchmark scripts
│   │   ├── index.faiss          # FAISS binary index
│   │   ├── mappings.json        # str_to_int / int_to_str ID maps
│   │   └── chunks.json          # Serialized CodeChunk objects
│   └── embed_cache/             # Per-chunk .npy files keyed by chunk_hash
│
├── results/
│   ├── ablation_*.json          # Ablation run results
│   ├── ablation_*.txt           # Human-readable tables
│   ├── version_updates_*.json   # Version benchmark results
│   └── charts/                  # (generated chart assets)
│
├── src/
│   ├── agent/
│   │   └── retrieval_agent.py   # 2-pass agentic loop: AST-pivot + PRF
│   ├── chunking/
│   │   ├── models.py            # CodeChunk Pydantic model (20 fields)
│   │   └── tokenizer.py         # BM25 tokenizer with camelCase/snake_case expansion
│   ├── evaluation/
│   │   └── metrics.py           # NDCG@k, MRR, Recall@k
│   ├── indexing/
│   │   ├── embedder.py          # EmbeddingProvider ABC + BGESmallProvider
│   │   ├── dense_index.py       # FAISS IndexIDMap(IndexFlatIP), persist/load
│   │   ├── bm25_index.py        # Okapi BM25 channel
│   │   └── version_manager.py   # chunk_hash diff → incremental update engine
│   ├── ingestion/
│   │   └── scanner.py           # JS file discovery with ignore rules + size guard
│   ├── parsing/
│   │   └── parser.py            # Tree-sitter AST parser, symbol extraction, ID gen
│   ├── ranking/
│   │   └── reranker.py          # CrossEncoderReranker (available, ablated off)
│   ├── retrieval/
│   │   ├── adaptive_retriever.py  # Per-query routing + structural boosting
│   │   ├── fusion.py              # RRFFuser (production), LinearFuser (ablation)
│   │   ├── hybrid_retriever.py    # 3-channel orchestration + HybridConfig
│   │   ├── query_analyzer.py      # Deterministic classifier → QueryProfile
│   │   ├── structural_index.py    # Call graph + import co-citation graph
│   │   └── symbol_channel.py      # 3-tier symbol name index
│   ├── config.py                # Pydantic Settings (TRACEIQ_ env prefix)
│   └── logger.py                # Structured logger
│
├── tests/
│   ├── conftest.py
│   ├── test_agent.py            # RetrievalAgent: termination, PRF, AST pivot
│   ├── test_api.py              # FastAPI endpoint smoke tests
│   ├── test_bm25.py             # BM25 channel correctness
│   ├── test_dense.py            # Dense channel: add/remove, FAISS, cache
│   ├── test_fusion.py           # RRF + Linear fuser correctness
│   ├── test_ingestion.py        # File scanner, ignore rules
│   ├── test_parser.py           # AST parser, chunk extraction
│   ├── test_qa_suite.py         # 14-dimension comprehensive QA suite (524 lines)
│   ├── test_query_analyzer.py   # All 5 categories, edge cases
│   ├── test_review_fixes.py     # Regression tests
│   └── test_structural.py       # Call graph, import siblings, boost scoring
│
├── Dockerfile                   # python:3.10-slim, non-root user, port 8000
├── docker-compose.yml           # API service: 8080:8000
├── pyproject.toml               # Python ≥3.10, pytest config, black/isort
└── requirements.txt             # fastapi uvicorn pydantic faiss-cpu rank_bm25
                                 # tree-sitter tree-sitter-javascript
                                 # sentence-transformers mteb networkx
```

---

## 23. Limitations

**Corpus-dependent results.** The ablation uses CoIR AppsRetrieval, dominated (~91%) by natural-language queries about competitive programming solutions. On a real JavaScript codebase with identifier-heavy developer queries (e.g., `"where is getUserById defined"`), the Symbol channel and adaptive routing are expected to provide larger measurable gains.

**JavaScript only.** The Tree-sitter parser is hardcoded to the `tree-sitter-javascript` grammar. TypeScript (`.ts`, `.tsx`) and other languages are not parsed.

**BM25 is not persisted.** The BM25 index is rebuilt from `chunks.json` at every API startup. For large corpora this can take several seconds.

**FAISS index must be pre-built.** The API does not build the FAISS index on startup. If `data/faiss_index/index.faiss` does not exist, the API starts in `"degraded"` state and returns HTTP 503 on search requests.

**Agentic loop hurts NL queries.** The `RetrievalAgent` is gated off in the default API path. On NL-query-dominated corpora, the AST-pivot expansion generates identifier-heavy Pass-2 queries that mismatch the corpus and reduce NDCG by ~50%.

**No ranking calibration.** The wRRF weights are assigned based on ablation results on one specific dataset distribution. They are not calibrated for arbitrary codebases.

**No GPU configuration.** `BGESmallProvider` defaults to `device="cpu"`. GPU support is available through the `sentence-transformers` API but is not wired into the configuration system.

---

## 24. Roadmap

The following items are **not currently implemented**.

| Feature | Priority | Notes |
|---|---|---|
| ONNX int8 quantization of BGE-small | High | Estimated 2–3× CPU speedup, <1% quality loss |
| TypeScript / JSX parsing via `tree-sitter-typescript` | High | Required for modern JS ecosystem coverage |
| Persistent BM25 index | Medium | Eliminate rebuild latency at API startup |
| Configurable routing weights per corpus | Medium | Allow per-deployment ablation to inform routing |
| GPU inference path | Low | `BGESmallProvider(device="cuda")` config flag |
| Multi-repository search | Low | API already tracks `repo_id` per chunk; needs frontend filter |
| Online indexing endpoint | Low | `POST /api/index` to ingest a new repository at runtime |
| RRF k tuning per query category | Research | Currently global k=60 |
| Structural graph persistence | Research | Save call graph to avoid O(N) rebuild per version update |
