# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**quick-llm** is a LangChain-based framework for quickly building and deploying Large Language Model (LLM) applications. It provides:
- Fluent API for building LLM chains with various configurations
- Built-in support for Retrieval-Augmented Generation (RAG)
- REST API endpoint generation with FastAPI
- Chat and generation workflows with sync, async, and streaming support
- Flexible input/output transformation

## Development Commands

### Setup
```bash
# Install dependencies (using uv package manager)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/chain_factory_test.py
pytest tests/chain_endpoints_test.py

# Run tests with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_name_pattern"
```

### Linting
```bash
# Run pylint
pylint src/quick_llm
```

### Build
```bash
# Build distribution packages (output in dist/)
uv build
```

## Architecture

### Core Components

The project is organized around four main architectural layers:

#### 1. ChainFactory (chain_factory.py)
Central factory class for building LLM chains using a fluent builder API.

**Key capabilities:**
- Configure language models and prompt templates
- Enable RAG (Retrieval-Augmented Generation) with vector stores/retrievers
- Parse structured JSON output using Pydantic models
- Transform inputs/outputs with custom functions
- Build three types of chains: Simple, RAG, and RAG-with-sources

**Typical usage pattern:**
```python
factory = (ChainFactory()
    .use_language_model(llm)
    .use_prompt_template(template)
    .use_json_model(OutputModel)  # Optional: for structured output
    .use_rag()  # Optional: enable RAG
    .use_vector_store(vector_store)  # Required if using RAG
)
chain = factory.build()
```

**Static factory methods:**
- `ChainFactory.for_json_model(model)` - Create factory for JSON output
- `ChainFactory.for_rag_with_sources(model)` - Create factory for RAG with source tracking

#### 2. ChainEndpoints (chain_endpoints.py)
Creates REST API endpoints from chains using FastAPI.

**Endpoint types:**
- `/api/generate` - Standard generation (POST)
- `/api/generate/stream` - Streaming generation (POST, NDJSON format)
- `/api/chat` - Standard chat (POST)
- `/api/chat/stream` - Streaming chat (POST, NDJSON format)

**Configuration:**
```python
app = FastAPI()
endpoints = (ChainEndpoints(app, chain)
    .with_generate_endpoint()  # Add generation endpoints
    .with_chat_endpoint(chat_provider)  # Add chat endpoints
    # OR
    .with_defaults()  # Add both with default paths
    .build()
)
```

#### 3. Chat Module (chat/)
Adapts chains to chat interfaces with message handling.

**ChatProvider** - Abstract base defining chat interface (send, send_async, send_stream)

**ChainChatProvider** - Concrete implementation that:
- Wraps a chain factory function for lazy initialization
- Transforms LangChain messages to/from chain input/output format
- Provides default transformers: messages → string, output → AIMessage
- Supports sync, async, and streaming modes

#### 4. Supporting Components

**PromptInputParser** (prompt_input_parser.py)
- Normalizes various input types (str, dict, BaseModel, messages) into dict format
- Used as first step in chain composition

**Type Definitions** (type_definitions.py)
- `ChainInputType` - Union of all accepted input types
- Generic type variables for chain outputs

### Chain Data Flow

**Simple Chain:**
```
Input → PromptInputParser → AdditionalValuesInjector →
PromptTemplate → LanguageModel → OutputCleaner → OutputTransformer → Output
```

**RAG Chain:**
```
Input → PromptInputParser → Retriever → DocumentFormatter →
AdditionalValuesInjector (with context) → PromptTemplate →
LanguageModel → OutputCleaner → OutputTransformer → Output
```

**RAG with Sources:**
- Same as RAG but preserves source documents in output
- Optionally formats them using final_answer_formatter

### Design Patterns

- **Builder Pattern**: ChainFactory uses method chaining for configuration
- **Factory Pattern**: Static factory methods and chain factory functions
- **Adapter Pattern**: ChainChatProvider adapts chains to chat interface
- **Composite Pattern**: Chains composed using LangChain Runnables (pipe operator `|`)
- **Strategy Pattern**: Pluggable transformers and formatters

## Testing Patterns

Tests use:
- **Parametrized tests** (`@pytest.mark.parametrize`) for testing multiple input types
- **FakeListChatModel** and **FakeListLLM** for deterministic testing without real LLM calls
- **FastAPI TestClient** for API endpoint testing
- Multiple chain types: string output, JSON output, RAG
- Coverage of sync, async, and streaming modes

When writing tests:
- Use parametrized fixtures for testing different input types (str, dict, BaseModel)
- Mock LLM responses with FakeListChatModel/FakeListLLM
- Use InMemoryVectorStore with FakeEmbeddings for RAG tests
- Test both successful and error cases
- Verify streaming responses use NDJSON format

## Key Files

| File | Purpose |
|------|---------|
| `src/quick_llm/chain_factory.py` | Core chain building logic, RAG orchestration |
| `src/quick_llm/chain_endpoints.py` | FastAPI integration, endpoint registration |
| `src/quick_llm/chat/chain_chat_provider.py` | Chat interface adapter |
| `src/quick_llm/chat/chat_provider.py` | Abstract chat interface |
| `src/quick_llm/prompt_input_parser.py` | Input format normalization |
| `src/quick_llm/type_definitions.py` | Type aliases |
| `tests/chain_factory_test.py` | ChainFactory unit tests |
| `tests/chain_endpoints_test.py` | API endpoint unit tests |

## Important Implementation Notes

### Circular Reference Prevention
The codebase was recently refactored to fix circular reference issues between modules. Be mindful of import dependencies.

### RAG Configuration
When using RAG:
- Either configure `vector_store` OR `retriever` (not both)
- If using `vector_store`, also configure `text_splitter` and `embeddings`
- `retrieval_query_builder` is optional (transforms input before retrieval)
- `context_formatter` is optional (formats retrieved documents)

### JSON Output
When using JSON output:
- Set a Pydantic model with `.use_json_model(Model)`
- The model's structure is automatically added to the prompt as `{format_instructions}`
- Output is parsed and validated against the model

### Streaming
Streaming endpoints:
- Return NDJSON format (one JSON object per line)
- Each line is a complete, parseable JSON object
- Use `GenerateResponse` or `ChatResponse` wrapper for consistency

### Input Transformation
ChainInputType supports:
- `str` - Direct string input
- `dict` - Dictionary with template variables
- `BaseModel` - Pydantic model (auto-converted to dict)
- `Sequence[MessageLikeRepresentation]` - List of messages

### Python Version
Project requires Python >=3.13 as specified in pyproject.toml.
