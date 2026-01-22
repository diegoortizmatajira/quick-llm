# Quick-LLM

`quick-llm` is a LangChain-based framework designed for developers to quickly
build and deploy applications around Large Language Models (LLMs). The
framework provides an intuitive interface for configuring and chaining
different components such as language models, prompt templates, vector
retrievers, and FastAPI endpoints. With built-in support for
Retrieval-Augmented Generation (RAG), streaming, and chat-based workflows,
`quick-llm` simplifies the process of creating flexible and scalable solutions
for various LLM applications.

## Features

- **Chain Factory**: Seamlessly build simple chains, RAG chains, and RAG chains
  with document source tracking.
- **FastAPI Integration**: Auto-generate REST API endpoints for standard and
  streaming workflows.
- **Flexible Input/Output Handling**: Transform inputs/outputs using custom
  functions, integrate structured JSON outputs with Pydantic models, and support
  for various input types.
- **Extensibility**: Configure retrievers, vector stores, context formatters,
  and more to meet diverse requirements.
- **Testing and Mocking**: Includes utilities like `FakeListChatModel` and
  `InMemoryVectorStore` for reliable testing.

---

## Installation

Follow the steps to get started with `quick-llm`:

### For Users

You can install the package directly from PyPI using `pip`:

```bash
pip install quick-llm-factory
```

### For Maintainers

Maintainers working on the package itself can install all required development dependencies:

```bash
git clone <repository_url>
cd quick-llm
uv sync -d dev
source .venv/bin/activate
```

This ensures all tools for testing, linting, and building the package are available.

### Step 3: Confirm Python Version

Ensure you're running Python version ≥ 3.13:

```bash
python --version
```

---

## Usage

Here, we demonstrate how to use the core features of `quick-llm`:

### Building a Simple Chain

```python
from quick_llm.chain_factory import ChainFactory

# Initialize language model (assumed pre-configured `llm` instance)
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI()

# Configure the chain
chain = (ChainFactory()
    .use_language_model(llm)
    .use_prompt_template("Translate this text to French: {text}")
    .build())

response = chain.invoke({"text": "Hello, world!"})
print(response)
```

### Enabling RAG (Retrieval-Augmented Generation)

```python
from quick_llm.chain_factory import ChainFactory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.fake_embeddings import FakeEmbeddings
from langchain.vectorstores import InMemoryVectorStore

# Example documents
documents = ["Document 1 content.", "Document 2 content."]

# Initialize components for RAG
splitter = RecursiveCharacterTextSplitter()
embeddings = FakeEmbeddings()
vector_store = InMemoryVectorStore.from_texts(documents, embeddings, splitter)

# Build RAG chain
chain = (ChainFactory()
    .use_language_model(llm)
    .use_prompt_template("Answer the query based on the context: {context}")
    .use_rag()
    .use_vector_store(vector_store)
    .build())

response = chain.invoke({"query": "What is Document 1 about?"})
print(response)
```

### Deploying REST API Endpoints

```python
from fastapi import FastAPI
from quick_llm.chain_endpoints import ChainEndpoints

app = FastAPI()

# Register chain endpoints
endpoints = (ChainEndpoints(app, chain)
    .with_generate_endpoint()
    .with_chat_endpoint()
    .build())
```

Run FastAPI application using `uvicorn`:

```bash
uvicorn main:app --reload
```

---

## Examples

### Example 1: Adding JSON Output Parsing

```python
from pydantic import BaseModel

class TranslationOutput(BaseModel):
    original: str
    translated: str

chain = (ChainFactory()
    .use_language_model(llm)
    .use_prompt_template("Translate '{text}' and include metadata.")
    .use_json_model(TranslationOutput)
    .build())

result = chain.invoke({"text": "Good morning!"})
print(result.translated)
```

### Example 2: Streaming with FastAPI

```python
# Include streaming endpoints
app = FastAPI()
endpoints = (ChainEndpoints(app, chain)
    .with_generate_endpoint(stream_endpoint="/api/generate/stream")
    .build())

# Streaming responses available at `/api/generate/stream`
```

---

## Contributing

Contributions are welcomed! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch-name`).
5. Create a pull request.

---

## License

This project is licensed under **MIT License**.
