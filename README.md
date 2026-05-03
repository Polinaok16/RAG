# 📚 RAG Pipeline — Question Generation & Answering over arXiv Papers

A **Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain** and **YandexGPT** that automatically downloads research papers from arXiv, generates questions from their content, and answers those questions using a vector-store-backed retriever.

---

## 📌 Overview

This project demonstrates an end-to-end RAG workflow:

1. **Download** PDF papers from arXiv via direct URL
2. **Generate questions** automatically from each paper using an LLM
3. **Index** the paper chunks in a vector store (Chroma + HuggingFace embeddings)
4. **Answer** both manually written and auto-generated questions using a retrieval-augmented LLM chain

---

## 🗂️ Project Structure

```
├── RAG.ipynb               # Main notebook with full pipeline
├── arxiv_articles/         # Downloaded PDF articles (created at runtime)
├── results/                # Trainer checkpoints (created at runtime)
└── logs/                   # Training logs (created at runtime)
```

---

## ⚙️ Tech Stack

| Component | Tool |
|---|---|
| Framework | [LangChain](https://python.langchain.com/) |
| LLM | [YandexGPT](https://yandex.cloud/en/docs/foundation-models/) |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace |
| Vector Store | [Chroma](https://www.trychroma.com/) |
| PDF Loader | PyMuPDF (`PyMuPDFLoader`) |
| Text Splitter | `RecursiveCharacterTextSplitter` |
| Runtime | Google Colab / Kaggle |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install langchain langchain_community langchain_chroma pymupdf yandexcloud
pip install sentence-transformers chromadb
```

### 2. Set your YandexGPT credentials

In the notebook, replace the placeholder values:

```python
llm = YandexGPT(api_key="YOUR_API_KEY", folder_id="YOUR_FOLDER_ID")
embeddings = YandexGPTEmbeddings(api_key="YOUR_API_KEY", folder_id="YOUR_FOLDER_ID")
```

> **Note:** If YandexGPT embeddings are unavailable, the notebook falls back to HuggingFace `all-MiniLM-L6-v2` embeddings automatically.

### 3. Run the notebook

Open `RAG.ipynb` and run all cells in order.

---

## 🔄 Pipeline Walkthrough

### Task 1 — Download Articles

Eight NLP/CV papers are downloaded as PDFs from arXiv and saved to `arxiv_articles/`.

```python
articles = [
    'https://arxiv.org/pdf/2412.14173',
    'https://arxiv.org/pdf/2412.14169',
    ...
]
```

---

### Task 2 — Manual Questions

Eight domain-specific questions are written manually, one per article, covering the core contribution of each paper.

---

### Task 3 — Automatic Question Generation

For each article:
- The document is split into chunks (`chunk_size=1500`, `overlap=200`)
- **3 random chunks** are sampled per article to save LLM budget
- Each chunk is sent to YandexGPT with the prompt:

```
Based on the article create three meaningful questions in English about the
following text. The questions must be written in English regardless of the
language of the source text: {text}
```

- All generated questions are grouped by source article
- A final set of **3 questions per article** is selected at random (`random.seed(42)`)

---

### Task 4 — RAG Pipeline

```
User question
     │
     ▼
 Retriever  ──►  Chroma vector store  ──►  Top-6 relevant chunks
     │
     ▼
 Prompt template  (context + question)
     │
     ▼
 YandexGPT  ──►  Answer
     │
     ▼
 StrOutputParser  ──►  Final answer string
```

**Chunking:** `RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)`

**Retrieval:** cosine-similarity search, top-6 chunks returned

**Prompt:**
```
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Context: {context}
Question: {question}
Answer:
```

---

## 📊 Articles Used

| # | arXiv ID | Topic |
|---|---|---|
| 1 | 2412.14173 | AniDoc — 2D animation colorization |
| 2 | 2412.14169 | NOVA — text-to-video/image generation |
| 3 | 2412.14170 | E-CAR — efficient autoregressive image generation |
| 4 | 2412.14172 | Humanoid pose control from Internet videos |
| 5 | 2412.14166 | MegaSynth — synthetic data for models |
| 6 | 2412.14162 | EFTs for Drell-Yan processes at LHC |
| 7 | 2412.14171 | MLLMs self-explanations on VSI-Bench |
| 8 | 2412.14167 | VideoDPO — video preference optimization |

---

## 📝 Notes

- Do **not** commit your `api_key` or `folder_id` to version control — use environment variables or a `.env` file instead.
- The notebook is designed for Google Colab; paths like `/content/arxiv_articles` may need adjustment for local runs.
- YandexGPT has rate limits; if you hit them, add `time.sleep()` between LLM calls.

---

## 📄 License

This project is for educational purposes.
