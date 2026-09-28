# Hybrid Retrieval-Augmented Generation (RAG) for Factual Solution Generation on GitHub Issues Using Sentence-BERT

A Retrieval-Augmented Generation (RAG) system for retrieving relevant GitHub Issues and generating concise, issue-level technical summaries using semantic retrieval, CrossEncoder reranking, and FLAN-T5.

## Overview

This project implements a RAG pipeline designed to help users find relevant solutions to technical problems documented in GitHub Issues.

The system combines:

- Fine-tuned Sentence-BERT for semantic query encoding and dense retrieval
- FAISS for vector similarity search
- CrossEncoder for candidate reranking
- Issue-level selection and context construction
- FLAN-T5 for generating factual technical summaries
- Gradio for an interactive web interface

The project was developed as an undergraduate thesis project in Computer Science.

## System Pipeline

```mermaid
flowchart LR
    A[User Query] --> B[Query Normalization]
    B --> C[Sentence-BERT]
    C --> D[FAISS Dense Retrieval]
    D --> E[Candidate Retrieval]
    E --> F[CrossEncoder Reranking]
    F --> G[Metadata Boosting]
    G --> H[Issue-Level Selection]
    H --> I[Context Construction]
    I --> J[FLAN-T5]
    J --> K[Generated Summary]
    K --> L[GitHub Issue References]
```

The pipeline follows a retrieve-rerank-generate approach. The retrieval stage identifies semantically relevant issue content, the reranking stage refines the candidate ranking, and the generation stage produces a concise summary from the selected issue context.

## Dataset

The system uses the GitHub Issues dataset provided in the companion repository:

[github-issues-dataset](https://github.com/briyanyehezkhiel/github-issues-dataset)

The dataset contains GitHub issue information including titles, issue bodies, answers, repository information, labels, and related metadata.

### Dataset Processing

The notebook performs the following processing steps:

1. Load the GitHub Issues dataset from the release file.
2. Keep closed issues.
3. Filter issues using technical labels:
   - `bug`
   - `error`
   - `exception`
   - `question`
   - `technical`
   - `help wanted`
4. Keep issues that contain answers.
5. Construct retrieval documents from issue bodies and answer chunks.

Dataset sizes recorded during the project:

| Stage | Records |
|---|---:|
| Initial dataset | 15,955 |
| After technical-label filtering | 13,665 |
| After answer filtering | 13,055 |
| Retrieval corpus vectors | 26,077 |

## Text Preprocessing

The preprocessing pipeline includes:

- HTML cleaning with BeautifulSoup
- Code block replacement/truncation
- Text normalization
- Whitespace normalization
- Sentence-based chunking
- Chunking with a maximum of 150 words

The retrieval corpus contains two main document types:

- `issue_body`
- `issue_answer`

Each corpus entry is associated with metadata such as issue ID, title, URL, repository, document type, likes, and label.

## Sentence-BERT Fine-Tuning

Sentence-BERT is fine-tuned to improve semantic retrieval for technical GitHub Issues.

The research uses `all-MiniLM-L6-v2` as the base model and trains it using pairs consisting of an issue title and its corresponding answer/solution.

The fine-tuning configuration includes:

| Parameter | Value |
|---|---|
| Base model | `all-MiniLM-L6-v2` |
| Training samples | 5,000 |
| Batch size | 32 |
| Epochs | 1 |
| Warmup steps | 100 |
| Loss function | `MultipleNegativesRankingLoss` |

The resulting model is then used to generate embeddings for both the retrieval corpus and user queries.

## Dense Retrieval

The normalized query is encoded using the fine-tuned Sentence-BERT model.

FAISS is used to perform dense vector similarity search and retrieve candidate chunks from the corpus.

The FAISS index uses normalized embeddings with inner-product similarity.

## CrossEncoder Reranking

The retrieved candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The CrossEncoder evaluates query-document pairs directly and produces relevance scores for the retrieved candidates.

## Metadata Boosting

The research pipeline also applies metadata-based score adjustment using GitHub reaction information.

Positive reactions such as:

- `plus_1`
- `heart`
- `rocket`
- `hooray`

can increase the score, while `minus_1` contributes a penalty.

This step is applied as an additional ranking signal before issue-level selection.

## Issue-Level Context Construction

Retrieved chunks are grouped by their original GitHub Issue rather than being treated only as independent text chunks.

The selected issue context can contain:

- Issue description
- Relevant answer/discussion chunks
- Supporting information from the same issue

The system then constructs and compresses the selected context so that it can be processed by the generation model.

## Generation

The generation component uses:

```text
google/flan-t5-large
```

FLAN-T5 generates a technical summary from the selected GitHub Issue context.

The generation configuration uses deterministic decoding with beam search and controls for repetition and output length.

The research configuration includes:

| Parameter | Value |
|---|---|
| Model | FLAN-T5-Large |
| Maximum input length | 1024 tokens |
| Maximum new tokens | 120 |
| Beam size | 4 |
| Sampling | Disabled |
| Repetition penalty | 1.2 |
| No-repeat n-gram size | 4 |
| Length penalty | 1.1 |

The generator is used to summarize information retrieved from GitHub Issues rather than to generate an unrestricted answer independently of the retrieved context.

## Retrieval Evaluation

The project evaluates retrieval using:

- Accuracy@1
- MRR@10
- nDCG@5
- Recall@5
- Recall@10

### Basic Retrieval Evaluation

The basic evaluation uses issue titles as queries to measure retrieval performance on sampled dataset records.

### Gold Query Evaluation

A separate gold-query evaluation uses manually written user-style queries associated with target issue IDs.

The gold-query evaluation measures:

- Accuracy@1
- MRR@10
- nDCG@5
- Recall@5
- Recall@10

The gold queries are designed to represent technical problems in a more natural form than directly using issue titles.

## End-to-End RAG Evaluation

The end-to-end RAG evaluation covers:

- Latency
- Coverage
- Faithfulness / Hallucination Rate
- BLEU
- ROUGE

The repository does not reproduce numeric evaluation results that are not stored in the notebook outputs.

## Interactive Interface

A Gradio interface is included for testing the complete RAG pipeline.

Example queries used with the interface include:

```text
http 403 websocket issue
connection reset by peer
windows socket stack problem
```

The interface accepts a technical query and displays:

- Retrieved GitHub Issues
- Issue titles
- Generated summaries
- GitHub issue references
- Runtime information

![GitHub Issue RAG Chatbot](https://github.com/user-attachments/assets/cbc777ed-7f64-4fd3-8a12-45d457d9c0fe)

## Technologies

- Python
- Pandas
- NumPy
- NLTK
- BeautifulSoup
- Sentence-Transformers
- FAISS
- CrossEncoder
- Hugging Face Transformers
- FLAN-T5
- PyTorch
- Gradio
- Scikit-learn
- Google Colab

## Project Structure

```text
github-issues-rag/
├── README.md
├── RAG_GitHub_Issues.ipynb
└── requirements.txt
```

The notebook contains the main experimental and inference workflow.

The inference workflow also uses trained model and retrieval artifacts that are kept separately from this repository, including the trained Sentence-BERT model, FAISS index, corpus, and metadata.

## How to Run

### 1. Open the notebook

Open:

```text
RAG_GitHub_Issues.ipynb
```

in Google Colab or a compatible Jupyter environment.

### 2. Install dependencies

Install the dependencies listed in:

```text
requirements.txt
```

The project uses libraries for data processing, NLP preprocessing, Sentence-BERT, FAISS, CrossEncoder reranking, FLAN-T5 generation, and Gradio.

A GPU-enabled environment is recommended for running the trained models and generation stage.

### 3. Prepare the trained artifacts

The inference workflow requires the trained Sentence-BERT model and retrieval artifacts used by the notebook.

These artifacts are not included in this repository. The notebook is provided as the main reference for the experimental and inference workflow.

### 4. Run the notebook

Execute the notebook cells in order to:

1. Load and preprocess the dataset.
2. Prepare the retrieval corpus.
3. Load the trained Sentence-BERT model.
4. Perform FAISS retrieval.
5. Rerank candidates using CrossEncoder.
6. Select relevant issues and construct context.
7. Generate technical summaries using FLAN-T5.
8. Launch the Gradio interface.

## Project Context

This project was developed as an undergraduate thesis in Computer Science.

The research focuses on using Retrieval-Augmented Generation to help users find relevant GitHub Issues and understand technical solutions through concise, context-grounded summaries.

The complete workflow covers:

```text
Data Filtering
    ↓
Text Preprocessing
    ↓
Sentence-BERT Fine-Tuning
    ↓
Retrieval Corpus Construction
    ↓
FAISS Dense Retrieval
    ↓
CrossEncoder Reranking
    ↓
Metadata-Based Ranking
    ↓
Issue-Level Context Construction
    ↓
FLAN-T5 Generation
    ↓
Generated Summary + References
```

## Companion Dataset

The dataset used by this project is maintained separately:

[github-issues-dataset](https://github.com/briyanyehezkhiel/github-issues-dataset)

The dataset repository contains the release files used by the project and is kept separate from the RAG implementation.

## Google Colab

The original notebook is also available in Google Colab:

[Open the RAG notebook in Google Colab](https://colab.research.google.com/drive/1ETBEArIFsMs4JJolQgKw_08oPqaTjwVU?usp=sharing)

## Note

This repository contains the research notebook and supporting documentation for the RAG project.

The trained model and retrieval artifacts are not included in the repository. The companion dataset is maintained separately.

The project was developed for academic research and portfolio documentation.
