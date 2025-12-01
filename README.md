# smallevals <img src="logo/smallevals_emoji.png" alt="logo" width="32" height="32"> - Small Language Models Evaluation Suite for RAG Systems

A lightweight evaluation framework powered by tiny ( really tiny <img src="logo/smallevals_emoji.png" alt="logo" width="32" height="32"> ) 0.6B models — runs 100% locally on CPU/GPU/MPS, extremely fast and cheap.

Evaluation tools requiring LLM-as-a-judge, that costs/doesn't scale easily. <img src="logo/smallevals_emoji.png" alt="logo" width="32" height="32"> evaluates in seconds in GPU, in minutes in any CPU  <img src="logo/smallevals_emoji.png" alt="logo" width="32" height="32"> <img src="logo/smallevals_emoji.png" alt="logo" width="32" height="32">!

## Evaluate Retrieval

Evaluation of RAG system includes retrieval and RAG stage, <img src="logo/smallevals_emoji.png" alt="logo" width="32" height="32"> attacks to test retrieval and RAG answers(in the near future)!

## Models

| Model Name | Task | Status | Link |
|------------|-------|--------|------|
| **QAG-0.6B** | Generate golden Q/A from chunks (synthetic evaluation data) | Available | [🤗](mburaksayici/golden_generate_qwen_0.6b_v2_gguf) |
| **CRC-0.6B** | Context relevance classifier (question ↔ retrieved chunk) | Incoming | — |
| **GJ-0.6B** | Groundedness / faithfulness judge (answer ↔ context) | Incoming | — |
| **ASM-0.6B** | Answer correctness / semantic similarity | Incoming | — |

**Current Focus**: Retrieval evaluation (QAG-0.5B). Generation evaluation models (CRC-0.5B, GJ-0.5B, ASM-0.5B) are future work.

## Installation

```bash
pip install smallevals
```

## Quick Start

### Evaluate Retrieval Quality (Python)

Connect to your favourite Vector DB (Milvus, Elastic, PGVector, Chroma, Pinecone, FAISS, Weawiate), attach your favourite embeddings, generate questions, and visualise results!

Under the hood, <img src="logo/smallevals_emoji.png" alt="logo" width="32" height="32"> generates question per chunk, and tries to retrieve it as a single-first relevant docs, calculate scores.

```python
from smallevals import evaluate_retrievals, SmallEvalsVDBConnection

vdb = SmallEvalsVDBConnection(
    connection=chroma_client,
    collection="my_collection",
    embedding=embedding
)

# Run evaluation
result = evaluate_retrievals(connection=vdb, top_k=10, n_chunks=200) # Generate question for 200 chunks, and test to retrieve them!
```
And evaluate results!

### Generate QA from Documents (CLI)

```bash
smallevals --docs-dir ./documents --num-questions 100
```


### **QAG-0.6B**

The model was trained on TriviaQA, SQuAD 2.0, Hand-curated synthetic data generated using Qwen-70B , generating a question from the chunk/doc. 


```
Given the passage below, extract ONE question/answer pair grounded strictly in a single atomic fact.

PASSAGE:
"Eiffel tower is built at 1989"

Return ONLY a JSON object.
```

```
{
  "question": "When was the Eiffel Tower completed?",
  "answer": "1889"
}
```

Known issues: 
- Model is trained on text/wiki data, bias towards well structured text.
- Dataset contains question that ask generic questions, dataset will be more carefully crafted in v3. 

### Other Models:

Other models to be trained to eliminate the need of external LLMs. 

**CRC-0.6B** : Context relevance classifier (question ↔ retrieved chunk)
**GJ-0.6B** : Groundedness / faithfulness judge (answer ↔ context)  
**ASM-0.6B** | Answer correctness / semantic similarity 
