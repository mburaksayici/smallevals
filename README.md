<p align="center">
  <img src="logo/smallevals_emoji_512_512.png" alt="smallevals cat logo" width="220">
</p>

# smallevals <img src="logo/smallevals_emoji_32_32.png" alt="logo" width="32" height="32"> - Small Language Models Evaluation Suite for RAG Systems

A lightweight evaluation framework powered by tiny ( really tiny <img src="logo/smallevals_emoji_32_32.png" alt="logo" width="32" height="32"> ) 0.6B models — runs 100% locally on CPU/GPU/MPS, attach any vector DB connection and run, fast and free.


Evaluation tools requiring LLM-as-a-judge or external, that costs/doesn't scale easily. <img src="logo/smallevals_emoji_32_32.png" alt="logo" width="32" height="32"> evaluates in seconds in GPU, in minutes in any CPU  <img src="logo/smallevals_emoji_32_32.png" alt="logo" width="32" height="32"> <img src="logo/smallevals_emoji_32_32.png" alt="logo" width="32" height="32">!


## Evaluate Retrieval

Evaluation of RAG system includes retrieval and RAG stage, <img src="logo/smallevals_emoji_32_32.png" alt="logo" width="32" height="32"> attacks to test retrieval and RAG answers(in the near future)!

## Models

| Model Name | Task | Status | Link |
|------------|-------|--------|------|
| **QAG-0.6B** | Generate golden Q/A from chunks or docs (synthetic evaluation data) | Available | [🤗](https://huggingface.co/mburaksayici/golden_generate_qwen_0.6b_v3) |
| **CRC-0.6B** | Context relevance classifier (question ↔ retrieved chunk) | Incoming | — |
| **GJ-0.6B** | Groundedness / faithfulness judge (answer ↔ context) | Incoming | — |
| **ASM-0.6B** | Answer correctness / semantic similarity | Incoming | — |

**Current Focus**: Retrieval evaluation (QAG-0.6B), after being sure the model generates correct answers and better questions for RAG(it does, but still room for improvement), the model will be the first model of pipeline leading to  (RAG) generation evaluation models (CRC-0.6B, GJ-0.6B, ASM-0.5B) which are the future work.



## Installation

```bash
pip install smallevals
```

## Quick Start

### Evaluate Retrieval Quality (Python)

Connect to your favourite Vector DB (Milvus, Elastic, PGVector, Chroma, Pinecone, FAISS, Weawiate), attach your favourite embeddings, generate questions, and visualise results!

Under the hood, <img src="logo/smallevals_emoji_32_32.png" alt="logo" width="32" height="32"> generates question per chunk, and tries to retrieve it as a single-first relevant docs, calculate scores.

```python
from smallevals import evaluate_retrievals, SmallEvalsVDBConnection

vdb = SmallEvalsVDBConnection(
    connection=chroma_client, # or elastic, milvus, pgvector, pinecone, faiss, weavite 
    collection="my_collection",
    embedding=embedding # hf embedding model 
)

# Run evaluation
result = evaluate_retrievals(connection=vdb, top_k=10, n_chunks=200) # Generate question for 200 chunks, and test to retrieve them!
```
And evaluate results!

```bash
smallevals dash --host 0.0.0.0 --port 8050 --debug
```

![smallevals demo](logo/demo.gif)


### Generate QA from Documents (CLI)

```bash
smallevals generate_qa --docs-dir ./documents --num-questions 100
```


### **QAG-0.6B**

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



### How does it work? 
Question Generator model, reads your chunk, assumes the chunk is the one that answers the question, and tries to match it back via Vector DB query. 

This allows directly to test your retrieval pipelines tied to your RAG systems. Whatever the complexity of your RAG system, you'll be sure if your vector queries works fine.

### Why this is a need? 
Other frameworks requiring APIs are costly, hard-to-scale, although they are better(for now). 


#### Known issues: 
- Model is trained on text/wiki data, bias towards well structured text.
- Dataset contains question that ask generic questions, dataset will be more carefully crafted in v3. 
- Some questions may be generic for the first version, leading a small decrease on the scores. 25$ led me to have this model. Let's see what I can do with more!
  
### Other Models:

Other models to be trained to eliminate the need of external LLMs. 

**CRC-0.6B** : Context relevance classifier (question ↔ retrieved chunk)
**GJ-0.6B** : Groundedness / faithfulness judge (answer ↔ context)  
**ASM-0.6B** | Answer correctness / semantic similarity 
