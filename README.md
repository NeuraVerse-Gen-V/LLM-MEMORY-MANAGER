# LLM-MEMORY-MANAGER
This project is to implement short,medium and long term memory context into any Ai model using Similarity detector, RAG-FAISS

# NOTE
THIS PROJECT IS UNDER DEVELOPEMENT , BUGS ARE EXPECTED!!!
---
# FILE STRUCTURE
```
manager.py              # main memory manager (handles short/medium/long)
files/                  # all memory files stored here
    ├── short/
    │   └── mem.pt # torch tensors for similarity
    │
    ├── medium/
    │   ├── summaries.json  # compressed summaries of short-term
    │   └── clusters.json   # optional: grouped convos
    │
    └── long_term/
        ├── faiss_index.bin # FAISS index
        ├── metadata.json   # id→text mapping
        └── embeddings.parquet
```
---
