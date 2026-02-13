#!/usr/bin/env python3
"""
Pre-compute embeddings for React facts dataset.
Saves to embeddings.json for faster experiment runs.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from vdd.ollama_embeddings import OllamaEmbeddings

def main():
    data_dir = Path(__file__).parent

    with open(data_dir / "react_facts.json") as f:
        data = json.load(f)

    embedder = OllamaEmbeddings(model="nomic-embed-text")
    print(f"Using model: {embedder.model}")
    print(f"Embedding dimension: {embedder.dimension}")

    embeddings = {
        "model": embedder.model,
        "dimension": embedder.dimension,
        "documents": {},
        "queries": {}
    }

    n_facts = len(data["facts"])
    for i, fact in enumerate(data["facts"]):
        fact_id = fact["id"]
        print(f"[{i+1}/{n_facts}] Processing: {fact_id}")

        # Embed query
        query = fact["query"]
        embeddings["queries"][fact_id] = embedder.embed(query).tolist()

        # Embed each version's document
        for version, content in fact["versions"].items():
            doc_key = f"{fact_id}_{version}"
            doc_text = content["document"]
            embeddings["documents"][doc_key] = {
                "text": doc_text,
                "answer": content["answer"],
                "version": version,
                "fact_id": fact_id,
                "embedding": embedder.embed(doc_text).tolist()
            }

    output_file = data_dir / "embeddings.json"
    with open(output_file, "w") as f:
        json.dump(embeddings, f, indent=2)

    print(f"\nSaved {len(embeddings['documents'])} document embeddings")
    print(f"Saved {len(embeddings['queries'])} query embeddings")
    print(f"Output: {output_file}")

if __name__ == "__main__":
    main()
