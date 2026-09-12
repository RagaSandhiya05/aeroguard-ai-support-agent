import json
import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DeltaHistoricalRetriever:
    """
    RAG Engine: Indexes Delta historical customer support resolution pairs.
    Retrieves the most semantically relevant historical resolutions for any incoming customer issue,
    grounding draft replies in official airline historical precedent.
    """
    def __init__(self, data_path: Optional[str] = None):
        if not data_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "delta_resolutions_sample.json")
            
        self.data_path = data_path
        self.corpus: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            max_features=8000
        )
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Corpus file not found: {self.data_path}")
            
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)
            
        # Index customer query texts
        docs = [item["customer_text"] for item in self.corpus]
        self.tfidf_matrix = self.vectorizer.fit_transform(docs)
        print(f"[DeltaHistoricalRetriever] Indexed {len(self.corpus)} historical Delta resolution pairs.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most similar historical customer queries and Delta's agent responses.
        """
        if not query or not query.strip():
            return []
            
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Top-k indices sorted descending
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            item = self.corpus[idx]
            results.append({
                "pair_id": item.get("pair_id", f"DL_{idx}"),
                "historical_customer_query": item["customer_text"],
                "historical_delta_reply": item["delta_reply"],
                "similarity_score": round(score, 4),
                "created_at": item.get("created_at", "")
            })
            
        return results

    def get_1nn_verbatim(self, query: str) -> str:
        """
        Baseline 1 Retriever: Returns the single top-1 nearest neighbor past agent reply verbatim.
        """
        matches = self.retrieve(query, top_k=1)
        if matches and matches[0]["similarity_score"] > 0.05:
            return matches[0]["historical_delta_reply"]
        return "@Delta Customer: Please DM us your confirmation code so our team can assist. *AA"
