from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticVectorRetriever:
    """جستجوی برداری معنایی متون مقالات با امبدینگ‌های چگال"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # مدل کوچک، سریع و استاندارد برای پردازش زبان طبیعی
        self.model = SentenceTransformer(model_name)
        self.documents: List[Dict[str, str]] = []
        self.embeddings: np.ndarray = np.array([])

    def index_documents(self, docs: List[Dict[str, str]]):
        """تولید و ذخیره امبدینگ برای تمام اسناد ورودی"""
        self.documents = docs
        texts = [d["text"] for d in docs]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)

    def search(self, query: str, top_k: int = 2) -> List[Dict]:
        """جستجوی کسینوسی و بازگرداندن مرتبط‌ترین متون"""
        if len(self.documents) == 0:
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            results.append({
                "doc_id": self.documents[idx]["id"],
                "text": self.documents[idx]["text"],
                "score": float(similarities[idx])
            })
        return results
