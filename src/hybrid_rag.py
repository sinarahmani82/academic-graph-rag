from typing import List, Dict
from src.graph_builder import ScientificKnowledgeGraph
from src.vector_retriever import SemanticVectorRetriever

class HybridGraphRAG:
    """ادغام خروجی جستجوی برداری با گراف دانش برای پاسخ به سوالات چندمرحله‌ای"""
    def __init__(self, knowledge_graph: ScientificKnowledgeGraph, vector_retriever: SemanticVectorRetriever):
        self.kg = knowledge_graph
        self.retriever = vector_retriever

    def query(self, question: str, target_entity: str = None, top_k: int = 2, graph_depth: int = 2) -> Dict:
        # ۱. استخراج متون مرتبط با جستجوی برداری
        semantic_docs = self.retriever.search(question, top_k=top_k)

        # ۲. پیمایش چندمرحله‌ای در گراف برای کشف پیوندهای علّی و ساختاری
        graph_triplets = []
        if target_entity:
            graph_triplets = self.kg.get_multi_hop_neighbors(target_entity, depth=graph_depth)

        # ۳. تجمیع زمینه (Context Fusion)
        synthesized_context = {
            "query": question,
            "semantic_passages": [doc["text"] for doc in semantic_docs],
            "relational_facts": [f"{h} --({r})--> {t}" for h, r, t in graph_triplets],
            "multi_hop_path_count": len(graph_triplets)
        }
        return synthesized_context
