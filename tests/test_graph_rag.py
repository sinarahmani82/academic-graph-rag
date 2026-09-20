import pytest
from src.graph_builder import ScientificKnowledgeGraph
from src.vector_retriever import SemanticVectorRetriever
from src.hybrid_rag import HybridGraphRAG

def test_knowledge_graph_multi_hop_traversal():
    """تست اعتبارسنجی پیمایش چندمرحله‌ای روابط در گراف دانش"""
    kg = ScientificKnowledgeGraph()
    kg.add_relation("Transformer", "utilizes", "Attention")
    kg.add_relation("Attention", "scales_with", "Sequence Length")

    # پیمایش با عمق ۲ باید هر دو رابطه زنجیره‌ای را کشف کند
    triplets = kg.get_multi_hop_neighbors("Transformer", depth=2)
    assert len(triplets) == 2, "Graph traversal failed to capture multi-hop relations!"
    
    relations = [r for _, r, _ in triplets]
    assert "utilizes" in relations
    assert "scales_with" in relations

def test_semantic_vector_search():
    """تست دقت شباهت کسینوسی در پیدا کردن مرتبط‌ترین سند"""
    retriever = SemanticVectorRetriever()
    test_docs = [
        {"id": "1", "text": "Deep convolutional neural networks are ideal for image classification."},
        {"id": "2", "text": "Reinforcement learning optimizes agent actions in an environment."}
    ]
    retriever.index_documents(test_docs)

    results = retriever.search("computer vision and convolutional filters", top_k=1)
    assert len(results) == 1
    # باید سند مربوط به شبکه کانولوشنال در صدر قرار گیرد
    assert results[0]["doc_id"] == "1"
    assert results[0]["score"] > 0.3

def test_hybrid_rag_integration():
    """تست ادغام خروجی متنی با فکت‌های گرافی در خروجی نهایی"""
    kg = ScientificKnowledgeGraph()
    kg.add_relation("ViT", "pruned_by", "Magnitude Pruning")

    retriever = SemanticVectorRetriever()
    retriever.index_documents([{"id": "1", "text": "Vision Transformers benefit from model pruning."}])

    rag = HybridGraphRAG(knowledge_graph=kg, vector_retriever=retriever)
    output = rag.query(question="How to optimize ViT?", target_entity="ViT", top_k=1, graph_depth=1)

    assert "semantic_passages" in output
    assert "relational_facts" in output
    assert len(output["semantic_passages"]) == 1
    assert len(output["relational_facts"]) == 1
