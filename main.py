from tabulate import tabulate
from src.graph_builder import ScientificKnowledgeGraph
from src.vector_retriever import SemanticVectorRetriever
from src.hybrid_rag import HybridGraphRAG

def main():
    print("=== Initializing Academic GraphRAG Framework ===\n")

    # ۱. ساخت دیتاست نمونه از مقالات علمی
    papers = [
        {"id": "P1", "text": "Vision Transformers (ViT) rely heavily on Multi-Head Self-Attention layers which require extensive memory."},
        {"id": "P2", "text": "Unstructured magnitude pruning removes near-zero weights, creating sparse neural networks."},
        {"id": "P3", "text": "Sparse networks significantly reduce energy footprint and memory access latency on Edge Devices."}
    ]

    # ۲. ساخت گراف دانش روابط علمی
    kg = ScientificKnowledgeGraph()
    kg.add_relation("Vision Transformer", "contains", "Multi-Head Attention")
    kg.add_relation("Multi-Head Attention", "optimized_by", "Magnitude Pruning")
    kg.add_relation("Magnitude Pruning", "induces", "Weight Sparsity")
    kg.add_relation("Weight Sparsity", "enables_deployment_on", "Edge Devices")
    kg.add_relation("Edge Devices", "demands", "Low Energy Consumption")

    # ۳. ایندکس متون در پایگاه داده برداری
    retriever = SemanticVectorRetriever()
    retriever.index_documents(papers)

    # ۴. ساخت پایپ‌لاین ترکیبی GraphRAG
    rag = HybridGraphRAG(knowledge_graph=kg, vector_retriever=retriever)

    # پرسش پژوهشی نیازمند به استنتاج چندمرحله‌ای:
    question = "How does pruning attention mechanisms facilitate edge deployment?"
    print(f"User Query: '{question}'\n")

    result = rag.query(question=question, target_entity="Vision Transformer", top_k=2, graph_depth=3)

    print("--- 1. Dense Semantic Passages (Vector Retrieval) ---")
    for i, p in enumerate(result["semantic_passages"], 1):
        print(f"[{i}] {p}")

    print("\n--- 2. Multi-Hop Graph Traversal Facts (Relational Context) ---")
    table_data = [[i+1, fact] for i, fact in enumerate(result["relational_facts"])]
    print(tabulate(table_data, headers=["#", "Discovered Knowledge Path"], tablefmt="grid"))

    print(f"\n[Summary]: Discovered {result['multi_hop_path_count']} relational hops connecting architectural design to hardware efficiency!")

if __name__ == "__main__":
    main()
