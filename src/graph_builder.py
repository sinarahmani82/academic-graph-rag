import networkx as nx
from typing import List, Dict, Tuple

class ScientificKnowledgeGraph:
    """ساخت و پیمایش گراف دانش مقالات علمی و مفاهیم هوش مصنوعی"""
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_relation(self, source: str, relation: str, target: str, metadata: Dict = None):
        """افزودن یک رابطه جهت‌دار بین دو مفهوم علمی"""
        self.graph.add_edge(source, target, relation=relation, **(metadata or {}))

    def get_multi_hop_neighbors(self, node: str, depth: int = 2) -> List[Tuple[str, str, str]]:
        """پیمایش چندمرحله‌ای در گراف برای کشف روابط پنهان یک مفهوم"""
        if node not in self.graph:
            return []

        discovered_triplets = []
        visited = set([node])
        current_level = [node]

        for _ in range(depth):
            next_level = []
            for current_node in current_level:
                for neighbor in self.graph.neighbors(current_node):
                    edge_data = self.graph.get_edge_data(current_node, neighbor)
                    relation = edge_data.get('relation', 'related_to')
                    discovered_triplets.append((current_node, relation, neighbor))
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
            current_level = next_level

        return discovered_triplets

    def get_graph_summary(self) -> Dict[str, int]:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges()
        }
