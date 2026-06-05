from typing import Dict, List, Tuple

import networkx as nx

from core.program import (
    AndNode,
    ApplyTransformNode,
    ClassifierNode,
    IfThenElseNode,
    Node,
    NotNode,
    OrNode,
    PredicateNode,
    Program,
    ThresholdNode,
)

from adapters.base_victim import BaseVictim


class StructuralRecoveryEvaluator:
    """Evaluates structural similarity between two programs using graph metrics."""

    def compute_dependency_graph(self, program: Program) -> nx.DiGraph:
        """Build a directed graph from the program AST.
        
        Nodes represent primitives/operators, edges represent data flow.
        """
        graph = nx.DiGraph()
        counter = [0]

        def _add_node(label: str) -> int:
            idx = counter[0]
            counter[0] += 1
            graph.add_node(idx, label=label)
            return idx

        def _walk(node: Node, parent_idx: int = -1) -> int:
            if isinstance(node, PredicateNode):
                label = f"pred:{node.primitive.name}:{node.primitive.parameters}"
                idx = _add_node(label)
                if parent_idx >= 0:
                    graph.add_edge(parent_idx, idx)
                return idx
            if isinstance(node, ClassifierNode):
                label = f"class:{node.primitive.name}"
                idx = _add_node(label)
                if parent_idx >= 0:
                    graph.add_edge(parent_idx, idx)
                return idx
            if isinstance(node, ThresholdNode):
                label = f"threshold:{node.classifier.name}:{node.threshold}"
                idx = _add_node(label)
                if parent_idx >= 0:
                    graph.add_edge(parent_idx, idx)
                return idx
            if isinstance(node, ApplyTransformNode):
                label = f"transform:{node.transform.name}"
                idx = _add_node(label)
                if parent_idx >= 0:
                    graph.add_edge(parent_idx, idx)
                _walk(node.inner, idx)
                return idx
            if isinstance(node, AndNode):
                idx = _add_node("AND")
                if parent_idx >= 0:
                    graph.add_edge(parent_idx, idx)
                _walk(node.left, idx)
                _walk(node.right, idx)
                return idx
            if isinstance(node, OrNode):
                idx = _add_node("OR")
                if parent_idx >= 0:
                    graph.add_edge(parent_idx, idx)
                _walk(node.left, idx)
                _walk(node.right, idx)
                return idx
            if isinstance(node, NotNode):
                idx = _add_node("NOT")
                if parent_idx >= 0:
                    graph.add_edge(parent_idx, idx)
                _walk(node.child, idx)
                return idx
            if isinstance(node, IfThenElseNode):
                idx = _add_node("IF-THEN-ELSE")
                if parent_idx >= 0:
                    graph.add_edge(parent_idx, idx)
                _walk(node.condition, idx)
                return idx
            return -1

        _walk(program.root)
        return graph

    def graph_edit_distance(self, prog1: Program, prog2: Program) -> float:
        """Compute approximate graph edit distance between two programs."""
        g1 = self.compute_dependency_graph(prog1)
        g2 = self.compute_dependency_graph(prog2)
        try:
            distance = nx.graph_edit_distance(
                g1, g2, node_match=lambda a, b: a.get("label") == b.get("label")
            )
            return float(distance) if distance is not None else float("inf")
        except Exception:
            return float("inf")

    def edge_precision_recall(
        self, prog1: Program, prog2: Program
    ) -> Dict[str, float]:
        """Compute precision and recall of edges between two program graphs."""
        g1 = self.compute_dependency_graph(prog1)
        g2 = self.compute_dependency_graph(prog2)

        edges1 = set(g1.edges())
        edges2 = set(g2.edges())

        if not edges2:
            return {
                "edge_precision": 1.0 if not edges1 else 0.0,
                "edge_recall": 1.0 if not edges1 else 0.0,
            }
        true_positives = len(edges1 & edges2)
        precision = true_positives / len(edges2) if edges2 else 0.0
        recall = true_positives / len(edges1) if edges1 else 0.0
        return {"edge_precision": precision, "edge_recall": recall}

    def recovery_score(
        self, victim: BaseVictim, discovered_program: Program
    ) -> Dict[str, float]:
        """Compute combined structural recovery score."""
        gt = victim.get_ground_truth_program()
        if gt is None:
            return {"recovery_score": 0.0}

        edge_metrics = self.edge_precision_recall(gt, discovered_program)
        try:
            ged = self.graph_edit_distance(gt, discovered_program)
        except Exception:
            ged = float("inf")
        ged_normalized = 1.0 / (1.0 + ged) if ged != float("inf") else 0.0
        combined = (
            edge_metrics["edge_precision"]
            + edge_metrics["edge_recall"]
            + ged_normalized
        ) / 3.0

        return {
            "recovery_score": combined,
            "graph_edit_distance": ged,
            **edge_metrics,
        }
