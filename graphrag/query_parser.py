"""Query parser for GraphRAG — convert natural-language questions to structured queries."""

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class GraphRAGQuery:
    raw_text: str
    target_entities: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)
    max_hops: int = 2


class QueryParser:
    _PATTERNS = [
        (re.compile(r"(?:why|how).*(rot13|base64|cipher|encode|decode)", re.I),
         ["rot13", "base64"], ["bypass", "decodes"]),
        (re.compile(r"(?:why|how).*(filter|keyword|block)", re.I),
         ["keyword_filter"], ["blocks", "triggers"]),
        (re.compile(r"(?:why|how).*(prefix|role.?play|roleplay)", re.I),
         ["add_prefix", "add_role_play"], ["bypass", "prefixes"]),
        (re.compile(r"(?:why|how).*(accept|refuse)", re.I),
         ["outcome"], ["causes"]),
    ]

    def parse(self, question: str) -> GraphRAGQuery:
        for pattern, entities, relations in self._PATTERNS:
            if pattern.search(question):
                return GraphRAGQuery(
                    raw_text=question,
                    target_entities=entities,
                    relation_types=relations,
                    max_hops=2,
                )
        terms = re.findall(r"'([^']*)'|\"([^\"]*)\"|([A-Z][a-z]+)", question)
        entities = [t[0] or t[1] or t[2] for t in terms if any(t)]
        return GraphRAGQuery(
            raw_text=question,
            target_entities=entities or ["transform", "predicate"],
            relation_types=[],
            max_hops=1,
        )
