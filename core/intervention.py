import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .primitive import PrimitiveRegistry, Transform
from .types import InterventionID, Prompt


@dataclass
class Intervention:
    base_prompt: Prompt
    transforms: List[Transform]
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: InterventionID = field(default_factory=lambda: str(uuid.uuid4()))
    _final_prompt: str = field(init=False, default="")

    @property
    def final_prompt(self) -> str:
        if not self._final_prompt:
            self._final_prompt = self.apply()
        return self._final_prompt

    @final_prompt.setter
    def final_prompt(self, value: str) -> None:
        self._final_prompt = value

    def apply(self) -> str:
        prompt = self.base_prompt
        for transform in self.transforms:
            prompt = transform.evaluate(prompt)
        self._final_prompt = prompt
        return prompt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "base_prompt": self.base_prompt,
            "transforms": [
                {"name": transform.name, "parameters": transform.parameters}
                for transform in self.transforms
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Intervention":
        registry = PrimitiveRegistry()
        transforms = []
        for transform_data in data.get("transforms", []):
            transforms.append(
                registry.get(transform_data["name"], transform_data.get("parameters", {}))
            )
        return Intervention(
            base_prompt=data["base_prompt"],
            transforms=transforms,
            metadata=data.get("metadata", {}),
            id=data.get("id", str(uuid.uuid4())),
        )
