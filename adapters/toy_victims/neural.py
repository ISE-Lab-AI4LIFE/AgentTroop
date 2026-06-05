from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from core.program import Program
from core.types import Outcome

from adapters.base_victim import BaseVictim


class SKLearnVictim(BaseVictim):
    """A victim using a logistic regression model trained on synthetic data.
    
    This victim does NOT have an explicit ground truth program.
    Its decision boundary is learned from data, so get_ground_truth_program()
    returns None. Evaluation against this victim is done behaviorally
    (via respond()) rather than structurally.
    """

    _DANGER_WORDS = [
        "bomb", "kill", "attack", "weapon", "explosive",
        "hack", "malware", "virus", "drug", "poison",
    ]

    def __init__(self, random_state: int = 42, training_size: int = 2000) -> None:
        super().__init__()
        self._program: Optional[Program] = None
        self._model = LogisticRegression(random_state=random_state)
        self._vectorizer = CountVectorizer(
            analyzer="char", ngram_range=(1, 3), max_features=1000
        )
        self._fitted = False
        self._random_state = random_state
        self._training_size = training_size

    def _generate_training_data(self) -> tuple[list[str], np.ndarray]:
        rng = np.random.default_rng(self._random_state)
        texts: list[str] = []
        labels: list[int] = []
        word_list = self._DANGER_WORDS
        for _ in range(self._training_size):
            length = rng.integers(10, 100)
            words: list[str] = []
            for _ in range(length // 5 + 1):
                if rng.random() < 0.15:
                    words.append(rng.choice(word_list))
                else:
                    word_len = rng.integers(3, 10)
                    word = "".join(
                        rng.choice(list("abcdefghijklmnopqrstuvwxyz"))
                        for _ in range(word_len)
                    )
                    words.append(word)
            text = " ".join(words)
            texts.append(text)
            has_danger = any(dw in text for dw in word_list)
            noise = rng.random() < 0.05
            label = 1 if (has_danger and not noise) else 0
            labels.append(label)
        return texts, np.array(labels, dtype=np.int64)

    def _train(self) -> None:
        X_text, y = self._generate_training_data()
        X = self._vectorizer.fit_transform(X_text)
        self._model.fit(X, y)
        self._fitted = True

    def respond(self, prompt: str) -> Outcome:
        if not self._fitted:
            self._train()
        X = self._vectorizer.transform([prompt])
        return int(self._model.predict(X)[0])

    def get_ground_truth_program(self) -> Optional[Program]:
        return None

    def get_metadata(self) -> dict:
        return {
            **super().get_metadata(),
            "type": "neural",
            "model": "LogisticRegression",
            "has_ground_truth": False,
            "training_size": self._training_size,
        }
