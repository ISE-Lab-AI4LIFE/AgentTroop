from typing import Callable, Dict, List, Tuple

import numpy as np

from adapters.base_victim import BaseVictim


class SampleEfficiencyEvaluator:
    """Evaluates how quickly a strategy converges to the correct program.
    
    Measures learning curves (accuracy vs number of interventions)
    and computes area-under-curve as a summary metric.
    """

    def compute_learning_curve(
        self,
        victim: BaseVictim,
        strategy: Callable[[BaseVictim, int], List[Tuple[int, float]]],
        max_interventions: int,
        n_runs: int = 3,
    ) -> Dict[str, List[float]]:
        """Run the strategy multiple times and aggregate learning curves.
        
        The strategy function receives (victim, budget) and returns a list
        of (num_interventions, accuracy) pairs.
        """
        all_curves: List[List[Tuple[int, float]]] = []
        for _ in range(n_runs):
            curve = strategy(victim, max_interventions)
            all_curves.append(sorted(curve, key=lambda x: x[0]))

        if not all_curves or not all_curves[0]:
            return {"x": [], "mean": [], "std": []}

        max_len = max(len(c) for c in all_curves)
        x_values = [c[-1][0] for c in all_curves if c]
        x = list(range(0, max_interventions + 1, max(1, max_interventions // 20)))

        y_matrix: List[List[float]] = []
        for xi in x:
            values = []
            for curve in all_curves:
                interp = np.interp(xi, [p[0] for p in curve], [p[1] for p in curve])
                values.append(interp)
            y_matrix.append(values)

        means = [float(np.mean(v)) for v in y_matrix]
        stds = [float(np.std(v)) for v in y_matrix]

        return {"x": x, "mean": means, "std": stds}

    def area_under_curve(self, learning_curve: Dict[str, List[float]]) -> float:
        """Compute the normalised area under the learning curve.
        
        AUC close to 1.0 indicates fast convergence to high accuracy.
        """
        x = learning_curve.get("x", [])
        y = learning_curve.get("mean", [])
        if not x or not y:
            return 0.0
        auc = float(np.trapz(y, x))
        max_auc = x[-1] * 1.0 if x else 1.0
        return auc / max_auc if max_auc > 0 else 0.0
