
from autogluon.core.metrics import make_scorer
import numpy as np

def map3(y_true, y_pred_probs):
    y_true = [[x] for x in y_true]
    y_pred_probs = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1].tolist()

    def ap3(y_true, y_pred_probs):
        y_pred_probs = y_pred_probs[:3]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(y_pred_probs):
            if p in y_true and p not in y_pred_probs[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        if not y_true:
            return 0.0

        return score

    return np.mean([ap3(a,p) for a,p in zip(y_true, y_pred_probs)])

ag_map3 = make_scorer(
    name='map3',
    score_func=map3,
    optimum=1,
    needs_proba=True,
    greater_is_better=True
)
