import math
import random


class SimpleTCN:
    """Very small classifier implemented with NumPy."""

    def __init__(self, input_size: int = 30, lr: float = 1e-3):
        self.lr = lr
        self.weights = [
            [random.gauss(0, 1) for _ in range(3)] for _ in range(input_size)
        ]
        self.bias = [0.0, 0.0, 0.0]

    def __call__(self, x):
        outputs = []
        for row in x:
            logits = []
            for j in range(3):
                s = sum(row[i] * self.weights[i][j] for i in range(len(row)))
                s += self.bias[j]
                logits.append(s)
            outputs.append(logits)
        return outputs

    def training_step(self, batch, batch_idx: int) -> float:
        x, y = batch
        logits = self(x)
        losses = []
        for logit, label in zip(logits, y):
            m = max(logit)
            exp = [math.exp(v - m) for v in logit]
            denom = sum(exp)
            prob = exp[label] / denom
            losses.append(-math.log(prob))
        loss = sum(losses) / len(losses)
        return loss

    def save(self, path: str) -> None:
        import json

        with open(path, "w") as f:
            json.dump({"weights": self.weights, "bias": self.bias}, f)

    @classmethod
    def load(cls, path: str) -> "SimpleTCN":
        import json

        with open(path) as f:
            data = json.load(f)
        model = cls(input_size=len(data["weights"]))
        model.weights = data["weights"]
        model.bias = data["bias"]
        return model
