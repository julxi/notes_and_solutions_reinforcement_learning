from dataclasses import dataclass


# Welford accumulator for running mean and variance_of_mean
@dataclass
class Welford:
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def sample_variance(self) -> float:
        return self.M2 / (self.n - 1)

    @property
    def variance_of_mean(self) -> float:
        sv = self.sample_variance
        return sv / self.n
