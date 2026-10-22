"""Pure controller utilities shared by synthetic and neural experiments."""
from dataclasses import dataclass
import numpy as np


def score(gradient, velocity, variance, batch, beta):
    mean = beta * velocity + (1 - beta) * gradient
    alignment = float(gradient @ mean)
    second_moment = float(mean @ mean) + (1-beta)**2 * variance / batch
    return max(alignment, 0.)**2 / second_moment if second_moment > 0 else 0.


def select(gradient, velocity, variance, batches, betas, price):
    if not batches or price < 0:
        raise ValueError('Need feasible batches and nonnegative price')
    # Sorted enumeration provides deterministic small-batch, low-beta tie breaks.
    candidates = [(score(gradient, velocity, variance, b, beta)-price*b,
                   b, beta) for b in sorted(batches) for beta in sorted(betas)]
    _, batch, beta = max(candidates, key=lambda x: x[0])
    return batch, beta, score(gradient, velocity, variance, batch, beta)


def pilot_statistics(gradients, microbatch_size):
    gradients = np.asarray(gradients)
    if len(gradients) < 2:
        raise ValueError('Variance estimation needs at least two microbatches')
    mean = gradients.mean(axis=0)
    variance = microbatch_size * np.sum((gradients-mean)**2)/(len(gradients)-1)
    return mean, float(variance)


@dataclass
class SearchResult:
    eta: float
    evaluations: int
    status: str


def armijo(theta, direction, gradient, objective, initial=1., max_eta=64.,
           max_trials=40):
    """Dyadic bidirectional Armijo. A cap is logged, never hidden as exact search."""
    alignment = float(gradient @ direction)
    if alignment <= 0 or not np.isfinite(alignment):
        return SearchResult(0., 0, 'non_descent')
    old = float(objective(theta))
    calls = 1
    def passes(eta):
        nonlocal calls
        calls += 1
        value = float(objective(theta-eta*direction))
        return np.isfinite(value) and value <= old-.5*eta*alignment
    eta = min(max(float(initial), 2.**-30), max_eta)
    if passes(eta):
        while calls < max_trials:
            if eta*2 > max_eta:
                return SearchResult(eta, calls, 'capped')
            if not passes(eta*2):
                return SearchResult(eta, calls, 'bracketed')
            eta *= 2
    else:
        while calls < max_trials:
            eta *= .5
            if passes(eta):
                return SearchResult(eta, calls, 'bracketed')
    raise RuntimeError('Armijo search exhausted trials; inspect scaling')
