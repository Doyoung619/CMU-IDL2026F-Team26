"""Pure controller utilities shared by synthetic and neural experiments."""
from dataclasses import dataclass
import numpy as np


def score(gradient, velocity, variance, batch, beta):
    mean = beta * velocity + (1 - beta) * gradient
    alignment = float(gradient @ mean)
    second_moment = float(mean @ mean) + (1-beta)**2 * variance / batch
    return max(alignment, 0.)**2 / second_moment if second_moment > 0 else 0.


def select(gradient, velocity, variance, batches, betas, price,
           rtol=1e-12, atol=1e-15, return_details=False):
    if not batches or price < 0:
        raise ValueError('Need feasible batches and nonnegative price')
    if rtol < 0 or atol < 0:
        raise ValueError('Tie tolerances must be nonnegative')
    # Compare every candidate to the global maximum, not pairwise, to avoid
    # nontransitive chains of near-ties. Sorted order favors smaller B then beta.
    candidates = [(score(gradient, velocity, variance, b, beta)-price*b,
                   b, beta) for b in sorted(batches) for beta in sorted(betas)]
    best = max(c[0] for c in candidates)
    tied = [c for c in candidates
            if best-c[0] <= atol+rtol*max(abs(best),abs(c[0]))]
    chosen, batch, beta = tied[0]
    result = (batch, beta, score(gradient, velocity, variance, batch, beta))
    if return_details:
        return (*result, dict(near_tie_count=len(tied), best_priced_score=best,
                             chosen_priced_score=chosen, priced_gap=best-chosen,
                             tolerance=atol+rtol*max(abs(best),abs(chosen))))
    return result


def residual_learning_rate(fixed_lr, previous_eta, policy='cap_previous'):
    if fixed_lr < 0 or previous_eta < 0:
        raise ValueError('Learning rates must be nonnegative')
    if policy == 'cap_previous':
        return min(fixed_lr, previous_eta)
    if policy == 'legacy_fixed':
        return fixed_lr
    raise ValueError(f'Unknown residual policy: {policy}')


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
