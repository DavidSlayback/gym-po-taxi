__all__ = ['generate_action_probability_matrix', 'vectorized_multinomial_with_rand', 'vectorized_multinomial_with_rng']

import numpy as np


def generate_action_probability_matrix(action_failure_probability: float = (1./3), action_n: int = 4) -> np.ndarray:
    """Generate a matrix of probabilities of categorical actions given some chance of failure

    With (1-action_failure_probability) chance, take chosen action. Distribute remaining probability among
    all other actions.

    Args:
        action_failure_probability: Probability that selected action fails
        action_n: Number of actions total
    Returns:
        (action_n,action_n) probabilities to draw from.
    """
    probs = np.full((action_n, action_n), action_failure_probability / (action_n - 1), dtype=np.float64)
    np.fill_diagonal(probs, 1 - action_failure_probability)
    return probs


def vectorized_multinomial_with_rand(selected_prob_matrix: np.ndarray, random_numbers: np.ndarray) -> np.ndarray:
    """Vectorized sample from [B,N] probabilitity matrix
    Lightly edited from https://stackoverflow.com/a/34190035/2504700
    Args:
        selected_prob_matrix: (Batch, p) size probability matrix (i.e. T[s,a] or O[s,a,s']
        random_numbers: (Batch,) size random numbers from np.random.rand()
    Returns:
        (Batch,) size sampled integers
    """
    s = selected_prob_matrix.cumsum(axis=1)  # Sum over p dim for accumulated probability
    return (s < np.expand_dims(random_numbers, axis=-1)).sum(axis=1)  # Returns first index where random number < accumulated probability


def vectorized_multinomial_with_rng(selected_prob_matrix: np.ndarray, rng: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    """Vectorized sample from [B,N] probabilitity matrix
    Lightly edited from https://stackoverflow.com/a/34190035/2504700
    Args:
        selected_prob_matrix: (Batch, p) size probability matrix (i.e. T[s,a] or O[s,a,s']
        rng: Random number generator from which to sample
    Returns:
        (Batch,) size sampled integers
    """
    random_numbers = rng.random(selected_prob_matrix.shape[0])
    s = selected_prob_matrix.cumsum(axis=1)  # Sum over p dim for accumulated probability
    return (s < np.expand_dims(random_numbers, axis=-1)).sum(axis=1)  # Returns first index where random number < accumulated probability