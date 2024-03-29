__all__ = [
    "create_action_probability_matrix",
    "add_gaussian_noise",
    "randomize_action_sign",
    "ACTION_NAMES_CARDINAL",
    "ACTION_NAMES_ORDINAL",
    "ACTIONS_ORDINAL",
    "ACTIONS_CARDINAL",
    "ACTIONS_CARDINAL_Z",
    "ACTIONS_ORDINAL_Z",
    "vectorized_multinomial_with_rng",
]
import numpy as np

# N, NE, E, SE, S, SW, W, NW
ACTIONS_ORDINAL = np.array(
    [
        [-1, 0],
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, -1],
    ]
)

ACTIONS_CARDINAL = ACTIONS_ORDINAL[::2]
ACTIONS_ORDINAL_Z = np.concatenate(
    (np.zeros((8, 1), dtype=int), ACTIONS_ORDINAL), -1
)  # Add z-dimension
ACTIONS_CARDINAL_Z = ACTIONS_ORDINAL_Z[::2]
ACTION_NAMES_ORDINAL = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ACTION_NAMES_CARDINAL = ACTION_NAMES_ORDINAL[::2]


def create_action_probability_matrix(
    action_n: int = 8, action_failure_probability: float = 0.2
):
    """Create action probability matrix for sampling"""
    probs = np.full(
        (action_n, action_n),
        action_failure_probability / (action_n - 1),
        dtype=np.float64,
    )
    np.fill_diagonal(probs, 1 - action_failure_probability)
    return probs


def add_gaussian_noise(
    actionsNDArray,
    action_std: float = 0.2,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """Add gaussian noise to continuous action. Sample per environment"""
    return actions + rng.normal(0.0, action_std, actions.shape)


def randomize_action_sign(
    actionsNDArray,
    action_failure_probability: float = 0.2,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """Akin to randomly failing discrete actions. Take input actions, and if failure, flip signs"""
    sign_flips = rng.random(actions.shape[0]) <= action_failure_probability
    multipliers = np.ones_like(actions)
    multipliers[(rng.random(actions.shape) > 0.5) & sign_flips[:, None]] = -1
    actions *= multipliers
    return actions


def vectorized_multinomial_with_rng(
    selected_prob_matrixNDArray, rng: np.random.Generator = np.random.default_rng()
) -> np.ndarray:
    """Vectorized sample from [B,N] probabilitity matrix
    Lightly edited from https://stackoverflow.com/a/34190035/2504700
    Args:
        selected_prob_matrix: (Batch, p) size probability matrix (i.e. T[s,a] or O[s,a,s']
        rng: Random number generator from which to sample
    Returns:
        (Batch,) size sampled integers
    """
    random_numbers = rng.random(selected_prob_matrix.shape[0])
    s = selected_prob_matrix.cumsum(
        axis=1
    )  # Sum over p dim for accumulated probability
    return (s < np.expand_dims(random_numbers, axis=-1)).sum(
        axis=1
    )  # Returns first index where random number < accumulated probability
