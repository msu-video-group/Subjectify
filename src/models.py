from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


def _answer_to_wins(left: str, right: str, answer: object) -> tuple[float, float]:
    if answer is None:
        return 0.5, 0.5

    if isinstance(answer, float) and math.isnan(answer):
        return 0.5, 0.5

    answer_text = str(answer).strip()
    if not answer_text:
        return 0.5, 0.5

    low = answer_text.lower()
    if low in {"left", "l"}:
        return 1.0, 0.0
    if low in {"right", "r"}:
        return 0.0, 1.0
    if answer_text == left:
        return 1.0, 0.0
    if answer_text == right:
        return 0.0, 1.0
    return 0.5, 0.5


def _wins_matrix(votes: pd.DataFrame, methods: list[str]) -> np.ndarray:
    index = {method: i for i, method in enumerate(methods)}
    wins = np.zeros((len(methods), len(methods)), dtype=float)

    for row in votes.itertuples(index=False):
        left = str(row.left_method)
        right = str(row.right_method)
        if left not in index or right not in index or left == right:
            continue

        left_score, right_score = _answer_to_wins(left, right, getattr(row, "answer", None))
        wins[index[left], index[right]] += left_score
        wins[index[right], index[left]] += right_score

    return wins


def _scores_to_frame(methods: list[str], scores: np.ndarray | list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "preset": methods,
            "subjective_score": np.asarray(scores, dtype=float),
        }
    )


def bradley_terry_scores(votes: pd.DataFrame, expected_methods: list[str], regularization: float = 0.1) -> pd.DataFrame:
    methods = list(expected_methods)
    if len(methods) < 2:
        return pd.DataFrame({"preset": methods, "subjective_score": [0.0] * len(methods)})

    wins = _wins_matrix(votes, methods)
    wins = wins + (np.ones(wins.shape) - np.eye(wins.shape[0])) * regularization

    def log_likelihood(hidden: np.ndarray) -> float:
        ranks = np.concatenate(([0.0], hidden))
        probabilities = special.expit(ranks[:, None] - ranks[None, :])
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        return float(np.sum(wins * np.log(probabilities)))

    result = optimize.minimize(
        lambda hidden: -log_likelihood(hidden),
        np.zeros(len(methods) - 1, dtype=float),
        method="SLSQP",
    )
    if not result.success:
        raise RuntimeError(f"Bradley-Terry optimization failed: {result.message}")

    ranks = np.concatenate(([0.0], result.x))
    ranks = ranks - np.min(ranks)
    return _scores_to_frame(methods, ranks)


def thurstone_scores(votes: pd.DataFrame, expected_methods: list[str], clamp_strategy: str = "laplace") -> pd.DataFrame:
    methods = list(expected_methods)
    if len(methods) < 2:
        return pd.DataFrame({"preset": methods, "subjective_score": [0.0] * len(methods)})

    wins = _wins_matrix(votes, methods)
    totals = wins + wins.T
    rows: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            total = totals[i, j]
            if total <= 0:
                continue

            if clamp_strategy == "laplace":
                probability = (wins[i, j] + 0.5) / (total + 1.0)
            else:
                eps = 1.0 / (2.0 * total)
                probability = min(max(wins[i, j] / total, eps), 1.0 - eps)

            row = np.zeros(len(methods), dtype=float)
            row[i] = 1.0
            row[j] = -1.0
            rows.append(row)
            targets.append(float(stats.norm.ppf(probability)))
            weights.append(float(total))

    if rows:
        design = np.vstack(rows)
        target = np.asarray(targets, dtype=float)
        weight_matrix = np.diag(np.asarray(weights, dtype=float))
        reduced_design = design[:, :-1]
        lhs = reduced_design.T @ weight_matrix @ reduced_design
        rhs = reduced_design.T @ weight_matrix @ target
        reduced_scores, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

        scores = np.zeros(len(methods), dtype=float)
        scores[:-1] = reduced_scores
    else:
        scores = np.zeros(len(methods), dtype=float)

    scores = scores - scores.mean()
    return _scores_to_frame(methods, scores)


def elo_scores(
    votes: pd.DataFrame,
    expected_methods: list[str],
    k_factor: float = 32.0,
    initial_rating: float = 1500.0,
    scale: float = 400.0,
) -> pd.DataFrame:
    methods = list(expected_methods)
    ratings = {method: float(initial_rating) for method in methods}

    for row in votes.itertuples(index=False):
        left = str(row.left_method)
        right = str(row.right_method)
        if left not in ratings or right not in ratings or left == right:
            continue

        left_score, right_score = _answer_to_wins(left, right, getattr(row, "answer", None))
        left_rating = ratings[left]
        right_rating = ratings[right]
        left_expected = 1.0 / (1.0 + 10.0 ** ((right_rating - left_rating) / scale))
        right_expected = 1.0 - left_expected
        ratings[left] = left_rating + k_factor * (left_score - left_expected)
        ratings[right] = right_rating + k_factor * (right_score - right_expected)

    return _scores_to_frame(methods, [ratings[method] for method in methods])


def copeland_scores(
    votes: pd.DataFrame,
    expected_methods: list[str],
    alpha: float = 0.5,
    normalize: bool = False,
    skip_uncompared: bool = True,
    tie_eps: float = 1e-9,
) -> pd.DataFrame:
    methods = list(expected_methods)
    if len(methods) < 2:
        return pd.DataFrame({"preset": methods, "subjective_score": [0.0] * len(methods)})

    wins = _wins_matrix(votes, methods)
    totals = wins + wins.T
    scores: list[float] = []

    for i in range(len(methods)):
        wins_count = 0
        ties_count = 0
        comparisons = 0

        for j in range(len(methods)):
            if i == j:
                continue

            total = totals[i, j]
            if total <= 0:
                if not skip_uncompared:
                    ties_count += 1
                    comparisons += 1
                continue

            comparisons += 1
            diff = wins[i, j] - wins[j, i]
            if diff > tie_eps:
                wins_count += 1
            elif abs(diff) <= tie_eps:
                ties_count += 1

        score = wins_count + alpha * ties_count
        if normalize and comparisons > 0:
            score /= comparisons
        scores.append(float(score))

    return _scores_to_frame(methods, scores)


def trueskill_scores(
    votes: pd.DataFrame,
    expected_methods: list[str],
    mu: float = 25.0,
    sigma: float | None = None,
    beta: float | None = None,
    tau: float | None = None,
    draw_probability: float = 0.0,
    conservative: bool = True,
) -> pd.DataFrame:
    try:
        import trueskill
    except ImportError as exc:
        raise ImportError("The 'trueskill' package is required for --model trueskill.") from exc

    methods = list(expected_methods)
    env = trueskill.TrueSkill(
        mu=mu,
        sigma=mu / 3.0 if sigma is None else sigma,
        beta=mu / 6.0 if beta is None else beta,
        tau=mu / 300.0 if tau is None else tau,
        draw_probability=draw_probability,
    )
    ratings = {method: env.create_rating() for method in methods}

    for row in votes.itertuples(index=False):
        left = str(row.left_method)
        right = str(row.right_method)
        if left not in ratings or right not in ratings or left == right:
            continue

        left_score, right_score = _answer_to_wins(left, right, getattr(row, "answer", None))
        left_rating = ratings[left]
        right_rating = ratings[right]
        if np.isclose(left_score, right_score):
            [new_left], [new_right] = env.rate([[left_rating], [right_rating]], ranks=[0, 0])
        elif left_score > right_score:
            [new_left], [new_right] = env.rate([[left_rating], [right_rating]], ranks=[0, 1])
        else:
            [new_right], [new_left] = env.rate([[right_rating], [left_rating]], ranks=[0, 1])
        ratings[left] = new_left
        ratings[right] = new_right

    scores = [
        rating.mu - 3.0 * rating.sigma if conservative else rating.mu
        for rating in (ratings[method] for method in methods)
    ]
    return _scores_to_frame(methods, scores)


MODEL_FUNCTIONS: dict[str, Callable[[pd.DataFrame, list[str]], pd.DataFrame]] = {
    "bradley-terry": bradley_terry_scores,
    "thurstone": thurstone_scores,
    "elo": elo_scores,
    "copeland": copeland_scores,
    "trueskill": trueskill_scores,
}


def scores_by_sequence(
    votes: pd.DataFrame,
    sequences: tuple[str, ...],
    presets: dict[str, list[str]],
    model_name: str = "bradley-terry",
) -> dict[str, pd.DataFrame]:
    if model_name not in MODEL_FUNCTIONS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {sorted(MODEL_FUNCTIONS)}")

    score_function = MODEL_FUNCTIONS[model_name]
    result: dict[str, pd.DataFrame] = {}
    for sequence in sequences:
        key = f"{sequence}@display={votes['strategy'].iloc[0]}"
        seq_votes = votes.loc[votes["test_case"].astype(str).str.startswith(f"{sequence}@")].copy()
        if seq_votes.empty:
            result[key] = pd.DataFrame(columns=["preset", "subjective_score"])
            continue
        methods = list(set(seq_votes["left_method"].astype(str)) | set(seq_votes["right_method"].astype(str)))
        result[key] = score_function(seq_votes, methods)
    return result
