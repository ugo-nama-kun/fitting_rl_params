"""Bandit environment simulation and SARSA parameter estimation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import math
import random


@dataclass
class SwitchingTwoArmBandit:
    prob_correct: float = 0.8
    prob_incorrect: float = 0.1
    switch_interval: int = 100
    n_trials: int = 400
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def optimal_arm(self, trial: int) -> int:
        block = trial // self.switch_interval
        return block % 2

    def sample_reward(self, action: int, trial: int) -> int:
        optimal = self.optimal_arm(trial)
        prob = self.prob_correct if action == optimal else self.prob_incorrect
        return 1 if self.rng.random() < prob else 0


@dataclass
class SarsaSoftmaxAgent:
    alpha: float
    beta: float
    gamma: float = 0.0
    n_actions: int = 2
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    q_values: List[float] = field(init=False)

    def __post_init__(self) -> None:
        self.q_values = [0.0 for _ in range(self.n_actions)]

    def softmax_probabilities(self) -> List[float]:
        preferences = [self.beta * q for q in self.q_values]
        max_pref = max(preferences)
        exp_prefs = [math.exp(pref - max_pref) for pref in preferences]
        total = sum(exp_prefs)
        return [value / total for value in exp_prefs]

    def select_action(self) -> Tuple[int, List[float]]:
        probs = self.softmax_probabilities()
        cumulative = 0.0
        r = self.rng.random()
        for index, prob in enumerate(probs):
            cumulative += prob
            if r <= cumulative:
                return index, probs
        return self.n_actions - 1, probs

    def update(self, action: int, reward: float) -> None:
        target = reward + self.gamma * self.q_values[action]
        self.q_values[action] += self.alpha * (target - self.q_values[action])


def simulate_agent(environment: SwitchingTwoArmBandit, agent: SarsaSoftmaxAgent) -> Tuple[List[int], List[int]]:
    actions: List[int] = [0] * environment.n_trials
    rewards: List[int] = [0] * environment.n_trials

    for t in range(environment.n_trials):
        action, _ = agent.select_action()
        reward = environment.sample_reward(action, t)
        agent.update(action, reward)

        actions[t] = action
        rewards[t] = reward

    return actions, rewards


def compute_log_likelihood(
    actions: Sequence[int],
    rewards: Sequence[int],
    alpha: float,
    beta: float,
    gamma: float = 0.0,
) -> float:
    q_values = [0.0, 0.0]
    log_likelihood = 0.0

    for action, reward in zip(actions, rewards):
        preferences = [beta * q for q in q_values]
        max_pref = max(preferences)
        exp_prefs = [math.exp(pref - max_pref) for pref in preferences]
        total = sum(exp_prefs)
        probs = [value / total for value in exp_prefs]

        action_prob = max(probs[action], 1e-12)
        log_likelihood += math.log(action_prob)

        target = reward + gamma * q_values[action]
        q_values[action] += alpha * (target - q_values[action])

    return log_likelihood


def evaluate_models(
    actions: Sequence[int],
    rewards: Sequence[int],
    alphas: Iterable[float],
    betas: Iterable[float],
) -> List[Tuple[float, float, float, float]]:
    n_params = 2
    n_samples = len(actions)
    evaluations: List[Tuple[float, float, float, float]] = []

    for alpha in alphas:
        for beta in betas:
            log_likelihood = compute_log_likelihood(actions, rewards, alpha, beta)
            aic = 2 * n_params - 2 * log_likelihood
            bic = n_params * math.log(n_samples) - 2 * log_likelihood
            evaluations.append((alpha, beta, log_likelihood, aic, bic))

    evaluations.sort(key=lambda item: item[2], reverse=True)
    return evaluations


def main() -> None:
    rng = random.Random(42)
    environment = SwitchingTwoArmBandit(rng=rng, n_trials=400)

    true_alpha = 0.4
    true_beta = 5.0
    agent_rng = random.Random(42)
    agent = SarsaSoftmaxAgent(alpha=true_alpha, beta=true_beta, rng=agent_rng)

    actions, rewards = simulate_agent(environment, agent)

    alphas = [round(x, 2) for x in [0.1 * i for i in range(1, 10)]]
    betas = [round(1.0 + i, 2) for i in range(10)]

    evaluations = evaluate_models(actions, rewards, alphas, betas)

    best_alpha, best_beta, best_log_like, best_aic, best_bic = evaluations[0]

    print("Best parameters based on log-likelihood:")
    print(f"  alpha={best_alpha:.2f}, beta={best_beta:.2f}")
    print(f"  log-likelihood={best_log_like:.2f}")
    print(f"  AIC={best_aic:.2f}, BIC={best_bic:.2f}")
    print()
    print("Top 5 parameter settings:")
    for alpha, beta, log_like, aic, bic in evaluations[:5]:
        print(
            f"alpha={alpha:.2f}, beta={beta:.2f}, "
            f"logL={log_like:.2f}, AIC={aic:.2f}, BIC={bic:.2f}"
        )


if __name__ == "__main__":
    main()
