"""Bandit environment simulation and SARSA parameter estimation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import math
import random

import matplotlib.pyplot as plt

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


@dataclass
class SimulationResult:
    actions: List[int]
    rewards: List[int]
    q_history: List[List[float]]


def simulate_agent(
    environment: SwitchingTwoArmBandit, agent: SarsaSoftmaxAgent
) -> SimulationResult:
    actions: List[int] = [0] * environment.n_trials
    rewards: List[int] = [0] * environment.n_trials
    q_history: List[List[float]] = [[0.0] * agent.n_actions for _ in range(environment.n_trials)]

    for t in range(environment.n_trials):
        action, _ = agent.select_action()
        reward = environment.sample_reward(action, t)
        agent.update(action, reward)

        actions[t] = action
        rewards[t] = reward
        q_history[t] = agent.q_values.copy()

    return SimulationResult(actions=actions, rewards=rewards, q_history=q_history)


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
) -> List[Tuple[float, float, float, float, float]]:
    n_params = 2
    n_samples = len(actions)
    evaluations: List[Tuple[float, float, float, float, float]] = []

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

    true_result = simulate_agent(environment, agent)
    optimal_arms = [environment.optimal_arm(t) for t in range(environment.n_trials)]

    alphas = [round(x, 2) for x in [0.1 * i for i in range(1, 10)]]
    betas = [round(1.0 + i, 2) for i in range(10)]

    evaluations = evaluate_models(true_result.actions, true_result.rewards, alphas, betas)

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

    estimated_environment = SwitchingTwoArmBandit(
        rng=random.Random(99),
        n_trials=environment.n_trials,
        prob_correct=environment.prob_correct,
        prob_incorrect=environment.prob_incorrect,
        switch_interval=environment.switch_interval,
    )
    estimated_agent = SarsaSoftmaxAgent(
        alpha=best_alpha,
        beta=best_beta,
        rng=random.Random(99),
    )
    estimated_result = simulate_agent(estimated_environment, estimated_agent)

    trials = list(range(environment.n_trials))

    fig1, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    column_titles = ["True agent", "Estimated agent"]
    results = [true_result, estimated_result]

    for col, (title, result) in enumerate(zip(column_titles, results)):
        q0 = [values[0] for values in result.q_history]
        q1 = [values[1] for values in result.q_history]

        axes[0, col].plot(trials, q0, label="Q(action 0)")
        axes[0, col].plot(trials, q1, label="Q(action 1)")
        axes[0, col].set_ylabel("Action value")
        axes[0, col].set_title(title)
        axes[0, col].legend(loc="upper right")

        axes[1, col].step(trials, result.actions, where="mid")
        axes[1, col].set_ylabel("Action")
        axes[1, col].set_yticks([0, 1])

        axes[2, col].step(trials, optimal_arms, where="mid", color="black")
        axes[2, col].set_ylabel("Optimal arm")
        axes[2, col].set_yticks([0, 1])

        axes[3, col].step(trials, result.rewards, where="mid", color="green")
        axes[3, col].set_ylabel("Reward")
        axes[3, col].set_yticks([0, 1])
        axes[3, col].set_xlabel("Trial")

    fig1.suptitle(
        "Figure 1: Time series of action values, actions, optimal arms, and rewards"
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.97])

    alpha_values = list(alphas)
    beta_values = list(betas)
    alpha_indices = {value: idx for idx, value in enumerate(alpha_values)}
    beta_indices = {value: idx for idx, value in enumerate(beta_values)}
    log_like_grid = [
        [float("nan")] * len(alpha_values) for _ in range(len(beta_values))
    ]
    for alpha, beta, log_like, _, _ in evaluations:
        row = beta_indices[beta]
        col = alpha_indices[alpha]
        log_like_grid[row][col] = log_like

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    alpha_step = alpha_values[1] - alpha_values[0] if len(alpha_values) > 1 else 1.0
    beta_step = beta_values[1] - beta_values[0] if len(beta_values) > 1 else 1.0
    extent = [
        alpha_values[0] - alpha_step / 2,
        alpha_values[-1] + alpha_step / 2,
        beta_values[0] - beta_step / 2,
        beta_values[-1] + beta_step / 2,
    ]
    c = ax2.imshow(
        log_like_grid,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    ax2.scatter([best_alpha], [best_beta], color="red", marker="x", s=100, label="Best")
    ax2.set_xlabel("alpha")
    ax2.set_ylabel("beta")
    ax2.set_xticks(alpha_values)
    ax2.set_yticks(beta_values)
    ax2.set_title("Figure 2: Log-likelihood surface")
    ax2.legend(loc="upper left")
    fig2.colorbar(c, ax=ax2, label="Log-likelihood")
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
