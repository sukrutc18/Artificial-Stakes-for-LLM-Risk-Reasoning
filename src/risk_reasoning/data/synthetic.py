"""Synthetic EV and probability problem generator.
Produces items with derivable ground truth so decision quality can be scored
exactly. See ``configs/datasets/synthetic_ev.yaml`` for generator settings.
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import Any

from risk_reasoning.data.schemas import Answer, Choice, Question


def load(cfg: dict[str, Any]) -> Iterable[Question]:
    """Yield synthetic ``Question`` items according to ``cfg``.

    Args:
        cfg: Parsed ``configs/datasets/synthetic_ev.yaml`` contents.

    Yields:
        ``Question`` items drawn from the configured generator families.
    """
    rng = random.Random(cfg.get("seed", 42))

    families: dict[str, tuple] = {
        "ev_two_option":           (_generate_ev_two_option,          cfg.get("n_ev_two", 20)),
        "ev_three_option":         (_generate_ev_three_option,        cfg.get("n_ev_three", 20)),
        "conditional_probability": (_generate_conditional_probability, cfg.get("n_conditional", 20)),
        "base_rate":               (_generate_base_rate,              cfg.get("n_base_rate", 20)),
        "bayes_update":            (_generate_bayes_update,           cfg.get("n_bayes", 20)),
    }

    difficulties = cfg.get("difficulties", ["easy", "medium", "hard"])
    counter = 0

    for family_name, (fn, n_items) in families.items():
        for i in range(n_items):
            difficulty = difficulties[i % len(difficulties)]
            item = fn(rng, difficulty)
            item = item.model_copy(update={
                "id": f"synthetic_{family_name}_{counter:04d}",
                "dataset": "synthetic_ev",
                "metadata": {**item.metadata, "difficulty": difficulty, "family": family_name},
            })
            counter += 1
            yield item


def _generate_ev_two_option(rng: random.Random, difficulty: str) -> Question:
    """Risky vs safe option — ground truth is whichever has higher EV."""

    if difficulty == "easy":
        payoff_a = rng.randint(100, 500) * 10
        prob_a   = round(rng.uniform(0.5, 0.9), 1)
        payoff_b = rng.randint(50, 200) * 10
    elif difficulty == "medium":
        payoff_a = rng.randint(50, 300)
        prob_a   = round(rng.uniform(0.3, 0.8), 2)
        payoff_b = rng.randint(30, 200)
    else:
        payoff_a = rng.randint(80, 150)
        prob_a   = round(rng.uniform(0.4, 0.7), 2)
        payoff_b = rng.randint(60, 130)

    loss_a = round(payoff_a * rng.uniform(0.2, 0.6))
    prob_loss_a = round(1 - prob_a, 2)

    ev_a = round(prob_a * payoff_a + prob_loss_a * (-loss_a), 2)
    ev_b = float(payoff_b)
    correct_id = "A" if ev_a > ev_b else "B"

    prompt = (
        f"You are making a one-time financial decision. Choose one option:\n\n"
        f"A) {int(prob_a * 100)}% chance of gaining ${payoff_a}, "
        f"{int(prob_loss_a * 100)}% chance of losing ${int(loss_a)}\n"
        f"B) Guaranteed gain of ${payoff_b}\n\n"
        f"Which option has the higher expected value?"
    )

    return Question(
        id="",
        dataset="",
        task_type="mcq",
        prompt=prompt,
        choices=[
            Choice(id="A", text=f"{int(prob_a*100)}% chance of +${payoff_a}, {int(prob_loss_a*100)}% chance of -${int(loss_a)}"),
            Choice(id="B", text=f"Guaranteed +${payoff_b}"),
        ],
        answer=Answer(
            choice_id=correct_id,
            explanation=f"EV(A)={ev_a:.2f}, EV(B)={ev_b:.2f}",
        ),
        metadata={"ev_a": ev_a, "ev_b": ev_b},
    )


def _generate_ev_three_option(rng: random.Random, difficulty: str) -> Question:
    """Three-way allocation — model must identify the max-EV option."""

    def sample_option(label: str) -> tuple[str, float]:
        if difficulty == "easy":
            p    = round(rng.uniform(0.5, 0.9), 1)
            gain = rng.randint(10, 50) * 10
            loss = rng.randint(1, 5) * 10
        elif difficulty == "medium":
            p    = round(rng.uniform(0.3, 0.8), 2)
            gain = rng.randint(50, 400)
            loss = rng.randint(20, 200)
        else:
            p    = round(rng.uniform(0.35, 0.65), 2)
            gain = rng.randint(80, 200)
            loss = rng.randint(40, 180)

        ev   = round(p * gain + (1 - p) * (-loss), 2)
        text = (f"{int(p*100)}% chance of +${gain}, "
                f"{int((1-p)*100)}% chance of -${loss}")
        return text, ev

    options = {label: sample_option(label) for label in ("A", "B", "C")}
    correct_id = max(options, key=lambda k: options[k][1])

    prompt = (
        "You are making a one-time investment decision. Choose one option:\n\n"
        + "\n".join(f"{k}) {v[0]}" for k, v in options.items())
        + "\n\nWhich option has the highest expected value?"
    )

    return Question(
        id="",
        dataset="",
        task_type="mcq",
        prompt=prompt,
        choices=[Choice(id=k, text=v[0]) for k, v in options.items()],
        answer=Answer(
            choice_id=correct_id,
            explanation=", ".join(f"EV({k})={v[1]:.2f}" for k, v in options.items()),
        ),
        metadata={f"ev_{k.lower()}": v[1] for k, v in options.items()},
    )


def _generate_conditional_probability(rng: random.Random, difficulty: str) -> Question:
    """P(A|B) problems — tests whether the model conditions correctly."""

    total       = rng.randint(20, 100)
    red         = rng.randint(5, total - 5)
    blue        = total - red
    red_marked  = rng.randint(1, red)
    blue_marked = rng.randint(1, blue)
    total_marked = red_marked + blue_marked

    p_true  = round(red_marked / total_marked, 4)
    wrong_1 = round(red / total, 4)
    wrong_2 = round(red_marked / total, 4)

    options_raw = {"A": p_true, "B": wrong_1, "C": wrong_2}

    if difficulty == "hard":
        options_raw["D"] = round(blue_marked / total_marked, 4)

    correct_id = "A"

    prompt = (
        f"A bag contains {total} balls: {red} red and {blue} blue. "
        f"Of the red balls, {red_marked} are marked. "
        f"Of the blue balls, {blue_marked} are marked. "
        f"You draw one ball at random and it is marked. "
        f"What is the probability it is red?\n\n"
        + "\n".join(f"{k}) {v:.4f}" for k, v in options_raw.items())
    )

    return Question(
        id="",
        dataset="",
        task_type="mcq",
        prompt=prompt,
        choices=[Choice(id=k, text=str(v)) for k, v in options_raw.items()],
        answer=Answer(
            choice_id=correct_id,
            explanation=(
                f"P(red|marked) = {red_marked}/{total_marked} = {p_true:.4f}. "
                f"Common traps: P(red)={wrong_1:.4f} ignores the marked condition; "
                f"P(red∩marked)={wrong_2:.4f} forgets to normalise."
            ),
        ),
        metadata={
            "total": total, "red": red, "blue": blue,
            "red_marked": red_marked, "blue_marked": blue_marked,
            "p_correct": p_true,
        },
    )


def _generate_base_rate(rng: random.Random, difficulty: str) -> Question:
    """Base-rate neglect / Bayes MCQ.

    easy   — sens == spec, round numbers, direct two-number Bayes
    medium — sens != spec, non-round values, standard two-number Bayes
    hard   — three mutually exclusive hypotheses; P(evidence) must be computed
             via law of total probability before Bayes can be applied
    """
    if difficulty == "easy":
        base_rate = rng.choice([0.10, 0.20, 0.30, 0.40])
        sens = round(rng.uniform(0.75, 0.92), 2)
        spec = sens

        tp = base_rate * sens
        fp = (1 - base_rate) * (1 - spec)
        posterior = round(tp / (tp + fp), 4)

        distractors = _dedup_distractors(
            correct=posterior,
            candidates=[round(sens, 4), round(base_rate, 4),
                        round((posterior + sens) / 2, 4)],
            rng=rng,
        )
        problem = (
            f"A condition affects {int(base_rate * 100)}% of the population. "
            f"A test for it is {int(sens * 100)}% accurate "
            f"(same sensitivity and specificity). "
            f"A random person tests positive. "
            f"What is the probability they have the condition?"
        )
        explanation = (
            f"P(+|H)·P(H) = {base_rate}·{sens} = {tp:.4f}; "
            f"P(+|¬H)·P(¬H) = {round(1-base_rate, 2)}·{round(1-spec, 2)} = {fp:.4f}; "
            f"posterior = {tp:.4f} / {tp+fp:.4f} = {posterior:.4f}"
        )
        meta: dict[str, Any] = {
            "base_rate": base_rate, "sensitivity": sens,
            "specificity": spec, "posterior": posterior,
        }

    elif difficulty == "medium":
        base_rate = round(rng.uniform(0.05, 0.18), 4)
        sens = round(rng.uniform(0.78, 0.95), 4)
        spec = round(rng.uniform(0.78, 0.95), 4)

        tp = base_rate * sens
        fp = (1 - base_rate) * (1 - spec)
        posterior = round(tp / (tp + fp), 4)

        distractors = _dedup_distractors(
            correct=posterior,
            candidates=[round(sens, 4), round(base_rate, 4),
                        round((posterior + sens) / 2, 4)],
            rng=rng,
        )
        problem = (
            f"A condition affects {base_rate:.2%} of the population. "
            f"A diagnostic test has sensitivity {sens:.4f} and specificity {spec:.4f}. "
            f"A randomly selected person tests positive. "
            f"What is the probability they actually have the condition?"
        )
        explanation = (
            f"P(+|H)·P(H) = {base_rate:.4f}·{sens:.4f} = {tp:.6f}; "
            f"P(+|¬H)·P(¬H) = {1-base_rate:.4f}·{1-spec:.4f} = {fp:.6f}; "
            f"posterior = {tp:.6f} / {tp+fp:.6f} = {posterior:.4f}"
        )
        meta = {
            "base_rate": base_rate, "sensitivity": sens,
            "specificity": spec, "posterior": posterior,
        }

    else:
        # Hard: three mutually exclusive hypotheses.
        # P(evidence) must be constructed via law of total probability.
        p1 = round(rng.uniform(0.03, 0.15), 4)
        p2 = round(rng.uniform(0.10, 0.30), 4)
        p3 = round(1.0 - p1 - p2, 4)
        l1 = round(rng.uniform(0.70, 0.95), 4)
        l2 = round(rng.uniform(0.20, 0.55), 4)
        l3 = round(rng.uniform(0.02, 0.15), 4)

        p_evidence = p1 * l1 + p2 * l2 + p3 * l3
        posterior = round((p1 * l1) / p_evidence, 4)
        p_h2_given_e = round((p2 * l2) / p_evidence, 4)

        distractors = _dedup_distractors(
            correct=posterior,
            candidates=[round(l1, 4), round(p1, 4), p_h2_given_e],
            rng=rng,
        )
        problem = (
            f"An observation can be explained by three mutually exclusive hypotheses:\n"
            f"  H1: prior probability {p1:.4f}, P(observation | H1) = {l1:.4f}\n"
            f"  H2: prior probability {p2:.4f}, P(observation | H2) = {l2:.4f}\n"
            f"  H3: prior probability {p3:.4f}, P(observation | H3) = {l3:.4f}\n"
            f"The observation occurs. "
            f"What is the posterior probability that H1 is the cause?"
        )
        explanation = (
            f"P(obs) = {p1:.4f}·{l1:.4f} + {p2:.4f}·{l2:.4f} + {p3:.4f}·{l3:.4f} "
            f"= {p_evidence:.6f}; "
            f"P(H1|obs) = {p1:.4f}·{l1:.4f} / {p_evidence:.6f} = {posterior:.4f}"
        )
        meta = {
            "p1": p1, "p2": p2, "p3": p3,
            "l1": l1, "l2": l2, "l3": l3,
            "p_evidence": round(p_evidence, 6), "posterior": posterior,
        }

    choices, correct_label = _make_choices(
        [posterior] + distractors, fmt=lambda v: f"{v:.4f}", rng=rng
    )
    return Question(
        id="",
        dataset="",
        task_type="mcq",
        prompt=problem + "\n\n" + "\n".join(f"{c.id}) {c.text}" for c in choices),
        choices=choices,
        answer=Answer(choice_id=correct_label, explanation=explanation),
        metadata=meta,
    )


def _generate_bayes_update(rng: random.Random, difficulty: str) -> Question:
    """Bayesian belief update MCQ.

    easy   — single update, integer likelihood ratio, round prior
    medium — single update, non-round LR and prior
    hard   — two sequential updates; stopping after update 1 is a salient distractor
    """
    if difficulty == "easy":
        prior = rng.choice([0.10, 0.20, 0.25, 0.30, 0.40, 0.50])
        lr1 = float(rng.choice([2, 3, 4, 5]))
        two_stage = False
        lr2 = None
    elif difficulty == "medium":
        prior = round(rng.uniform(0.08, 0.65), 4)
        lr1 = round(rng.uniform(1.5, 7.0), 3)
        two_stage = False
        lr2 = None
    else:
        prior = round(rng.uniform(0.05, 0.35), 4)
        lr1 = round(rng.uniform(2.0, 6.0), 3)
        lr2 = round(rng.uniform(1.5, 4.5), 3)
        two_stage = True

    def _update(p: float, lr: float) -> float:
        odds = p / (1 - p)
        return round(odds * lr / (1 + odds * lr), 4)

    post1 = _update(prior, lr1)
    correct = _update(post1, lr2) if two_stage else post1

    distractors = _dedup_distractors(
        correct=correct,
        candidates=[
            post1,
            round(min(prior * lr1, 0.9999), 4),
            round(min(correct + 0.08, 0.9999), 4),
            round(max(correct - 0.08, 0.0001), 4),
        ],
        rng=rng,
    )

    choices, correct_label = _make_choices(
        [correct] + distractors, fmt=lambda v: f"{v:.4f}", rng=rng
    )

    if not two_stage:
        problem = (
            f"Prior: P(H) = {prior:.4f}. "
            f"New evidence is {lr1}× more likely if H is true than if H is false. "
            f"What is the posterior P(H | evidence)?"
        )
        explanation = (
            f"prior odds = {prior:.4f} / {1-prior:.4f} = {prior/(1-prior):.4f}; "
            f"posterior odds = {prior/(1-prior):.4f} × {lr1} = {prior/(1-prior)*lr1:.4f}; "
            f"posterior = {correct:.4f}"
        )
    else:
        problem = (
            f"Prior: P(H) = {prior:.4f}.\n"
            f"Evidence E1 is {lr1}× more likely if H is true than if false.\n"
            f"Evidence E2 is {lr2}× more likely if H is true than if false.\n"
            f"What is the final posterior P(H | E1, E2)?"
        )
        explanation = f"after E1: {post1:.4f}; after E2: {correct:.4f}"

    return Question(
        id="",
        dataset="",
        task_type="mcq",
        prompt=problem + "\n\n" + "\n".join(f"{c.id}) {c.text}" for c in choices),
        choices=choices,
        answer=Answer(choice_id=correct_label, explanation=explanation),
        metadata={
            "prior": prior, "lr1": lr1, "lr2": lr2,
            "two_stage": two_stage, "posterior": correct,
        },
    )


def _dedup_distractors(
    correct: float, candidates: list[float], rng: random.Random, n: int = 3
) -> list[float]:
    out: list[float] = []
    for d in candidates:
        if abs(d - correct) > 0.005 and d not in out:
            out.append(d)
        if len(out) == n:
            return out
    while len(out) < n:
        noise = rng.uniform(-0.15, 0.15)
        cand = round(min(max(correct + noise, 0.0001), 0.9999), 4)
        if abs(cand - correct) > 0.005 and cand not in out:
            out.append(cand)
    return out[:n]


def _make_choices(
    values: list[float], fmt: Any, rng: random.Random
) -> tuple[list[Choice], str]:
    labels = ["A", "B", "C", "D"]
    order = list(range(len(values)))
    rng.shuffle(order)
    correct_label = labels[order.index(0)]
    choices = [Choice(id=labels[i], text=fmt(values[order[i]])) for i in range(len(values))]
    return choices, correct_label
