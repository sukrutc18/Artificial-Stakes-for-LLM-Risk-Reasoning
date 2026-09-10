"""Tests for the synthetic EV / probability generator."""
from __future__ import annotations

import pytest
from risk_reasoning.data import synthetic
from risk_reasoning.data.schemas import Question

# ---------------------------------------------------------------------------
# Shared config helper
# ---------------------------------------------------------------------------

def _cfg(seed: int = 42, **overrides) -> dict:
    base = {
        "seed": seed,
        "difficulties": ["easy", "medium", "hard"],
        "n_ev_two": 0,
        "n_ev_three": 0,
        "n_conditional": 0,
        "n_base_rate": 0,
        "n_bayes": 0,
    }
    return {**base, **overrides}


# ---------------------------------------------------------------------------
# load() — general
# ---------------------------------------------------------------------------

def test_load_yields_nonempty() -> None:
    cfg = _cfg(n_ev_two=5)
    items = list(synthetic.load(cfg))
    assert len(items) == 5


def test_every_item_has_ground_truth() -> None:
    cfg = _cfg(n_conditional=5)
    for q in synthetic.load(cfg):
        assert q.answer is not None
        assert q.answer.choice_id is not None


def test_all_items_are_question_instances() -> None:
    cfg = _cfg(n_ev_two=3, n_ev_three=3, n_conditional=3)
    assert all(isinstance(q, Question) for q in synthetic.load(cfg))


def test_all_items_have_unique_ids() -> None:
    cfg = _cfg(n_ev_two=5, n_ev_three=5, n_conditional=5)
    ids = [q.id for q in synthetic.load(cfg)]
    assert len(ids) == len(set(ids)), "Duplicate IDs found"


def test_dataset_field_is_synthetic_ev() -> None:
    cfg = _cfg(n_ev_two=3)
    assert all(q.dataset == "synthetic_ev" for q in synthetic.load(cfg))


def test_task_type_is_mcq() -> None:
    cfg = _cfg(n_ev_two=3, n_ev_three=3, n_conditional=3)
    assert all(q.task_type == "mcq" for q in synthetic.load(cfg))


def test_choices_are_populated() -> None:
    cfg = _cfg(n_ev_two=5)
    for q in synthetic.load(cfg):
        assert q.choices is not None
        assert len(q.choices) >= 2


def test_correct_choice_id_exists_in_choices() -> None:
    cfg = _cfg(n_ev_two=5, n_ev_three=5, n_conditional=5)
    for q in synthetic.load(cfg):
        choice_ids = {c.id for c in q.choices}
        assert q.answer.choice_id in choice_ids, (
            f"{q.id}: answer '{q.answer.choice_id}' not in choices {choice_ids}"
        )


def test_answer_has_explanation() -> None:
    cfg = _cfg(n_ev_two=3, n_ev_three=3, n_conditional=3)
    for q in synthetic.load(cfg):
        assert q.answer.explanation is not None
        assert len(q.answer.explanation) > 0


def test_difficulty_stamped_in_metadata() -> None:
    cfg = _cfg(n_ev_two=6, difficulties=["easy", "medium", "hard"])
    for q in synthetic.load(cfg):
        assert q.metadata.get("difficulty") in {"easy", "medium", "hard"}


def test_family_stamped_in_metadata() -> None:
    cfg = _cfg(n_ev_two=3, n_ev_three=3, n_conditional=3)
    for q in synthetic.load(cfg):
        assert "family" in q.metadata


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_produces_same_items() -> None:
    cfg = _cfg(n_ev_two=5, n_ev_three=5, n_conditional=5)
    run_1 = [(q.id, q.answer.choice_id) for q in synthetic.load(cfg)]
    run_2 = [(q.id, q.answer.choice_id) for q in synthetic.load(cfg)]
    assert run_1 == run_2


def test_different_seeds_produce_different_items() -> None:
    cfg_a = _cfg(seed=1,  n_ev_two=10)
    cfg_b = _cfg(seed=99, n_ev_two=10)
    prompts_a = [q.prompt for q in synthetic.load(cfg_a)]
    prompts_b = [q.prompt for q in synthetic.load(cfg_b)]
    assert prompts_a != prompts_b


# ---------------------------------------------------------------------------
# _generate_ev_two_option
# ---------------------------------------------------------------------------

def test_ev_two_option_correct_answer_matches_higher_ev() -> None:
    cfg = _cfg(n_ev_two=20)
    for q in synthetic.load(cfg):
        ev_a = q.metadata["ev_a"]
        ev_b = q.metadata["ev_b"]
        expected = "A" if ev_a > ev_b else "B"
        assert q.answer.choice_id == expected, (
            f"{q.id}: EV(A)={ev_a}, EV(B)={ev_b} but answer={q.answer.choice_id}"
        )


def test_ev_two_option_has_exactly_two_choices() -> None:
    cfg = _cfg(n_ev_two=10)
    for q in synthetic.load(cfg):
        assert len(q.choices) == 2


def test_ev_two_option_metadata_keys_present() -> None:
    cfg = _cfg(n_ev_two=5)
    for q in synthetic.load(cfg):
        assert "ev_a" in q.metadata
        assert "ev_b" in q.metadata


def test_ev_two_option_choice_labels_are_a_and_b() -> None:
    cfg = _cfg(n_ev_two=5)
    for q in synthetic.load(cfg):
        ids = {c.id for c in q.choices}
        assert ids == {"A", "B"}


# ---------------------------------------------------------------------------
# _generate_ev_three_option
# ---------------------------------------------------------------------------

def test_ev_three_option_correct_answer_is_max_ev() -> None:
    cfg = _cfg(n_ev_three=20)
    for q in synthetic.load(cfg):
        evs = {
            k.replace("ev_", "").upper(): v
            for k, v in q.metadata.items()
            if k.startswith("ev_")
        }
        expected = max(evs, key=evs.get)
        assert q.answer.choice_id == expected, (
            f"{q.id}: EVs={evs} but answer={q.answer.choice_id}"
        )


def test_ev_three_option_has_exactly_three_choices() -> None:
    cfg = _cfg(n_ev_three=10)
    for q in synthetic.load(cfg):
        assert len(q.choices) == 3


def test_ev_three_option_metadata_keys_present() -> None:
    cfg = _cfg(n_ev_three=5)
    for q in synthetic.load(cfg):
        assert "ev_a" in q.metadata
        assert "ev_b" in q.metadata
        assert "ev_c" in q.metadata


def test_ev_three_option_choice_labels_are_a_b_c() -> None:
    cfg = _cfg(n_ev_three=5)
    for q in synthetic.load(cfg):
        ids = {c.id for c in q.choices}
        assert ids == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# _generate_conditional_probability
# ---------------------------------------------------------------------------

def test_conditional_probability_answer_is_a() -> None:
    cfg = _cfg(n_conditional=10)
    for q in synthetic.load(cfg):
        assert q.answer.choice_id == "A"


def test_conditional_probability_p_correct_in_metadata() -> None:
    cfg = _cfg(n_conditional=10)
    for q in synthetic.load(cfg):
        assert "p_correct" in q.metadata
        p = q.metadata["p_correct"]
        assert 0.0 < p < 1.0


def test_conditional_probability_hard_has_four_choices() -> None:
    cfg = _cfg(n_conditional=15, difficulties=["hard"])
    for q in synthetic.load(cfg):
        assert len(q.choices) == 4


def test_conditional_probability_easy_medium_has_three_choices() -> None:
    cfg = _cfg(n_conditional=10, difficulties=["easy", "medium"])
    for q in synthetic.load(cfg):
        assert len(q.choices) == 3


def test_conditional_probability_metadata_counts_consistent() -> None:
    cfg = _cfg(n_conditional=10)
    for q in synthetic.load(cfg):
        m = q.metadata
        total_marked = m["red_marked"] + m["blue_marked"]
        expected_p = round(m["red_marked"] / total_marked, 4)
        assert abs(m["p_correct"] - expected_p) < 1e-6, (
            f"{q.id}: p_correct={m['p_correct']} but computed={expected_p}"
        )


# ---------------------------------------------------------------------------
# _generate_base_rate  (Shrey)
# ---------------------------------------------------------------------------

def test_base_rate_correct_label_in_choices() -> None:
    cfg = _cfg(n_base_rate=15)
    for q in synthetic.load(cfg):
        assert q.answer.choice_id in {c.id for c in q.choices}


def test_base_rate_four_distinct_choices() -> None:
    cfg = _cfg(n_base_rate=15)
    for q in synthetic.load(cfg):
        texts = [c.text for c in q.choices]
        assert len(texts) == 4
        assert len(set(texts)) == 4


def test_base_rate_posterior_in_range() -> None:
    cfg = _cfg(n_base_rate=15)
    for q in synthetic.load(cfg):
        assert 0 < q.metadata["posterior"] < 1


def test_base_rate_hard_has_three_hypotheses() -> None:
    cfg = _cfg(n_base_rate=9, difficulties=["hard"])
    for q in synthetic.load(cfg):
        m = q.metadata
        assert "p1" in m and "p2" in m and "p3" in m
        assert abs(m["p1"] + m["p2"] + m["p3"] - 1.0) < 1e-4


def test_base_rate_easy_medium_math_correct() -> None:
    cfg = _cfg(n_base_rate=10, difficulties=["easy", "medium"])
    for q in synthetic.load(cfg):
        m = q.metadata
        tp = m["base_rate"] * m["sensitivity"]
        fp = (1 - m["base_rate"]) * (1 - m["specificity"])
        expected = round(tp / (tp + fp), 4)
        assert abs(m["posterior"] - expected) < 1e-4, (
            f"{q.id}: expected {expected} but got {m['posterior']}"
        )


def test_base_rate_hard_math_correct() -> None:
    cfg = _cfg(n_base_rate=9, difficulties=["hard"])
    for q in synthetic.load(cfg):
        m = q.metadata
        p_ev = m["p1"] * m["l1"] + m["p2"] * m["l2"] + m["p3"] * m["l3"]
        expected = round((m["p1"] * m["l1"]) / p_ev, 4)
        assert abs(m["posterior"] - expected) < 1e-4


# ---------------------------------------------------------------------------
# _generate_bayes_update  (Shrey)
# ---------------------------------------------------------------------------

def test_bayes_update_correct_label_in_choices() -> None:
    cfg = _cfg(n_bayes=15)
    for q in synthetic.load(cfg):
        assert q.answer.choice_id in {c.id for c in q.choices}


def test_bayes_update_four_distinct_choices() -> None:
    cfg = _cfg(n_bayes=15)
    for q in synthetic.load(cfg):
        texts = [c.text for c in q.choices]
        assert len(set(texts)) == 4


def test_bayes_update_hard_has_lr2() -> None:
    cfg = _cfg(n_bayes=9, difficulties=["hard"])
    for q in synthetic.load(cfg):
        assert q.metadata["two_stage"] is True
        assert q.metadata["lr2"] is not None


def test_bayes_update_easy_medium_no_lr2() -> None:
    cfg = _cfg(n_bayes=10, difficulties=["easy", "medium"])
    for q in synthetic.load(cfg):
        assert q.metadata["two_stage"] is False
        assert q.metadata["lr2"] is None


def test_bayes_update_math_correct() -> None:
    cfg = _cfg(n_bayes=15)
    for q in synthetic.load(cfg):
        m = q.metadata
        def _update(p: float, lr: float) -> float:
            odds = p / (1 - p)
            return round(odds * lr / (1 + odds * lr), 4)
        post1 = _update(m["prior"], m["lr1"])
        expected = _update(post1, m["lr2"]) if m["two_stage"] else post1
        assert abs(m["posterior"] - expected) < 1e-4, (
            f"{q.id}: expected {expected} but got {m['posterior']}"
        )


def test_bayes_update_posterior_in_range() -> None:
    cfg = _cfg(n_bayes=15)
    for q in synthetic.load(cfg):
        assert 0 < q.metadata["posterior"] < 1
