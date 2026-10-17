"""Tests for the frozen unit split (shikra/data/split.py)."""

from __future__ import annotations


def test_no_unit_overlap() -> None:
    """No unit may appear in two splits."""
    from shikra.data.split import make_split

    split = make_split()
    train_set = {(s, u) for s, u in split["train"]}
    calib_set = {(s, u) for s, u in split["calib"]}
    replay_set = {(s, u) for s, u in split["replay"]}

    assert train_set.isdisjoint(calib_set), "Train/calib overlap!"
    assert train_set.isdisjoint(replay_set), "Train/replay overlap!"
    assert calib_set.isdisjoint(replay_set), "Calib/replay overlap!"


def test_total_count_200() -> None:
    """The splits must sum to exactly 200 units (100 FD001 + 100 FD003)."""
    from shikra.data.split import make_split

    split = make_split()
    total = len(split["train"]) + len(split["calib"]) + len(split["replay"])
    assert total == 200, f"Expected 200, got {total}"


def test_deterministic_seed() -> None:
    """Calling make_split() twice with seed 2026 must yield identical results."""
    from shikra.data.split import make_split

    s1 = make_split()
    s2 = make_split()
    assert s1 == s2, "Split is not deterministic!"


def test_split_proportions() -> None:
    """Each subset should have 60/20/20 split (within rounding)."""
    from shikra.data.split import make_split

    split = make_split()
    for subset in ("FD001", "FD003"):
        n_train = sum(1 for s, _ in split["train"] if s == subset)
        n_calib = sum(1 for s, _ in split["calib"] if s == subset)
        n_replay = sum(1 for s, _ in split["replay"] if s == subset)
        total = n_train + n_calib + n_replay
        assert total == 100, f"{subset}: expected 100 total, got {total}"
        assert n_train == 60, f"{subset}: expected 60 train, got {n_train}"
        assert n_calib == 20, f"{subset}: expected 20 calib, got {n_calib}"
        assert n_replay == 20, f"{subset}: expected 20 replay, got {n_replay}"
