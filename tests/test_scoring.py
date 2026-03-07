"""Tests for the scoring engine — added to as Stage 6 is implemented."""

import pytest
from src.scoring.engine import label_score


def test_label_score_buy():
    assert label_score(70) == "Buy"


def test_label_score_hold():
    assert label_score(50) == "Hold"


def test_label_score_dont_buy():
    assert label_score(20) == "Don't Buy"
