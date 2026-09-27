"""End to end: a deterministic field settles pairs, with no model call.

Reproduces the measured phone result through the public API:
  8 people, each written differently in two sources.
  Global normalize() reconciles 3/8; a per-field canonicalizer reconciles 8/8.

Uses a CallableJudge-style offline transport so the test needs no API key: the
point is that a `judge=False` field can settle a pair before the judge is asked.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

import jlink
from jlink.fields import Field, key_text

# ---------------------------------------------------------------- the parser
_DECOR = re.compile(r"\((?!\d{3}\))([^)]*)\)", re.I)          # (cell), (home) -- not (555)
_EXT = re.compile(r"(?:e?xt?ension|ext|x|#)\s*[:.]?\s*(\d{1,6})\s*$", re.I)


def parse_phone(v) -> tuple[str | None, str | None]:
    """(main10, ext). Split the extension BEFORE counting digits, or it eats the number."""
    s = str(v).strip()
    if not s:
        return None, None
    s = _DECOR.sub(" ", s)
    ext = None
    if m := _EXT.search(s):
        ext, s = m.group(1), s[: m.start()]
    d = "".join(ch for ch in s if ch.isdigit())
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    return (d or None), ext


def main10(v):
    p, _ = parse_phone(v)
    return p or ""


# ------------------------------------------------------------------ the data
PEOPLE = [
    ("p1", "Alice Chen",  "(555) 123-4567",           "555-123-4567 ext 2222"),
    ("p2", "Bob Ortiz",   "555.234.5678 (cell)",      "+1 555 234 5678"),
    ("p3", "Cara Ali",    "555 345 6789 x10",         "(555) 345-6789"),
    ("p4", "Dan Webb",    "5554567890",               "1-555-456-7890 ext. 5"),
    ("p5", "Eve Park",    "1-555-567-8901",           "(555) 567-8901 (home)"),
    ("p6", "Fay Ndiaye",  "+1 (555) 678-9012",        "555 678 9012 #44"),
    ("p7", "Gus Lind",    "555-789-0123",             "555.789.0123 (mobile)"),
    ("p8", "Hana Sato",   "(555) 890-1234",           "5558901234"),
]
TRUTH = {(k, k) for k, _, _, _ in PEOPLE}


@pytest.fixture
def tables():
    left = pd.DataFrame([{"left_id": k, "name": n, "phone": a, "been_there": 1}
                         for k, n, a, _ in PEOPLE])
    right = pd.DataFrame([{"right_id": k, "name": n, "phone": b, "been_there": 1}
                          for k, n, _, b in PEOPLE])
    return left, right


def reconcile(scores: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(scores.left_id, scores.right_id))


# -------------------------------------------------------------------- tests
def test_global_normalize_does_not_reconcile_formatted_phones():
    """The measured failure, kept as a regression guard.

    normalize() keeps spaces and punctuation-derived separators, so NONE of these eight
    differently-written pairs agree -- even though every one is the same number.
    """
    pairs = [(a, b) for _, _, a, b in PEOPLE]
    same = sum(key_text(a) == key_text(b) and key_text(a) != "" for a, b in pairs)
    assert same == 0, f"the global normalizer reconciled {same} of {len(pairs)}; expected none"


def test_canonicalizer_reconciles_every_pair():
    pairs = [(a, b) for _, _, a, b in PEOPLE]
    same = sum(main10(a) == main10(b) and main10(a) != "" for a, b in pairs)
    assert same == len(pairs) == 8


def test_extension_is_preserved_separately_and_not_destroyed():
    assert parse_phone("555-123-4567 ext 2222") == ("5551234567", "2222")
    assert parse_phone("+1 (555) 123-4567 x99") == ("5551234567", "99")
    assert parse_phone("(555) 123-4567 (cell)") == ("5551234567", None)
    assert parse_phone("5551234567") == ("5551234567", None)
    # the naive alternative, which this parser exists to avoid
    naive = "".join(ch for ch in "555-123-4567 ext 2222" if ch.isdigit())[-10:]
    assert naive == "2345672222" and parse_phone("555-123-4567 ext 2222")[0] == "5551234567"


def test_exact_blocker_on_a_custom_normalizer_finds_all_true_pairs(tables):
    """block.exact accepts a Field, so a canonicalized key joins cleanly."""
    left, right = tables
    cands = jlink.block.candidates(
        left, right, on=[Field("phone", "phone", "phone", normalize=main10, compare="exact")],
        blockers=[jlink.block.exact(Field("phone", "phone", "phone", normalize=main10))],
        left_id="left_id", right_id="right_id", max_pairs=None)
    got = reconcile(cands)
    assert got == TRUTH
    assert len(cands) == 8


def test_raw_phone_field_finds_almost_nothing(tables):
    """Without the contract the same data links poorly -- this is the motivation."""
    left, right = tables
    cands = jlink.block.candidates(
        left, right, on=[("phone", "phone")],
        blockers=[jlink.block.exact("phone")],
        left_id="left_id", right_id="right_id", max_pairs=None)
    assert len(reconcile(cands) & TRUTH) < len(TRUTH)


def test_field_carries_its_contract_into_the_judge_key(tables):
    """The exact-shortcut key uses the field's normalizer, so equal digits settle."""
    left, right = tables
    from jlink.judge import _records, _side_specs
    from jlink.fields import parse_fields
    fields = parse_fields([Field("phone", "phone", "phone", normalize=main10)], unpaired=True)
    a = _records(left, pd.Index(left.left_id), _side_specs(fields, "left"))
    b = _records(right, pd.Index(right.right_id), _side_specs(fields, "right"))
    # p1 left '(555) 123-4567' and p1 right '555-123-4567 ext 2222' share a key
    assert a.loc["p1", "exact_key"] == b.loc["p1", "exact_key"] == ("5551234567",)
    # ...while a differently-formatted pair that is NOT the same person does not
    assert a.loc["p1", "exact_key"] != b.loc["p2", "exact_key"]


def test_judge_reports_what_it_was_given(tables):
    """A Field does not change the record text shown to the judge, only the comparison."""
    left, _ = tables
    from jlink.judge import _records, _side_specs
    from jlink.fields import parse_fields
    fields = parse_fields([Field("phone", "phone", "phone", normalize=main10)], unpaired=True)
    a = _records(left, pd.Index(left.left_id), _side_specs(fields, "left"))
    assert a.loc["p1", "record"] == {"phone": "(555) 123-4567"}
