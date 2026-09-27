"""Per-field contract: normalizers, comparison kinds, and deterministic settling.

Motivated by a measured failure: normalize() keeps spaces, so '5551234567' and
'555 123 4567' become DIFFERENT keys, and a naive digits-only rule corrupts
extensions ('... ext 2222' -> '2345672222').
"""

from __future__ import annotations

import pandas as pd
import pytest

from jlink.fields import Field, key_text, parse_fields, parse_on


def digits(v):
    return "".join(ch for ch in str(v) if ch.isdigit())


def last10(v):
    d = digits(v)
    return d[-10:] if len(d) >= 10 else d


# ------------------------------------------------------------------ parsing
def test_bare_column_and_pair_still_parse_as_before():
    assert parse_on(["name", ("city", "town")]) == [("name", "name", "name"), ("city", "city", "town")]
    assert [f.triple for f in parse_fields(["name", ("city", "town")])] == \
        [("name", "name", "name"), ("city", "city", "town")]


def test_field_triple_matches_plain_pair():
    assert Field("city", "city", "town").triple == ("city", "city", "town")


def test_fields_mix_with_plain_items():
    got = parse_fields(["name", Field("phone", "phone", "contact")])
    assert [f.label for f in got] == ["name", "phone"]
    assert got[0].compare == "text" and got[0].judge is True and got[0].normalizer() is not last10
    assert got[1].normalizer() is not last10
    custom = parse_fields([Field("phone", "phone", "contact", normalize=last10)])
    assert custom[0].normalizer() is last10


def test_field_validation():
    with pytest.raises(ValueError, match="`compare` must be one of"):
        parse_fields([Field("x", "x", "x", compare="fuzzy")])
    with pytest.raises(ValueError, match="`normalize` must be callable"):
        parse_fields([Field("x", "x", "x", normalize="nope")])
    with pytest.raises(ValueError, match="`judge` must be a boolean"):
        parse_fields([Field("x", "x", "x", judge="yes")])
    with pytest.raises(ValueError, match="label must be a nonempty"):
        parse_fields([Field("", "", "")])


def test_one_sided_field_still_needs_unpaired():
    with pytest.raises(ValueError, match="one side only"):
        parse_fields([Field("text", "text", None)])
    # a positive case needs both sides labelled, as parse_on has always required
    got = parse_fields([Field("text", "text", None), Field("place", None, "place")], unpaired=True)
    assert [f.triple for f in got] == [("text", "text", None), ("place", None, "place")]
    # a lone one-sided field leaves the other side unlabelled; the same error parse_on gives
    with pytest.raises(ValueError, match="right records no field"):
        parse_fields([Field("text", "text", None)], unpaired=True)


def test_field_rejects_two_missing_sides():
    with pytest.raises(ValueError, match="each `on` item"):
        parse_fields([Field("x", None, None)], unpaired=True)


# -------------------------------------------------------------- normalization
def test_key_text_accepts_a_field_normalizer():
    assert key_text("555-123-4567", last10) == "5551234567"
    assert key_text("5551234567", last10) == "5551234567"
    assert key_text(5551234567, last10) == "5551234567"


def test_a_normalizer_must_handle_the_extension_itself():
    """key_text cannot know an extension is there; the field's rule must strip it.

    This is the documented trap: last10() alone turns '... ext 2222' into '2345672222'
    because the extension digits sit at the end of the string.
    """
    assert key_text("+1 (555) 123-4567 x99", last10) == "5123456799"   # extension ate the tail

    def main10(v):
        s = str(v).split(" ext ")[0].split(" x")[0]
        return last10(s)

    assert key_text("555-123-4567 ext 2222", main10) == "5551234567"
    assert key_text("+1 (555) 123-4567 x99", main10) == "5551234567"


def test_key_text_default_is_unchanged():
    assert key_text("Société Générale & Co.") == "societe generale and co"
    assert key_text(1985) == key_text("1985.0") == "1985"


def test_the_measured_failure_is_now_fixable():
    """Without a field normalizer these are different keys; with last10 they are one."""
    formatted, bare = "(555) 123-4567", "5551234567"
    assert key_text(formatted) != key_text(bare)          # today's behaviour
    assert key_text(formatted, last10) == key_text(bare, last10) == "5551234567"  # with the contract


def test_naive_digits_alone_would_corrupt_extensions():
    """Documents why the parser must split the extension BEFORE counting digits."""
    naive = lambda v: (lambda d: d[-10:] if len(d) >= 10 else d)(digits(v))
    assert naive("555-123-4567 ext 2222") == "2345672222"   # wrong

    def main10(v):
        s = str(v).split(" ext ")[0].split(" x")[0]
        return last10(s)

    assert main10("555-123-4567 ext 2222") == "5551234567"  # right


# ------------------------------------------------------------------ config
def test_field_to_config_is_json_safe_and_names_custom_normalizers():
    cfg = Field("phone", "phone", "contact", normalize=last10, compare="exact", judge=False).to_config()
    assert cfg["custom_normalizer"] is True and cfg["compare"] == "exact" and cfg["judge"] is False
    assert Field("name", "name", "name").to_config()["custom_normalizer"] is False
