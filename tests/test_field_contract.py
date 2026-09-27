"""Per-field contract: a field's own normalizer, compare kind and judging policy.

A field's rule is used everywhere the field becomes text: the blocking keys, the ``sim`` score,
the group keys and the judge's view. Two small, generic columns stand in for any structured
field, so the mechanism is tested without a domain story.
"""

from __future__ import annotations

import pandas as pd
import pytest

import jlink
from jlink.fields import Field, key_text, parse_fields, parse_on


def digits(v):
    """A generic per-field rule: keep the digits, drop the formatting."""
    return "".join(ch for ch in str(v) if ch.isdigit())


@pytest.fixture
def tables():
    left = pd.DataFrame({"lid": [1, 2], "ref": ["A-100", "B-200"]})
    right = pd.DataFrame({"rid": [10, 20], "code": ["A100", "B200"]})
    return left, right


# ------------------------------------------------------------------ parsing
def test_plain_items_still_mean_what_they_meant():
    assert parse_on(["name", ("city", "town")]) == [("name", "name", "name"), ("city", "city", "town")]
    assert Field("city", "city", "town").triple == ("city", "city", "town")


def test_fields_mix_with_plain_items():
    got = parse_fields(["name", Field("ref", "ref", "code", normalize=digits)])
    assert [f.label for f in got] == ["name", "ref"]
    assert got[0].normalizer() is jlink.fields.normalize and got[1].normalizer() is digits


def test_field_validation_rejects_a_bad_contract():
    for bad, message in ((Field("x", "x", "x", compare="fuzzy"), "`compare` must be one of"),
                         (Field("x", "x", "x", normalize="nope"), "`normalize` must be callable"),
                         (Field("x", "x", "x", judge="yes"), "`judge` must be a boolean"),
                         (Field("", "", ""), "nonempty")):
        with pytest.raises(ValueError, match=message):
            parse_fields([bad])


def test_one_sided_fields_parse_only_when_both_sides_are_labelled():
    with pytest.raises(ValueError, match="one side only"):
        parse_fields([Field("text", "text", None)])
    got = parse_fields([Field("text", "text", None), Field("place", None, "place")], unpaired=True)
    assert [f.triple for f in got] == [("text", "text", None), ("place", None, "place")]
    with pytest.raises(ValueError, match="right records no field"):
        parse_fields([Field("text", "text", None)], unpaired=True)   # a lone one-sided field
    with pytest.raises(ValueError, match="each `on` item"):
        parse_fields([Field("x", None, None)], unpaired=True)


# -------------------------------------------------------------- the rule is used
def test_a_field_rule_matches_where_the_default_rule_does_not():
    formatted, bare = "A-100", "A100"
    assert key_text(formatted) != key_text(bare)              # the shared text rule splits them
    assert key_text(formatted, digits) == key_text(bare, digits) == "100"  # the field's rule joins them


def test_key_text_whole_numbers_survive_the_shared_rule():
    """Naming the shared normalize is not a custom rule, so 1985 meets 1985.0 as before."""
    assert key_text("1985.0") == key_text(1985) == "1985"
    assert key_text("1985.0", jlink.fields.normalize) == "1985"


def test_default_blocking_carries_the_field_rule(tables):
    """A Field in `on` with no explicit blockers must key the field its own way."""
    left, right = tables
    linker = jlink.Linker(entity="record", on=[Field("ref", "ref", "code", normalize=digits)])
    cands = linker.candidates(left, right, left_id="lid", right_id="rid")
    assert {(1, 10), (2, 20)} <= set(zip(cands.left_id, cands.right_id))
    assert cands.attrs["blocking"]["passes"][0]["config"]["fields"][0]["custom_normalizer"] is True


def test_the_rule_scores_sim_too(tables):
    left, right = tables
    cands = jlink.block.candidates(left, right,
                                   on=[Field("ref", "ref", "code", normalize=digits, compare="exact")],
                                   left_id="lid", right_id="rid")
    assert (cands.sim == 1.0).all(), "both sides canonicalize to the same digits"


def test_compare_kind_picks_the_default_pass():
    passes = jlink.block.default_passes(
        [Field("name", "name", "name"), Field("ref", "ref", "code", compare="exact", normalize=digits)])
    assert [p.name for p in passes] == ["ngrams:name", "ngrams-reverse:name", "exact:ref=code"]


# --------------------------------------------------------- settling without the model
def test_judge_false_settles_a_pair_with_no_model_call(tables):
    from fakes import FakeJev

    left, right = tables
    fake = FakeJev()
    linker = jlink.Linker(entity="record", cache=False, on=[
        Field("ref", "ref", "code", normalize=digits, compare="exact", judge=False)])
    result = linker.link(left, right, left_id="lid", right_id="rid",
                         how="many-to-many", progress=False, transport=fake.transport)
    assert set(zip(result.links.left_id, result.links.right_id)) == {(1, 10), (2, 20)}
    assert result.meter.calls == 0 and fake.bodies == []
    assert set(result.links["source"]) == {"exact"}


def test_judge_false_must_exist_on_both_sides(tables):
    left, right = tables
    cands = pd.DataFrame({"left_id": [1], "right_id": [10], "sim": [0.5]})
    with pytest.raises(ValueError, match="must exist on both sides"):
        jlink.judge(cands, left, right, on=[("ref", "ref"),
                                            Field("code", "code", None, normalize=digits, judge=False)],
                    entity="record", left_id="lid", right_id="rid",
                    budget=0.0, cache=False, progress=False)


# ------------------------------------------------------------------ provenance
def test_the_field_contract_is_recorded_and_json_safe():
    import json
    from jlink.linker import _normalization_summary

    specs = parse_fields([("name", "name"), Field("ref", "ref", "code", normalize=digits)])
    assert _normalization_summary(specs) == "jlink.fields.normalize_v1+custom:ref"
    assert _normalization_summary(parse_fields(["name"])) == "jlink.fields.normalize_v1"
    json.dumps([s.to_config() for s in specs])   # a callable must never reach settings.json
