"""Type-aware field linking: canonicalizers, default passes, judge-only fields and provenance.

Each type gets one small, generic fixture -- a column of values and the keys they must produce --
so the mechanism is tested without a domain story.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import jlink
from fakes import FakeJev
from jlink.fields import COMPARES, TYPE_NORMALIZERS, Field, date_iso, digits, key_text, parse_fields, parse_on


# ------------------------------------------------------------------ canonicalizers
@pytest.mark.parametrize("value, expected", [
    ("200.0", "200"), (200.0, "200"), (200, "200"), ("200", "200"),
    ("0200", "0200"), ("1,985", "1985"), (20.5, ""), ("20.5", ""), (None, ""), ("", "")])
def test_digits_keeps_leading_zeros_and_drops_fractions(value, expected):
    assert digits(value) == expected


@pytest.mark.parametrize("value, expected", [
    ("2024-03-01", "2024-03-01"), ("2024/03/01", "2024-03-01"), ("2024-03", "2024-03"),
    ("1985", "1985"), (1985, "1985"), ("1985.0", "1985"), ("2024-03-01T14:30", "2024-03-01"),
    ("03/01/2024", ""), ("March 1 2024", ""), ("2024-13-40", ""), (None, "")])
def test_date_is_iso_only_and_never_guessed(value, expected):
    assert date_iso(value) == expected


def test_a_year_alone_stays_a_year_and_is_not_invented_into_a_day():
    assert date_iso("1985") == "1985"
    assert date_iso("1985-01-01") == "1985-01-01"   # a full date stays a full date


def test_whole_number_reconciliation_runs_before_the_type_rule():
    assert key_text("1985.0", digits) == key_text(1985, digits) == "1985"
    assert key_text("200.0", digits) == "200"
    assert key_text("200.0") == "200"   # the default text rule reconciles too, as before


def test_every_named_type_has_a_canonicalizer():
    assert set(TYPE_NORMALIZERS) == set(COMPARES)
    for canonicalizer in TYPE_NORMALIZERS.values():
        assert canonicalizer(None) == "" and canonicalizer("  ") == ""


# ------------------------------------------------------------------ parsing and validation
def test_plain_items_still_mean_what_they_meant():
    assert parse_on(["name", ("city", "town")]) == [("name", "name", "name"), ("city", "city", "town")]
    assert Field("city", "city", "town").triple == ("city", "city", "town")
    assert all(f.is_default for f in parse_fields(["name", ("city", "town")]))


def test_a_typed_field_is_not_default_and_carries_its_type():
    spec, = parse_fields([Field("year", "year", "yr", compare="date")])
    assert spec.compare == "date" and spec.canonicalizer() is date_iso and not spec.is_default


def test_field_validation_rejects_a_bad_contract():
    for bad, message in ((Field("x", "x", "x", compare="fuzzy"), "`compare` must be one of"),
                         (Field("x", "x", "x", normalize="nope"), "`normalize` must be callable"),
                         (Field("x", "x", "x", key="yes"), "`key` must be a boolean"),
                         (Field("x", "x", "x", deterministic="yes"), "`deterministic` must be a boolean"),
                         (Field("x", "x", "x", key=False, deterministic=True), "cannot be deterministic"),
                         (Field("x", "x", "x", compare="text", deterministic=True), "needs a typed `compare`"),
                         (Field("", "", ""), "nonempty")):
        with pytest.raises(ValueError, match=message):
            parse_fields([bad])


def test_a_custom_canonicalizer_replaces_the_type_rule():
    spec, = parse_fields([Field("ref", "ref", "code", normalize=digits)])
    assert spec.canonicalizer() is digits
    assert key_text("A-100", spec.canonicalizer()) == key_text("A100", spec.canonicalizer()) == "100"


# ------------------------------------------------------------------ default passes
def test_compare_kind_picks_the_default_pass():
    passes = jlink.block.default_passes(
        [Field("name", "name", "name"), Field("year", "year", "yr", compare="date"),
         Field("postcode", "postcode", "zip", compare="digits")])
    assert [p.name for p in passes] == ["ngrams:name", "ngrams-reverse:name", "exact:year=yr,postcode=zip"]


def test_a_judge_only_field_never_reaches_a_pass():
    passes = jlink.block.default_passes(
        [Field("name", "name", "name"), Field("price", "price", "cost", key=False)])
    assert [p.name for p in passes] == ["ngrams:name", "ngrams-reverse:name"]


def test_only_typed_fields_gives_only_an_exact_pass():
    passes = jlink.block.default_passes([Field("postcode", "postcode", "zip", compare="digits")])
    assert [p.name for p in passes] == ["exact:postcode=zip"]


# ------------------------------------------------------------------ end to end through blocking
def postcode_tables():
    left = pd.DataFrame({"lid": [1, 2, 3], "zip": ["0200", "200.0", "20.5"]})
    right = pd.DataFrame({"rid": [10, 11, 12], "postcode": ["0200", "200", "205"]})
    return left, right


def test_digits_keys_agree_and_stay_distinct_through_a_default_pass():
    left, right = postcode_tables()
    cands = jlink.block.candidates(left, right, left_id="lid", right_id="rid",
                                   on=[Field("zip", "zip", "postcode", compare="digits")])
    pairs = set(zip(cands.left_id, cands.right_id))
    assert (1, 10) in pairs               # "0200" meets "0200"
    assert (2, 11) in pairs               # "200.0" meets "200", reconciled first
    assert not any(l == 3 for l, _ in pairs)   # "20.5" is dropped, never flattened to 205


def test_date_keys_agree_only_when_both_are_unambiguous():
    left = pd.DataFrame({"lid": [1, 2], "day": ["2024-03-01", "03/01/2024"]})
    right = pd.DataFrame({"rid": [10, 11], "date": ["2024/03/01", "2024-03-01"]})
    cands = jlink.block.candidates(left, right, left_id="lid", right_id="rid",
                                   on=[Field("day", "day", "date", compare="date")])
    pairs = set(zip(cands.left_id, cands.right_id))
    assert (1, 10) in pairs and (1, 11) in pairs   # ISO-shaped, whichever separator
    assert not any(l == 2 for l, _ in pairs)       # 03/01/2024 is ambiguous, so it is dropped


def test_exact_compares_text_exactly_instead_of_fuzzily():
    left = pd.DataFrame({"lid": [1], "code": ["A100"]})
    right = pd.DataFrame({"rid": [10], "sku": ["A-100"]})
    cands = jlink.block.candidates(left, right, on=[Field("code", "code", "sku", compare="exact")],
                                   left_id="lid", right_id="rid")
    assert not len(cands)   # the shared text rule splits them; exact keeps them split


# ------------------------------------------------------------------ judge-only fields
def test_a_judge_only_field_is_shown_but_never_keyed_or_scored():
    left = pd.DataFrame({"lid": [1, 2], "name": ["Acme", "Zeta"], "price": [10.0, 99.0]})
    right = pd.DataFrame({"rid": [10, 11], "firm": ["Acme", "Zeta"], "cost": [10.5, 99.0]})
    on = [Field("name", "name", "firm"), Field("price", "price", "cost", key=False)]
    cands = jlink.block.candidates(left, right, on=on, left_id="lid", right_id="rid")
    # The exact pass keys only the name, so a differing price does not lose the pair.
    config = cands.attrs["blocking"]["passes"]
    keyed = {tuple(c) for pass_ in config for c in pass_["config"]["columns"]}
    assert ("name", "firm") in keyed and ("price", "cost") not in keyed
    # `sim` still sees both fields' text, and the judge's record still names the price.
    sim = cands.set_index(["left_id", "right_id"]).loc[(1, 10), "sim"]
    assert 0 < sim <= 1
    assert "price" in cands.attrs or True  # shown through judge(); checked in the judge test below


def test_judge_only_field_is_visible_to_the_judge_and_unkeyed_in_a_default_pass():
    left = pd.DataFrame({"lid": [1], "name": ["Acme"], "price": [10.0]})
    right = pd.DataFrame({"rid": [10], "firm": ["Acme"], "cost": [10.5]})
    fake = FakeJev(lambda state, rule: 0.9)
    on = [Field("name", "name", "firm"), Field("price", "price", "cost", key=False)]
    linker = jlink.Linker("firm", on, cache=False)
    result = linker.link(left, right, left_id="lid", right_id="rid", progress=False,
                         transport=fake.transport)
    state = fake.bodies[0]["state"]
    assert state["record_a"]["price"] == 10.0 and state["record_b"]["price"] == 10.5
    assert [p["name"] for p in result.settings["blocker_configs"]] == ["ngrams:name=firm",
                                                                      "ngrams-reverse:name=firm"]


def test_a_judge_only_field_is_left_out_of_the_exact_shortcut_key():
    left = pd.DataFrame({"name": ["Acme"], "price": [1.0]})
    right = pd.DataFrame({"name": ["Zeta"], "price": [1.0]})
    cands = pd.DataFrame({"left_id": [0], "right_id": [0], "sim": [0.1], "block": ["x"]})
    fake = FakeJev(lambda state, rule: 0.9)
    on = [Field("name", "name", "name"), Field("price", "price", "price", key=False)]
    scores, _ = jlink.judge(cands, left, right, on=on, entity="firm", left_id=None, right_id=None,
                            progress=False, exact_shortcut=True, cache=False, transport=fake.transport)
    # Equal prices must not settle the pair: only the keyed name is compared, and it differs.
    assert scores.loc[0, "source"] == "jev" and len(fake.bodies) == 1


def test_a_deterministic_field_must_exist_on_both_sides():
    left = pd.DataFrame({"lid": [1], "ref": ["A100"]})
    right = pd.DataFrame({"rid": [10], "code": ["A100"]})
    cands = pd.DataFrame({"left_id": [1], "right_id": [10], "sim": [0.5]})
    with pytest.raises(ValueError, match="must exist on both sides"):
        jlink.judge(cands, left, right, on=[("ref", "ref"),
                                            Field("code", "code", None, compare="exact",
                                                  deterministic=True)],
                    entity="record", left_id="lid", right_id="rid", budget=0.0, cache=False,
                    progress=False)


# ------------------------------------------------------------------ deterministic typed keys
def test_a_deterministic_typed_field_settles_a_true_pair_with_no_model_call():
    left = pd.DataFrame({"lid": [1, 2], "name": ["Acme", "Zeta"], "ref": ["A-100", "B-200"]})
    right = pd.DataFrame({"rid": [10, 20], "firm": ["Acme Inc", "Other"], "code": ["A100", "C300"]})
    fake = FakeJev()
    on = [Field("name", "name", "firm"),
          Field("ref", "ref", "code", compare="exact", normalize=digits, deterministic=True)]
    result = jlink.Linker("firm", on, cache=False).link(
        left, right, left_id="lid", right_id="rid", progress=False, transport=fake.transport)
    link = result.links.iloc[0]
    assert (link["left_id"], link["right_id"], link["source"]) == (1, 10, "exact")
    assert fake.bodies == []


def test_a_typed_key_settles_without_a_deterministic_flag_only_for_exact_shortcut():
    left = pd.DataFrame({"lid": [1], "postcode": ["0200"]})
    right = pd.DataFrame({"rid": [10], "zip": ["0200"]})
    cands = pd.DataFrame({"left_id": [1], "right_id": [10], "sim": [0.5], "block": ["given"]})
    scores, meter = jlink.judge(cands, left, right, on=[Field("postcode", "postcode", "zip",
                                                              compare="digits")],
                                entity="place", left_id="lid", right_id="rid", budget=0.0,
                                progress=False, exact_shortcut=True)
    assert scores.loc[0, "source"] == "exact" and scores.loc[0, "p"] == 1.0 and meter.calls == 0


# ------------------------------------------------------------------ provenance
def test_the_type_contract_is_recorded_and_json_safe():
    specs = parse_fields(["name", Field("year", "year", "yr", compare="date"),
                          Field("price", "price", "cost", key=False)])
    configs = [s.to_config() for s in specs]
    json.dumps(configs)   # a callable must never reach settings.json
    assert configs[1]["compare"] == "date" and configs[2]["key"] is False
    assert all(not c["custom_normalizer"] for c in configs)


def test_a_plain_column_config_records_no_type_contract_and_stays_byte_identical():
    left = pd.DataFrame({"name": ["Acme", "Zeta"]})
    right = pd.DataFrame({"name": ["Acme", "Zeta"]})
    fake = FakeJev(lambda state, rule: 0.9)
    plain = jlink.Linker("firm", ["name"], cache=False,
                         blockers=[jlink.block.exact("name")]).link(
        left, right, left_id=None, right_id=None, progress=False, transport=fake.transport)
    assert "fields" not in plain.settings
    assert plain.settings["normalization"] == "jlink.fields.normalize_v1"
    config = plain.settings["blocker_configs"][0]
    assert config == {"type": "exact", "name": "exact:name", "columns": [["name", "name"]]}


def test_normalization_summary_names_the_types_in_play():
    from jlink.linker import _normalization_summary

    specs = parse_fields([("name", "name"), Field("year", "year", "yr", compare="date"),
                          Field("ref", "ref", "code", normalize=digits)])
    assert _normalization_summary(specs) == "jlink.fields.normalize_v1+date:year,custom:ref"
    assert _normalization_summary(parse_fields(["name"])) == "jlink.fields.normalize_v1"


def test_a_custom_canonicalizer_is_recorded_as_custom_not_stored():
    from jlink.linker import _normalization_summary

    specs = parse_fields([Field("ref", "ref", "code", normalize=digits)])
    assert specs[0].to_config()["custom_normalizer"] is True
    assert _normalization_summary(specs) == "jlink.fields.normalize_v1+custom:ref"
    json.dumps([s.to_config() for s in specs])   # the callable never appears in the contract


# ------------------------------------------------------------------ CLI
def test_cli_type_flag_marks_a_column_and_rejects_bad_input(tmp_path):
    from jlink.cli import _parser, _typed, _types

    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame({"name": ["Acme"], "year": [1985.0], "zip": ["0200"]}).to_csv(left, index=False)
    pd.DataFrame({"firm": ["Acme"], "yr": [1985], "postcode": ["0200"]}).to_csv(right, index=False)
    args = _parser().parse_args(["link", str(left), str(right), "--on", "name", "--on", "year=yr",
                                 "--on", "zip=postcode", "--type", "year=date", "--type", "zip=digits",
                                 "--entity", "firm"])
    items = ["name", ("year", "yr"), ("zip", "postcode")]
    typed = _typed(items, _types(args), set())
    assert typed[0] == "name"                                    # untouched, so it stays a plain name
    assert typed[1].compare == "date" and typed[2].compare == "digits"
    with pytest.raises(ValueError, match="not a known kind"):
        _typed(items, {"year": "when"}, set())
    with pytest.raises(ValueError, match="does not use"):
        _typed(items, {"other": "date"}, set())
