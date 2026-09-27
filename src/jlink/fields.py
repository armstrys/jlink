"""How records are described: the `on` argument, its types, and the text used for similarity.

A field names a *type* -- what kind of value it holds -- and the type supplies two things: a
canonicalizer, which turns a value into the string used for keys and comparison, and a default
blocking pass. The default type is ``text``, whose canonicalizer is :func:`normalize`; that keeps
an ``on`` argument of plain column names behaving exactly as it always has.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date as _date
from typing import Callable

import numpy as np
import pandas as pd

On = list  # items are "column", ("left_column", "right_column"), or a Field
_PUNCT = re.compile(r"[^\w\s]|_")
_SPACE = re.compile(r"\s+")
_WHOLE_DECIMAL = re.compile(r"[+-]?\d+\.0+")
_NON_DIGIT = re.compile(r"\D")
# A year-first date, optionally followed by a time. Year-first is ISO-shaped and unambiguous; a
# month-first or day-first form such as 03/01/2024 depends on a convention we refuse to guess.
_ISO_DATE = re.compile(r"^(\d{4})(?:([-/.])(\d{1,2})(?:\2(\d{1,2}))?)?(?:[T ].*)?$")

COMPARES = ("text", "exact", "digits", "date")


@dataclass(frozen=True)
class Field:
    """One ``on`` item, with its type and its own judging policy.

    A plain column name or ``(left, right)`` pair is equivalent to a Field with defaults, so an
    existing ``on`` argument means exactly what it meant before. A type is named by ``compare``:
    ``text`` (the default), ``exact`` (text matched exactly instead of fuzzily), ``digits``
    (postcode, ID, VAT, EAN) or ``date`` (ISO 8601). ``normalize`` replaces the type's
    canonicalizer with your own callable -- the escape hatch for anything the named types do not
    cover, such as a locale-specific date. ``key=False`` marks a field the judge sees but that is
    never keyed: a price, description or note. ``deterministic=True`` marks a typed field whose
    equal, present key may settle a pair without asking the model.
    """

    label: str
    left: str | None
    right: str | None
    compare: str = "text"
    normalize: Callable[[object], str] | None = None
    key: bool = True
    deterministic: bool = False

    @property
    def triple(self) -> tuple[str, str | None, str | None]:
        """The ``(label, left, right)`` view used by code that predates Field."""
        return (self.label, self.left, self.right)

    def canonicalizer(self) -> Callable[[object], str]:
        """The callable that turns this field's values into keys, custom rule first."""
        return self.normalize or TYPE_NORMALIZERS[self.compare]

    @property
    def is_default(self) -> bool:
        """True when nothing about this field departs from a plain column name."""
        return (self.compare == "text" and self.normalize is None and self.key
                and not self.deterministic)

    def to_config(self) -> dict:
        """A JSON-safe description; a custom canonicalizer is recorded, never stored."""
        return {"label": self.label, "left": self.left, "right": self.right, "compare": self.compare,
                "key": bool(self.key), "deterministic": bool(self.deterministic),
                "custom_normalizer": self.normalize is not None}


def parse_fields(on, *, unpaired: bool = False) -> list[Field]:
    """Every ``on`` item as a :class:`Field`.

    Accepts a column name, a ``(left, right)`` pair, a one-sided ``(left, None)`` or
    ``(None, right)``, or a Field, in any mixture. ``parse_on`` is the tuple view of this.
    """
    if isinstance(on, (str, Field)):
        on = [on]
    if not on:
        raise ValueError("`on` must name at least one field to compare")
    out: list[Field] = []
    for item in on:
        if isinstance(item, Field):
            _check_field(item, unpaired=unpaired)
            out.append(item)
            continue
        pair = tuple(item) if isinstance(item, (tuple, list)) and len(item) == 2 else ()
        match (isinstance(item, str), pair, sum(c is None for c in pair)):
            case (True, _, _):
                out.append(Field(item, item, item))
            case (_, (), _):
                raise ValueError("each `on` item is a column name, a (left, right) pair of names, or a "
                                 f"one-sided (left, None) or (None, right); got {item!r}")
            case (_, (left, right), 0) if all(isinstance(c, str) for c in pair):
                out.append(Field(left, left, right))
            case (_, (left, right), 1) if any(isinstance(c, str) for c in pair):
                if not unpaired:
                    raise ValueError(f"the field {pair!r} exists on one side only, but this step compares a "
                                     "left column with a right column; one-sided fields are only shown to "
                                     "the judge")
                out.append(Field(left or right, left, right))
            case _:
                raise ValueError("each `on` item is a column name, a (left, right) pair of names, or a "
                                 f"one-sided (left, None) or (None, right); got {item!r}")
    if any(f.left is None or f.right is None for f in out):
        for side, labels in (("left", [f.label for f in out if f.left is not None]),
                             ("right", [f.label for f in out if f.right is not None])):
            if not labels:
                raise ValueError(f"`on` gives the {side} records no field; "
                                 "the judge needs at least one per side")
            repeated = [label for label in labels if labels.count(label) > 1]
            if repeated:
                raise ValueError(f"`on` would show the {side} record two fields labeled {repeated[0]!r}; "
                                 "paired fields are labeled by their left column, so rename a column")
    return out


def _check_field(field: Field, *, unpaired: bool) -> None:
    """Validate one caller-supplied Field before it is used."""
    if not isinstance(field.label, str) or not field.label:
        raise ValueError(f"a Field's label must be a nonempty column name; got {field.label!r}")
    if field.normalize is not None and not callable(field.normalize):
        raise ValueError(f"field {field.label!r}: `normalize` must be callable")
    if field.compare not in COMPARES:
        raise ValueError(f"field {field.label!r}: `compare` must be one of {', '.join(COMPARES)}; "
                         f"got {field.compare!r}")
    if not isinstance(field.key, bool):
        raise ValueError(f"field {field.label!r}: `key` must be a boolean")
    if not isinstance(field.deterministic, bool):
        raise ValueError(f"field {field.label!r}: `deterministic` must be a boolean")
    if field.deterministic and not field.key:
        raise ValueError(f"field {field.label!r}: a `key=False` field is shown only to the judge, so it "
                         "cannot be deterministic")
    if field.deterministic and field.compare == "text":
        raise ValueError(f"field {field.label!r}: `deterministic` needs a typed `compare` (exact, digits "
                         "or date), because text alone does not establish identity")
    if field.left is None and field.right is None:
        raise ValueError(f"each `on` item is a column name, a (left, right) pair of names, or a "
                         f"one-sided (left, None) or (None, right); got {(field.left, field.right)!r}")
    if (field.left is None or field.right is None) and not unpaired:
        raise ValueError(f"the field {(field.left, field.right)!r} exists on one side only, but this step "
                         "compares a left column with a right column; one-sided fields are only shown to "
                         "the judge")


def parse_on(on, *, unpaired: bool = False) -> list[tuple[str, str | None, str | None]]:
    """[(label, left_column, right_column), ...]. The label is the left column name.

    With ``unpaired=True`` an item may also be ``(left, None)`` or ``(None, right)``: a field that
    only one side has. It is shown to the judge under its own column name; the absent side is None.
    Steps that compare a left column with a right column keep the default and reject such items.
    """
    return [field.triple for field in parse_fields(on, unpaired=unpaired)]


def side_fields(fields: list, side: str) -> list[tuple[str, str]]:
    """[(label, column), ...] for the fields that the left or the right records have.

    Accepts :class:`Field` objects or the older ``(label, left, right)`` triples.
    """
    index = 1 if side == "left" else 2
    out = []
    for field in fields:
        if isinstance(field, Field):
            label, column = field.label, (field.left if index == 1 else field.right)
        else:
            label, column = field[0], field[index]
        if column is not None:
            out.append((label, column))
    return out


def check_columns(frame: pd.DataFrame, columns: list[str], side: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"the {side} data has no column {missing[0]!r}; its columns are {list(frame.columns)}")


def normalize(text: object) -> str:
    """Casefold, strip accents, turn & into 'and', drop punctuation, collapse whitespace."""
    if text is None or (isinstance(text, float) and text != text) or text is pd.NA:
        return ""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold().replace("&", " and ")
    return _SPACE.sub(" ", _PUNCT.sub(" ", s)).strip()


def clean(value):
    """A JSON-ready value, or None if missing. Whole floats become ints: a year read as 1985.0 is 1985."""
    if isinstance(value, (np.generic,)):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value) if value.is_integer() and abs(value) < 1e15 else value
    if isinstance(value, (bool, int)):
        return value
    text = str(value).strip()
    return text or None


def reconcile(value):
    """Whole-number presentation as one string, before a type rule sees it: 1985.0 and "1985.0" -> 1985.

    A numeric column with one missing value is float in pandas, and a table written from it says
    "1985.0". Reconciling first is what lets ``digits("200.0")`` be ``"200"`` rather than ``"2000"``.
    """
    value = clean(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and _WHOLE_DECIMAL.fullmatch(value):
        return value[:value.index(".")]
    return value


def key_text(value: object, normalizer: Callable[[object], str] | None = None) -> str:
    """One key component: whole-number presentation reconciled, then the field's rule applied.

    ``normalizer`` is the field's canonicalizer. Without one the shared text rule applies, so a
    plain column's key is built exactly as before.
    """
    value = reconcile(value)
    if value is None:
        return ""
    return (normalizer or normalize)(value)


def digits(value: object) -> str:
    """The digits of a whole number, leading zeros kept; anything with a fractional part is dropped.

    ``"0200"`` stays ``"0200"``, so postcodes beginning with a zero are not folded into other
    postcodes. A genuine fraction is not a whole number: ``20.5`` returns empty rather than ``"205"``.
    """
    value = reconcile(value)
    if value is None:
        return ""
    text = str(value)
    if "." in text:
        return ""
    return _NON_DIGIT.sub("", text)


def date_iso(value: object) -> str:
    """An ISO 8601 date as ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD``; anything ambiguous is empty.

    A year alone canonicalizes to its own four digits. A month-first or day-first form such as
    ``03/01/2024`` has no single meaning without a convention, so it is dropped, never guessed;
    a caller who needs one passes their own ``normalize`` callable.
    """
    value = reconcile(value)
    if value is None:
        return ""
    found = _ISO_DATE.match(str(value).strip())
    if not found:
        return ""
    year, _, month, day = found.groups()
    if month is None:
        return year
    if day is None:
        return f"{year}-{int(month):02d}"
    try:
        return _date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return ""


# Each named type's canonicalizer. `text` and `exact` share the text rule and differ only in the
# default pass; the CLI's names are these keys.
TYPE_NORMALIZERS: dict[str, Callable[[object], str]] = {
    "text": normalize, "exact": normalize, "digits": digits, "date": date_iso}
# Each named type's default blocking pass: text searches n-grams, every other type keys exactly.
TYPE_PASSES = {"text": "ngrams", "exact": "exact", "digits": "exact", "date": "exact"}


def record_text(frame: pd.DataFrame, columns: list[str],
                normalizers: list[Callable[[object], str] | None] | None = None) -> pd.Series:
    """One text per row: the listed columns, each canonicalized, joined by a space.

    Each column's own canonicalizer is used where one is given; without one the shared text rule
    applies, so plain columns score exactly as before.
    """
    normalizers = normalizers or [None] * len(columns)
    if not columns:
        return pd.Series([""] * len(frame), index=frame.index)
    # astype(object) keeps the .str accessor working when a frame is empty or a column is all-missing floats
    parts = [frame[c].astype(object).map(n or normalize).astype(object)
             for c, n in zip(columns, normalizers)]
    text = parts[0]
    for part in parts[1:]:
        text = text.str.cat(part, sep=" ")
    return text.str.strip()


def ids(frame: pd.DataFrame, id_column: str | None, side: str) -> pd.Index:
    """The frame's IDs, checked for uniqueness."""
    if id_column is None:
        values = frame.index
    else:
        check_columns(frame, [id_column], side)
        values = pd.Index(frame[id_column])
    if values.has_duplicates:
        where = f"column {id_column!r}" if id_column else "index"
        raise ValueError(f"the {side} data's {where} has duplicate IDs; IDs must be unique")
    return values
