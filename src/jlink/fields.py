"""How records are described: the `on` argument, and the normalized text used for similarity."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

On = list  # items are "column", ("left_column", "right_column"), or a Field
_PUNCT = re.compile(r"[^\w\s]|_")
_SPACE = re.compile(r"\s+")
_WHOLE_DECIMAL = re.compile(r"[+-]?\d+\.0+")

COMPARES = ("text", "exact", "number", "date")


@dataclass(frozen=True)
class Field:
    """One ``on`` item, with its own normalization, comparison rule and judging policy.

    A plain column name or ``(left, right)`` pair is equivalent to a Field with defaults, so
    existing ``on`` arguments mean exactly what they meant before. Use a Field to say that a
    column should be compared its own way -- for example a reference number compared by its digits,
    with ``digits_only`` imported from :mod:`jlink`:

        on=["name", Field("ref", "ref", "code", normalize=digits_only, compare="exact")]

    ``normalize`` replaces :func:`normalize` for this field everywhere the field is used: the
    blocking keys, the exact shortcut and the text shown to the judge. ``compare`` names the
    kind of comparison so default blocking can pick sensible passes. ``judge=False`` marks a
    field as deterministic: a pair may be settled from it without asking the model.
    """

    label: str
    left: str | None
    right: str | None
    normalize: Callable[[object], str] | None = None
    compare: str = "text"
    judge: bool = True

    @property
    def triple(self) -> tuple[str, str | None, str | None]:
        """The ``(label, left, right)`` view used by code that predates Field."""
        return (self.label, self.left, self.right)

    def normalizer(self) -> Callable[[object], str]:
        """This field's normalizer, falling back to the shared :func:`normalize`."""
        return self.normalize or normalize

    def to_config(self) -> dict:
        """JSON-safe description; a callable normalizer is recorded, not stored."""
        return {"label": self.label, "left": self.left, "right": self.right,
                "compare": self.compare, "judge": bool(self.judge),
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
            field = item
            if not isinstance(field.label, str) or not field.label:
                raise ValueError(f"a Field's label must be a nonempty column name; got {field.label!r}")
            if field.normalize is not None and not callable(field.normalize):
                raise ValueError(f"field {field.label!r}: `normalize` must be callable")
            if field.compare not in COMPARES:
                raise ValueError(f"field {field.label!r}: `compare` must be one of "
                                 f"{', '.join(COMPARES)}; got {field.compare!r}")
            if not isinstance(field.judge, bool):
                raise ValueError(f"field {field.label!r}: `judge` must be a boolean")
            if (field.left is None) and (field.right is None):
                raise ValueError(f"each `on` item is a column name, a (left, right) pair of names, or a "
                                 f"one-sided (left, None) or (None, right); got {(field.left, field.right)!r}")
            if field.left is None or field.right is None:
                if not unpaired:
                    raise ValueError(f"the field {(field.left, field.right)!r} exists on one side only, "
                                     "but this step compares a left column with a right column; "
                                     "one-sided fields are only shown to the judge")
            out.append(field)
        else:
            pair = tuple(item) if isinstance(item, (tuple, list)) and len(item) == 2 else ()
            if isinstance(item, str):
                out.append(Field(item, item, item))
            elif pair and all(isinstance(c, str) for c in pair):
                out.append(Field(pair[0], pair[0], pair[1]))
            elif pair and sum(c is None for c in pair) == 1 and any(isinstance(c, str) for c in pair):
                if not unpaired:
                    raise ValueError(f"the field {pair!r} exists on one side only, but this step compares a left "
                                     "column with a right column; one-sided fields are only shown to the judge")
                out.append(Field(pair[0] or pair[1], pair[0], pair[1]))
            else:
                raise ValueError("each `on` item is a column name, a (left, right) pair of names, or a one-sided "
                                 f"(left, None) or (None, right); got {item!r}")
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
        label, column = (field.label, field.left if index == 1 else field.right) \
            if isinstance(field, Field) else (field[0], field[index])
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


def key_text(value: object, normalizer: Callable[[object], str] | None = None) -> str:
    """Normalized text of one exact or group key component; empty if missing.

    Whole numbers agree however they are stored. A numeric column with one missing value is
    float in pandas, and a table written from it says "1985.0", so the integer 1985, the float
    1985.0 and the texts "1985" and "1985.0" all give "1985".

    A field's own ``normalizer`` replaces the text rule, and the whole-number reconciliation above
    is then skipped: the field's rule sees the original value, so it is not surprised by an int
    where it was given a string. The shared :func:`normalize` itself is not a custom rule, so
    naming it -- as the CLI's ``normalize`` rule does -- keeps the default behavior exactly.
    """
    if normalizer is not None and normalizer is not normalize:
        return "" if (cleaned := clean(value)) is None else normalizer(cleaned)
    value = clean(value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)  # clean() leaves whole floats from 1e15 up as floats; sixteen-digit keys exist
    elif isinstance(value, str) and _WHOLE_DECIMAL.fullmatch(value):
        value = value[:value.index(".")]
    return normalize(value)


def record_text(frame: pd.DataFrame, columns: list[str],
                normalizers: list[Callable[[object], str] | None] | None = None) -> pd.Series:
    """One normalized string per row: the listed columns joined by a space.

    Each column's own normalizer is used where one is given; without one the shared text rule
    applies, exactly as before.
    """
    normalizers = normalizers or [None] * len(columns)
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


# Named normalizers for callers that cannot pass a callable, such as a command line. Each
# takes a value and returns the string used for comparison. Keys are the names a CLI accepts.
def digits_only(value: object) -> str:
    """Keep only digits: 'A-100' -> '100'."""
    return re.sub(r"\D", "", str(value))


def phone_main(value: object, keep_last: int = 10) -> str:
    """A phone's main number: drop a trailing extension, then keep the last `keep_last` digits.

    The extension must be removed before digits are counted, or its digits are taken for the
    end of the number: '555-123-4567 ext 2222' would otherwise become '2345672222'.
    """
    text = re.sub(r"(?:e?xt?ension|ext|x|#)\s*[:.]?\s*\d{1,6}\s*$", " ", str(value), flags=re.I)
    text = re.sub(r"\((?!\d{3}\))[^)]*\)", " ", text)     # (cell), (home) -- not (555)
    digits = re.sub(r"\D", "", text)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits[-keep_last:] if len(digits) >= keep_last else digits


def phone_extension(value: object) -> str:
    """The extension digits of a phone string, or '' when it has none."""
    found = re.search(r"(?:e?xt?ension|ext|x|#)\s*[:.]?\s*(\d{1,6})\s*$", str(value), flags=re.I)
    return found.group(1) if found else ""


NORMALIZERS = {"normalize": normalize, "digits": digits_only, "phone": phone_main,
               "extension": phone_extension}
