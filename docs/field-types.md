# Field types and custom canonicalizers

A field's **type** says what kind of value it holds. The type supplies a *canonicalizer* -- the
rule that turns a value into the string used as a key -- and a default blocking pass. `text` is
the default, so a plain `on=["name"]` behaves exactly as it always has. The named types are the
basic defaults; anything beyond them is a callable you pass yourself.

| `compare=` | canonical key | default pass |
| --- | --- | --- |
| `text` | `normalize` (casefold, strip accents, `&`→`and`, drop punctuation) | n-grams, both directions |
| `exact` | `normalize` | exact key |
| `digits` | digit characters of a whole number, leading zeros kept | exact key |
| `date` | ISO 8601 as `YYYY`, `YYYY-MM` or `YYYY-MM-DD` | exact key |

```python
from jlink import Linker
from jlink.fields import Field

Linker(entity="person", on=[
    "name",
    Field("postcode", "zip", "postal_code", compare="digits"),
    Field("born", "date_of_birth", "dob", compare="date"),
    Field("price", "price", "cost", key=False),          # judge-only
    Field("ref", "ref", "code", normalize=my_rule),      # custom canonicalizer
])
```

## Drop, never guess

A canonicalizer returns the empty string for anything it cannot read **unambiguously**, and an
empty key never pairs. This mirrors the blocking window pass, whose policy is "nothing is
repaired: a value that does not parse is dropped and counted, never guessed." The reason is
asymmetric cost: a wrong canonicalization *silently merges* distinct records, and that error is
invisible downstream, whereas dropping a value only loses a match that a later pass or the judge
may still recover.

So `date` reads ISO-shaped, year-first dates only. `03/01/2024` is March 1st in one convention
and January 3rd in another; without being told, `date` leaves it unmatched rather than pick one.
`20.5` under `digits` is not a whole number, so it is dropped rather than flattened to `205`.
And a locale-specific format -- `1. März 2024` -- is never inferred.

Whole-number presentation *is* reconciled before the type rule runs, because `200.0` and `200`
are the same number however the column was stored, not a format convention: a numeric column with
one missing value becomes float in pandas and a table written from it says `"200.0"`. So
`digits` gives `"200.0"` and `200.0` and `"200"` the same key `"200"`, while `"0200"` stays
distinct because it has no fractional part and its leading zero carries information (postcodes).

## Writing a custom canonicalizer

A canonicalizer is any callable from a value to a string. Pass it as `normalize=`. It replaces the
type's rule everywhere the field is used: blocking keys, `sim`, the group keys and the judge's
text. A saved run records only that a custom rule was used, never the code.

The protocol, in one line:

```python
canonicalizer(value) -> str    # empty string means "drop this value"
```

### Worked example: a phone number, and a truncation trap

Phone numbers are a good example of why the library ships no phone canonicalizer: formatting
conventions are national, ambiguous and lossy, so a defensible rule depends on the data. Here is
one you can adapt. The trap it teaches is real and applies to every canonicalizer that keeps a
*suffix*.

```python
import re

def phone_main(value, keep_last=10):
    """A phone's main number: drop a trailing extension, then keep the last ten digits.

    The extension must be removed before digits are counted. Otherwise its digits are taken
    for the end of the number: '555-123-4567 ext 2222' would become '2345672222'.
    """
    text = re.sub(r"(?:e?xt?ension|ext|x|#)\s*[:.]?\s*\d{1,6}\s*$", " ", str(value), flags=re.I)
    text = re.sub(r"\((?!\d{3}\))[^)]*\)", " ", text)          # (cell), (home) -- not (555)
    digits = re.sub(r"\D", "", text)
    if len(digits) == 11 and digits.startswith("1"):           # a US country code
        digits = digits[1:]
    return digits[-keep_last:] if len(digits) >= keep_last else digits
```

Used as `Field("phone", "phone", "telephone", compare="exact", normalize=phone_main)`. Note that
this rule is US-centric (the country-code strip) and lossy (it discards a real extension), which
is exactly why it is documentation and not an export. If you need the extension as part of the
key, canonicalize it separately, or keep the country-code stripping out of a shared key. The
"drop, never guess" principle argues for returning `""` when a value does not look like a phone
number at all, rather than emitting whatever digits it happens to contain.

## Judge-only fields

`key=False` marks a field as shown to the judge but never keyed: it stays out of the blocking
passes' keys, out of `sim`, and out of the exact shortcut. This is the mechanism for a price,
a description or a note -- columns the benchmark layouts name in `on` so the judge can weigh them,
but which are not identifiers. The judge always sees the fields the caller named: `key=False`
changes only the keys, never the record sent to the model.

## Deterministic typed keys

`deterministic=True` on a typed field (`exact`, `digits` or `date`) lets an equal, present key
settle a pair with no model call, recorded as `source="exact"`. It must exist on both sides, and
the key is built from the marked fields only, so a differing name does not block a pair whose
typed key is conclusive. Use it only when that key establishes identity in your data: the whole
point of `exact_shortcut` and `deterministic` being opt-in is that equal text is usually not
enough.
