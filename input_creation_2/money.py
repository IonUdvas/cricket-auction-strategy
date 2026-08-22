"""
One parser for auction money, in lakh.

There were two, and they disagreed.

    auction_replay_engine._parse_price
        re.match(r"([\\d.]+)\\s*(CR|CRORE|L|LAKH)?", value.upper())
        -- unit OPTIONAL, so a bare "13.00" silently becomes 13 lakh
           rather than raising, and "13.00 Crore" parses as 13 lakh
           because the regex is not anchored at the end and (CR) is
           matched by the leading "Cr" of "Crore"... which happens to
           be right, by luck, for that spelling only.

    player_features.demographics._parse_money
        re.match(r"([\\d.]+)\\s*(cr|l)\\b", text, re.I)
        -- unit REQUIRED with a word boundary, so "13.00 Crore" fails
           the regex (\\b between "r" and "o" does not hold), falls
           through to float("13.00 Crore"), raises ValueError, and
           returns None. A silently dropped salary.

So the same string could become 1300 lakh in the replay engine and
None in the salary history. Neither spelling occurs in the shipped
files today -- I checked all 278 distinct `Amount` values and every
`basePrice` / `auctionPrice` across the nine editions, and every one
is "<number> Cr", "<number> L" or "--" -- so this is latent rather
than active. It is worth closing anyway: the failure is silent in
both directions, and the two functions are edited independently.

Rules, in one place:

  * "2.00 Cr" / "2 CRORE" / "2cr"      -> 200.0
  * "30.00 L" / "30 LAKH" / "30l"      -> 30.0
  * "--", "", None, NaN                -> None
  * a bare number                      -> lakh, but only when
                                          `require_unit` is False
  * anything else                      -> None, and recorded

`require_unit=True` is the safer default for new call sites. The two
existing call sites pass False, because both currently accept bare
numbers and changing that is a behaviour change rather than a fix.
"""

from __future__ import annotations

import re

import pandas as pd

LAKH_PER_CRORE = 100.0

# Anchored at BOTH ends. The un-anchored version is what let
# "13.00 Crore" match as "13.00 Cr" plus trailing junk.
_MONEY = re.compile(
    r"^\s*([\d]+(?:\.\d+)?)\s*(CR|CRORE|CRORES|L|LAKH|LAKHS|LACS)?\s*$",
    re.IGNORECASE,
)

_CRORE_UNITS = {"CR", "CRORE", "CRORES"}
_LAKH_UNITS = {"L", "LAKH", "LAKHS", "LACS"}

# The spellings that mean "no price", as opposed to "unparseable".
_NULL_TOKENS = {"", "--", "-", "N/A", "NA", "NAN", "NONE", "TBD"}


def parse_money(value, require_unit=False, unparseable=None):
    """
    Money string -> float lakh, or None.

    `unparseable` : optional list. Anything that is neither a null
        token nor a valid amount is appended to it. Pass one in to
        report what was dropped instead of dropping it in silence --
        which is what both original implementations did.
    """
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "")

    if text.upper() in _NULL_TOKENS:
        return None

    match = _MONEY.match(text)
    if match is None:
        if unparseable is not None:
            unparseable.append(value)
        return None

    amount = float(match.group(1))
    unit = (match.group(2) or "").upper()

    if unit in _CRORE_UNITS:
        return amount * LAKH_PER_CRORE

    if unit in _LAKH_UNITS:
        return amount

    # No unit at all.
    if require_unit:
        if unparseable is not None:
            unparseable.append(value)
        return None

    return amount
