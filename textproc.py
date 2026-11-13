"""
Pure text post-processing for WisperWin — no Qt, no audio, no I/O.
Extracted from wisperwin.py so the logic that gets tuned most often
(vocabulary correction, hallucination stripping) is unit-testable
without importing the heavy app module.
"""

import difflib
import re

# Whisper silence hallucinations — these phrases appear when the model gets
# quiet/empty audio. Strip them before processing or pasting.
HALLUCINATIONS = {
    "thank you", "thanks for watching", "thanks for watching!",
    "thank you for watching", "thank you for watching!",
    "you", ".", "...", "bye", "bye!", "goodbye",
    "please subscribe", "like and subscribe",
    "subtitles by", "transcribed by",
}

# Fuzzy-match tuning for single-word vocabulary correction.
FUZZY_RATIO_MIN = 0.82
FUZZY_WORD_MIN_LEN = 4


def strip_hallucination(text: str) -> str:
    t = text.lower().strip()
    # Check the raw form first — punctuation-only hallucinations like "..."
    # would otherwise rstrip down to "" and never match the set.
    if t in HALLUCINATIONS or t.rstrip(".!?,") in HALLUCINATIONS:
        return ""
    return text


def apply_vocabulary(text: str, vocab: list) -> str:
    """
    Post-processing: enforce exact spelling of vocabulary words.
    Handles two cases:
      1. Exact case-insensitive match  →  fix capitalisation
      2. Fuzzy match (ratio ≥ 0.82, word ≥ 4 chars)  →  fix spelling
         e.g. Whisper writes "Peter" but vocab has "Pieter"
    Multi-word phrases are matched first (longest to shortest), on word
    boundaries only — so "Claude Code" never rewrites part of "claude coder".
    """
    if not vocab or not text:
        return text

    # ── Multi-word phrases (replace whole phrase, case-insensitive) ───────
    multi = sorted([v for v in vocab if " " in v], key=len, reverse=True)
    for phrase in multi:
        text = re.sub(rf"\b{re.escape(phrase)}\b", phrase, text,
                      flags=re.IGNORECASE)

    # ── Single words ──────────────────────────────────────────────────────
    singles = [v for v in vocab if " " not in v]
    if not singles:
        return text

    # Tokenise preserving punctuation attached to words
    tokens = re.split(r"(\s+)", text)
    result = []
    for token in tokens:
        if not token.strip():          # whitespace token — keep as-is
            result.append(token)
            continue
        # Separate leading/trailing punctuation
        m = re.match(r"^([^\w]*)(\w[\w'-]*)([^\w]*)$", token)
        if not m:
            result.append(token)
            continue
        pre, word, post = m.group(1), m.group(2), m.group(3)

        best_vocab = None
        best_ratio = 0.0
        word_lower  = word.lower()
        for vw in singles:
            vw_lower = vw.lower()
            if word_lower == vw_lower:          # exact case-insensitive
                best_vocab  = vw
                best_ratio  = 1.0
                break
            if len(word) >= FUZZY_WORD_MIN_LEN and len(vw) >= FUZZY_WORD_MIN_LEN:
                ratio = difflib.SequenceMatcher(None, word_lower, vw_lower).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_vocab  = vw

        if best_vocab and best_ratio >= FUZZY_RATIO_MIN:
            result.append(pre + best_vocab + post)
        else:
            result.append(token)

    return "".join(result)
