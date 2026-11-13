"""Unit tests for WisperWin's pure text/hotkey helpers.

Run from the WisperWin folder:  python -m unittest discover tests -v
(stdlib only — works with any Python 3.10+, no venv needed)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from textproc import apply_vocabulary, strip_hallucination  # noqa: E402
from hotkey_util import parse_hotkey, parse_hotkey_safe, MOD_CONTROL, MOD_SHIFT  # noqa: E402


class TestStripHallucination(unittest.TestCase):
    def test_strips_known_hallucinations(self):
        self.assertEqual(strip_hallucination("Thank you."), "")
        self.assertEqual(strip_hallucination("thanks for watching!"), "")
        self.assertEqual(strip_hallucination("you"), "")
        self.assertEqual(strip_hallucination("..."), "")

    def test_keeps_real_text(self):
        self.assertEqual(strip_hallucination("Thank you for the update on Kubernetes."),
                         "Thank you for the update on Kubernetes.")
        self.assertEqual(strip_hallucination("Bye for now, more tomorrow."),
                         "Bye for now, more tomorrow.")

    def test_empty(self):
        self.assertEqual(strip_hallucination(""), "")


class TestApplyVocabulary(unittest.TestCase):
    VOCAB = ["GitHub", "Kubernetes", "Claude Code", "Pieter", "Figma", "PM"]

    def test_no_vocab_passthrough(self):
        self.assertEqual(apply_vocabulary("hello world", []), "hello world")

    def test_exact_case_fix(self):
        self.assertEqual(apply_vocabulary("we use figma daily", self.VOCAB),
                         "we use Figma daily")

    def test_fuzzy_spelling_fix(self):
        # "Peter" vs vocab "Pieter" — SequenceMatcher ratio 0.909 ≥ 0.82
        self.assertEqual(apply_vocabulary("ask Peter about it", self.VOCAB),
                         "ask Pieter about it")

    def test_punctuation_preserved(self):
        self.assertEqual(apply_vocabulary("Is it github?", self.VOCAB),
                         "Is it GitHub?")

    def test_multiword_phrase_case_fix(self):
        self.assertEqual(apply_vocabulary("open claude code now", self.VOCAB),
                         "open Claude Code now")

    def test_multiword_respects_word_boundaries(self):
        # Regression: without \b anchors, "claude coder" was rewritten from
        # inside — the phrase must only match whole words.
        self.assertEqual(apply_vocabulary("the claude coders meetup", self.VOCAB),
                         "the claude coders meetup")

    def test_short_words_exact_only(self):
        # "PM" is under the 4-char fuzzy floor: exact matches fix case,
        # near-misses are left alone.
        self.assertEqual(apply_vocabulary("ping the pm", self.VOCAB),
                         "ping the PM")
        self.assertEqual(apply_vocabulary("it went pfft", self.VOCAB),
                         "it went pfft")

    def test_empty_text(self):
        self.assertEqual(apply_vocabulary("", self.VOCAB), "")


class TestParseHotkey(unittest.TestCase):
    def test_default(self):
        hk = parse_hotkey("ctrl+space")
        self.assertEqual(hk["mods"], {"ctrl"})
        self.assertEqual(hk["vk"], 0x20)
        self.assertEqual(hk["mod_flags"], MOD_CONTROL)
        self.assertEqual(hk["label"], "Ctrl+Space")

    def test_multi_modifier(self):
        hk = parse_hotkey("ctrl+shift+space")
        self.assertEqual(hk["mods"], {"ctrl", "shift"})
        self.assertEqual(hk["mod_flags"], MOD_CONTROL | MOD_SHIFT)

    def test_function_key(self):
        hk = parse_hotkey("alt+f9")
        self.assertEqual(hk["vk"], 0x78)   # VK_F9
        self.assertEqual(hk["label"], "Alt+F9")

    def test_letter_key(self):
        self.assertEqual(parse_hotkey("win+d")["vk"], ord("D"))

    def test_case_and_alias_tolerance(self):
        hk = parse_hotkey("Control+SPACE")
        self.assertEqual(hk["mods"], {"ctrl"})

    def test_invalid_raises(self):
        for bad in ("", "ctrl+", "ctrl", "ctrl+bananas", "ctrl+shift", None):
            with self.assertRaises((ValueError, TypeError, AttributeError)):
                parse_hotkey(bad)

    def test_safe_fallback(self):
        self.assertEqual(parse_hotkey_safe("garbage+++")["label"], "Ctrl+Space")
        self.assertEqual(parse_hotkey_safe(None)["label"], "Ctrl+Space")


if __name__ == "__main__":
    unittest.main()
