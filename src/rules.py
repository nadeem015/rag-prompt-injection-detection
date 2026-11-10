"""Transparent, explainable rule-based prompt-injection indicators.

DESIGN NOTES (research integrity)
---------------------------------
* The rules below were written from general knowledge of published attack
  patterns (OWASP LLM01 examples, BIPIA / InjecAgent attack styles) BEFORE the
  final-experiment texts were inspected, and were then frozen.  The SHA-256 of
  this file is recorded in results/final/manifests/ so any later edit is visible.
* This is a RE-IMPLEMENTATION.  The author's original ``src/detector.py`` (used
  in the early project phases) was not available when this package was built.
* Rules are English-only lexical patterns.  They cannot catch paraphrase,
  other languages, or semantically disguised instructions.
* Score = noisy-OR over matched indicators:  1 - prod(1 - w_i).
  A text is flagged when score >= RULE_THRESHOLD (default 0.50).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List

RULE_THRESHOLD = 0.50

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)


def normalize(text: str) -> str:
    """NFKC-normalise, drop zero-width characters, collapse whitespace."""
    t = unicodedata.normalize("NFKC", text or "").translate(_ZERO_WIDTH)
    return re.sub(r"\s+", " ", t).strip()


# (category, name, weight, compiled regex)
_I = re.IGNORECASE
_RAW = [
    # --- instruction override -------------------------------------------------
    ("instruction_override", "ignore_previous", 0.80,
     r"\b(ignore|disregard|forget|override|bypass|discard)\b[^.\n]{0,40}\b(previous|prior|above|earlier|preceding|all|any|your|the)\b[^.\n]{0,30}\b(instructions?|rules?|prompts?|guidelines?|directives?|polic(?:y|ies)|constraints?|context)\b"),
    ("instruction_override", "new_instructions", 0.55,
     r"\b(new|updated|revised|replacement)\s+(instructions?|rules?|directives?)\s*[:\-]"),
    ("instruction_override", "from_now_on", 0.35,
     r"\bfrom now on\b[^.\n]{0,60}\b(you|always|only|never|must|will)\b"),
    ("instruction_override", "wipe_rules", 0.60,
     r"\b(wipe|erase|clear|reset|delete|replace)\b[^.\n]{0,30}\b(memory|rules?|instructions?|conflicting|prior|previous)\b"),
    # --- role hijack / jailbreak ----------------------------------------------
    ("role_hijack", "you_are_now", 0.50,
     r"\byou are (now|no longer)\b"),
    ("role_hijack", "no_restrictions", 0.70,
     r"\b(without|no|free of|ignoring)\s+(any\s+)?(restrictions?|limitations?|filters?|safety|guardrails?|censorship)\b"),
    ("role_hijack", "jailbreak_terms", 0.75,
     r"\b(jailbreak(?:ed)?|developer mode|DAN mode|do anything now|god mode|unfiltered mode)\b"),
    ("role_hijack", "pretend_unrestricted", 0.45,
     r"\b(pretend|act|behave)\b[^.\n]{0,25}\b(as if|you have no|unrestricted|without)\b"),
    # --- system prompt exfiltration -------------------------------------------
    ("prompt_exfiltration", "reveal_system_prompt", 0.80,
     r"\b(reveal|print|show|display|repeat|output|leak|disclose|tell me)\b[^.\n]{0,40}\b(system prompt|hidden (?:instructions?|prompt)|initial (?:instructions?|prompt)|your (?:instructions?|rules)|developer message)\b"),
    # --- data exfiltration -----------------------------------------------------
    ("data_exfiltration", "send_to_destination", 0.65,
     r"\b(send|forward|email|e-mail|post|upload|transmit|exfiltrate|leak)\b[^.\n]{0,60}\b(to|at)\b[^.\n]{0,30}(https?://|www\.|[\w.+-]+@[\w-]+\.[\w.]+)"),
    ("data_exfiltration", "secret_terms", 0.35,
     r"\b(api[ _-]?keys?|passwords?|credentials?|secret tokens?|private keys?|session cookies?|access tokens?)\b[^.\n]{0,40}\b(send|include|append|forward|reveal|share|output)\b"),
    ("data_exfiltration", "markdown_image_url", 0.60,
     r"!\[[^\]]*\]\(\s*https?://[^)\s]*[?&=][^)]*\)"),
    # --- authority / provenance spoofing --------------------------------------
    ("authority_claim", "fake_system_tag", 0.60,
     r"(^|\s)(\[\s*system\s*\]|<\s*system\s*>|system\s*:|### ?system|admin(?:istrator)?\s+(?:override|notice|message))"),
    ("authority_claim", "authorised_claim", 0.40,
     r"\b(authori[sz]ed by|approved by|official notice from|message from)\b[^.\n]{0,30}\b(admin|administrator|developer|security team|system|operator|openai|anthropic)\b"),
    ("authority_claim", "priority_claim", 0.35,
     r"\b(highest priority|top priority|takes precedence|supersedes?|overrid(?:es|ing))\b[^.\n]{0,40}\b(all|any|previous|other|system)\b"),
    # --- tool / action abuse ---------------------------------------------------
    ("tool_abuse", "invoke_tool", 0.45,
     r"\b(call|invoke|use|trigger|run|execute)\b[^.\n]{0,25}\b(the )?(function|tool|api|command|script|shell)\b[^.\n]{0,40}\b(with|using|to)\b"),
    ("tool_abuse", "destructive_action", 0.55,
     r"\b(delete|remove|drop|wipe|erase|truncate)\b[^.\n]{0,25}\b(all|every|entire)\b[^.\n]{0,25}\b(files?|records?|data|emails?|tables?|database|messages?)\b"),
    ("tool_abuse", "money_transfer", 0.55,
     r"\b(transfer|wire|send|pay)\b[^.\n]{0,20}\b(funds?|money|\$\s?\d|payment)\b"),
    # --- output / behaviour control -------------------------------------------
    ("output_control", "respond_only_with", 0.40,
     r"\b(respond|reply|answer|acknowledge|output)\b[^.\n]{0,10}\bonly (?:with|by saying)\b"),
    ("output_control", "conceal_from_user", 0.65,
     r"\b(do not|don't|never|without)\b[^.\n]{0,20}\b(tell|inform|mention|notify|alert|reveal)\b[^.\n]{0,25}\b(user|human|operator|anyone)\b"),
    ("output_control", "hidden_behaviour", 0.35,
     r"\b(silently|secretly|covertly|quietly)\b"),
    # --- chat-template / delimiter forgery ------------------------------------
    ("delimiter_forgery", "chat_tokens", 0.70,
     r"(<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[INST\]|\[/INST\]|<<SYS>>|<\|endoftext\|>)"),
    # --- obfuscation -----------------------------------------------------------
    ("obfuscation", "decode_and_follow", 0.55,
     r"\b(decode|decipher|rot13|base64)\b[^.\n]{0,40}\b(and )?(follow|execute|obey|run|apply)\b"),
]

INDICATORS = [(c, n, w, re.compile(p, _I)) for c, n, w, p in _RAW]


@dataclass
class RuleResult:
    score: float
    flagged: bool
    matches: List[dict] = field(default_factory=list)

    @property
    def categories(self) -> List[str]:
        return sorted({m["category"] for m in self.matches})


class RuleDetector:
    """Explainable rule detector (noisy-OR of weighted regex indicators)."""

    def __init__(self, threshold: float = RULE_THRESHOLD):
        self.threshold = float(threshold)

    def detect(self, text: str) -> RuleResult:
        t = normalize(text)
        matches, keep = [], 1.0
        for cat, name, w, rx in INDICATORS:
            m = rx.search(t)
            if m:
                matches.append({"category": cat, "indicator": name, "weight": w,
                                "span": m.group(0)[:120]})
                keep *= (1.0 - w)
        score = 1.0 - keep
        return RuleResult(score=score, flagged=score >= self.threshold, matches=matches)
