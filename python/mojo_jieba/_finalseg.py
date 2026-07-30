"""Jieba-compatible BMES fallback for unknown Han-character runs."""

from __future__ import annotations

import re

from jieba.finalseg.prob_emit import P as emit_P
from jieba.finalseg.prob_start import P as start_P
from jieba.finalseg.prob_trans import P as trans_P

MIN_FLOAT = -3.14e100
PREV_STATUS = {"B": "ES", "M": "MB", "S": "SE", "E": "BM"}
FORCE_SPLIT_WORDS: set[str] = set()

_RE_HAN = re.compile(r"([\u4E00-\u9FD5]+)")
_RE_SKIP = re.compile(r"([a-zA-Z0-9]+(?:\.\d+)?%?)")


def _viterbi(sentence: str) -> list[str]:
    scores = {
        state: start_P[state] + emit_P[state].get(sentence[0], MIN_FLOAT)
        for state in "BMES"
    }
    paths = {state: [state] for state in "BMES"}
    for char in sentence[1:]:
        next_scores: dict[str, float] = {}
        next_paths: dict[str, list[str]] = {}
        for state in "BMES":
            emission = emit_P[state].get(char, MIN_FLOAT)
            score, previous = max(
                (
                    scores[prev] + trans_P[prev].get(state, MIN_FLOAT) + emission,
                    prev,
                )
                for prev in PREV_STATUS[state]
            )
            next_scores[state] = score
            next_paths[state] = paths[previous] + [state]
        scores, paths = next_scores, next_paths
    _, state = max((scores[state], state) for state in "ES")
    return paths[state]


def _cut_han(sentence: str):
    states = _viterbi(sentence)
    begin = 0
    next_index = 0
    for index, (char, state) in enumerate(zip(sentence, states)):
        if state == "B":
            begin = index
        elif state == "E":
            yield sentence[begin : index + 1]
            next_index = index + 1
        elif state == "S":
            yield char
            next_index = index + 1
    if next_index < len(sentence):
        yield sentence[next_index:]


def cut(sentence: str):
    for block in _RE_HAN.split(sentence):
        if _RE_HAN.fullmatch(block):
            for word in _cut_han(block):
                if word in FORCE_SPLIT_WORDS:
                    yield from word
                else:
                    yield word
        else:
            for item in _RE_SKIP.split(block):
                if item:
                    yield item


def add_force_split(word: str) -> None:
    FORCE_SPLIT_WORDS.add(word)
