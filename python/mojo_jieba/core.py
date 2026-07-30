"""Python-facing jieba API backed by the Mojo route-selection kernel."""

from __future__ import annotations

import importlib.resources
import logging
import math
import os
import re
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from . import _finalseg
from ._lib import route as native_route
from ._lib import route_batch as native_route_batch

DEFAULT_DICT = None
DEFAULT_DICT_NAME = "dict.txt"

_RE_USERDICT = re.compile(r"^(.+?)( [0-9]+)?( [a-z]+)?$", re.U)
_RE_ENG = re.compile(r"[a-zA-Z0-9]", re.U)
_RE_HAN_DEFAULT = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&._%\-]+)", re.U)
_RE_SKIP_DEFAULT = re.compile(r"(\r\n|\s)", re.U)

logger = logging.getLogger("mojo_jieba")


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _default_dict_path() -> str:
    return os.fspath(importlib.resources.files("jieba").joinpath(DEFAULT_DICT_NAME))


@dataclass(frozen=True)
class _Vocabulary:
    edge_chars: np.ndarray
    edge_offsets: np.ndarray
    edge_children: np.ndarray
    node_weights: np.ndarray
    root_children: np.ndarray
    max_length: int


class Tokenizer:
    def __init__(self, dictionary=DEFAULT_DICT):
        self.lock = threading.RLock()
        self.dictionary = (
            DEFAULT_DICT if dictionary == DEFAULT_DICT else os.path.abspath(dictionary)
        )
        self.FREQ: dict[str, int] = {}
        self.total = 0
        self.user_word_tag_tab: dict[str, str] = {}
        self.initialized = False
        self.tmp_dir = None
        self.cache_file = None
        self._vocabulary: _Vocabulary | None = None

    def __repr__(self):
        return f"<Tokenizer dictionary={self.dictionary!r}>"

    @staticmethod
    def gen_pfdict(file) -> tuple[dict[str, int], int]:
        frequencies: dict[str, int] = {}
        total = 0
        for lineno, raw_line in enumerate(file, 1):
            try:
                line = raw_line.strip()
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                word, frequency = line.split(" ")[:2]
                frequency = int(frequency)
            except ValueError as error:
                raise ValueError(
                    f"invalid dictionary entry at Line {lineno}: {raw_line!r}"
                ) from error
            frequencies[word] = frequency
            total += frequency
            for length in range(1, len(word) + 1):
                frequencies.setdefault(word[:length], 0)
        file.close()
        return frequencies, total

    def get_dict_file(self):
        path = (
            _default_dict_path()
            if self.dictionary == DEFAULT_DICT
            else self.dictionary
        )
        return open(path, "rb")

    def initialize(self, dictionary=None):
        if dictionary:
            absolute = os.path.abspath(dictionary)
            if self.dictionary == absolute and self.initialized:
                return
            self.dictionary = absolute
            self.initialized = False
        with self.lock:
            if self.initialized:
                return
            self.FREQ, self.total = self.gen_pfdict(self.get_dict_file())
            self._vocabulary = None
            self.initialized = True

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def _build_vocabulary(self) -> _Vocabulary:
        words = list(self.FREQ.items())
        node_index = {word: index + 1 for index, (word, _) in enumerate(words)}
        parents = np.fromiter(
            (
                node_index[word[:-1]] if len(word) > 1 else 0
                for word, _ in words
            ),
            dtype=np.int64,
        )
        edge_chars = np.fromiter(
            (ord(word[-1]) for word, _ in words), dtype=np.uint32
        )
        edge_children = np.arange(1, len(words) + 1, dtype=np.int64)
        root_children = np.full(65_536, -1, dtype=np.int64)
        root_edges = (parents == 0) & (edge_chars <= 65_535)
        root_children[edge_chars[root_edges]] = edge_children[root_edges]
        order = np.lexsort((edge_chars, parents))
        edge_chars = np.ascontiguousarray(edge_chars[order])
        edge_children = np.ascontiguousarray(edge_children[order])
        parents = parents[order]
        edge_offsets = np.empty(len(words) + 2, dtype=np.int64)
        edge_offsets[0] = 0
        np.cumsum(
            np.bincount(parents, minlength=len(words) + 1),
            out=edge_offsets[1:],
        )
        log_total = math.log(self.total)
        node_weights = np.empty(len(words) + 1, dtype=np.float64)
        node_weights[0] = -1.0e300
        node_weights[1:] = np.fromiter(
            (
                math.log(freq) - log_total if freq > 0 else -1.0e300
                for _, freq in words
            ),
            dtype=np.float64,
        )
        return _Vocabulary(
            edge_chars,
            edge_offsets,
            edge_children,
            node_weights,
            root_children,
            max((len(word) for word, _ in words), default=1),
        )

    def _ensure_vocabulary(self) -> _Vocabulary:
        self.check_initialized()
        if self._vocabulary is None:
            with self.lock:
                if self._vocabulary is None:
                    self._vocabulary = self._build_vocabulary()
        return self._vocabulary

    def _route(self, sentence: str) -> tuple[np.ndarray, np.ndarray]:
        vocabulary = self._ensure_vocabulary()
        chars = np.frombuffer(sentence.encode("utf-32-le"), dtype=np.uint32)
        route = np.empty(len(chars), dtype=np.int64)
        scores = np.empty(len(chars) + 1, dtype=np.float64)
        native_route(
                chars,
                vocabulary.edge_chars,
                vocabulary.edge_offsets,
                vocabulary.edge_children,
                vocabulary.node_weights,
                vocabulary.root_children,
                vocabulary.max_length,
                -math.log(self.total),
                route,
                scores,
        )
        return route, scores

    def _route_blocks(
        self, sentence: str, spans: list[tuple[int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        vocabulary = self._ensure_vocabulary()
        chars = np.frombuffer(sentence.encode("utf-32-le"), dtype=np.uint32)
        route = np.empty(len(chars), dtype=np.int64)
        scores = np.empty(len(chars) + 1, dtype=np.float64)
        if not spans:
            scores[0] = 0.0
            return route, scores
        starts = np.fromiter((start for start, _ in spans), dtype=np.int64)
        ends = np.fromiter((end for _, end in spans), dtype=np.int64)
        native_route_batch(
            chars,
            starts,
            ends,
            vocabulary.edge_chars,
            vocabulary.edge_offsets,
            vocabulary.edge_children,
            vocabulary.node_weights,
            vocabulary.root_children,
            vocabulary.max_length,
            -math.log(self.total),
            route,
            scores,
        )
        return route, scores

    def get_DAG(self, sentence):
        self.check_initialized()
        dag = {}
        for start in range(len(sentence)):
            ends = []
            end = start
            fragment = sentence[start]
            while end < len(sentence) and fragment in self.FREQ:
                if self.FREQ[fragment]:
                    ends.append(end)
                end += 1
                fragment = sentence[start : end + 1]
            dag[start] = ends or [start]
        return dag

    def calc(self, sentence, DAG, route):
        route[len(sentence)] = (0, 0)
        log_total = math.log(self.total)
        for index in range(len(sentence) - 1, -1, -1):
            route[index] = max(
                (
                    math.log(self.FREQ.get(sentence[index : end + 1]) or 1)
                    - log_total
                    + route[end + 1][0],
                    end,
                )
                for end in DAG[index]
            )

    def _cut_dag_no_hmm(
        self,
        sentence: str,
        route: np.ndarray | None = None,
        offset: int = 0,
    ):
        if route is None:
            route, _ = self._route(sentence)
        index = 0
        buffer = ""
        while index < len(sentence):
            end = int(route[offset + index]) + 1 - offset
            word = sentence[index:end]
            if _RE_ENG.match(word) and len(word) == 1:
                buffer += word
            else:
                if buffer:
                    yield buffer
                    buffer = ""
                yield word
            index = end
        if buffer:
            yield buffer

    def _cut_dag(
        self,
        sentence: str,
        route: np.ndarray | None = None,
        offset: int = 0,
    ):
        if route is None:
            route, _ = self._route(sentence)
        index = 0
        buffer = ""
        while index < len(sentence):
            end = int(route[offset + index]) + 1 - offset
            word = sentence[index:end]
            if end - index == 1:
                buffer += word
            else:
                if buffer:
                    if len(buffer) == 1:
                        yield buffer
                    elif not self.FREQ.get(buffer):
                        yield from _finalseg.cut(buffer)
                    else:
                        yield from buffer
                    buffer = ""
                yield word
            index = end
        if buffer:
            if len(buffer) == 1:
                yield buffer
            elif not self.FREQ.get(buffer):
                yield from _finalseg.cut(buffer)
            else:
                yield from buffer

    def _cut_all(self, sentence: str):
        dag = self.get_DAG(sentence)
        old_end = -1
        eng_scan = False
        eng_buffer = ""
        for start, ends in dag.items():
            if eng_scan and not _RE_ENG.match(sentence[start]):
                eng_scan = False
                yield eng_buffer
            if len(ends) == 1 and start > old_end:
                word = sentence[start : ends[0] + 1]
                if _RE_ENG.match(word):
                    if not eng_scan:
                        eng_scan = True
                        eng_buffer = word
                    else:
                        eng_buffer += word
                if not eng_scan:
                    yield word
                old_end = ends[0]
            else:
                for end in ends:
                    if end > start:
                        yield sentence[start : end + 1]
                        old_end = end
        if eng_scan:
            yield eng_buffer

    def cut(self, sentence, cut_all=False, HMM=True, use_paddle=False):
        if use_paddle:
            raise NotImplementedError("Paddle mode is not covered by mojo-jieba")
        sentence = _decode(sentence)
        blocks = _RE_HAN_DEFAULT.split(sentence)
        route = None
        if not cut_all:
            position = 0
            spans = []
            for block_index, block in enumerate(blocks):
                block_end = position + len(block)
                if block_index % 2:
                    spans.append((position, block_end))
                position = block_end
            route, _ = self._route_blocks(sentence, spans)
        position = 0
        for block_index, block in enumerate(blocks):
            if not block:
                continue
            if block_index % 2:
                if cut_all:
                    yield from self._cut_all(block)
                elif HMM:
                    yield from self._cut_dag(block, route, position)
                else:
                    yield from self._cut_dag_no_hmm(block, route, position)
            else:
                for item in _RE_SKIP_DEFAULT.split(block):
                    if _RE_SKIP_DEFAULT.fullmatch(item):
                        yield item
                    elif not cut_all:
                        yield from item
                    else:
                        yield item
            position += len(block)

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))

    def cut_for_search(self, sentence, HMM=True):
        self.check_initialized()
        for word in self.cut(sentence, HMM=HMM):
            if len(word) > 2:
                for index in range(len(word) - 1):
                    gram = word[index : index + 2]
                    if self.FREQ.get(gram):
                        yield gram
            if len(word) > 3:
                for index in range(len(word) - 2):
                    gram = word[index : index + 3]
                    if self.FREQ.get(gram):
                        yield gram
            yield word

    def lcut_for_search(self, *args, **kwargs):
        return list(self.cut_for_search(*args, **kwargs))

    def tokenize(self, unicode_sentence, mode="default", HMM=True):
        if not isinstance(unicode_sentence, str):
            raise ValueError("jieba: the input parameter should be unicode.")
        start = 0
        for word in self.cut(unicode_sentence, HMM=HMM):
            width = len(word)
            if mode != "default":
                if width > 2:
                    for index in range(width - 1):
                        gram = word[index : index + 2]
                        if self.FREQ.get(gram):
                            yield gram, start + index, start + index + 2
                if width > 3:
                    for index in range(width - 2):
                        gram = word[index : index + 3]
                        if self.FREQ.get(gram):
                            yield gram, start + index, start + index + 3
            yield word, start, start + width
            start += width

    def load_userdict(self, file):
        self.check_initialized()
        close = False
        if isinstance(file, (str, os.PathLike)):
            file = open(file, "rb")
            close = True
        try:
            for lineno, raw_line in enumerate(file, 1):
                line = raw_line.strip()
                if isinstance(line, bytes):
                    line = line.decode("utf-8").lstrip("\ufeff")
                if not line:
                    continue
                match = _RE_USERDICT.match(line)
                if match is None:
                    raise ValueError(f"invalid user dictionary entry at Line {lineno}")
                word, frequency, tag = match.groups()
                self.add_word(
                    word,
                    frequency.strip() if frequency else None,
                    tag.strip() if tag else None,
                )
        finally:
            if close:
                file.close()

    def add_word(self, word, freq=None, tag=None):
        self.check_initialized()
        word = _decode(word)
        frequency = int(freq) if freq is not None else self.suggest_freq(word, False)
        self.FREQ[word] = frequency
        self.total += frequency
        if tag:
            self.user_word_tag_tab[word] = tag
        for length in range(1, len(word) + 1):
            self.FREQ.setdefault(word[:length], 0)
        self._vocabulary = None
        if frequency == 0:
            _finalseg.add_force_split(word)

    def del_word(self, word):
        self.add_word(word, 0)

    def suggest_freq(self, segment, tune=False):
        self.check_initialized()
        total = float(self.total)
        frequency = 1.0
        if isinstance(segment, (str, bytes)):
            word = _decode(segment)
            for part in self.cut(word, HMM=False):
                frequency *= self.FREQ.get(part, 1) / total
            result = max(
                int(frequency * self.total) + 1, self.FREQ.get(word, 1)
            )
        else:
            parts = tuple(map(_decode, segment))
            word = "".join(parts)
            for part in parts:
                frequency *= self.FREQ.get(part, 1) / total
            result = min(int(frequency * self.total), self.FREQ.get(word, 0))
        if tune:
            self.add_word(word, result)
        return result

    def set_dictionary(self, dictionary_path):
        absolute = os.path.abspath(dictionary_path)
        if not os.path.isfile(absolute):
            raise Exception("jieba: file does not exist: " + absolute)
        with self.lock:
            self.dictionary = absolute
            self.initialized = False
            self._vocabulary = None


def set_log_level(level):
    logger.setLevel(level)
