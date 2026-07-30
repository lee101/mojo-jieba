"""Jieba-compatible Chinese word segmentation accelerated by Mojo."""

from __future__ import annotations

from .core import DEFAULT_DICT, Tokenizer, set_log_level

__version__ = "0.1.0"
__license__ = "MIT"

dt = Tokenizer()

get_FREQ = lambda key, default=None: dt.FREQ.get(key, default)
add_word = dt.add_word
calc = dt.calc
cut = dt.cut
lcut = dt.lcut
cut_for_search = dt.cut_for_search
lcut_for_search = dt.lcut_for_search
del_word = dt.del_word
get_DAG = dt.get_DAG
get_dict_file = dt.get_dict_file
initialize = dt.initialize
load_userdict = dt.load_userdict
set_dictionary = dt.set_dictionary
suggest_freq = dt.suggest_freq
tokenize = dt.tokenize
user_word_tag_tab = dt.user_word_tag_tab
setLogLevel = set_log_level


def enable_parallel(processnum=None):
    raise NotImplementedError("parallel mode is not covered by mojo-jieba")


def disable_parallel():
    return None


__all__ = [
    "DEFAULT_DICT",
    "Tokenizer",
    "add_word",
    "calc",
    "cut",
    "cut_for_search",
    "del_word",
    "disable_parallel",
    "dt",
    "enable_parallel",
    "get_DAG",
    "get_FREQ",
    "get_dict_file",
    "initialize",
    "lcut",
    "lcut_for_search",
    "load_userdict",
    "set_dictionary",
    "setLogLevel",
    "suggest_freq",
    "tokenize",
]
