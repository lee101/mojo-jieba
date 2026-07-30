"""Behavioral parity with jieba 0.42.1."""

from __future__ import annotations

import inspect
import io

import numpy as np
import pytest

import jieba as upstream
import mojo_jieba as mojo
from mojo_jieba import _lib


SENTENCES = [
    "",
    "我来到北京清华大学",
    "他来到了网易杭研大厦",
    "小明硕士毕业于中国科学院计算所，后在日本京都大学深造",
    "南京市长江大桥",
    "今天天气不错",
    "丰田太省了",
    "知识就是力量",
    "结婚的和尚未结婚的",
    "据《日经亚洲评论》网站报道",
    "this is a test 123.45%",
    "C++与C#、Node.js和AT&T",
    "你好\r\n世界\t再见",
    "甲𠀀乙",
]


@pytest.fixture(scope="module", autouse=True)
def initialize_tokenizers():
    mojo.initialize()
    upstream.initialize()


@pytest.mark.parametrize("sentence", SENTENCES)
@pytest.mark.parametrize("hmm", [False, True])
def test_accurate_cut_parity(sentence, hmm):
    assert mojo.lcut(sentence, HMM=hmm) == upstream.lcut(sentence, HMM=hmm)


@pytest.mark.parametrize("sentence", SENTENCES)
def test_full_mode_parity(sentence):
    assert mojo.lcut(sentence, cut_all=True) == upstream.lcut(
        sentence, cut_all=True
    )


@pytest.mark.parametrize("sentence", SENTENCES)
@pytest.mark.parametrize("hmm", [False, True])
def test_search_mode_parity(sentence, hmm):
    assert mojo.lcut_for_search(sentence, HMM=hmm) == upstream.lcut_for_search(
        sentence, HMM=hmm
    )


@pytest.mark.parametrize("mode", ["default", "search"])
@pytest.mark.parametrize("hmm", [False, True])
def test_tokenize_offsets_parity(mode, hmm):
    sentence = "永和服装饰品有限公司，欢迎新老客户"
    assert list(mojo.tokenize(sentence, mode=mode, HMM=hmm)) == list(
        upstream.tokenize(sentence, mode=mode, HMM=hmm)
    )


def test_cut_returns_generator_and_bytes_are_accepted():
    result = mojo.cut("中文")
    assert inspect.isgenerator(result)
    assert list(result) == upstream.lcut("中文")
    assert mojo.lcut("中文".encode()) == upstream.lcut("中文".encode())


def test_cut_for_search_returns_generator():
    result = mojo.cut_for_search("南京市长江大桥")
    assert inspect.isgenerator(result)
    assert list(result) == upstream.lcut_for_search("南京市长江大桥")


def test_tokenize_rejects_bytes_like_upstream():
    with pytest.raises(ValueError, match="unicode"):
        list(mojo.tokenize(b"abc"))


def test_unsupported_execution_modes_fail_explicitly():
    with pytest.raises(NotImplementedError, match="Paddle"):
        mojo.lcut("中文", use_paddle=True)
    with pytest.raises(NotImplementedError, match="parallel"):
        mojo.enable_parallel()


def test_dag_parity():
    sentence = "南京市长江大桥欢迎您"
    assert mojo.get_DAG(sentence) == upstream.get_DAG(sentence)


def test_public_calc_parity():
    sentence = "南京市长江大桥欢迎您"
    ours_dag = mojo.get_DAG(sentence)
    theirs_dag = upstream.get_DAG(sentence)
    ours_route = {}
    theirs_route = {}
    mojo.calc(sentence, ours_dag, ours_route)
    upstream.calc(sentence, theirs_dag, theirs_route)
    assert {key: value[1] for key, value in ours_route.items()} == {
        key: value[1] for key, value in theirs_route.items()
    }
    assert [ours_route[i][0] for i in range(len(sentence) + 1)] == pytest.approx(
        [theirs_route[i][0] for i in range(len(sentence) + 1)]
    )


def test_mojo_route_matches_upstream_dynamic_program():
    sentence = "南京市长江大桥和北京清华大学都欢迎你"
    endpoints, scores = mojo.dt._route(sentence)
    dag = upstream.get_DAG(sentence)
    reference = {}
    upstream.calc(sentence, dag, reference)
    assert endpoints.tolist() == [reference[i][1] for i in range(len(sentence))]
    assert scores.tolist() == pytest.approx(
        [reference[i][0] for i in range(len(sentence) + 1)]
    )


def test_custom_dictionary_tokenizer_parity(tmp_path):
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text(
        "研究 1000 v\n研究生 3000 n\n生命 2500 n\n命 10 n\n起源 2000 n\n",
        encoding="utf-8",
    )
    ours = mojo.Tokenizer(dictionary)
    theirs = upstream.Tokenizer(dictionary)
    sentence = "研究生命起源"
    assert ours.lcut(sentence, HMM=False) == theirs.lcut(sentence, HMM=False)
    assert ours.get_DAG(sentence) == theirs.get_DAG(sentence)


def test_non_bmp_custom_dictionary_parity(tmp_path):
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("𠀀 10 n\n𠀀甲 100 n\n甲 10 n\n", encoding="utf-8")
    ours = mojo.Tokenizer(dictionary)
    theirs = upstream.Tokenizer(dictionary)
    assert ours.lcut("𠀀甲𠀀", HMM=False) == theirs.lcut(
        "𠀀甲𠀀", HMM=False
    )


def test_add_and_delete_word_parity(tmp_path):
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text(
        "云原生 100 n\n数据库 100 n\n云 80 n\n原生 80 n\n",
        encoding="utf-8",
    )
    ours = mojo.Tokenizer(dictionary)
    theirs = upstream.Tokenizer(dictionary)
    sentence = "云原生数据库"
    ours.add_word("云原生数据库", 100_000, "n")
    theirs.add_word("云原生数据库", 100_000, "n")
    assert ours.lcut(sentence, HMM=False) == theirs.lcut(sentence, HMM=False)
    assert ours.user_word_tag_tab == theirs.user_word_tag_tab
    ours.del_word("云原生数据库")
    theirs.del_word("云原生数据库")
    assert ours.lcut(sentence, HMM=False) == theirs.lcut(sentence, HMM=False)


def test_load_userdict_parity(tmp_path):
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text(
        "创新 100 n\n办 10 v\n创新办 1 n\n",
        encoding="utf-8",
    )
    ours = mojo.Tokenizer(dictionary)
    theirs = upstream.Tokenizer(dictionary)
    userdict = "创新办学 99999 n\n云计算 8000 nz\n"
    ours.load_userdict(io.BytesIO(userdict.encode()))
    theirs.load_userdict(io.BytesIO(userdict.encode()))
    sentence = "创新办学发展云计算"
    assert ours.lcut(sentence, HMM=False) == theirs.lcut(sentence, HMM=False)


@pytest.mark.parametrize(
    "segment",
    ["中出", ("中", "将"), "台中", ("台", "中")],
)
def test_suggest_freq_parity(segment):
    ours = mojo.Tokenizer()
    theirs = upstream.Tokenizer()
    assert ours.suggest_freq(segment) == theirs.suggest_freq(segment)


def test_suggest_freq_tune_changes_segmentation():
    tokenizer = mojo.Tokenizer()
    before = tokenizer.lcut("如果放到post中将出错。", HMM=False)
    tokenizer.suggest_freq(("中", "将"), tune=True)
    after = tokenizer.lcut("如果放到post中将出错。", HMM=False)
    assert before != after
    assert "中" in after and "将" in after


def test_set_dictionary_and_repr(tmp_path):
    dictionary = tmp_path / "small.txt"
    dictionary.write_text("测试 100 n\n文本 80 n\n", encoding="utf-8")
    tokenizer = mojo.Tokenizer()
    tokenizer.set_dictionary(dictionary)
    assert tokenizer.lcut("测试文本", HMM=False) == ["测试", "文本"]
    assert str(dictionary) in repr(tokenizer)


def test_missing_dictionary_raises(tmp_path):
    with pytest.raises(Exception, match="does not exist"):
        mojo.Tokenizer().set_dictionary(tmp_path / "missing.txt")


def test_hash_table_handles_random_dictionary_routes(tmp_path):
    rng = np.random.default_rng(7)
    alphabet = np.array(list("甲乙丙丁戊己庚辛壬癸"))
    words = {"".join(rng.choice(alphabet, size=int(rng.integers(1, 7)))) for _ in range(500)}
    dictionary = tmp_path / "random.txt"
    dictionary.write_text(
        "".join(f"{word} {index + 1} n\n" for index, word in enumerate(words)),
        encoding="utf-8",
    )
    ours = mojo.Tokenizer(dictionary)
    theirs = upstream.Tokenizer(dictionary)
    for _ in range(10):
        sentence = "".join(rng.choice(alphabet, size=60))
        assert ours.lcut(sentence, HMM=False) == theirs.lcut(sentence, HMM=False)


def test_simd_child_scan_tail_parity(tmp_path):
    dictionary = tmp_path / "wide.txt"
    dictionary.write_text(
        "甲丁 100 n\n"
        "甲丙 100 n\n"
        "甲乙 100 n\n"
        "甲己 100 n\n"
        "甲庚 100000 n\n"
        "甲戊 100 n\n",
        encoding="utf-8",
    )
    ours = mojo.Tokenizer(dictionary)
    theirs = upstream.Tokenizer(dictionary)
    sentence = "甲庚甲戊甲辛"
    assert ours.lcut(sentence, HMM=False) == theirs.lcut(sentence, HMM=False)


def test_ffi_rejects_wrong_dtype_and_noncontiguous_arrays():
    tokenizer = mojo.Tokenizer()
    vocabulary = tokenizer._ensure_vocabulary()
    chars = np.frombuffer("中文".encode("utf-32-le"), dtype=np.uint32)
    output = np.empty(2, dtype=np.int64)
    scores = np.empty(3, dtype=np.float64)
    arguments = (
        vocabulary.edge_chars,
        vocabulary.edge_offsets,
        vocabulary.edge_children,
        vocabulary.node_weights,
        vocabulary.root_children,
        vocabulary.max_length,
        -np.log(tokenizer.total),
        output,
        scores,
    )
    with pytest.raises(TypeError, match="chars must have dtype"):
        _lib.route(chars.astype(np.int64), *arguments)
    with pytest.raises(ValueError, match="C-contiguous"):
        _lib.route(chars[::-1], *arguments)
    readonly_scores = scores.copy()
    readonly_scores.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        _lib.route(chars, *arguments[:-1], readonly_scores)
    with pytest.raises(ValueError, match="character count"):
        _lib.route(chars, *arguments[:-2], output[:1], scores)


def test_ffi_batch_rejects_touching_spans():
    tokenizer = mojo.Tokenizer()
    vocabulary = tokenizer._ensure_vocabulary()
    chars = np.frombuffer("中文测试".encode("utf-32-le"), dtype=np.uint32)
    with pytest.raises(ValueError, match="separated"):
        _lib.route_batch(
            chars,
            np.array([0, 2], dtype=np.int64),
            np.array([2, 4], dtype=np.int64),
            vocabulary.edge_chars,
            vocabulary.edge_offsets,
            vocabulary.edge_children,
            vocabulary.node_weights,
            vocabulary.root_children,
            vocabulary.max_length,
            -np.log(tokenizer.total),
            np.empty(4, dtype=np.int64),
            np.empty(5, dtype=np.float64),
        )


@pytest.mark.parametrize("repeats", [1, 800])
def test_batched_route_serial_and_parallel_parity(repeats):
    unit = (
        "南京市长江大桥欢迎您。"
        "小明硕士毕业于中国科学院计算所，后在日本京都大学深造。"
        "他来到了网易杭研大厦。"
    )
    sentence = unit * repeats
    assert mojo.lcut(sentence, HMM=False) == upstream.lcut(
        sentence, HMM=False
    )
