import json

import pytest
from conftest import SAMPLE_SCRIPT, ScriptedLLM

from vidpipe.narration import build_cues, to_srt
from vidpipe.script import ScriptError, generate_script, validate


def test_generate_script_valid_first_try(good_llm):
    s = generate_script(good_llm, "ローカルLLM", n_scenes=3)
    assert s.title.startswith("ローカルLLM")
    assert len(s.scenes) == 3


def test_invalid_output_is_retried_with_error_feedback():
    llm = ScriptedLLM("not json", json.dumps(SAMPLE_SCRIPT, ensure_ascii=False))
    s = generate_script(llm, "x", n_scenes=3)
    assert len(s.scenes) == 3
    assert "前回の出力の問題" in llm.prompts[1]


def test_gives_up_after_retries():
    bad = dict(SAMPLE_SCRIPT, scenes=SAMPLE_SCRIPT["scenes"][:1])
    with pytest.raises(ScriptError, match="expected 3 scenes"):
        generate_script(ScriptedLLM(json.dumps(bad)), "x", n_scenes=3)


def test_validation_rules():
    with pytest.raises(ScriptError, match="not allowed"):
        validate(dict(SAMPLE_SCRIPT, title="<b>bad</b>"), 3)
    long_tags = dict(SAMPLE_SCRIPT, tags=["タグ" * 20] * 30)
    assert sum(len(t) + 1 for t in validate(long_tags, 3).tags) <= 500


def test_cues_follow_real_audio_durations():
    cues = build_cues(["はじめに。次に、本題です。", "まとめ。"], [3.0, 1.0])
    assert cues[0].start == 0
    assert cues[-1].end == pytest.approx(4.0, abs=0.01)
    scene2 = [c for c in cues if c.text == "まとめ。"][0]
    assert scene2.start == pytest.approx(3.0, abs=0.01)  # starts exactly when scene 2 audio starts


def test_long_clause_is_wrapped():
    cues = build_cues(["あ" * 80], [4.0], max_chars=28)
    assert all(len(c.text) <= 36 for c in cues) and len(cues) >= 2


def test_srt_format():
    srt = to_srt(build_cues(["こんにちは。"], [61.5]))
    assert srt.startswith("1\n00:00:00,000 --> 00:01:01,500\nこんにちは。")


def test_thumbnail_title_wrap_never_splits_numbers_or_words():
    from vidpipe.editor import wrap_title

    assert wrap_title("短いタイトル") == ["短いタイトル"]
    assert wrap_title("ローカルLLM入門：Ollamaで社内AIを動かす") == [
        "ローカルLLM入門",
        "Ollamaで社内AIを動かす",
    ]
    first, second = wrap_title("生成AIで業務効率化する10の方法とその注意点を解説")
    assert "10" in first + second and not first.endswith("1")
