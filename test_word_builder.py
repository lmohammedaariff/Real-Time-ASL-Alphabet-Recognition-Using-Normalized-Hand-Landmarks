import pytest
from src.word_builder.word_builder import WordBuilder

def test_word_builder_cooldown_and_controls():
    w = WordBuilder(1.0)
    assert w.add_letter("H", now=10)
    assert not w.add_letter("E", now=10.5)
    assert w.add_letter("E", now=11.1)
    w.space(); w.add_letter("I", now=12.2)
    assert w.text == "HE I"
    w.delete(); assert w.text == "HE "
    w.clear(); assert w.text == ""

def test_invalid_letter():
    with pytest.raises(ValueError): WordBuilder().add_letter("1", now=5)
