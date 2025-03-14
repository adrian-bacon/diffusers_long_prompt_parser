import pytest
from diffusers_long_prompt_parser.block_of_tokens import BlockOfTokens


def test_init():
    block = BlockOfTokens(0, 1, 2, token_block_length=3)
    assert block.tokens == [0, 1, 1, 1, 2]
    assert block.multipliers == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_str():
    block = BlockOfTokens(0, 1, 2, token_block_length=3)
    assert str(block) == "tokens: [0, 1, 1, 1, 2], multipliers: [1.0, 1.0, 1.0, 1.0, 1.0]"


def test_len():
    block = BlockOfTokens(0, 1, 2, token_block_length=3)
    assert len(block) == 0


def test_add_token():
    block = BlockOfTokens(0, 1, 2, token_block_length=3)
    block.add_token(10, 5.0)
    assert block.tokens == [0, 10, 1, 1, 2]
    assert block.multipliers == [1.0, 5.0, 1.0, 1.0, 1.0]
    assert len(block) == 1


def test_add_token_full_block():
    block = BlockOfTokens(0, 1, 2, token_block_length=3)
    for i in range(3):
        block.add_token(10, 5.0)
    assert block.tokens == [0, 10, 10, 10, 2]
    assert block.multipliers == [1.0, 5.0, 5.0, 5.0 , 1.0]


def test_is_full():
    block = BlockOfTokens(0, 1, 2, token_block_length=3)
    assert not block.is_full()
    for i in range(3):
        block.add_token(10, 5.0)
    assert block.is_full()


def test_add_token_after_full_block():
    block = BlockOfTokens(0, 1, 2, token_block_length=3)
    for i in range(3):
        block.add_token(10, 5.0)

    with pytest.raises(RuntimeError) as e:
        block.add_token(100, 50.0)  # Trying to add a token after the block is full
    assert str(e.value) == "Cannot add token 100 and weight 50.0 to block because it is full"
    assert block.tokens == [0, 10, 10, 10, 2]
    assert block.multipliers == [1.0, 5.0, 5.0, 5.0, 1.0]
