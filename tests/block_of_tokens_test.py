import pytest
from diffusers_long_prompt_parser.block_of_tokens import BlockOfTokens  # Replace 'your_module' with the actual name of your module


def test_init():
    block = BlockOfTokens(0, 1, 2, token_block_length=3)
    assert block.tokens == [0, 1, 2]
    assert block.multipliers == [1.0, 1.0, 1.0]


def test_str():
    block = BlockOfTokens(0, 1, 2, token_block_length=3)
    assert str(block) == "tokens: [0, 1, 2]\nmultipliers: [1.0, 1.0, 1.0]"


def test_len():
    block = BlockOfTokens(0, 1, 2, token_block_length=3)
    assert len(block) == 0


def test_add_token():
    block = BlockOfTokens(0, 1, 2, token_block_length=5)
    block.add_token(10, 5.0)
    assert block.tokens == [0, 1, 2, 10]
    assert block.multipliers == [1.0, 1.0, 1.0, 5.0]
    assert len(block) == 1


def test_add_token_full_block():
    block = BlockOfTokens(0, 1, 2, token_block_length=3)
    for i in range(3):
        block.add_token(i * 10, 5.0)
    assert block.tokens == [0, 10, 20]
    assert block.multipliers == [1.0, 5.0, 5.0]


def test_is_full():
    block = BlockOfTokens(0, 1, 2, token_block_length=3)
    assert not block.is_full()
    for i in range(3):
        block.add_token(i * 10, 5.0)
    assert block.is_full()


def test_add_token_after_full_block():
    block = BlockOfTokens(0, 1, 2, token_block_length=3)
    for i in range(3):
        block.add_token(i * 10, 5.0)
    block.add_token(100, 50.0)  # Trying to add a token after the block is full
    assert block.tokens == [0, 10, 20]
    assert block.multipliers == [1.0, 5.0, 5.0]
