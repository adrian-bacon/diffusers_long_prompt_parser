from typing import Tuple
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedModel
from diffusers_long_prompt_parser \
    .long_prompt_parser import (LongPromptParserSd1x)


def assemble_prompt_to_string(prompt: str | list = "") -> str:
    """
    Assemble prompt string from list of strings if supplied.

    Args:
        prompt (str | list): Prompt to be assembled.

    Return:
        str: Assembled prompt.
    """
    if isinstance(prompt, str):
        prompt_list = [prompt]
    else:
        prompt_list = prompt

    prompt_string = ''
    for text in prompt_list:
        prompt_string += ', '
        prompt_string += text

    return prompt_string


def initialize_prompt_parser_sd1x(tokenizer: PreTrainedTokenizer,
                                  text_encoder: PreTrainedModel,
                                  device: str = "cpu") -> LongPromptParserSd1x:
    """
    Initialize a long prompt parser for stable diffusion 1.x.
    This applies to SD 1.0-1.5.

    Args:
        tokenizer (PreTrainedTokenizer): Pre-trained tokenizer.
        text_encoder (PreTrainedModel): Pre-trained text encoder.
        device (str, optional): Device to use. Defaults to "cpu".

    Return:
        LongPromptParserSd1x: Long prompt parser initialized.
    """

    return LongPromptParserSd1x(tokenizer=tokenizer,
                                text_encoder=text_encoder,
                                device=device)


def get_embeddings_sd1x(
        prompt_parser: LongPromptParserSd1x,
        prompt: str | list = "",
        negative_prompt: str | list = ""
) -> Tuple[Tensor, Tensor]:
    """
    Get the positive and negative prompt embeddings

    Args:
        prompt_parser (LongPromptParserSd1x):
        prompt (str): Prompt to be assembled.
        negative_prompt (str): Prompt to be assembled.

    Returns:
        Tuple[Tensor, Tensor]: Embeddings and negative prompt embeddings.
    """
    return prompt_parser(assemble_prompt_to_string(prompt),
                         assemble_prompt_to_string(negative_prompt))
