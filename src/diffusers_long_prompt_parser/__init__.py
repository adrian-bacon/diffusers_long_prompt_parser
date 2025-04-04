from typing import Tuple
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedModel
from diffusers_long_prompt_parser \
    .long_prompt_parser import (LongPromptParserSd1x, LongPromptParserSd2x,
                                LongPromptParserSdxl)


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
                                  weight_normalization: str = 'none',
                                  device: str = "cpu") -> LongPromptParserSd1x:
    """
    Initialize a long prompt parser for stable diffusion 1.x.
    This applies to SD 1.0-1.5.

    Args:
        tokenizer (PreTrainedTokenizer): Pre-trained tokenizer.
        text_encoder (PreTrainedModel): Pre-trained text encoder.
        weight_normalization (str, optional): Normalization strategy.
        device (str, optional): Device to use. Defaults to "cpu".

    Return:
        LongPromptParserSd1x: Long prompt parser initialized.
    """

    return LongPromptParserSd1x(tokenizer=tokenizer,
                                text_encoder=text_encoder,
                                weight_normalization=weight_normalization,
                                device=device)


def initialize_prompt_parser_sd2x(tokenizer: PreTrainedTokenizer,
                                  text_encoder: PreTrainedModel,
                                  weight_normalization: str = 'none',
                                  device: str = "cpu") -> LongPromptParserSd2x:
    """
    Initialize a long prompt parser for stable diffusion 2.x.
    This applies to SD 2.0-2.1.

    Args:
        tokenizer (PreTrainedTokenizer): Pre-trained tokenizer.
        text_encoder (PreTrainedModel): Pre-trained text encoder.
        weight_normalization (str, optional): Normalization strategy.
        device (str, optional): Device to use. Defaults to "cpu".

    Return:
        LongPromptParserSd2x: Long prompt parser initialized.
    """

    return LongPromptParserSd2x(tokenizer=tokenizer,
                                text_encoder=text_encoder,
                                weight_normalization=weight_normalization,
                                device=device)


def initialize_prompt_parser_sdxl(tokenizer: PreTrainedTokenizer,
                                  text_encoder: PreTrainedModel,
                                  tokenizer_2: PreTrainedTokenizer,
                                  text_encoder_2: PreTrainedModel,
                                  weight_normalization: str = 'none',
                                  device: str = "cpu") -> LongPromptParserSdxl:
    """
    Initialize a long prompt parser for stable diffusion XL.

    Args:
        tokenizer (PreTrainedTokenizer): Pre-trained tokenizer.
        text_encoder (PreTrainedModel): Pre-trained text encoder.
        weight_normalization (str, optional): Normalization strategy.
        device (str, optional): Device to use. Defaults to "cpu".

    Return:
        LongPromptParserSdxl: Long prompt parser initialized.
    """

    return LongPromptParserSdxl(tokenizer=tokenizer,
                                text_encoder=text_encoder,
                                tokenizer_2=tokenizer_2,
                                text_encoder_2=text_encoder_2,
                                weight_normalization=weight_normalization,
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


def get_embeddings_sd2x(
        prompt_parser: LongPromptParserSd2x,
        prompt: str | list = "",
        negative_prompt: str | list = ""
) -> Tuple[Tensor, Tensor]:
    """
    Get the positive and negative prompt embeddings

    Args:
        prompt_parser (LongPromptParserSd2x):
        prompt (str): Prompt to be assembled.
        negative_prompt (str): Prompt to be assembled.

    Returns:
        Tuple[Tensor, Tensor]: Embeddings and negative prompt embeddings.
    """
    return prompt_parser(positive_text=assemble_prompt_to_string(prompt),
                         negative_text=assemble_prompt_to_string(
                             negative_prompt))


def get_embeddings_sdxl(
        prompt_parser: LongPromptParserSdxl,
        prompt: str | list = "",
        negative_prompt: str | list = "",
        prompt_2: str | list = "",
        negative_prompt_2: str | list = "",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Get the positive and negative prompt embeddings

    Args:
        prompt_parser (LongPromptParserSdxl):
        prompt (str): The primary positive prompt.
        negative_prompt (str): The primary negative prompt.
        prompt_2 (str): The secondary positive prompt.
        negative_prompt_2 (str): The secondary negative prompt.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Embeddings and negative prompt
        embeddings.
    """
    if ((prompt_2 != "" or (isinstance(prompt_2, list) and len(prompt_2) > 0))
            or (negative_prompt_2 != "" or (isinstance(negative_prompt_2, list)
                                            and len(negative_prompt_2) > 0))):
        return prompt_parser(
            positive_text=assemble_prompt_to_string(prompt),
            negative_text=assemble_prompt_to_string(negative_prompt),
            positive_text_2=assemble_prompt_to_string(prompt_2),
            negative_text_2=assemble_prompt_to_string(negative_prompt_2)
        )
    else:
        return prompt_parser(
            positive_text=assemble_prompt_to_string(prompt),
            negative_text=assemble_prompt_to_string(negative_prompt)
        )
