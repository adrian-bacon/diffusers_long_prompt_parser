from typing import Tuple
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedModel
from diffusers_long_prompt_parser.long_prompt_parser import LongPromptParserSd1x


def assemble_prompt_to_string(prompt: str | list = "") -> str:
    """
    Assemble prompt string from list of strings.

    :param prompt: the string or list of strings to assemble.
    :return: the assembled prompt
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
    Initialize a long prompt parser for stable diffusion 1.x.  The applies to SD 1.0-1.5.

    :param tokenizer: the stable diffusion tokenizer.
    :param text_encoder: the stable diffusion tokenizer.
    :param device: the torch device to use.
    :return: a LongPromptParserSd1x object.
    """

    return LongPromptParserSd1x(tokenizer=tokenizer, text_encoder=text_encoder, device=device)


def get_embeddings_sd1x(
        prompt_parser: LongPromptParserSd1x,
        prompt: str | list = "",
        negative_prompt: str | list = ""
) -> Tuple[Tensor, Tensor]:
    """
    Get the positive and negative prompt embeddings
    :param prompt_parser: the instance of LongPromptParserSd1x object.
    :param prompt: the positive prompt string or list of strings.
    :param negative_prompt: the negative prompt string or list of strings.
    :return: the positive and negative prompt embeddings.
    """
    return prompt_parser(assemble_prompt_to_string(prompt), assemble_prompt_to_string(negative_prompt))

