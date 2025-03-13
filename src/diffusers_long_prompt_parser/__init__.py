import torch
from typing import Union, List
from transformers import PreTrainedTokenizer, PreTrainedModel
from diffusers_long_prompt_parser.long_prompt_parser import LongPromptParser


def assemble_prompt_to_string(prompt: str | list = "") -> str:
    if isinstance(prompt, str):
        prompt_list = [prompt]
    else:
        prompt_list = prompt

    prompt_string = ''
    for text in prompt_list:
        prompt_string += text

    return prompt_string


def initialize_prompt_parser(tokenizer: Union[PreTrainedTokenizer, List[PreTrainedTokenizer]],
                             text_encoder: Union[PreTrainedModel, List[PreTrainedModel]],
                             device: str = "cpu") -> LongPromptParser:

    return LongPromptParser(tokenizer=tokenizer, text_encoder=text_encoder, device=device)


def parse_prompt(
        prompt_parser: LongPromptParser,
        prompt: str | list = "",
        negative_prompt: str | list = ""
) -> list[torch.Tensor]: return prompt_parser(
    assemble_prompt_to_string(prompt),
    assemble_prompt_to_string(negative_prompt))

