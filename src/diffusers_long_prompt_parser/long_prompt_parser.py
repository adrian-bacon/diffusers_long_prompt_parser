import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Tuple
from diffusers_long_prompt_parser.block_of_tokens import BlockOfTokens
from diffusers_long_prompt_parser.prompt_attention_chunker import prompt_attention_chunker

GPU_DEVICE = "cuda:0"
CPU_DEVICE = "cpu"


class LongPromptParserSd1x:
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 text_encoder: PreTrainedModel,
                 token_block_length: int = 75,
                 weight_normalization: str = 'none',
                 device: str = None
                 ):
        """
        A Stable diffusion 1.x prompt weigher and long prompt parser

        :param tokenizer: the stable diffusion 1.x tokenizer
        :param text_encoder: the stable diffusion 1.x text encoder
        :param token_block_length: the length of the block of tokens. Defaults to 75 tokens
        :param weight_normalization: the weight normalization to use. Defaults to 'none', but can also be 'mean' and 'max'
        :param device: the torch device to use. Defaults to CPU if not specified.
        """

        if device is None:
            self.device = torch.device(GPU_DEVICE if torch.cuda.is_available() else CPU_DEVICE)
        else:
            self.device = device

        print(f"Initializing LongPromptParserSd1x: token_block_length={token_block_length} device={self.device}")

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.text_encoder.eval()
        self.token_block_length = token_block_length
        self.id_start = self.tokenizer.bos_token_id
        self.id_end = self.tokenizer.eos_token_id
        self.id_pad = self.tokenizer.pad_token_id
        if weight_normalization in ['none', 'mean', 'max']:
            self.weight_normalization = weight_normalization
        else:
            self.weight_normalization = 'none'

    def get_next_token_block(self, token_block: BlockOfTokens, token_blocks: list) -> BlockOfTokens:
        """
        Appends token_block to token_blocks and returns a new BlockOfTokens.

        :param token_block: The current token block to append
        :param token_blocks: The list of token blocks so far
        :return: The next token block
        """
        token_blocks.append(token_block)

        return BlockOfTokens(self.id_start, self.id_pad, self.id_end, self.token_block_length)

    def tokenize_text(self, text: str) -> List[BlockOfTokens]:
        """
        Tokenizes a list of strings into a list of TokenBlocks

        :param text: the string to parse and tokenize to token_ids.
        :return: the list of TokenBlocks
        """
        weight_multiplier = 1.0
        token_blocks = []
        token_block = BlockOfTokens(self.id_start, self.id_pad, self.id_end, self.token_block_length)
        tokenized_attention = []

        # each attention chunk can be one or more token_ids, so break them out so that they can assembled into blocks.
        for text, weight in prompt_attention_chunker(text):
            for token_id in self.tokenizer(text=text, truncation=False, add_special_tokens=False)['input_ids']:
                tokenized_attention.append([token_id, text, weight])

        # handle any weight normalization needed here
        if self.weight_normalization == 'max':
            max_value = 1.0
            for _, _, weight in tokenized_attention:
                if weight > max_value:
                    max_value = weight
            weight_multiplier = 1.0 / max_value
        elif self.weight_normalization == 'mean':
            sum_value = 0.0
            total_count = 0.0
            for _, _, weight in tokenized_attention:
                sum_value += weight
                total_count += 1
            weight_multiplier = 1.0 / (sum_value / total_count)

        # now take the token ids and segment them out into blocks.
        for token_id, text, weight in tokenized_attention:
            if text == 'BREAK' and weight == -1:
                token_block = self.get_next_token_block(token_block, token_blocks)
                continue

            if token_block.add_token(token_id, weight * weight_multiplier).is_full():
                token_block = self.get_next_token_block(token_block, token_blocks)

        if token_block.current_number_of_tokens > 0 or len(token_blocks) == 0:
            self.get_next_token_block(token_block, token_blocks)

        return token_blocks

    def encode_token_block(self, token_block: BlockOfTokens) -> torch.Tensor:
        """
        Encodes a token block into a tensor of token ids.

        :param token_block: the token block to encode
        :return: the encoded token ids
        """

        torch_tokens = torch.asarray([token_block.tokens])

        encoded_tokens = self.text_encoder(input_ids=torch_tokens)[0].squeeze(0)

        for i in range(len(encoded_tokens)):
            encoded_tokens[i] = encoded_tokens[i] * token_block.multipliers[i]

        return encoded_tokens.unsqueeze(0)

    def __call__(self, positive_text: str, negative_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a positive and negative text into prompt embeddings for stable diffusion 1.x

        :param positive_text:
        :param negative_text:
        :return:
        """
        positive_torch = []
        negative_torch = []

        with torch.no_grad():
            # tokenize our text
            positive_token_blocks = self.tokenize_text(positive_text)
            negative_token_blocks = self.tokenize_text(negative_text)

            # make the positive token blocks and the negative token blocks the same number of chunks
            block_count = max(len(positive_token_blocks), len(negative_token_blocks))

            while len(positive_token_blocks) < block_count:
                positive_token_blocks.append(
                    BlockOfTokens(self.id_start, self.id_pad, self.id_end, self.token_block_length))
            while len(negative_token_blocks) < block_count:
                negative_token_blocks.append(
                    BlockOfTokens(self.id_start, self.id_pad, self.id_end, self.token_block_length))

            # now encode our tokens with the text encoder
            for token_block in positive_token_blocks:
                positive_torch.append(self.encode_token_block(token_block))

            for token_block in negative_token_blocks:
                negative_torch.append(self.encode_token_block(token_block))

            return torch.hstack(positive_torch), torch.hstack(negative_torch)
