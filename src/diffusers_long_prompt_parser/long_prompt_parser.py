import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Tuple
from diffusers_long_prompt_parser.block_of_tokens import BlockOfTokens
from diffusers_long_prompt_parser.prompt_attention_chunker \
    import prompt_attention_chunker

GPU_DEVICE = "cuda:0"
CPU_DEVICE = "cpu"


class LongPromptParserSd1x:
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 text_encoder: PreTrainedModel,
                 token_block_length: int = 75,
                 weight_normalization: str = 'none',
                 device: str = CPU_DEVICE
                 ):
        """
        A Stable diffusion 1.x prompt weigher and long prompt parser

        Args:
            tokenizer (PreTrainedTokenizer): the model tokenizer
            text_encoder (PreTrainedModel): the model text encoder
            token_block_length (int, optional): the number of block tokens
            weight_normalization (str, optional): the weight normalization
            device (str, optional): the torch device to use

        Notes:
            - The token_block_length defaults to 75, not including start and
              end tokens
            - The weight_normalization defaults to 'none', but can be 'mean' or
              'max'
        """

        if device is None:
            self.device = torch.device(GPU_DEVICE if torch.cuda.is_available()
                                       else CPU_DEVICE)
        else:
            self.device = device

        print(f"Initializing Diffusers Long Prompt Parser: "
              f"token_block_length={token_block_length} device={self.device}")

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

    def get_next_token_block(
            self,
            token_block: BlockOfTokens,
            token_blocks: list,
            id_start: int,
            id_pad: int,
            id_end: int,
            token_block_length: int) -> BlockOfTokens:
        """
        Appends token_block to token_blocks and returns a new BlockOfTokens.

        Args:
            token_block (BlockOfTokens): the block of tokens
            token_blocks (list): the list of blocks of tokens
            id_start (int): the start of the id
            id_pad (int): the padding id
            id_end (int): the end of the id

        Returns:
            BlockOfTokens: a new block of tokens
        """
        token_blocks.append(token_block)

        return BlockOfTokens(
            id_start,
            id_pad,
            id_end,
            token_block_length)

    def tokenize_text(self,
                      text: str,
                      tokenizer: PreTrainedTokenizer,
                      id_start: int,
                      id_pad: int,
                      id_end: int,
                      token_block_length: int) -> List[BlockOfTokens]:
        """
        Tokenizes a list of strings into a list of TokenBlocks

        Args:
            text (str): the text to tokenize
            tokenizer (PreTrainedTokenizer): the model tokenizer
            id_start (int): the start of the id
            id_pad (int): the padding id
            id_end (int): the end of the id
            token_block_length (int): the number of tokens per block

        Returns:
            List[BlockOfTokens]: a list of TokenBlocks
        """
        weight_multiplier = 1.0
        token_blocks = []
        token_block = BlockOfTokens(
            id_start,
            id_pad,
            id_end,
            token_block_length)

        tokenized_attention = []

        # each attention chunk can be one or more token_ids, so break them out
        # so that they can assembled into blocks.
        for text, weight in prompt_attention_chunker(text):
            for token_id in tokenizer(
                    text=text,
                    truncation=False,
                    add_special_tokens=False)['input_ids']:
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
                token_block = self.get_next_token_block(
                    token_block,
                    token_blocks,
                    id_start,
                    id_pad,
                    id_end,
                    token_block_length)
                continue

            if token_block.add_token(
                    token_id, weight * weight_multiplier).is_full():
                token_block = self.get_next_token_block(
                    token_block,
                    token_blocks,
                    id_start,
                    id_pad,
                    id_end,
                    token_block_length)

        if token_block.current_number_of_tokens > 0 or len(token_blocks) == 0:
            self.get_next_token_block(
                token_block,
                token_blocks,
                id_start,
                id_pad,
                id_end,
                token_block_length)

        return token_blocks

    def encode_token_block(self,
                           token_block: BlockOfTokens,
                           text_encoder: PreTrainedModel,
                           output_hidden_states: bool = False,
                           return_pooled: bool = False) -> torch.Tensor:
        """
        Encodes a token block into a tensor of token ids.

        Args:
            token_block (BlockOfTokens): the token block,
            text_encoder (PreTrainedModel): the model text encoder
            output_hidden_states (bool, optional): output hidden states or not
            return_pooled (bool, optional): return pooled hidden states or not

        Returns:
            torch.Tensor: the encoded token ids
        """

        torch_tokens = torch.asarray([token_block.tokens])

        if ((output_hidden_states and return_pooled)
                or (output_hidden_states is False and return_pooled is False)):
            encoded_tokens = text_encoder(
                input_ids=torch_tokens,
                output_hidden_states=output_hidden_states)[0].squeeze(0)

        elif output_hidden_states is True and return_pooled is False:
            encoded_tokens = text_encoder(
                input_ids=torch_tokens,
                output_hidden_states=output_hidden_states
            ).hidden_states[-2].squeeze(0)
        else:
            raise ValueError('Unsupported Configuration, cannot return pooled '
                             'with output hiddend states false.')

        if return_pooled is False:
            for i in range(len(encoded_tokens)):
                encoded_tokens[i] \
                    = encoded_tokens[i] * token_block.multipliers[i]

        return encoded_tokens.unsqueeze(0)

    @torch.no_grad()
    def __call__(self,
                 positive_text: str = "",
                 negative_text: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a positive and negative text into prompt embeddings.

        Args:
            positive_text (str): the positive text
            negative_text (str): the negative text

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the encoded token ids
        """
        print("Encoding prompt")
        positive_torch = []
        negative_torch = []

        # tokenize our text
        positive_token_blocks = self.tokenize_text(
            positive_text,
            self.tokenizer,
            self.id_start,
            self.id_pad,
            self.id_end,
            self.token_block_length)

        negative_token_blocks = self.tokenize_text(
            negative_text,
            self.tokenizer,
            self.id_start,
            self.id_pad,
            self.id_end,
            self.token_block_length)

        # make the positive token blocks and the negative token blocks the
        # same number of chunks
        block_count = max(len(positive_token_blocks),
                          len(negative_token_blocks))

        while len(positive_token_blocks) < block_count:
            positive_token_blocks.append(
                BlockOfTokens(self.id_start,
                              self.id_pad,
                              self.id_end,
                              self.token_block_length))

        while len(negative_token_blocks) < block_count:
            negative_token_blocks.append(
                BlockOfTokens(self.id_start,
                              self.id_pad,
                              self.id_end,
                              self.token_block_length))

        # now encode our tokens with the text encoder
        for token_block in positive_token_blocks:
            positive_torch.append(
                self.encode_token_block(token_block, self.text_encoder))

        for token_block in negative_token_blocks:
            negative_torch.append(
                self.encode_token_block(token_block, self.text_encoder))

        return torch.hstack(positive_torch), torch.hstack(negative_torch)


class LongPromptParserSd2x(LongPromptParserSd1x):

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 text_encoder: PreTrainedModel,
                 token_block_length: int = 75,
                 weight_normalization: str = 'none',
                 device: str = CPU_DEVICE
                 ):
        """
        A Stable diffusion 2.x prompt weigher and long prompt parser

        Args:
            tokenizer (PreTrainedTokenizer): the model tokenizer
            text_encoder (PreTrainedModel): the model text encoder
            token_block_length (int, optional): the number of block tokens
            weight_normalization (str, optional): the weight normalization
            device (str, optional): the torch device to use

        Notes:
            - The token_block_length defaults to 75, not including start and
              end tokens
            - The weight_normalization defaults to 'none', but can be 'mean' or
              'max'
        """
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            token_block_length=token_block_length,
            weight_normalization=weight_normalization,
            device=device)


class LongPromptParserSdxl(LongPromptParserSd2x):

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 text_encoder: PreTrainedModel,
                 tokenizer_2: PreTrainedTokenizer,
                 text_encoder_2: PreTrainedModel,
                 token_block_length: int = 75,
                 weight_normalization: str = 'none',
                 device: str = CPU_DEVICE
                 ):
        """
        A Stable diffusion XL prompt weigher and long prompt parser

        Args:
            tokenizer (PreTrainedTokenizer): the model tokenizer
            text_encoder (PreTrainedModel): the model text encoder
            token_block_length (int, optional): the number of block tokens
            weight_normalization (str, optional): the weight normalization
            device (str, optional): the torch device to use

        Notes:
            - The token_block_length defaults to 75, not including start and
              end tokens
            - The weight_normalization defaults to 'none', but can be 'mean' or
              'max'
        """
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            token_block_length=token_block_length,
            weight_normalization=weight_normalization,
            device=device)

        self.tokenizer_2 = tokenizer_2
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_2.eval()

        self.id_start_2 = self.tokenizer.bos_token_id
        self.id_end_2 = self.tokenizer.eos_token_id
        self.id_pad_2 = self.tokenizer.pad_token_id

    @torch.no_grad()
    def __call__(self,
                 positive_text: str = "",
                 negative_text: str = "",
                 positive_text_2: str = "",
                 negative_text_2: str = "") -> (
            Tuple)[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Processes a positive and negative text into prompt embeddings.

        Args:
            positive_text (str): The primary positive text
            negative_text (str): the primary negative text
            positive_text_2 (str): the secondary positive text
            negative_text_2 (str): the secondary negative text

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            the encoded token ids
        """
        print("Encoding prompt")
        prompt = positive_text
        negative_prompt = negative_text

        if positive_text_2 == "" and negative_text_2 == "":
            prompt_2 = positive_text
            negative_prompt_2 = negative_text
        else:
            prompt_2 = positive_text_2
            negative_prompt_2 = negative_text_2

        pooled_prompt = f"{positive_text}, {positive_text_2}"
        negative_pooled_prompt = f"{negative_text}, {negative_text_2}"

        positive_torch = []
        negative_torch = []
        positive_torch_2 = []
        negative_torch_2 = []

        positive_torch_pooled = []
        negative_torch_pooled = []

        positive_torch_3 = []
        negative_torch_3 = []

        # tokenize our text
        positive_token_blocks = self.tokenize_text(
            prompt,
            self.tokenizer,
            self.id_start,
            self.id_pad,
            self.id_end,
            self.token_block_length)
        negative_token_blocks = self.tokenize_text(
            negative_prompt,
            self.tokenizer,
            self.id_start,
            self.id_pad,
            self.id_end,
            self.token_block_length)

        positive_token_blocks_2 = self.tokenize_text(
            prompt_2,
            self.tokenizer_2,
            self.id_start_2,
            self.id_pad_2,
            self.id_end_2,
            self.token_block_length)

        negative_token_blocks_2 = self.tokenize_text(
            negative_prompt_2,
            self.tokenizer_2,
            self.id_start_2,
            self.id_pad_2,
            self.id_end_2,
            self.token_block_length)

        positive_token_blocks_pooled = self.tokenize_text(
            pooled_prompt,
            self.tokenizer_2,
            self.id_start_2,
            self.id_pad_2,
            self.id_end_2,
            self.token_block_length)

        negative_token_blocks_pooled = self.tokenize_text(
            negative_pooled_prompt,
            self.tokenizer_2,
            self.id_start_2,
            self.id_pad_2,
            self.id_end_2,
            self.token_block_length)

        # make the positive token blocks and the negative token blocks the
        # same number of chunks
        block_count_1 = max(len(positive_token_blocks),
                            len(negative_token_blocks))
        block_count_2 = max(len(positive_token_blocks_2),
                            len(negative_token_blocks_2))
        block_count = max(block_count_1, block_count_2)

        while len(positive_token_blocks) < block_count:
            positive_token_blocks.append(
                BlockOfTokens(self.id_start,
                              self.id_pad,
                              self.id_end,
                              self.token_block_length))

        while len(negative_token_blocks) < block_count:
            negative_token_blocks.append(
                BlockOfTokens(self.id_start,
                              self.id_pad,
                              self.id_end,
                              self.token_block_length))

        while len(positive_token_blocks_2) < block_count:
            positive_token_blocks_2.append(
                BlockOfTokens(self.id_start_2,
                              self.id_pad_2,
                              self.id_end_2,
                              self.token_block_length))

        while len(negative_token_blocks_2) < block_count:
            negative_token_blocks_2.append(
                BlockOfTokens(self.id_start_2,
                              self.id_pad_2,
                              self.id_end_2,
                              self.token_block_length))

        # now encode our tokens with the text encoders
        for token_block in positive_token_blocks:
            positive = self.encode_token_block(token_block, self.text_encoder)
            positive_torch.append(positive)

        for token_block in negative_token_blocks:
            negative = self.encode_token_block(token_block, self.text_encoder)
            negative_torch.append(negative)

        for token_block in positive_token_blocks_2:
            positive = self.encode_token_block(
                token_block,
                self.text_encoder_2,
                output_hidden_states=True,
                return_pooled=False)
            positive_torch_2.append(positive)

        for token_block in negative_token_blocks_2:
            negative = self.encode_token_block(
                token_block,
                self.text_encoder_2,
                output_hidden_states=True,
                return_pooled=False)
            negative_torch_2.append(negative)

        # TODO: we're getting all the chunks for pooled embeds but only
        #       using one, effectively truncating it to 75 tokens. Figure out a
        #       way to stack them the same way we do the positive and negative
        #       embeds so there's unlimited long prompts for the pooled embeds
        for token_block in positive_token_blocks_pooled:
            positive = self.encode_token_block(
                token_block,
                self.text_encoder_2,
                output_hidden_states=True,
                return_pooled=True)
            positive_torch_pooled.append(positive)

        for token_block in negative_token_blocks_pooled:
            negative = self.encode_token_block(
                token_block,
                self.text_encoder_2,
                output_hidden_states=True,
                return_pooled=True)
            negative_torch_pooled.append(negative)

        # step through and concat our positive and negative torches with the
        # positive and negative torch 2s.
        for i in range(len(positive_torch)):
            positive_torch_3.append(
                torch.concat(
                    [positive_torch[i], positive_torch_2[i]], dim=-1))

            negative_torch_3.append(
                torch.concat(
                    [negative_torch[i], negative_torch_2[i]], dim=-1))

        return torch.hstack(positive_torch_3), \
            torch.hstack(negative_torch_3), \
            positive_torch_pooled[0], \
            negative_torch_pooled[0]
