import torch

from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Union, List
from diffusers_long_prompt_parser.block_of_tokens import BlockOfTokens
from diffusers_long_prompt_parser.prompt_attention_chunker import prompt_attention_chunker

GPU_DEVICE = "cuda:0"
CPU_DEVICE = "cpu"
CLIP_STOP_AT_LAST_LAYERS = 1


class LongPromptParser:
    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, List[PreTrainedTokenizer]],
                 text_encoder: Union[PreTrainedModel, List[PreTrainedModel]],
                 token_block_length: int = 75,
                 device: str = None
                 ):

        # device should be either "cuda:0" or "cpu"
        if device is None:
            self.device = torch.device(GPU_DEVICE if torch.cuda.is_available() else CPU_DEVICE)
        else:
            self.device = device

        print(f"Initializing prompt parser using device: {self.device}")

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.text_encoder.eval()
        self.token_block_length = token_block_length
        self.is_trainable = False
        self.input_key = 'txt'
        self.return_pooled = False
        self.id_start = self.tokenizer.bos_token_id
        self.id_end = self.tokenizer.eos_token_id
        self.id_pad = self.tokenizer.pad_token_id

        self.encoder_embeddings = self.text_encoder.text_model.embeddings
        self.ids = self.tokenizer(',', max_length=1, return_tensors="pt", add_special_tokens=False, truncation=True)["input_ids"]
        self.embedded = self.encoder_embeddings(self.ids.to(self.encoder_embeddings.token_embedding.weight.device)).squeeze(0)
        self.shape = self.embedded.shape[1]

        print(f"token_block length: {self.token_block_length}, id_start: {self.id_start}, id_end: {self.id_end}, id_pad: {self.id_pad}")

        # print(f"text encoder config: {self.text_encoder.config}")

    def get_next_token_block(self, token_block: BlockOfTokens, token_blocks: list) -> BlockOfTokens:
        """
        Appends token_block to token_blocks and returns a new BlockOfTokens.

        :param token_block: The current token block to append
        :param token_blocks: The list of token blocks so far
        :return: The next token block
        """
        token_blocks.append(token_block)

        return BlockOfTokens(self.id_start, self.id_pad, self.id_end, self.token_block_length)

    def tokenize_text(self, text: str) -> list:
        """
        Tokenizes a list of strings into a list of TokenBlocks
        :param text: the string to parse and tokenize to token_ids.
        :return: the list of TokenBlocks
        """
        token_blocks = []
        token_block = BlockOfTokens(self.id_start, self.id_pad, self.id_end, self.token_block_length)

        token_ids = []
        attention_chunks = prompt_attention_chunker(text)

        print(f"attention_chunks: {attention_chunks}")

        for text, _ in attention_chunks:
            token_ids.append(self.tokenizer(text=text, truncation=False, add_special_tokens=False)["input_ids"])

        for token_ids, (text, weight) in zip(token_ids, attention_chunks):
            if text == 'BREAK' and weight == -1:
                token_block = self.get_next_token_block(token_block, token_blocks)
                continue

            position = 0
            while position < len(token_ids):
                if token_block.add_token(token_ids[position], weight).is_full():
                    token_block = self.get_next_token_block(token_block, token_blocks)

                position += 1

        if token_block.current_number_of_tokens > 0 or len(token_blocks) == 0:
            self.get_next_token_block(token_block, token_blocks)

        return token_blocks

    def encode_tokens(self, token_ids: list, token_multipliers: list):
        # print(f"tokens: {token_ids}")
        # print(f"token_multipliers: {token_multipliers}")
        torch_tokens = torch.asarray(token_ids).to(self.device)
        torch_token_multipliers = torch.asarray(token_multipliers).to(self.device)

        # print(f"transformer: {self.text_encoder.text_model}")
        encoded_tokens = self.text_encoder(input_ids=torch_tokens, output_hidden_states=-CLIP_STOP_AT_LAST_LAYERS)

        # print(f"encoded_tokens: {encoded_tokens}")

        z = encoded_tokens.last_hidden_state

        # print(f"z before: {z}")

        pooled = getattr(z, 'pooled', None)

        original_mean = z.mean()
        z = z * torch_token_multipliers.reshape(torch_token_multipliers.shape + (1,)).expand(z.shape).to(self.device)
        new_mean = z.mean()
        z = z * (original_mean / new_mean)

        if pooled is not None:
             z.pooled = pooled

        # print(f"z after: {z}")

        return z

    def __call__(self, positive_text: str, negative_text: str):
        positive_torch = []
        negative_torch = []

        try:
            with torch.no_grad():
                # tokenize our text
                positive_token_blocks = self.tokenize_text(positive_text)
                negative_token_blocks = self.tokenize_text(negative_text)

                # make the positive token blocks and the negative token blocks the same number of chunks
                block_count = max(len(positive_token_blocks), len(negative_token_blocks))
                # print(f"{block_count} blocks of 75 tokens to encode")

                while len(positive_token_blocks) < block_count:
                    positive_token_blocks.append(BlockOfTokens(self.id_start, self.id_pad, self.id_end, self.token_block_length))
                while len(negative_token_blocks) < block_count:
                    negative_token_blocks.append(BlockOfTokens(self.id_start, self.id_pad, self.id_end, self.token_block_length))

                # now encode our tokens with the text encoder
                for token_block in positive_token_blocks:
                    t = self.encode_tokens([token_block.tokens], [token_block.multipliers])
                    # print(f"t: {t}")
                    positive_torch.append(t)

                for token_block in negative_token_blocks:
                    negative_torch.append(self.encode_tokens([token_block.tokens], [token_block.multipliers]))

                # return the stacked embeddings
                if self.return_pooled:
                    return (torch.hstack(positive_torch), positive_torch[0].pooled), (torch.hstack(negative_torch), negative_torch[0].pooled)
                else:
                    return torch.hstack(positive_torch), torch.hstack(negative_torch)

        except RuntimeError as e:
            print(f"error: {e}")
            return [], []
