class BlockOfTokens:

    def __init__(self, start_token: int, pad_token: int, end_token: int,
                 token_block_length: int = 75):
        """
        This object contains a block of token ids and their weights

        Args:
            start_token (int): the start token of the block
            pad_token (int): the padding token of the block
            end_token (int): the end token of the block
            token_block_length (int): the length of the block
        """
        self.tokens = [pad_token] * (token_block_length + 2)
        self.tokens[0] = start_token
        self.tokens[-1] = end_token
        self.multipliers = [float(1.0)] * (token_block_length + 2)
        self.token_block_length = token_block_length
        self.current_block_index = 1
        self.current_number_of_tokens = 0

    def __str__(self) -> str:
        return f"tokens: {self.tokens}, multipliers: {self.multipliers}"

    def __len__(self) -> int:
        return self.current_number_of_tokens

    def add_token(self, token: int, weight: float):
        """
        Add a token to the block

        Args:
        token (int): the token to add
        weight (float): the weight to add
        """
        if self.is_full():
            raise RuntimeError(f"Cannot add token {token} and weight {weight} "
                               f"to block because it is full")

        self.tokens[self.current_block_index] = token
        self.multipliers[self.current_block_index] = weight
        self.current_block_index += 1
        self.current_number_of_tokens += 1

        return self

    def is_full(self) -> bool:
        """
        Check if the block is full

        Returns:
            bool: whether the block is full
        """
        return self.current_number_of_tokens == self.token_block_length
