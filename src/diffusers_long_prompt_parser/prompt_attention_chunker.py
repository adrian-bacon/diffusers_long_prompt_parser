import re
from typing import List

prompt_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

prompt_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def apply_multiplier(start_position: int, multiplier: float, result: []):
    """
    Applies the calculated prompt multiplier to the start position.

    Args:
        start_position (int): The position of the start position.
        multiplier (float): The multiplier to be applied.
        result (list): A list containing the result of the calculation.
    """
    for p in range(start_position, len(result)):
        result[p][1] *= multiplier


def prompt_attention_chunker(prompt: str = "") -> List[List[str | float]]:
    """
    Takes a string that follows the A1111 prompting convention and
    returns an array of strings and their weights.

    Args:
        prompt (str): The prompt that should be parsed.

    Returns:
        List[List[str | float]]: A list of the words and weights
    """
    result = []
    split_result = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    for m in prompt_attention.finditer(prompt):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            result.append([text[1:], 1.0])

        elif text == '(':
            round_brackets.append(len(result))

        elif text == '[':
            square_brackets.append(len(result))

        elif weight is not None and round_brackets:
            apply_multiplier(round_brackets.pop(), float(weight), result)

        elif text == ')' and round_brackets:
            apply_multiplier(round_brackets.pop(),
                             round_bracket_multiplier,
                             result)

        elif text == ']' and square_brackets:
            apply_multiplier(square_brackets.pop(),
                             square_bracket_multiplier,
                             result)

        else:
            parts = re.split(prompt_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    result.append(["BREAK", -1])
                result.append([part, 1.0])

    for position in round_brackets:
        apply_multiplier(position, round_bracket_multiplier, result)

    for position in square_brackets:
        apply_multiplier(position, square_bracket_multiplier, result)

    if len(result) == 0:
        return [["", 1.0]]

    # we want to return each word and their weight, this way when we go to
    # tokenize it, we don't get long runs that the tokenizer doesn't support.
    for w in result:
        split_text = w[0].strip().split()
        for t in split_text:
            if len(t.strip()) > 0:
                split_result.append([t, w[1]])

    return split_result
