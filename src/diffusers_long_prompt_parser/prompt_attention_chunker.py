import re

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
    for p in range(start_position, len(result)):
        result[p][1] *= multiplier


def prompt_attention_chunker(prompt: str = ""):
    """
    Takes a string that follows the A1111 prompting convention and returns an array of strings and their weights.
    :param prompt:
    :return:
    """
    result = []
    round_brackets = []
    square_brackets = []
    plus_signs = []
    minus_signs = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1
    plus_sign_multiplier = 1.1
    minus_sign_multiplier = 1 / 1.1

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
            apply_multiplier(round_brackets.pop(), round_bracket_multiplier, result)
        elif text == ']' and square_brackets:
            apply_multiplier(square_brackets.pop(), square_bracket_multiplier, result)
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
        result = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(result):
        if result[i][1] == result[i + 1][1]:
            result[i][0] += result[i + 1][0]
            result.pop(i + 1)
        else:
            i += 1

    return result

if __name__ == "__main__":
    prompt = "the quick (brown fox) (jumped:1.0) (over :1.0)(the :1.1)(lazy dog), the lazy dog [barked] at the quick brown fox BREAK the yellow cat watched them both"
    print()
    print()

    print(prompt_attention_chunker(prompt))
    print()

    print(prompt)
    print()
