"""A small, mixed benchmark prompt set (written for this project).

Speculative decoding speedups depend heavily on the text being generated:
code and structured output are predictable (high acceptance), open-ended
creative text is not. The set mixes both so averages are not flattering.
"""

PROMPTS = {
    "code": [
        "Write a Python function that returns the n-th Fibonacci number using memoization.",
        "Write a Python class implementing an LRU cache with get and put methods.",
        "Write a SQL query that returns the five customers with the highest total order value.",
        "Implement binary search in Python and explain its time complexity in one sentence.",
    ],
    "structured": [
        "List the planets of the solar system in order from the Sun, one per line.",
        "Convert this to JSON with keys name, age and city: Alice is 31 and lives in Austin.",
        "Give a numbered list of five steps to make a cup of tea.",
        "Write the multiplication table of 7 from 7x1 to 7x10.",
    ],
    "explain": [
        "Explain what a hash table is to a first-year computer science student.",
        "Why is decoding in large language models usually limited by memory bandwidth?",
        "Explain the difference between latency and throughput with an everyday example.",
        "What does a compiler do? Answer in three short paragraphs.",
    ],
    "open": [
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Describe a city a thousand years in the future.",
        "Write a persuasive paragraph about why people should learn to cook.",
        "Invent a new board game and describe its rules.",
    ],
}


def all_prompts() -> list[tuple[str, str]]:
    return [(cat, p) for cat, ps in PROMPTS.items() for p in ps]


def encode_prompt(tokenizer, text: str) -> list[int]:
    """Apply the chat template when the tokenizer has one."""
    if getattr(tokenizer, "chat_template", None):
        return list(tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True))
    return list(tokenizer.encode(text))
