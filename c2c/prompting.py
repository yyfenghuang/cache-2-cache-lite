"""The MMLU prompt, defined once.

No I/O, no model loading. Tokenizing is neither.

This module exists because of what went wrong one tier below. The projection
was fitted on wikitext and graded on multiple choice questions, and
`results/distribution_shift.json` measured how little of it survived the trip.
The fuser is trained on question-shaped text precisely to close that gap, and
the gap only actually closes if the text it trains on is the same text it is
graded on, down to the newlines.

Two copies of a prompt template drift. They drift silently, because both
copies keep producing well-formed prompts and the only symptom is a result
that is slightly worse than it should be, which is indistinguishable from the
mechanism not working. So there is one copy, here, and both the training
script and the grading script import it.

The last line is "Answer:" with no trailing space. The option letters are
scored with a leading space, so the space belongs to the token being
predicted rather than to the prompt. Moving it would change which token the
comparison is between.
"""

from __future__ import annotations

__all__ = ["OPTIONS", "build_prompt", "option_token_ids"]

OPTIONS = ("A", "B", "C", "D")


def build_prompt(subject: str, question: str, choices) -> str:
    topic = subject.replace("_", " ")
    lines = [
        f"The following are multiple choice questions (with answers) about {topic}.",
        "",
        question.strip(),
    ]
    lines += [f"{letter}. {choice}" for letter, choice in zip(OPTIONS, choices)]
    lines.append("Answer:")
    return "\n".join(lines)


def option_token_ids(tokenizer) -> list[int]:
    """One token per option, or the comparison is not between like things."""
    ids = []
    for letter in OPTIONS:
        encoded = tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(
                f"option {letter!r} encodes to {len(encoded)} tokens, so the "
                "four scores would not be comparable"
            )
        ids.append(int(encoded[0]))
    if len(set(ids)) != len(ids):
        raise ValueError(f"option letters collide on token ids {ids}")
    return ids