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

Both ends of that convention live here: how the question is written, and how
the answer is read back off the logits it produces. They are one agreement
and splitting them across two files is how they come apart.
"""

from __future__ import annotations

import torch

__all__ = [
    "OPTIONS",
    "build_prompt",
    "decompose_loss",
    "option_token_ids",
]

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


# The identity below is exact in real arithmetic and accurate to rounding in
# float32. The tolerance is relative, not absolute: the quantities being
# compared are unbounded above, and an absolute tolerance would pass
# everywhere the numbers are small and fail everywhere they are large without
# either outcome saying anything about correctness.
IDENTITY_TOLERANCE = 1e-4


def decompose_loss(logits: torch.Tensor, letters: list[int], answer: int) -> dict:
    """Split one question's loss into the part the grader sees and the rest.

    Write `M` for the probability mass on the four option letters together
    and `c` for the correct one. Then

        -log p(c)  =  -log( p(c) / M )  +  -log M

    The left side is what gets trained and reported. The first term on the
    right lives in the same space as the argmax the grader takes, so it is
    the only part that can change an answer. The second is the cost of
    spending probability anywhere other than on the four letters, and the
    grader normalises it away entirely.

    The two terms sum to the whole exactly, which is what makes this an
    attribution rather than an interpretation. Measured on the first fused
    run, the second term carried 76 percent of the improvement.

    The identity is asserted per question rather than trusted, because a
    decomposition that silently fails to close would report two plausible
    numbers that happen not to be about the same quantity.
    """
    letter_logits = logits[letters]
    total = torch.logsumexp(logits, dim=0)
    letter_total = torch.logsumexp(letter_logits, dim=0)

    full = float(total - logits[letters[answer]])
    four_way = float(letter_total - letter_logits[answer])
    letter_mass = float(total - letter_total)

    if abs(full - (four_way + letter_mass)) > IDENTITY_TOLERANCE * max(
        1.0, abs(full)
    ):
        raise ValueError(
            f"the decomposition does not close: {full} against "
            f"{four_way} + {letter_mass}. These are supposed to be one "
            "quantity written two ways, so a gap is a bug here and not a "
            "property of the model."
        )

    chosen = int(torch.argmax(letter_logits))
    return {
        "full": full,
        "four_way": four_way,
        "letter_mass": letter_mass,
        "mass_on_letters": float(torch.exp(letter_total - total)),
        "chosen": chosen,
        "correct": chosen == answer,
    }