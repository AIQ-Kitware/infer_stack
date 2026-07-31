"""
Response modes: what the mock says, as opposed to whether it is right.

The default mode answers from the simulator, which is what an evaluation
card wants. The others exist because a card's scoring, parsing and
robustness paths are hard to exercise against a well-behaved model, and
several of them are more entertaining than the thing they replace.

Each mode is a function of a :class:`ModeContext` returning
``(text, finish_reason)``. They are deliberately deterministic -- the
sampling seed reaches them -- so a run that misbehaves can be reproduced.

Some are jokes that turned out to be useful:

``sycophant``
    Always agrees, enthusiastically. A card that mistakes agreement for
    correctness scores this at 100%, which is exactly the bug worth
    catching before a real sycophantic model does it more subtly.
``echo``
    Returns the prompt it was given. The fastest way to see what a client
    actually sent after templating, without reading a transcript.
``thinking``
    Wraps the answer in ``<think>`` ... ``</think>``. Clients that strip
    reasoning segments need something to strip.
``truncated``
    Stops mid-sentence with ``finish_reason='length'``. Exercises the
    "ran out of budget" path that otherwise only appears at 3am.
``markov``
    Babbles a bigram chain built from the prompt. On-topic, fluent, and
    meaningless -- so answer extraction has to actually work rather than
    bailing out on obviously-unparseable text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .simulator import unit_hash

__all__ = ['MODES', 'ModeContext', 'resolve_mode', 'available_modes']


@dataclass
class ModeContext:
    """Everything a mode may use to decide what to say."""

    model_id: str
    prompt: str
    seed: str
    sample_index: int
    #: What the simulator would have said.
    simulated_text: str
    #: Whether the simulated answer matches the gold answer.
    is_correct: bool
    #: The gold answer, when the question is in the answer key.
    gold: Optional[str] = None
    #: The request's messages, verbatim. ``prompt`` is their flattened
    #: form, which is what keying uses; a mode that wants to show a client
    #: what it sent should use these instead.
    messages: list = field(default_factory=list)

    def choice(self, options, salt: str = ''):
        """Deterministically pick one of ``options``."""
        options = list(options)
        index = int(
            unit_hash(self.seed, salt, self.model_id, self.prompt,
                      self.sample_index) * len(options)
        )
        return options[min(index, len(options) - 1)]


def _simulate(ctx: ModeContext):
    return ctx.simulated_text, 'stop'


_AGREEMENT = [
    "You're absolutely correct!",
    "You're absolutely right, and I apologize for the confusion.",
    "Great catch — you're completely correct.",
    "Yes, exactly right. Well spotted!",
    "You make an excellent point, and you're right.",
]


def _sycophant(ctx: ModeContext):
    return ctx.choice(_AGREEMENT, 'sycophant'), 'stop'


_EIGHT_BALL = [
    'It is certain.', 'Reply hazy, try again.', 'Don\'t count on it.',
    'Outlook good.', 'Cannot predict now.', 'Signs point to yes.',
    'My sources say no.', 'Concentrate and ask again.',
    'Without a doubt.', 'Very doubtful.',
]


def _magic_8ball(ctx: ModeContext):
    return ctx.choice(_EIGHT_BALL, '8ball'), 'stop'


def _echo(ctx: ModeContext):
    """Return the content the client sent, without role decoration.

    ``ctx.prompt`` is the flattened form used for keying and carries
    ``role:`` prefixes; those are an artifact of keying, not of the
    request, so echoing them back would misreport what was sent.
    """
    contents = [str(m.get('content', '')) for m in ctx.messages]
    return ('\n'.join(contents) if contents else ctx.prompt), 'stop'


def _thinking(ctx: ModeContext):
    reasoning = ctx.choice([
        'Let me work through this step by step.',
        'Hmm, I need to be careful here.',
        'Okay, breaking this down.',
        'Wait, let me reconsider.',
    ], 'think')
    return (f'<think>\n{reasoning}\n</think>\n\n{ctx.simulated_text}', 'stop')


def _truncated(ctx: ModeContext):
    text = f'{ctx.simulated_text} and therefore the answer must be'
    return text[: max(8, len(text) // 2)], 'length'


def _empty(ctx: ModeContext):
    return '', 'stop'


def _confidently_wrong(ctx: ModeContext):
    wrong = f'definitely-not-{ctx.gold}' if ctx.gold else ctx.simulated_text
    preamble = ctx.choice([
        'I am certain the answer is',
        'Without question, the answer is',
        'This is unambiguously',
    ], 'confident')
    return f'{preamble} {wrong}. There is no room for doubt.', 'stop'


#: Words a walk prefers to stop after, so output ends like a sentence
#: rather than being guillotined mid-clause.
_TERMINALS = frozenset('.!?')


def _markov(ctx: ModeContext):
    """
    Babble a bigram chain built from the prompt's own vocabulary.

    The result is on-topic, fluent-looking, and meaningless -- which is
    exactly what makes it useful. A mock that answers ``answer-3f9c`` is
    trivially unparseable, so a scorer that quietly fails to extract an
    answer looks the same as one that extracts a wrong answer. Text drawn
    from the prompt's own words defeats that: it has the register, the
    vocabulary and the length of a real response, so extraction and
    scoring have to actually work rather than bailing out early.

    It also catches the opposite bug. A scorer that rewards topical
    overlap with the question -- lexical F1 against the prompt, say --
    will score this well, and it deserves not to.

    Deterministic like every other mode: each step's choice is a hash of
    the seed, model, prompt, sample index and step number.
    """
    tokens = re.findall(r"[\w'\u2019-]+|[.,!?;:]", ctx.prompt)
    if len(tokens) < 6:
        # Too little to chain on; babbling from three words is not funny,
        # it is just broken.
        return ctx.simulated_text, 'stop'

    bigrams: dict = {}
    for current, following in zip(tokens, tokens[1:]):
        bigrams.setdefault(current, []).append(following)

    # Prefer starting on a capitalized word, the way a sentence would.
    openers = [t for t in tokens if t[:1].isupper()] or tokens
    word = ctx.choice(openers, 'markov:start')

    out = [word]
    limit = 12 + int(unit_hash(ctx.seed, 'markov:len', ctx.model_id,
                               ctx.prompt, ctx.sample_index) * 24)
    for step in range(limit):
        options = bigrams.get(word)
        if not options:
            break
        index = int(
            unit_hash(ctx.seed, f'markov:{step}', ctx.model_id, ctx.prompt,
                      ctx.sample_index) * len(options)
        )
        word = options[min(index, len(options) - 1)]
        out.append(word)
        if word in _TERMINALS and step >= 6:
            break

    text = ' '.join(out)
    # Undo the spaces the tokenizer split before punctuation.
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    if text[-1:] not in _TERMINALS:
        text += '.'
    return text, 'stop'


def _pirate(ctx: ModeContext):
    return (f"Arr! After much deliberation on the high seas, I reckon "
            f"it be {ctx.simulated_text}, matey!"), 'stop'


#: Registry of response modes, keyed by the name a config uses.
MODES: dict[str, Callable[[ModeContext], Any]] = {
    'simulate': _simulate,
    'sycophant': _sycophant,
    'magic_8ball': _magic_8ball,
    'echo': _echo,
    'thinking': _thinking,
    'truncated': _truncated,
    'empty': _empty,
    'confidently_wrong': _confidently_wrong,
    'markov': _markov,
    'pirate': _pirate,
}


def available_modes() -> list:
    """Sorted mode names, for error messages and ``--help``."""
    return sorted(MODES)


def resolve_mode(name: Optional[str]):
    """
    Look up a mode by name.

    Args:
        name (str | None): mode name; None or empty means ``simulate``.

    Returns:
        Callable[[ModeContext], tuple]

    Raises:
        ValueError: naming a mode that does not exist, rather than
            silently falling back -- a typo'd mode would otherwise look
            like a model that behaves normally.
    """
    if not name:
        return MODES['simulate']
    try:
        return MODES[name]
    except KeyError:
        raise ValueError(
            f'unknown response mode {name!r}; available: {available_modes()}'
        ) from None
