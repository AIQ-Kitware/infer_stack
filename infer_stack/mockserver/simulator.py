"""
Deterministic response simulator for the mock inference server.

The point of this module is to be *boring and reproducible*: given the same
server seed, model, prompt and sample index, it always returns the same
completion.  Nothing here calls out to a real model.

Why it is not just "return a fixed string"
------------------------------------------

Evaluation cards measure how model behaviour *varies*:

* Self-consistency cards sample the same prompt K times and score how often
  the answers agree.  A server that returns one fixed string makes that
  score identically 1.0, so the card passes while measuring nothing.
* Cards that predict accuracy from a behavioural signal fit a curve across
  a cohort.  A server where every model is equally accurate makes that fit
  degenerate, so the card's headline number is meaningless.

So the simulator models two per-model traits and lets them drive both
correctness and agreement:

``ability``
    Probability-ish scale for getting a question right.  Combined with a
    per-question difficulty so that hard questions are hard for everyone
    and strong models are better everywhere.
``consistency``
    Probability that a resample of the same latent question returns the
    same answer.  Defaults to a function of ``ability`` so that stronger
    models are also steadier -- the correlation real cards rely on.

Because both derive from ``ability`` by default, a cohort configured only
with abilities still produces the accuracy/consistency correlation that
makes an accuracy-prediction card non-trivial.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    'ModelProfile',
    'Simulator',
    'SimulatedCompletion',
    'default_consistency_for',
    'unit_hash',
]


def unit_hash(*parts: Any) -> float:
    """
    Deterministically map arbitrary parts onto a float in ``[0, 1)``.

    This is the only source of randomness in the simulator.  It is a hash
    rather than an RNG so that results never depend on call order, which
    keeps parallel and resumed runs identical.

    Args:
        *parts: values to fold into the digest; each is str()-ed.

    Returns:
        float: a value in ``[0, 1)``.

    Example:
        >>> from infer_stack.mockserver.simulator import unit_hash
        >>> a = unit_hash('seed', 'model-a', 'question-1')
        >>> b = unit_hash('seed', 'model-a', 'question-1')
        >>> assert a == b, 'must be reproducible'
        >>> assert a != unit_hash('seed', 'model-b', 'question-1')
        >>> assert 0.0 <= a < 1.0
    """
    digest = hashlib.blake2b(
        '\x1f'.join(str(part) for part in parts).encode('utf-8'),
        digest_size=8,
    ).digest()
    (value,) = struct.unpack('<Q', digest)
    return value / float(1 << 64)


def default_consistency_for(ability: float) -> float:
    """
    Derive a plausible self-consistency rate from an ability score.

    Stronger models are steadier, but even a perfect model is not perfectly
    reproducible at nonzero temperature, and even a weak one repeats itself
    sometimes -- so the mapping is compressed into ``[0.45, 0.95]`` rather
    than spanning the whole unit interval.

    Args:
        ability (float): the model's ability in ``[0, 1]``.

    Returns:
        float: a consistency rate in ``[0.45, 0.95]``.

    Example:
        >>> from infer_stack.mockserver.simulator import default_consistency_for
        >>> assert default_consistency_for(0.0) == 0.45
        >>> assert default_consistency_for(1.0) == 0.95
        >>> assert default_consistency_for(0.5) == 0.70
    """
    ability = min(max(float(ability), 0.0), 1.0)
    return 0.45 + 0.50 * ability


@dataclass
class ModelProfile:
    """
    Behavioural configuration for one simulated model.

    Attributes:
        model_id (str): the id clients ask for.
        ability (float): competence in ``[0, 1]``.
        consistency (float | None): agreement rate for resamples of the same
            latent question.  Defaults to :func:`default_consistency_for`.
        failure_rate (float): fraction of requests answered with an HTTP
            error instead of a completion, for exercising retry paths.
        failure_status (int): status code used for injected failures.
        latency_s (float): artificial delay per request, in seconds.
        extra (dict): free-form metadata echoed back on ``/v1/models``.
    """

    model_id: str
    ability: float = 0.5
    consistency: float | None = None
    failure_rate: float = 0.0
    failure_status: int = 503
    latency_s: float = 0.0
    #: Response mode; see :mod:`infer_stack.mockserver.modes`.
    mode: str = 'simulate'
    #: Serving settings a vLLM-backed endpoint would advertise or enforce.
    max_model_len: int | None = None
    served_model_name: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        from .modes import resolve_mode
        # Validate eagerly: a typo'd mode would otherwise behave like a
        # normal model and be discovered only by a puzzling result.
        resolve_mode(self.mode)
        self.ability = min(max(float(self.ability), 0.0), 1.0)
        if self.consistency is None:
            self.consistency = default_consistency_for(self.ability)
        self.consistency = min(max(float(self.consistency), 0.0), 1.0)

    @classmethod
    def coerce(cls, model_id: str, data: Mapping[str, Any] | None) -> 'ModelProfile':
        """
        Build a profile from a config mapping.

        Args:
            model_id (str): the model id.
            data (Mapping | None): the model's config block.

        Returns:
            ModelProfile
        """
        data = dict(data or {})
        known = {
            'ability',
            'consistency',
            'failure_rate',
            'failure_status',
            'latency_s',
            'mode',
            'max_model_len',
            'served_model_name',
        }
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            model_id=model_id,
            ability=data.get('ability', 0.5),
            consistency=data.get('consistency'),
            failure_rate=data.get('failure_rate', 0.0),
            failure_status=data.get('failure_status', 503),
            latency_s=data.get('latency_s', 0.0),
            mode=data.get('mode', 'simulate'),
            max_model_len=data.get('max_model_len'),
            served_model_name=data.get('served_model_name'),
            extra=extra,
        )


@dataclass
class SimulatedCompletion:
    """
    One simulated answer plus the reasoning that produced it.

    Attributes:
        text (str): the completion body sent to the client.
        is_correct (bool): whether it matches the gold answer.
        latent_key (str): the question identity used for keying.
        should_fail (bool): whether this request should be an injected
            error instead of a completion.
    """

    text: str
    is_correct: bool
    latent_key: str
    should_fail: bool = False
    finish_reason: str = 'stop'


class Simulator:
    """
    Turn a chat request into a deterministic completion.

    Args:
        profiles (Mapping[str, ModelProfile]): configured models.
        answer_key (Mapping[str, str] | None): maps a *question id* to its
            gold answer.  When a prompt contains a known question's text,
            all calls about that question share one latent identity -- so a
            direct answer and a decomposed re-ask agree exactly as often as
            the model's ``consistency`` says they should.
        questions (Mapping[str, str] | None): maps question id to the
            question text used for substring matching.
        seed (str): server-wide seed; change it to get a different but
            equally reproducible world.

    Example:
        >>> from infer_stack.mockserver.simulator import ModelProfile, Simulator
        >>> sim = Simulator(
        ...     profiles={'strong': ModelProfile('strong', ability=0.95),
        ...               'weak': ModelProfile('weak', ability=0.05)},
        ...     answer_key={'q1': 'Paris'},
        ...     questions={'q1': 'What is the capital of France?'},
        ...     seed='demo',
        ... )
        >>> msgs = [{'role': 'user',
        ...          'content': 'What is the capital of France?'}]
        >>> strong = sim.complete('strong', msgs, temperature=0.0, sample_index=0)
        >>> weak = sim.complete('weak', msgs, temperature=0.0, sample_index=0)
        >>> assert strong.latent_key == 'q1'
        >>> assert strong.is_correct and not weak.is_correct
        >>> # Reproducible.
        >>> again = sim.complete('strong', msgs, temperature=0.0, sample_index=0)
        >>> assert again.text == strong.text
    """

    def __init__(
        self,
        profiles: Mapping[str, ModelProfile],
        answer_key: Mapping[str, str] | None = None,
        questions: Mapping[str, str] | None = None,
        seed: str = 'infer-stack-mock',
        composition: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.profiles = dict(profiles)
        self.answer_key = dict(answer_key or {})
        self.questions = dict(questions or {})
        self.composition = {k: tuple(v) for k, v in (composition or {}).items()}
        self.seed = seed

        # A question may register several surface forms. That matters for
        # decomposition-style cards: a question asked directly and the same
        # question re-asked in composed form look nothing alike as strings,
        # but should resolve to one identity -- and only when the composed
        # form carries the *correct* intermediate answer. Registering the
        # correctly-composed string as a second form gives exactly that,
        # so direct/decomposed agreement tracks model ability instead of
        # being trivially 1.0.
        forms = []
        for question_id, value in self.questions.items():
            texts = value if isinstance(value, (list, tuple)) else [value]
            forms.extend((question_id, str(text)) for text in texts)

        # Longest form first, so a form containing another form's text
        # still matches the more specific one.
        self._match_order = sorted(
            forms, key=lambda kv: len(kv[1]), reverse=True
        )

    def resolve_profile(self, model_id):
        """
        Find a profile by id or by its ``served_model_name`` alias.

        A deployment often serves a model under a shorter name than its
        repository id; clients then ask for the alias.
        """
        profile = self.profiles.get(model_id)
        if profile is not None:
            return profile
        for candidate in self.profiles.values():
            if candidate.served_model_name == model_id:
                return candidate
        return None

    def latent_key_for(self, prompt: str) -> str:
        """
        Identify which latent question a prompt is about.

        Args:
            prompt (str): the flattened prompt text.

        Returns:
            str: a registered question id when the prompt quotes a known
                question, otherwise a hash of the prompt itself.
        """
        haystack = ' '.join(prompt.split()).lower()
        for question_id, text in self._match_order:
            needle = ' '.join(str(text).split()).lower()
            if needle and needle in haystack:
                return question_id
        return 'anon:' + hashlib.blake2b(
            haystack.encode('utf-8'), digest_size=8
        ).hexdigest()

    def knows(self, model_id: str, latent_key: str, profile=None) -> bool:
        """
        Whether ``model_id`` can answer ``latent_key`` correctly.

        Ability beats difficulty -> correct.  Difficulty is shared across
        models, which is what couples their errors together.

        A question declared in ``composition`` is *compositional*: the
        model gets it right only if it can also do every step it decomposes
        into.  Without this, a step's outcome and its parent's outcome are
        independent draws, and any feature derived from the decomposition
        carries no information about whether the final answer is right --
        so a card claiming that such a feature predicts correctness would
        measure a true null no matter how much data it was given.

        Args:
            model_id (str): the answering model.
            latent_key (str): the question identity.
            profile (ModelProfile | None): resolved from ``model_id`` when
                omitted.

        Returns:
            bool
        """
        if profile is None:
            profile = self.profiles[model_id]

        own = unit_hash(self.seed, 'knows', model_id, latent_key) < (
            _skill_vs_difficulty(
                profile.ability, self.difficulty_for(latent_key)
            )
        )
        if not own:
            return False

        # Recurse into declared steps. Depth is bounded by the config, and
        # a self-referential entry would otherwise loop forever.
        for step_key in self.composition.get(latent_key, ()):
            if step_key == latent_key:
                continue
            if not self.knows(model_id, step_key, self.profiles[model_id]):
                return False
        return True

    def difficulty_for(self, latent_key: str) -> float:
        """
        Per-question difficulty in ``[0, 1]``, stable across models.

        Shared difficulty is what makes model scores correlate: without it,
        every model's errors would be independent and a cohort-level fit
        would have nothing to learn.

        Args:
            latent_key (str): the question identity.

        Returns:
            float
        """
        return unit_hash(self.seed, 'difficulty', latent_key)

    def complete(
        self,
        model_id: str,
        messages: Sequence[Mapping[str, Any]],
        temperature: float = 0.0,
        sample_index: int = 0,
    ) -> SimulatedCompletion:
        """
        Produce one deterministic completion.

        Args:
            model_id (str): which configured model is answering.
            messages (Sequence[Mapping]): OpenAI-style chat messages.
            temperature (float): sampling temperature.  At 0 the model is
                treated as greedy, so resamples never diverge.
            sample_index (int): which of ``n`` samples this is.

        Returns:
            SimulatedCompletion
        """
        profile = self.resolve_profile(model_id)
        if profile is None:
            raise KeyError(model_id)

        prompt = flatten_messages(messages)
        latent_key = self.latent_key_for(prompt)

        should_fail = profile.failure_rate > 0.0 and unit_hash(
            self.seed, 'failure', model_id, prompt, sample_index
        ) < profile.failure_rate

        knows = self.knows(model_id, latent_key, profile)

        # Greedy decoding is reproducible by definition, so only a nonzero
        # temperature lets a resample drift off the model's usual answer.
        drifts = False
        if temperature > 0.0 and sample_index > 0:
            drifts = unit_hash(
                self.seed, 'drift', model_id, latent_key, sample_index
            ) >= profile.consistency

        is_correct = knows and not drifts
        gold = self.answer_key.get(latent_key)

        if is_correct and gold is not None:
            text = str(gold)
        elif is_correct:
            text = _synthetic_answer(self.seed, latent_key, 'gold')
        else:
            # A wrong answer still has to be *stably* wrong for a
            # consistent model, so key the distractor on the drift index.
            variant = sample_index if drifts else 0
            text = _synthetic_answer(
                self.seed, latent_key, f'{model_id}:{variant}'
            )

        from .modes import ModeContext, resolve_mode

        context = ModeContext(
            model_id=model_id, prompt=prompt, seed=self.seed,
            sample_index=sample_index, simulated_text=text,
            is_correct=is_correct, gold=gold,
            messages=list(messages),
        )
        text, finish_reason = resolve_mode(profile.mode)(context)

        return SimulatedCompletion(
            text=text,
            is_correct=is_correct,
            latent_key=latent_key,
            should_fail=should_fail,
            finish_reason=finish_reason,
        )


def _skill_vs_difficulty(ability: float, difficulty: float) -> float:
    """
    Combine ability and difficulty into a correctness probability.

    A simple bounded blend: ability dominates, difficulty shifts it. Kept
    deliberately simple -- this is a test fixture, not psychometrics.
    """
    raw = 0.15 + 0.9 * ability - 0.35 * difficulty
    return min(max(raw, 0.02), 0.98)


def _synthetic_answer(seed: str, latent_key: str, salt: str) -> str:
    """Stable short pseudo-answer for prompts with no gold answer."""
    digest = hashlib.blake2b(
        f'{seed}|{latent_key}|{salt}'.encode('utf-8'), digest_size=4
    ).hexdigest()
    return f'answer-{digest}'


def flatten_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """
    Flatten chat messages into the text the simulator keys on.

    Args:
        messages (Sequence[Mapping]): OpenAI-style chat messages.

    Returns:
        str: newline-joined ``role: content`` lines.

    Example:
        >>> from infer_stack.mockserver.simulator import flatten_messages
        >>> flatten_messages([{'role': 'user', 'content': 'hi'}])
        'user: hi'
    """
    lines = []
    for message in messages:
        content = message.get('content', '')
        if isinstance(content, list):
            # OpenAI content-parts form.
            content = ' '.join(
                part.get('text', '')
                for part in content
                if isinstance(part, Mapping)
            )
        lines.append(f'{message.get("role", "user")}: {content}')
    return '\n'.join(lines)
