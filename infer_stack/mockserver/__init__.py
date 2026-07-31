"""
A deterministic, OpenAI-compatible mock inference server.

Lets evaluation cards, HELM runs and deployment plumbing be exercised
end-to-end with no GPU, no API key and no network -- while still producing
*varied* behaviour across models and questions, so the statistics a card
computes are exercised rather than short-circuited.

See :mod:`infer_stack.mockserver.simulator` for why fixed responses are not
good enough, and :mod:`infer_stack.mockserver.server` for the HTTP surface.

Warning:
    A card that passes against this server has demonstrated that its
    plumbing and statistics work.  It has demonstrated nothing about any
    real model.  Runs against the mock should be marked as such and must
    never be submitted as evaluation results.
"""

from .server import MockServer, build_simulator
from .simulator import (
    ModelProfile,
    SimulatedCompletion,
    Simulator,
    default_consistency_for,
    flatten_messages,
    unit_hash,
)

__all__ = [
    'MockServer',
    'ModelProfile',
    'SimulatedCompletion',
    'Simulator',
    'build_simulator',
    'default_consistency_for',
    'flatten_messages',
    'unit_hash',
]
