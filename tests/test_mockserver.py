"""
Tests for the deterministic mock inference server.

The properties worth pinning down are the ones a naive mock gets wrong and
that silently corrupt evaluation results rather than failing loudly:
reproducibility, per-model variation, per-question difficulty coupling,
and self-consistency that actually tracks the configured rate.
"""

import json
import urllib.error
import urllib.request

import pytest

from infer_stack.mockserver import (
    ModelProfile,
    MockServer,
    Simulator,
    default_consistency_for,
)


COHORT = {
    'seed': 'test-seed',
    'models': {
        'weak': {'ability': 0.05},
        'middling': {'ability': 0.5},
        'strong': {'ability': 0.95},
    },
    'questions': {f'q{i}': f'Question number {i} about topic {i}?' for i in range(40)},
    'answer_key': {f'q{i}': f'gold-{i}' for i in range(40)},
}


def _simulator(config=None):
    from infer_stack.mockserver.server import build_simulator

    return build_simulator(config or COHORT)


def _ask(sim, model_id, question_id, temperature=0.0, sample_index=0):
    messages = [{'role': 'user', 'content': COHORT['questions'][question_id]}]
    return sim.complete(
        model_id, messages, temperature=temperature, sample_index=sample_index
    )


# -- determinism -----------------------------------------------------------


def test_identical_requests_give_identical_answers():
    sim = _simulator()
    first = _ask(sim, 'middling', 'q7')
    second = _ask(sim, 'middling', 'q7')
    assert first.text == second.text
    assert first.is_correct == second.is_correct


def test_two_servers_with_the_same_seed_agree():
    a, b = _simulator(), _simulator()
    for question_id in list(COHORT['questions'])[:10]:
        assert _ask(a, 'strong', question_id).text == _ask(
            b, 'strong', question_id
        ).text


def test_changing_the_seed_changes_the_world():
    other = dict(COHORT, seed='different-seed')
    baseline = _simulator()
    changed = _simulator(other)
    answers_differ = any(
        _ask(baseline, 'middling', q).text != _ask(changed, 'middling', q).text
        for q in list(COHORT['questions'])[:20]
    )
    assert answers_differ


# -- the properties a fixed-response mock would break ----------------------


def test_ability_orders_accuracy_across_the_cohort():
    # A cohort where every model scores the same makes an accuracy
    # prediction fit degenerate, and the card would measure nothing.
    sim = _simulator()
    rates = {}
    for model_id in ('weak', 'middling', 'strong'):
        correct = sum(
            _ask(sim, model_id, q).is_correct for q in COHORT['questions']
        )
        rates[model_id] = correct / len(COHORT['questions'])

    assert rates['weak'] < rates['middling'] < rates['strong'], rates


def test_difficulty_is_shared_across_models():
    # Errors must be correlated across the cohort; independent errors would
    # leave a cohort-level fit with nothing to learn.
    sim = _simulator()
    hard = max(COHORT['questions'], key=sim.difficulty_for)
    easy = min(COHORT['questions'], key=sim.difficulty_for)
    assert sim.difficulty_for(hard) > sim.difficulty_for(easy)
    # The same question object is scored identically regardless of model.
    assert sim.difficulty_for(hard) == sim.difficulty_for(hard)


def test_greedy_decoding_never_drifts():
    sim = _simulator()
    base = _ask(sim, 'middling', 'q3', temperature=0.0, sample_index=0)
    for sample_index in range(1, 6):
        resample = _ask(
            sim, 'middling', 'q3', temperature=0.0, sample_index=sample_index
        )
        assert resample.text == base.text


def test_sampling_diverges_and_tracks_the_consistency_rate():
    # Self-consistency cards sample the same prompt K times and score how
    # often the answers agree.  If resamples never diverge the score is
    # identically 1.0 and the card silently measures nothing.
    sim = _simulator()
    agreements = {}
    for model_id in ('weak', 'strong'):
        agree = 0
        for question_id in COHORT['questions']:
            base = _ask(sim, model_id, question_id, temperature=0.7)
            resample = _ask(
                sim, model_id, question_id, temperature=0.7, sample_index=1
            )
            agree += (base.text == resample.text)
        agreements[model_id] = agree / len(COHORT['questions'])

    assert agreements['strong'] > agreements['weak'], agreements
    # Not degenerate in either direction.
    assert 0.0 < agreements['weak'] < 1.0


def test_consistency_defaults_to_a_function_of_ability():
    assert default_consistency_for(0.0) == pytest.approx(0.45)
    assert default_consistency_for(1.0) == pytest.approx(0.95)
    profile = ModelProfile('m', ability=0.8)
    assert profile.consistency == pytest.approx(default_consistency_for(0.8))


def test_explicit_consistency_overrides_the_default():
    profile = ModelProfile('m', ability=0.8, consistency=0.1)
    assert profile.consistency == pytest.approx(0.1)


# -- latent question identity ---------------------------------------------


def test_rephrased_prompts_about_a_known_question_share_an_identity():
    # A card that re-asks the same underlying question in a decomposed form
    # must be able to agree with its own direct answer; that only works if
    # both calls resolve to the same latent question.
    sim = _simulator()
    direct = [{'role': 'user', 'content': COHORT['questions']['q5']}]
    wrapped = [
        {'role': 'system', 'content': 'Answer concisely.'},
        {
            'role': 'user',
            'content': (
                f'Passage: some context.\n'
                f'{COHORT["questions"]["q5"]}\n'
                f'Answer with a short phrase.'
            ),
        },
    ]
    assert (
        sim.complete('strong', direct).latent_key
        == sim.complete('strong', wrapped).latent_key
        == 'q5'
    )


def test_unknown_prompts_get_a_stable_anonymous_identity():
    sim = _simulator()
    messages = [{'role': 'user', 'content': 'something never registered'}]
    first = sim.complete('strong', messages)
    second = sim.complete('strong', messages)
    assert first.latent_key.startswith('anon:')
    assert first.latent_key == second.latent_key


def test_unknown_model_is_rejected():
    sim = _simulator()
    with pytest.raises(KeyError):
        sim.complete('not-configured', [{'role': 'user', 'content': 'hi'}])


# -- HTTP surface ----------------------------------------------------------


def _post(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def test_openai_chat_completions_shape():
    with MockServer(COHORT, port=0) as server:
        body = _post(
            server.url + '/v1/chat/completions',
            {
                'model': 'strong',
                'messages': [
                    {'role': 'user', 'content': COHORT['questions']['q1']}
                ],
                'temperature': 0.0,
            },
        )
    assert body['object'] == 'chat.completion'
    assert body['model'] == 'strong'
    assert body['choices'][0]['message']['role'] == 'assistant'
    assert isinstance(body['choices'][0]['message']['content'], str)


def test_n_samples_returns_n_choices():
    with MockServer(COHORT, port=0) as server:
        body = _post(
            server.url + '/v1/chat/completions',
            {
                'model': 'weak',
                'messages': [
                    {'role': 'user', 'content': COHORT['questions']['q2']}
                ],
                'temperature': 0.7,
                'n': 3,
            },
        )
    assert len(body['choices']) == 3
    assert [c['index'] for c in body['choices']] == [0, 1, 2]


def test_health_and_models_endpoints():
    with MockServer(COHORT, port=0) as server:
        assert _get(server.url + '/health')['status'] == 'ok'
        models = _get(server.url + '/v1/models')
    assert {m['id'] for m in models['data']} == {'weak', 'middling', 'strong'}


def test_requests_are_recorded_verbatim_for_parity_checks():
    # Recording exact payloads is the reason this is a server rather than
    # an in-process fake: two clients can be diffed against each other to
    # prove they present the same prompt to the model.
    with MockServer(COHORT, port=0) as server:
        payload = {
            'model': 'strong',
            'messages': [{'role': 'user', 'content': 'hello there'}],
            'temperature': 0.0,
        }
        _post(server.url + '/v1/chat/completions', payload)
        recorded = _get(server.url + '/__mock__/requests')['requests']

    assert len(recorded) == 1
    assert recorded[0]['body'] == payload
    assert recorded[0]['path'] == '/v1/chat/completions'


def test_recorded_requests_can_be_reset():
    with MockServer(COHORT, port=0) as server:
        _post(
            server.url + '/v1/chat/completions',
            {'model': 'strong', 'messages': [{'role': 'user', 'content': 'a'}]},
        )
        _post(server.url + '/__mock__/reset', {})
        recorded = _get(server.url + '/__mock__/requests')['requests']
    assert recorded == []


def test_repeated_sampling_requests_diverge_over_http():
    # Real clients draw K samples by sending the same body K times -- the
    # OpenAI API has no "sample index". If the mock keyed only on request
    # content those K calls would collapse to one answer and every
    # self-consistency score would be identically 1.0.
    body = {
        'model': 'weak',
        'messages': [{'role': 'user', 'content': COHORT['questions']['q4']}],
        'temperature': 0.7,
    }
    with MockServer(COHORT, port=0) as server:
        answers = [
            _post(server.url + '/v1/chat/completions', body)['choices'][0][
                'message'
            ]['content']
            for _ in range(8)
        ]
    assert len(set(answers)) > 1, answers


def test_repeated_greedy_requests_stay_identical_over_http():
    body = {
        'model': 'weak',
        'messages': [{'role': 'user', 'content': COHORT['questions']['q4']}],
        'temperature': 0.0,
    }
    with MockServer(COHORT, port=0) as server:
        answers = [
            _post(server.url + '/v1/chat/completions', body)['choices'][0][
                'message'
            ]['content']
            for _ in range(8)
        ]
    assert len(set(answers)) == 1, answers


def test_a_strong_model_resamples_more_stably_than_a_weak_one():
    def _spread(model_id, server):
        distinct = 0
        for question_id in list(COHORT['questions'])[:25]:
            body = {
                'model': model_id,
                'messages': [
                    {'role': 'user',
                     'content': COHORT['questions'][question_id]}
                ],
                'temperature': 0.7,
            }
            answers = {
                _post(server.url + '/v1/chat/completions', body)['choices'][0][
                    'message'
                ]['content']
                for _ in range(3)
            }
            distinct += len(answers)
        return distinct

    with MockServer(COHORT, port=0) as server:
        strong_spread = _spread('strong', server)
        weak_spread = _spread('weak', server)

    assert strong_spread < weak_spread, (strong_spread, weak_spread)


def test_n_greater_than_one_reserves_a_contiguous_block():
    # A client asking for n=3 in one request and a client sending three
    # requests should both advance the counter by three, so the two styles
    # cannot silently share sample indices.
    body = {
        'model': 'weak',
        'messages': [{'role': 'user', 'content': COHORT['questions']['q6']}],
        'temperature': 0.7,
        'n': 3,
    }
    with MockServer(COHORT, port=0) as server:
        first = _post(server.url + '/v1/chat/completions', body)
        second = _post(server.url + '/v1/chat/completions', body)

    first_texts = [c['message']['content'] for c in first['choices']]
    second_texts = [c['message']['content'] for c in second['choices']]
    assert first_texts != second_texts


def test_unknown_model_over_http_is_a_404_naming_the_configured_models():
    with MockServer(COHORT, port=0) as server:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(
                server.url + '/v1/chat/completions',
                {'model': 'nope', 'messages': [{'role': 'user', 'content': 'a'}]},
            )
        assert excinfo.value.code == 404
        detail = json.loads(excinfo.value.read())['error']['message']
    assert 'strong' in detail


def test_failure_injection_returns_the_configured_status():
    config = dict(
        COHORT,
        models={'flaky': {'ability': 0.5, 'failure_rate': 1.0,
                          'failure_status': 429}},
    )
    with MockServer(config, port=0) as server:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(
                server.url + '/v1/chat/completions',
                {'model': 'flaky',
                 'messages': [{'role': 'user', 'content': 'a'}]},
            )
    assert excinfo.value.code == 429


def test_simulator_requires_profiles_to_be_configured():
    sim = Simulator(profiles={}, seed='x')
    with pytest.raises(KeyError):
        sim.complete('anything', [{'role': 'user', 'content': 'a'}])


# -- raw completions endpoint ---------------------------------------------


def test_completions_endpoint_shape():
    # Clients whose prompts are already rendered (a chat template applied
    # upstream) must not have a second template applied by the server, so
    # they use /v1/completions rather than /v1/chat/completions.
    with MockServer(COHORT, port=0) as server:
        body = _post(
            server.url + '/v1/completions',
            {'model': 'strong', 'prompt': COHORT['questions']['q1'],
             'temperature': 0.0},
        )
    assert body['object'] == 'text_completion'
    choice = body['choices'][0]
    assert isinstance(choice['text'], str)
    assert 'message' not in choice, 'completions return text, not a message'


def test_completions_and_chat_agree_on_the_same_prompt():
    prompt = COHORT['questions']['q3']
    with MockServer(COHORT, port=0) as server:
        completion = _post(
            server.url + '/v1/completions',
            {'model': 'strong', 'prompt': prompt, 'temperature': 0.0},
        )['choices'][0]['text']
        chat = _post(
            server.url + '/v1/chat/completions',
            {'model': 'strong',
             'messages': [{'role': 'user', 'content': prompt}],
             'temperature': 0.0},
        )['choices'][0]['message']['content']
    assert completion == chat


def test_completions_n_indices_are_per_response():
    # OpenAI indices are 0..n-1 within a response, not a running counter.
    body = {'model': 'weak', 'prompt': COHORT['questions']['q2'],
            'temperature': 0.7, 'n': 3}
    with MockServer(COHORT, port=0) as server:
        first = _post(server.url + '/v1/completions', body)
        second = _post(server.url + '/v1/completions', body)
    assert [c['index'] for c in first['choices']] == [0, 1, 2]
    assert [c['index'] for c in second['choices']] == [0, 1, 2]
    # ...but the samples themselves still advance.
    assert ([c['text'] for c in first['choices']]
            != [c['text'] for c in second['choices']])
