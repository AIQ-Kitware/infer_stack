"""
Tests for the deterministic mock inference server.

The properties worth pinning down are the ones a naive mock gets wrong and
that silently corrupt evaluation results rather than failing loudly:
reproducibility, per-model variation, per-question difficulty coupling,
and self-consistency that actually tracks the configured rate.
"""

import json
import re
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


# -- response modes --------------------------------------------------------


def _mode_cohort(mode, **over):
    config = dict(COHORT)
    config['models'] = {'m': dict({'ability': 0.9, 'mode': mode}, **over)}
    return config


def test_sycophant_mode_always_agrees():
    # A card that mistakes agreement for correctness scores this at 100%,
    # which is the bug worth catching before a real model does it subtly.
    with MockServer(_mode_cohort('sycophant'), port=0) as server:
        texts = {
            _post(server.url + '/v1/completions',
                  {'model': 'm', 'prompt': COHORT['questions'][f'q{i}']}
                  )['choices'][0]['text']
            for i in range(8)
        }
    assert texts, 'got responses'
    assert all('correct' in t.lower() or 'right' in t.lower() or
               'excellent' in t.lower() for t in texts), texts


def test_echo_mode_returns_the_prompt_it_was_sent():
    # The fastest way to see what a client actually sent after templating.
    prompt = 'Passage: x\nQuestion: y\nAnswer concisely.'
    with MockServer(_mode_cohort('echo'), port=0) as server:
        body = _post(server.url + '/v1/completions',
                     {'model': 'm', 'prompt': prompt})
    assert body['choices'][0]['text'] == prompt


def test_thinking_mode_emits_a_strippable_reasoning_segment():
    with MockServer(_mode_cohort('thinking'), port=0) as server:
        text = _post(server.url + '/v1/completions',
                     {'model': 'm', 'prompt': COHORT['questions']['q1']}
                     )['choices'][0]['text']
    assert text.startswith('<think>')
    assert '</think>' in text


def test_truncated_mode_reports_length_not_stop():
    with MockServer(_mode_cohort('truncated'), port=0) as server:
        choice = _post(server.url + '/v1/completions',
                       {'model': 'm', 'prompt': COHORT['questions']['q1']}
                       )['choices'][0]
    assert choice['finish_reason'] == 'length'


def test_modes_are_still_deterministic():
    for mode in ('sycophant', 'magic_8ball', 'pirate', 'confidently_wrong'):
        payload = {'model': 'm', 'prompt': COHORT['questions']['q2']}
        with MockServer(_mode_cohort(mode), port=0) as server:
            first = _post(server.url + '/v1/completions', payload)
        with MockServer(_mode_cohort(mode), port=0) as server:
            second = _post(server.url + '/v1/completions', payload)
        assert (first['choices'][0]['text']
                == second['choices'][0]['text']), mode


def test_an_unknown_mode_is_rejected_at_config_time():
    # A typo would otherwise behave like a normal model and be discovered
    # only by a puzzling result.
    with pytest.raises(ValueError, match='unknown response mode'):
        ModelProfile.coerce('m', {'mode': 'sychophant'})


# -- authentication --------------------------------------------------------


AUTH_COHORT = dict(COHORT, api_keys=['sk-test-key'])


def _post_with(url, payload, token=None):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'), headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def test_a_valid_key_is_accepted():
    with MockServer(AUTH_COHORT, port=0) as server:
        body = _post_with(server.url + '/v1/completions',
                          {'model': 'strong', 'prompt': 'hi'},
                          token='sk-test-key')
    assert body['choices']


def test_a_missing_key_is_rejected_the_way_openai_does():
    with MockServer(AUTH_COHORT, port=0) as server:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post_with(server.url + '/v1/completions',
                       {'model': 'strong', 'prompt': 'hi'})
        assert excinfo.value.code == 401
        error = json.loads(excinfo.value.read())['error']
    assert error['code'] == 'invalid_api_key'
    assert 'Authorization header' in error['message']


def test_a_wrong_key_is_rejected():
    with MockServer(AUTH_COHORT, port=0) as server:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post_with(server.url + '/v1/completions',
                       {'model': 'strong', 'prompt': 'hi'}, token='sk-wrong')
        assert excinfo.value.code == 401


def test_auth_is_off_unless_asked_for():
    with MockServer(COHORT, port=0) as server:
        assert _post(server.url + '/v1/completions',
                     {'model': 'strong', 'prompt': 'hi'})['choices']


def test_health_stays_open_so_readiness_probes_work():
    with MockServer(AUTH_COHORT, port=0) as server:
        assert _get(server.url + '/health')['status'] == 'ok'


# -- vLLM serving settings -------------------------------------------------


def test_models_advertises_vllm_style_metadata():
    config = dict(COHORT)
    config['models'] = {'Qwen/Qwen3-8B': {'ability': 0.7,
                                          'max_model_len': 32768,
                                          'served_model_name': 'qwen3'}}
    with MockServer(config, port=0) as server:
        entry = _get(server.url + '/v1/models')['data'][0]
    assert entry['id'] == 'qwen3', 'served under its alias'
    assert entry['root'] == 'Qwen/Qwen3-8B'
    assert entry['owned_by'] == 'vllm'
    assert entry['max_model_len'] == 32768


def test_a_model_can_be_requested_by_its_served_alias():
    config = dict(COHORT)
    config['models'] = {'Qwen/Qwen3-8B': {'ability': 0.9,
                                          'served_model_name': 'qwen3'}}
    with MockServer(config, port=0) as server:
        by_alias = _post(server.url + '/v1/completions',
                         {'model': 'qwen3', 'prompt': 'hi'})
        by_id = _post(server.url + '/v1/completions',
                      {'model': 'Qwen/Qwen3-8B', 'prompt': 'hi'})
    assert by_alias['choices'][0]['text'] == by_id['choices'][0]['text']


def test_exceeding_max_model_len_is_a_context_length_error():
    # Otherwise only reachable with a genuinely long prompt, which makes the
    # client's overflow path effectively untested.
    config = dict(COHORT)
    config['models'] = {'m': {'ability': 0.9, 'max_model_len': 64}}
    with MockServer(config, port=0) as server:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(server.url + '/v1/completions',
                  {'model': 'm', 'prompt': 'x' * 400, 'max_tokens': 256})
        assert excinfo.value.code == 400
        error = json.loads(excinfo.value.read())['error']
    assert error['code'] == 'context_length_exceeded'
    assert 'maximum context length is 64' in error['message']


def test_markov_mode_babbles_from_the_prompts_own_vocabulary():
    # The point of this mode: output that looks like a real response, so a
    # scorer that quietly fails to extract an answer cannot hide behind
    # obviously-unparseable text.
    prompt = ('Passage: The mitochondrion is the powerhouse of the cell, '
              'generating most of the chemical energy needed to power '
              'biochemical reactions. Question: Which organelle generates '
              'chemical energy? Answer with a short phrase.')
    with MockServer(_mode_cohort('markov'), port=0) as server:
        text = _post(server.url + '/v1/completions',
                     {'model': 'm', 'prompt': prompt})['choices'][0]['text']

    prompt_words = set(re.findall(r"[\w'-]+", prompt.lower()))
    said = re.findall(r"[\w'-]+", text.lower())
    assert said, 'produced something'
    assert set(said) <= prompt_words, (
        f'markov must only use the prompt vocabulary; strays: '
        f'{sorted(set(said) - prompt_words)}')
    assert len(said) > 3, 'not a single word'
    assert text.rstrip()[-1] in '.!?', 'ends like a sentence'


def test_markov_mode_is_deterministic_but_varies_across_samples():
    prompt = ('The quick brown fox jumps over the lazy dog while the quick '
              'red fox watches the lazy cat sleep near the warm fire.')
    body = {'model': 'm', 'prompt': prompt, 'temperature': 0.7, 'n': 5}
    with MockServer(_mode_cohort('markov'), port=0) as server:
        first = [c['text'] for c in
                 _post(server.url + '/v1/completions', body)['choices']]
    with MockServer(_mode_cohort('markov'), port=0) as server:
        again = [c['text'] for c in
                 _post(server.url + '/v1/completions', body)['choices']]
    assert first == again, 'same seed, same babble'
    assert len(set(first)) > 1, 'samples should not all collapse to one'


def test_markov_mode_falls_back_when_there_is_nothing_to_chain_on():
    # Babbling from three words is not funny, it is just broken.
    with MockServer(_mode_cohort('markov'), port=0) as server:
        text = _post(server.url + '/v1/completions',
                     {'model': 'm', 'prompt': 'hi'})['choices'][0]['text']
    assert text and 'hi' not in text.lower(), text
