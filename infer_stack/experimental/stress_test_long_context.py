#!/usr/bin/env python3
"""
Long-context stress test for a vLLM/OpenAI-compatible endpoint.

What it does:
1. Generates a long synthetic corpus with random "needle" facts embedded throughout.
2. Uses the Qwen tokenizer chat template to size the prompt near a target token budget.
3. Sends retrieval and arithmetic questions to the model with requests.
4. Checks the returned answers and prints PASS/FAIL.

Typical usage:

  python stress_test_long_context.py \
    --base-url http://127.0.0.1:18000/v1 \
    --model qwen3.5-122b-a10b-fp8-262k \
    --max-context 262144 \
    --reserved-output 512

If your API key is in generated/.env as VLLM_BACKEND_API_KEY, it will be auto-detected.
Otherwise pass --api-key explicitly.

Requirements:
  pip install requests transformers
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


@dataclass
class Fact:
    marker: str
    secret_value: int
    secret_city: str
    secret_word: str


def load_api_key(cli_value: str | None) -> str | None:
    if cli_value:
        return cli_value
    env_key = os.environ.get('VLLM_BACKEND_API_KEY') or os.environ.get(
        'OPENAI_API_KEY'
    )
    if env_key:
        return env_key
    env_path = Path('generated/.env')
    if env_path.exists():
        text = env_path.read_text(encoding='utf-8')
        for line in text.splitlines():
            if line.startswith('VLLM_BACKEND_API_KEY='):
                return line.split('=', 1)[1].strip()
    return None


def strip_think_tags(text: str) -> str:
    # Remove <think>...</think> blocks if present.
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()


def normalize_answer(text: str) -> str:
    text = strip_think_tags(text)
    text = text.strip()
    # Remove wrapping code fences or quotes.
    text = text.strip('`').strip()
    text = text.strip('"').strip("'").strip()
    return text


def random_word(min_len: int = 4, max_len: int = 10) -> str:
    return ''.join(
        random.choices(
            string.ascii_lowercase, k=random.randint(min_len, max_len)
        )
    )


def filler_paragraph(num_words: int = 120) -> str:
    return ' '.join(random_word() for _ in range(num_words))


def build_facts(n: int) -> list[Fact]:
    cities = [
        'Lima',
        'Oslo',
        'Accra',
        'Perth',
        'Quito',
        'Hobart',
        'Dakar',
        'Tallinn',
        'Salta',
        'Cork',
        'Uppsala',
        'Medellin',
        'LuangPrabang',
    ]
    facts: list[Fact] = []
    for _ in range(n):
        marker = str(uuid.uuid4())[:8].upper()
        facts.append(
            Fact(
                marker=marker,
                secret_value=random.randint(100000, 999999),
                secret_city=random.choice(cities),
                secret_word=random_word(7, 12).upper(),
            )
        )
    return facts


def build_corpus(target_sections: int, facts: list[Fact]) -> str:
    """
    Build a long boring corpus with fact blocks inserted at spread-out positions.
    """
    sections = [
        f'SECTION {i}\n{filler_paragraph()}\n' for i in range(target_sections)
    ]
    # Spread facts through the document.
    positions = [
        int(i * target_sections / max(len(facts), 1)) for i in range(len(facts))
    ]
    positions = [min(target_sections - 1, max(0, p)) for p in positions]

    for pos, fact in zip(positions, facts):
        sections[pos] += (
            f'\nMARKER {fact.marker}\n'
            f'SECRET_VALUE {fact.secret_value}\n'
            f'SECRET_CITY {fact.secret_city}\n'
            f'SECRET_WORD {fact.secret_word}\n'
            f'END_MARKER {fact.marker}\n'
        )

    return '\n'.join(sections)


def chat_token_count(tokenizer: Any, user_content: str) -> int:
    ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': user_content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(ids)


def fit_corpus_to_budget(
    tokenizer: Any,
    base_instruction: str,
    corpus_template: str,
    target_prompt_tokens: int,
) -> str:
    """
    Binary-search the number of repeated filler sections until the whole prompt
    fits just under target_prompt_tokens.
    """
    lo, hi = 1, 4000
    best = corpus_template

    while lo <= hi:
        mid = (lo + hi) // 2
        test_corpus = corpus_template.format(
            extra_sections=('\nEXTRA\n' + filler_paragraph() + '\n') * mid
        )
        user_content = (
            f'{base_instruction}\n\nBEGIN CORPUS\n{test_corpus}\nEND CORPUS'
        )
        n = chat_token_count(tokenizer, user_content)
        if n <= target_prompt_tokens:
            best = test_corpus
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def make_questions(facts: list[Fact]) -> list[dict[str, Any]]:
    """
    Build a small suite of retrieval and aggregation questions.
    """
    qs: list[dict[str, Any]] = []

    # Exact value retrieval
    f0 = facts[0]
    qs.append(
        {
            'kind': 'value',
            'question': (
                f'Using only the provided corpus, what is the SECRET_VALUE for marker {f0.marker}? '
                'Reply with only the integer.'
            ),
            'expected': str(f0.secret_value),
        }
    )

    # Exact city retrieval
    f1 = facts[len(facts) // 2]
    qs.append(
        {
            'kind': 'city',
            'question': (
                f'Using only the provided corpus, what is the SECRET_CITY for marker {f1.marker}? '
                'Reply with only the city name.'
            ),
            'expected': f1.secret_city,
        }
    )

    # Exact word retrieval from later in the corpus
    f2 = facts[-1]
    qs.append(
        {
            'kind': 'word',
            'question': (
                f'Using only the provided corpus, what is the SECRET_WORD for marker {f2.marker}? '
                'Reply with only the word.'
            ),
            'expected': f2.secret_word,
        }
    )

    # Arithmetic over multiple markers
    picks = [facts[1], facts[len(facts) // 3], facts[-2]]
    total = sum(f.secret_value for f in picks)
    markers = ', '.join(f.marker for f in picks)
    qs.append(
        {
            'kind': 'sum',
            'question': (
                f'Using only the provided corpus, add the SECRET_VALUE values for markers {markers}. '
                'Reply with only the integer sum.'
            ),
            'expected': str(total),
        }
    )

    return qs


def call_chat_completion(
    session: requests.Session,
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    question: str,
    max_tokens: int,
    temperature: float,
) -> str:
    url = base_url.rstrip('/') + '/chat/completions'
    payload: dict[str, Any] = {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': (
                    'Answer using only the provided corpus. '
                    'If the requested value is not present, reply with NOT_FOUND.'
                ),
            },
            {
                'role': 'user',
                'content': f'{prompt}\n\nQUESTION:\n{question}',
            },
        ],
        'max_tokens': max_tokens,
        'temperature': temperature,
    }
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    response = session.post(
        url, headers=headers, data=json.dumps(payload), timeout=3600
    )
    response.raise_for_status()
    data = response.json()
    return data['choices'][0]['message']['content']


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:18000/v1')
    parser.add_argument('--model', required=True)
    parser.add_argument(
        '--tokenizer', default=None, help='Defaults to --model if omitted.'
    )
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--max-context', type=int, required=True)
    parser.add_argument('--reserved-output', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--num-facts', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-corpus', default='haystack_generated.txt')
    parser.add_argument('--save-answers', default='answer_key_generated.json')
    args = parser.parse_args()

    random.seed(args.seed)
    api_key = load_api_key(args.api_key)
    tokenizer_name = args.tokenizer or args.model

    print(f'Loading tokenizer: {tokenizer_name}', file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    facts = build_facts(args.num_facts)

    # Base corpus with distributed facts. We will append extra filler during sizing.
    corpus_without_extra = build_corpus(target_sections=1800, facts=facts)
    corpus_template = corpus_without_extra + '\n{extra_sections}'

    base_instruction = (
        'You will be given a long corpus delimited by BEGIN CORPUS and END CORPUS. '
        'Use only that corpus to answer the later question.'
    )

    # Keep a little slack so the specific question still fits.
    target_prompt_tokens = args.max_context - args.reserved_output - 1024
    fitted_corpus = fit_corpus_to_budget(
        tokenizer=tokenizer,
        base_instruction=base_instruction,
        corpus_template=corpus_template,
        target_prompt_tokens=target_prompt_tokens,
    )

    # Save generated materials for inspection.
    Path(args.save_corpus).write_text(fitted_corpus, encoding='utf-8')
    Path(args.save_answers).write_text(
        json.dumps([fact.__dict__ for fact in facts], indent=2),
        encoding='utf-8',
    )

    prompt = f'{base_instruction}\n\nBEGIN CORPUS\n{fitted_corpus}\nEND CORPUS'
    prompt_tokens = chat_token_count(tokenizer, prompt)
    print(f'Prompt tokens: {prompt_tokens}')
    print(f'Reserved output tokens: {args.reserved_output}')
    print(f'Total budget target: {args.max_context}')
    print(f'Saved corpus to: {args.save_corpus}')
    print(f'Saved answer key to: {args.save_answers}')

    questions = make_questions(facts)
    session = requests.Session()

    passed = 0
    for i, q in enumerate(questions, start=1):
        print(f'\n=== Test {i}: {q["kind"]} ===')
        print('Question:', q['question'])
        try:
            raw = call_chat_completion(
                session=session,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                prompt=prompt,
                question=q['question'],
                max_tokens=args.reserved_output,
                temperature=args.temperature,
            )
        except Exception as ex:
            print(f'FAIL: request error: {ex}')
            continue

        got = normalize_answer(raw)
        expected = str(q['expected']).strip()

        # Gentle normalization for exact-answer tasks.
        if q['kind'] in {'value', 'sum'}:
            m = re.search(r'-?\d+', got)
            got_cmp = m.group(0) if m else got
        else:
            got_cmp = got.splitlines()[0].strip()

        ok = got_cmp == expected
        print('Expected:', expected)
        print('Got:', got_cmp)
        print('Raw output:', raw[:1000].replace('\n', '\\n'))
        print('Result:', 'PASS' if ok else 'FAIL')
        passed += int(ok)

    print(f'\nPassed {passed}/{len(questions)} tests.')
    return 0 if passed == len(questions) else 1


if __name__ == '__main__':
    raise SystemExit(main())
