"""Single-endpoint OpenAI-compatible chat example for presentation demos."""
import argparse
import json
import os
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt')
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.timeout <= 0:
        parser.error('max-tokens and timeout must be positive')
    models = [s.strip() for s in os.environ.get('INFER_STACK_MODELS', '').split(',') if s.strip()]
    base = os.environ.get('OPENAI_BASE_URL', '')
    if len(models) != 1 or not base:
        parser.error('acquire one chat endpoint and source its lease.env, or use infer-stack run')
    body = json.dumps({'model': models[0], 'messages': [
        {'role': 'user', 'content': args.prompt}], 'max_tokens': args.max_tokens,
        'stream': False}).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get('OPENAI_API_KEY')
    if key:
        headers['Authorization'] = 'Bearer ' + key
    request = urllib.request.Request(base.rstrip('/') + '/chat/completions', data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        # Provider response bodies may contain credentials or private input.
        raise SystemExit(f'Chat request failed: HTTP {exc.code}') from None
    except (urllib.error.URLError, TimeoutError, ValueError):
        raise SystemExit('Chat request failed or returned invalid JSON') from None
    try:
        choice = result['choices'][0]
        message = choice['message']
    except (KeyError, IndexError, TypeError):
        raise SystemExit('Chat response has no usable choice') from None
    print(f'Endpoint: {models[0]}')
    content = message.get('content')
    reasoning = message.get('reasoning_content') or message.get('reasoning')
    if content:
        print(content)
    elif reasoning:
        print('[Reasoning only; increase --max-tokens for a final answer.]')
        print(reasoning)
    else:
        print('[No text returned.]')
    if result.get('usage'):
        print('Usage:', json.dumps(result['usage']))


if __name__ == '__main__':
    main()
