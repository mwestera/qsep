import json


def retry_until_parse(pipe, chat_start, parser, n_retries):
    n_try = 0
    result = None
    errors = []
    while result is None and n_try < n_retries:
        n_try += 1
        raw = pipe([chat_start])[0][0]['generated_text'][-1]['content']
        try:
            result = parser(raw)
        except ValueError as e:
            errors.append(str(e))
            continue
        else:
            return result
    else:
        raise ValueError(' | '.join(errors))


def parse_json_list_of_strings(raw):
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError
    if not isinstance(result, list):
        raise ValueError
    if any(not isinstance(x, str) for x in result):
        raise ValueError
    return result


def iter_chat_starts(texts, examples, system_prompt):
    examples_chat = []
    for example in examples:
        examples_chat.append({"role": "user", "content": example['prompt']})
        examples_chat.append({"role": "assistant", "content": json.dumps(example['response'])})

    for text in texts:
        text = text.strip()
        chat_start = [
            {"role": "system", "content": system_prompt},
            *examples_chat,
            {"role": "user", "content": text},
        ]
        yield chat_start
