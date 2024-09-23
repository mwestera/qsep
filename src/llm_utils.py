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


def make_chat_start(prompt, examples, system_prompt):
    examples_chat = []
    for example in examples:    # TODO This is executed anew for each prompt...
        examples_chat.append({"role": "user", "content": example['prompt']})
        examples_chat.append({"role": "assistant", "content": example['response']})

    chat_start = [
        {"role": "system", "content": system_prompt},
        *examples_chat,
        {"role": "user", "content": prompt.strip()},
    ]
    return chat_start
