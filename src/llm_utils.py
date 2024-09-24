import json
import logging


def retry_until_parse(pipe, chat_start, parser, n_retries):
    n_try = 0
    result = None
    errors = []
    logging.info(f'Prompt: {chat_start[-1]["content"]}')
    while result is None and n_try < n_retries:
        n_try += 1
        raw = pipe([chat_start])[0][0]['generated_text'][-1]['content']
        logging.info(f'(Attempt {n_try}): Model says: {raw}')
        try:
            result = parser(raw)
        except ValueError as e:
            errors.append(str(e))
            continue
        else:
            return result
    else:
        raise ValueError('Max number of retries.')


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
