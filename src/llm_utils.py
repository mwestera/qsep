import json
import logging


def retry_until_parse(pipe, chat_start, parser, n_retries, fail_ok=False, try_skip_first_line=True):
    """
    :param try_skip_first_line: Sometimes LLMs preface their (otherwise fine) answer by "Here is the answer:" etc.
    """
    n_try = 0
    result = None
    errors = []
    logging.info(f'Prompt: {chat_start[-1]["content"]}'.replace('\n', '//'))
    while result is None and n_try < n_retries:
        n_try += 1
        raw = pipe([chat_start])[0][0]['generated_text'][-1]['content']
        logging.info(f'(Attempt {n_try}): Model says: {raw}'.replace('\n', '//'))
        try:
            result = parser(raw)
        except ValueError as e1:    # TODO: refactor
            if try_skip_first_line:
                try:
                    raw_lines = raw.splitlines()
                    if len(raw_lines) > 1:
                        result = parser('\n'.join(raw_lines[1:]))
                except ValueError as e2:
                    errors.append(str(e1) + '; ' + str(e2))
                    continue
            else:
                errors.append(str(e1))
                continue
        return result
    else:
        if not fail_ok:
            raise ValueError(f'Max number of retries ({"; ".join(errors)})')
        else:
            logging.warning(f'Max number of retries ({"; ".join(errors)})')
            return None


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
