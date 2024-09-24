# QSep: Question separator #

A command-line tool for breaking down potentially composite questions into their subquestions, using an LLM.

Currently intended for Dutch, but easily adapted.

## Install ##

Ideally in a virtual environment (or use `pipx`):

```bash
pip install git+https://github.com/mwestera/qsep
```

This will make available two commands: 

- `qsep` tasks an LLM with splitting a composite question into self-contained subquestions.
- `qspan` tasks an LLM with finding the span (start, end) in the original composite question, from which a sub-question derives.

Typically, you'd want to call `qsep` with `--validate`, which automatically invokes `qspan` as a second step.

## Usage ##

Given a text file `questions.txt` containing composite questions to break down, one per line:

```text
Hoe werkt dat en waarom?
Wie ben je en hoe oud ben je?
```

You can feed it into `qsep` like this (for example; default temperature is 0.1):

```bash
cat questions.txt | qsep --temp .3
```

This will output one subquestion per line, the outputs for different inputs separated by empty lines. Alternatively, add `--list` to get a single-line JSON list per input, instead of potentially multiple lines.

```bash
$ cat questions.txt | qsep --list
```

Include `--validate` to retrieve the spans from which each subquestion derives:

```bash
cat questions.txt | qsep --validate
```

You can add `--json` to get JSON lines as output. Without `--validate` (or with `--list`), this does the same as `--list`. With `--validate`, it results in potentially multiple JSON lines per input:  

```bash
cat questions.txt | qsep --json --validate
```

