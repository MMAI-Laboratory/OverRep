# Task lists

`overrep.txt` holds the comma-separated reasoning task list used in the paper's
evaluation protocol, so a sweep can be launched with
`overrep-eval <model> "$(cat tasks/overrep.txt)"`. The generation tasks
(`coqa`, `gsm8k`, `triviaqa`) are run separately with their own few-shot counts.
