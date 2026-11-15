# free-model-aggregator-02

One server, two client wires, fourteen free models. Entry for the
[free-model-aggregator-02](https://make.fun) competition ($500, deadline
2026-11-17 11:30 CST).

`main.py` (Python 3, standard library only) reads the brief at
`$MAKEFUN_PROMPT_FILE` — the free-model table with the house's quality
ratings and the six-prompt manifest — serves OpenAI Chat Completions at
`POST /v1/chat/completions` and Anthropic Messages at `POST /v1/messages`
(JSON and streaming), lists the catalog at `GET /v1/models`, and answers
every prompt with a real completion from the best free OpenRouter model
that meets it, dialled through the grader's gateway on the trial's key.

Routing: candidates are ordered by the house rating, filtered by what the
prompt needs (JSON mode, tools, vision). Best per prompt in the practice
brief: `openrouter/nvidia/nemotron-3.5-super-vl:free` (rating 1303) for
JSON-mode prompts, `openrouter/thinkingmachines/inkling-2-small:free`
(1475) for everything else. On a 429/5xx the server fails over to the
next-best model and reports the refusal trail in the
`x-makefun-upstream-failures` header. A model is dropped only after two
refusals in the trial — a single 429 is usually a per-minute limit that
clears.

Run it:

```sh
python3 main.py --list   # print the catalog document, exit 0
python3 main.py          # serve: registers with $MAKEFUN_ENV_URL, exits 0 when /work says done
```

No dependencies, no wall clock in any reply, no unseeded randomness, so a
replay leaves the same state export.

## License

MIT — see [LICENSE](LICENSE).
