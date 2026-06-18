# CLAUDE.md

## Project overview

Prototype pipeline that infers likely countries of origin for a forename + surname. It combines two public name datasets into lookup tables, then uses an LLM (via HuggingFace router) to rank candidate countries by probability.

---

## Stack

| Layer | Tool |
|-------|------|
| Runtime | Python 3.13, `uv` |
| Data | pandas, pycountry, names-dataset |
| Script | openai SDK pointed at HuggingFace router (`base_url="https://router.huggingface.co/v1"`) |
| Model | `CohereLabs/command-a-reasoning-08-2025:cohere` |
| Text normalisation | unicodedata (stdlib), unidecode, transliterate |
| Eval | scikit-learn (train/test split) |
| Config | `.env` file → `HF_TOKEN` |

---

## File map

```
create_clean_dataset.py   # one-time ETL: builds data/final_*.csv from raw CSVs
main.py                   # core inference: lookup + LLM call + formatted output
utils/cleaning.py         # normalize_name + romanize_if_needed
evaluate_method.py        # train/test eval with distractor countries
data/
  common-forenames-by-country.csv   # raw sigpwned source
  common-surnames-by-country.csv    # raw sigpwned source
  final_forenames.csv               # built by create_clean_dataset.py
  final_surnames.csv                # built by create_clean_dataset.py
```

---

## Workflow

```
# 1. Build lookup tables (one-time, slow — hits the names-dataset library)
uv run python create_clean_dataset.py

# 2. Run inference (edit main.py:258-260 to change the input name)
uv run python main.py

# 3. Evaluate LLM accuracy against held-out names
uv run python evaluate_method.py --test-size 0.2 --max-samples 50
```

Inference pipeline per name:
1. Romanise non-Latin script → normalise (lowercase, strip diacritics)
2. Lookup in `final_forenames.csv` / `final_surnames.csv` → candidate country list
3. LLM call with candidates → ranked probabilities
4. Return combined result dict

---

## Hard rules

- **No LLM call if lookup returns nothing** — `llm_call` is only invoked when `country_list` is non-empty (`main.py:182`).
- **HF_TOKEN required for LLM** — without it, inference skips the LLM step silently; evaluation will fail.
- **Lookup tables must exist** — run `create_clean_dataset.py` before `main.py` or `evaluate_method.py`.
- **Data sources intersected, not unioned** — only names confirmed by both sigpwned CSV and `names-dataset` are kept. Don't relax this without revisiting noise levels.
- **Eval uses distractors** — the evaluator adds random distractor countries so the LLM can't just pick from a pre-filtered correct list. Keep `n_distractors > 0`.

---

## Coding guidelines

**1. Think before coding — don't assume, surface tradeoffs**

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.

**2. Simplicity first — minimum code that solves the problem**

- No features beyond what was asked.
- No abstractions for single-use code.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

**3. Surgical changes — touch only what you must**

- Don't improve adjacent code, comments, or formatting.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove only imports/variables YOUR changes made unused.

**4. Goal-driven execution — define success criteria, loop until verified**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```

---

**These guidelines work when:** diffs are clean, changes trace to the request, and clarifying questions come before implementation rather than after mistakes.
