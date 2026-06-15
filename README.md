# Name Origin — Country Likelihood from Forenames and Surnames

An MVP pipeline that takes a forename and surname and returns a probability distribution over the countries where that name is likely to originate.

**Example** — given `"John Smith"`, the system might return:

| Country   | Probability |
|-----------|-------------|
| United Kingdom | 0.85   |
| United States  | 0.10   |
| Australia      | 0.05   |

---

## How it works

```
Input: forename + surname
       ↓
Normalise (lowercase, strip accents, collapse hyphens)
       ↓
Transliterate if non-Latin script (Cyrillic, Arabic, …)
       ↓
Lookup in forename + surname tables → candidate countries
       ↓
LLM ranks candidates by probability
       ↓
Output: ranked country list with probabilities
```

The lookup tables are built by intersecting two public datasets:

| Dataset | What it provides |
|---------|-----------------|
| [sigpwned/popular-names-by-country](https://github.com/sigpwned/popular-names-by-country-dataset) | Popular forenames and surnames per country |
| [names-dataset (PyPI)](https://pypi.org/project/names-dataset/) | Large first/last name corpus with country distributions |

Only names confirmed by **both** sources are kept, reducing noise.

---

## Handling ambiguity

Many names are common across multiple countries due to linguistic, historical, or colonial overlap:

| Name    | Possible Countries             |
|---------|--------------------------------|
| Silva   | Portugal, Brazil, Mozambique   |
| Lee     | China, Korea, United States    |
| Sofia   | Italy, Spain, Bulgaria         |

The LLM is given the full candidate list and asked to rank by likelihood, allowing multi-country output.

### Transliteration

Names from non-Latin scripts are romanised before lookup:

- Cyrillic: `Михаил` → `Mikhail`
- Arabic: `محمد` → `Muhammad`

---

## Usage

### Install dependencies

```bash
uv sync
```

### Run inference

```bash
uv run python main.py
```

Edit `main.py` lines 258–260 to change the input name.

### Run evaluation

```bash
uv run python evaluate_method.py --test-size 0.2
```

Requires `HF_TOKEN` set in a `.env` file (HuggingFace inference token):

```
HF_TOKEN=hf_...
```

Optional flags:

```bash
# Limit samples (faster for testing)
uv run python evaluate_method.py --max-samples 50

# Control number of distractor countries added per name (default 5)
uv run python evaluate_method.py --n-distractors 5
```

---

## Evaluation methodology

The evaluation tests the LLM's ability to identify the correct country of origin for a name, given a mixed candidate list.

**Setup (80/20 train/test split on unique names):**

1. Split the lookup tables into train (80%) and test (20%) by name — names are deduplicated before splitting so no name appears in both sets.
2. For each test name, take its true countries and add **N random distractor countries** sampled from the full country pool.
3. Pass the shuffled candidate list (true + distractors) to the LLM.
4. Mark as correct if the LLM's top-ranked country is in the true country list.

This avoids the trivial 100%-accuracy result that occurs when the candidate list contains only the correct answers.

**Output files:**

| File | Contents |
|------|----------|
| `evaluation_results.csv` | Per-name predictions, true countries, distractors used, correctness |
| `evaluation_summary.txt` | Aggregate accuracy and coverage statistics |

---

## Limitations

- **Data coverage**: only names present in both source datasets are included. Rare or regional names may be missing.
- **Single ground truth**: the dataset records all countries where a name is common, not a single "primary" country. Accuracy is measured as top-1 prediction ∈ true country set.
- **Transliteration variants**: `Muhammad`, `Mohamed`, `Mohamad` are treated as separate entries. Variant normalisation is not exhaustive.
- **LLM dependency**: results depend on the model used (currently `CohereLabs/command-a-reasoning-08-2025` via HuggingFace router). Changing the model may affect accuracy.

---

## Future enhancements

- Scrape official national statistics sites for more comprehensive name coverage.
- Add [Hobson/surname-nationality](https://huggingface.co/datasets/Hobson/surname-nationality) as a third data source.
- Weight country probabilities by name frequency, not just presence.
- Evaluate with an independent held-out dataset for more reliable accuracy estimates.
