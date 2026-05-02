# Support Triage Agent

A terminal-based AI agent that classifies, retrieves, and triages support tickets across three product ecosystems: **HackerRank**, **Claude**, and **Visa**.

---

## Architecture

The agent uses a **3-stage pipeline**:

### 1. **Classifier** (`classifier.py`)
- Infers company domain from ticket text (HackerRank, Claude, Visa)
- Classifies request type: `product_issue`, `feature_request`, `bug`, `invalid`
- Identifies product area (e.g., `assessments`, `claude_api`, `transaction_disputes`)
- Detects high-risk signals (fraud, account compromise, security, prompt injection)
- Uses signal-matching heuristics (no ML required)

### 2. **Retriever** (`retriever.py`)
- Loads and chunks the provided support corpus (from `data/`)
- Builds a **BM25 index** for each company domain
- Retrieves top-K (default 5) most relevant documentation chunks
- Formats chunks as corpus context for the LLM prompt
- **No external network calls** — corpus is static and local

### 3. **Generator** (`generator.py`)
- Uses **Google Gemini 1.5 Flash** to reason over the ticket + corpus context
- Produces structured JSON output with:
  - `status`: `replied` or `escalated`
  - `product_area`: most relevant support category
  - `response`: grounded, user-facing answer (or escalation message)
  - `justification`: concise reasoning with corpus attribution
  - `request_type`: final classification
- Enforces hard override: all high-risk tickets are escalated regardless of model output
- Implements retry logic with exponential backoff for transient errors

---

## Setup

### Prerequisites
- Python 3.9+
- `pip` or `uv` (package manager)

### Install Dependencies

```bash
cd code/
pip install -r ../requirements.txt
# OR with uv (faster):
uv pip install -r ../requirements.txt
```

### Environment Variables

Create a `.env` file in the repo root (already in `.gitignore`):

```bash
cp ../.env.example .env
```

Edit `.env` and add your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

Or export directly in the terminal:

```bash
export GEMINI_API_KEY=your_api_key
# (Windows: set GEMINI_API_KEY=your_api_key)
```

---

## Running the Agent

### Process all tickets

```bash
python main.py
```

This reads `../support_tickets/support_tickets.csv` and writes results to `../support_tickets/output.csv`.

### Process a custom input file

```bash
python main.py --input ../support_tickets/sample_support_tickets.csv --output my_output.csv
```

### Expected Output

The agent writes a CSV with these columns:

| Column         | Example                                                  |
| -------------- | -------------------------------------------------------- |
| `status`       | `replied`                                                |
| `product_area` | `assessments`                                            |
| `response`     | "You can reschedule your assessment by contacting..." |
| `justification`| "Based on [Source 1], assessments can be rescheduled..." |
| `request_type` | `product_issue`                                          |

---

## Design Decisions

### Why Google Gemini 1.5 Flash?

- **Fast inference**: Free tier supports 15 requests/minute; Flash is optimized for speed
- **Structured output**: Good JSON generation with temperature=0.0
- **Cost-effective**: Flash tier is cheaper than other general-purpose models
- **No external APIs needed**: Single LLM call per ticket; no multi-turn loops

### Why BM25 for Retrieval?

- **No embeddings required**: Avoids dependency on vector databases or embedding models
- **Deterministic**: Same query always returns the same results (no randomness)
- **Efficient**: Fast over local corpus (no network calls)
- **Transparent**: Signal-based scoring is interpretable and auditable

### Why Hard Escalation Override for High-Risk?

High-risk tickets (fraud, identity theft, account compromise, security vulnerabilities) are **always escalated** — both in the prompt and enforced in code after LLM output. This ensures safety even if the model is misdirected by malicious input.

### Prompt Injection Protection

The classifier includes signal-matching for common injection patterns:
- "ignore previous instructions", "reveal your rules", "act as", "jailbreak", etc.
- Tickets matching these patterns are flagged as `invalid` and escalated

---

## Reproducibility

### Pinned Dependencies

All package versions are fixed in `../requirements.txt`:

```
google-genai==1.74.0
python-dotenv==1.0.1
(other packages...)
```

Run the same command on any machine, and you'll get byte-identical results (within Gemini's non-determinism at temperature=0.0).

### Model Configuration

```python
MODEL_NAME   = "gemini-1.5-flash"
TEMPERATURE  = 0.0  # Deterministic (no sampling)
MAX_RETRIES  = 3
RETRY_DELAY  = 5  # seconds, exponential backoff
```

### No Hardcoded Secrets

All API keys are read from environment variables only. No hardcoded keys anywhere.

---

## Module Reference

### `classifier.py`
- `classify(issue, subject, company_field)` → `ClassificationResult`
  - Extracts company, product area, request type, and risk signals

### `retriever.py`
- `retrieve(query, company, top_k=5)` → `RetrievalResult`
  - Queries BM25 index and returns top-K chunks with scores
- `format_context(result)` → `str`
  - Formats chunks into corpus context for the LLM prompt
- `BM25Index` (class)
  - Efficient rank-based retrieval over chunks

### `generator.py`
- `generate(issue, subject, company, product_area, ...)` → `GeneratorOutput`
  - Calls Gemini and returns the five output fields
  - Implements retry logic and high-risk override

### `main.py`
- Entry point: reads CSV, processes each row, writes output CSV

---

## Safety & Escalation

The agent escalates tickets when:

1. **High-risk signals detected** (fraud, security, identity theft, billing disputes)
2. **Corpus has no relevant documentation** (out-of-scope)
3. **LLM unable to answer safely** (ambiguous, hallucination risk)
4. **Prompt injection detected** (requests to reveal rules, ignore instructions, etc.)
5. **Invalid/spam tickets** (hello, test message, out-of-scope chatter)

Escalated tickets receive a polite message: *"A human support agent will review your case and follow up shortly."*

---

## Troubleshooting

### `GEMINI_API_KEY not set`

**Error:** `EnvironmentError: GEMINI_API_KEY environment variable is not set`

**Fix:** Export the key before running:
```bash
export GEMINI_API_KEY=your_key
python main.py
```

Or use `.env`:
```bash
cp ../.env.example .env
# Edit .env with your key
python main.py
```

### Rate Limit (429 errors)

Gemini free tier is 15 requests/minute. The agent includes a `REQUEST_DELAY = 4.5` seconds between tickets to stay within limits.

If you hit a 429:
- The agent retries up to 3 times with exponential backoff
- Consider increasing `REQUEST_DELAY` in `main.py` if running on large batches

### Missing Corpus Files

**Error:** `FileNotFoundError: data/hackerrank/...`

**Fix:** Ensure `data/` directory exists with all three subdirectories (`hackerrank/`, `claude/`, `visa/`). The corpus is shipped in the repo root.

---


## License & Attribution

Built for **HackerRank Orchestrate** (May 1–2, 2026). Uses provided support corpus only.
