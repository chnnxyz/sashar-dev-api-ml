# sashar-dev-api-ml

Python **FastAPI** backend for **sashar.dev** — powers the Time Series & ML playground
(`/ml/v1`) and the LLM pipeline (`/llm/v1`) on **:8002**. Reads its datasets from the
shared sqlite database and does all model training, forecasting, hyperparameter
optimization, and the tokenize→encode→embed→generate LLM walkthrough.

## Stack
- FastAPI + uvicorn, Pydantic v2
- scikit-learn, LightGBM, statsmodels (Holt-Winters), CPU-only PyTorch (MLP + GRU, ≤50 epochs)
- hyperopt (TPE), sklearn ParameterGrid (grid search), pygad (genetic algorithm)
- transformers + **distilgpt2**, spaCy (`en_core_web_sm`) for the LLM steps

## Setup (Debian/Ubuntu VM)
```bash
xargs -a apt_requirements.txt sudo apt-get install -y   # libgomp1, build-essential, python3-dev
poetry install                                          # resolves CPU torch from the pytorch-cpu source
poetry run python -m nltk.downloader wordnet omw-1.4    # WordNet corpus for cluster labels (bundle offline)
poetry run python -m app.seed_datasets                  # seed datasets into the shared sqlite (run once)

# Development:
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8002
# Production (gunicorn process manager + uvicorn workers):
poetry run gunicorn app.main:app -c gunicorn.conf.py
```
> First LLM call downloads distilgpt2 from the HuggingFace hub (~350 MB, cached afterwards).

## Configuration (env)
| Var | Default | Purpose |
|-----|---------|---------|
| `PORT` | `8002` | uvicorn port |
| `SASHAR_DB_PATH` | `<repo-parent>/sashar.db` | Shared sqlite file (same file the Go service uses) |
| `ALLOWED_ORIGINS` | site + localhost dev | CORS allow-list |
| `MODEL_NAME` | `distilgpt2` | HuggingFace causal LM for the LLM pipeline |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy model for tokenization |
| `MAX_EPOCHS` | `50` | Hard cap on torch training loops |
| `INTERNAL_API_SECRET` | *(unset)* | If set, `/ml/v1` HTTP routes require header `X-Internal-Secret` |

## Endpoints
**ML / TS** (`/ml/v1`)
- `POST /run-regression` · `POST /run-classification` · `POST /run-clustering` → `MLRunResult`
- `POST /run-forecast` → `TSRunResult`
- `POST /optimize-params` → `OptimizeResult`
- `GET /datasets` · `GET /datasets/{id}?seed=` → dataset rows + train/test split
- `WS /ws/optimize` → streams `{iteration, rmse, params}` per trial
- `WS /ws/train` → streams `{epoch, loss}` for MLP / GRU training

**LLM** (`/llm/v1`)
- `POST /tokenize` → `{tokens}` (spaCy)
- `POST /encode` → `{ids}` (GPT-2 vocab)
- `POST /embed` → `{points:[{x,y,label}]}` (distilgpt2 embeddings → PCA 2D)
- `POST /generate` → `{outputTokens}` (distilgpt2)

## Datasets
Seeded from sklearn built-ins (iris, breast_cancer, diabetes, and California Housing when
fetchable) plus synthetic generators (blobs, moons, ETTh1) and the verbatim Air Passengers
series. Each is stratified-truncated to ≤1000 rows and stored long-format in the shared
`data` table. See `app/seed_datasets.py`.

## Notes
Response fields are snake_case in Python but serialize to the camelCase JSON the frontend
expects (e.g. `trainRMSE`, `testPredictions`, `outputTokens`) via Pydantic aliases.
Optimization of the torch/statsmodels models (mlp, rnn, tes, dbscan) uses a fast
deterministic surrogate objective so the no-GPU demo stays responsive.
