# MiniMaestro: Conversational Music Recommender System (RecSys Challenge 2026)

**Team Overfit & Chill.** MiniMaestro is a conversational music recommender built for the [ACM RecSys Challenge 2026 (TalkPlay)](https://nlp4musa.github.io/music-crs-challenge/). Each turn, it plans a structured query, pools candidates from multiple retrieval sources, reranks them with a LambdaRank model, and generates a natural-language response, reusing a single open-weight Qwen3-8B model for both the planning and generation stages.

> **Paper:** *MiniMaestro: Resource-Conscious Conversational Music Recommendation with a Single Open-Weight 8B Model.* Simran Sundrani and Mohan Bhambhani (Team Overfit & Chill), RecSys Challenge 2026: Music-CRS (TalkPlay).

**Best results on the blind benchmarks:**

| Metric | Blind A | Blind B (graded) |
|--------|---------|------------------|
| Composite score | **0.53** | **0.48** |
| nDCG@20 | 0.41 | 0.33 |
| Catalog Diversity | 0.03 | 0.03 |
| Lexical Diversity | 0.73 | 0.73 |
| LLM-Judge | 4.30 / 5.0 | 4.15 / 5.0 |

Blind B is the graded set that determined the final ranking; its breakdown matches Table 2 of the paper.

Composite = 0.50×nDCG@20 + 0.10×CatalogDiversity + 0.10×LexicalDiversity + 0.30×LLM-Judge

---

## Reproducing the Submission (Blind B)

These steps regenerate the `prediction.json` we submitted for Blind Dataset B. They are written to be environment-agnostic, so adapt the paths to wherever you run them (local CUDA machine, RunPod, or any GPU host). A CUDA GPU with roughly 24 GB of memory is recommended to hold the Qwen3-8B model; the full run is I/O and GPU bound, not something that finishes in a few minutes.

### 1. Environment and data

```bash
# Clone
git clone https://github.com/simransu/recsys-2026-music-crs.git
cd recsys-2026-music-crs

# Create an environment and install (Python 3.10+)
uv venv .venv --python=3.10 && source .venv/bin/activate
uv pip install -e .                             # or: pip install -e .
pip install flash-attn --no-build-isolation     # GPU only, optional but faster

# Authenticate for the gated TalkPlay HuggingFace datasets
export HF_HOME=/path/to/hf_cache                # any writable cache dir
huggingface-cli login
```

All data (conversations, track/user metadata, precomputed track/user embeddings) is pulled from the [TalkPlay HuggingFace collection](https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge) listed under [Datasets](#datasets); nothing else is required. If the structured-JSON query planner errors on a fresh GPU box, force-reinstall its dependency: `pip install --force-reinstall lm-format-enforcer`.

### 2. LambdaRank reranker — use the shipped model *or* retrain

**Option A — use the shipped model (reproduces the exact reranker we submitted).**

```bash
mkdir -p cache
cp models/lambdarank_model_blindsetB.txt cache/lambdarank_model.txt
```

`models/lambdarank_model_blindsetB.txt` is the exact LightGBM LambdaRank model behind the reported Blind B score, checked into this repo so you do not have to retrain to reproduce the submission. Inference reads the reranker from `./cache/lambdarank_model.txt` (see `lambdarank_model_path` in `config/qwen3_8b_multi_source_blindset_B.yaml`), so this copy is all that is needed.

**Option B — retrain the reranker from scratch.**

```bash
# Cache the 8B planner outputs once, then train
python scripts/precompute_planner.py --config config/lambdarank_training.yaml
python scripts/train_lambdarank.py \
  --config config/lambdarank_training.yaml \
  --goal_filter --last_n_turns 1 --num_rounds 1000
# writes ./cache/lambdarank_model.txt (the path inference reads)
```

### 3. Run Blind B inference

```bash
python run_inference_blindset.py \
  --tid qwen3_8b_multi_source_blindset_B \
  --eval_dataset blindset_B \
  --batch_size 10 --retrieval_batch_size 10
# writes exp/inference/blindset_B/qwen3_8b_multi_source_blindset_B.json
```

For Blind A, use `--tid qwen3_8b_multi_source_blindset_A --eval_dataset blindset_A`. The `--eval_dataset` flag is required and must match the config, since it selects both the input split and the output directory.

Optionally, confirm no response came back empty:

```bash
python -c "
import json
d = json.load(open('exp/inference/blindset_B/qwen3_8b_multi_source_blindset_B.json'))
empty = [e['session_id'][:8] for e in d if not e['predicted_response'].strip()]
print(f'{len(empty)} empty responses:', empty)
"
```

### 4. Build the submission archive

```bash
cp exp/inference/blindset_B/qwen3_8b_multi_source_blindset_B.json prediction.json
zip submission.zip prediction.json    # prediction.json sits at the archive root
```

### Reproducibility note

Scores will land close to the reported numbers but **will not reproduce bit-for-bit**, by design. Two sources of run-to-run variation are expected:

- **Stochastic LLM generation.** Query planning and response generation both run an 8B model with sampling (temperature 0.7, top-p 0.9), so the planner's structured queries and the final natural-language responses can differ slightly between runs. That shifts the retrieval pools and the reranked lists, so nDCG@20 and the LLM-as-a-Judge score fluctuate within a small band rather than matching exactly.
- **Reranker retraining.** LambdaRank training uses no held-out split and no early stopping, so a from-scratch retrain (Option B) can settle on a slightly different model than the one we shipped. Use Option A (the checked-in model) to reproduce the exact reranker; use Option B only if you want to verify the training pipeline itself.

---

## System Architecture

```
User conversation turn
        │
        ▼
┌───────────────────────┐
│   Qwen3-8B Planner    │  Structured JSON: bm25_query, artist_names,
│  (query planning)     │  genre_tags, mood_phrases, year_terms, ...
└──────────┬────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              Multi-Source Candidate Retrieval           │
│                                                         │
│  BM25 (weighted fields)                                 │
│  BGE dense retrieval                                    │
│  Qwen3 dense retrieval                                  │
│  BPR user-to-item (CF)                                  │
│  Item-to-item embeddings                                │
│    ├─ CF-BPR embeddings                                 │
│    ├─ Image (SigLIP2)                                   │
│    ├─ Audio (LAION-CLAP)                                │
│    ├─ Metadata (Qwen3-Embedding)                        │
│    ├─ Attributes (Qwen3-Embedding)                      │
│    └─ Lyrics (Qwen3-Embedding)                          │
│  Artist / album shortcut retrieval                      │
│  Entity matching                                        │
│  Session co-occurrence                                  │
│  Train-thought BM25                                     │
│                                                         │
│  All source lists unioned into one candidate pool       │
└──────────────────────────┬──────────────────────────────┘
                           │  union of all source lists
                           ▼
          ┌────────────────────────────────────────┐
          │          LambdaRank Reranker           │
          │  (LightGBM)                            │
          │                                        │
          │  Features per candidate:               │
          │  · rank in each source list            │
          │  · presence flag per source (binary)   │
          │  · artist/album match to anchors       │
          │  · entity match signals                │
          │  · turn number, anchor count           │
          └──────────────┬─────────────────────────┘
                         │  top-20
                         ▼
          ┌────────────────────────────┐
          │   Qwen3-8B Response Gen    │  Thinking mode + template
          │   (natural language)       │  routing (discovery /
          └────────────────────────────┘  expert / conversational)
```

---

## Components

### 1. Multi-Source Retrieval (`mcrs/retrieval_modules/`)

| Module | Description |
|--------|-------------|
| `bm25.py` | Weighted BM25 over track metadata fields (track name ×4, artist ×3, album ×2, date ×1, tags ×1). Field weighting is implemented by repeating field text at index time. |
| `bert.py` | Dense retrieval using `BAAI/bge-small-en-v1.5` sentence embeddings |
| `qwen3_dense.py` | Zero-shot dense retrieval using `Qwen/Qwen3-Embedding-0.6B` |
| `user_to_item.py` | Collaborative filtering via BPR user embeddings (warm user personalization) |
| `item_to_item.py` | Multimodal I2I expansion over six precomputed embedding spaces: CF-BPR, image (SigLIP2), audio (LAION-CLAP), and three Qwen3-Embedding text views (metadata, attributes, lyrics) |
| `session_cooccurrence.py` | Session-level co-occurrence signals from training data |
| `train_thought_bm25.py` | BM25 over training-set rationale/thought annotations |
| `multi_source.py` | Internally fuses BM25 + BGE + BPR via RRF to produce a base candidate list; all sources are then unioned for LambdaRank |

Each source retrieves candidates independently. All source lists are unioned into a single candidate pool passed to LambdaRank. LambdaRank uses each source's per-candidate rank as a feature; there is no RRF step in the final ranking path.

### 2. LLM Query Planner (`mcrs/lm_modules/qwen3.py`)

Qwen3-8B runs in thinking mode and generates a structured JSON plan per conversation turn:

```json
{
  "bm25_query": "upbeat indie rock summer road trip",
  "artist_names": ["The Strokes", "Arctic Monkeys"],
  "genre_tags": ["indie rock", "alternative"],
  "mood_phrases": ["energetic", "feel-good"],
  "year_terms": ["2000s", "2010s"],
  "track_titles": [],
  "album_names": [],
  "negative_constraints": ["no sad songs"]
}
```

The plan replaces the raw query for BM25 and is also injected into the evidence block shown to the response generator, improving recommendation relevance.

Plans are **precomputed and cached** before training/inference to avoid running the 8B model twice (`scripts/precompute_planner.py`).

### 3. LambdaRank Reranker (`scripts/train_lambdarank.py`)

A LightGBM LambdaRank model trained on the challenge-provided conversations, one example per session (the session's last `MOVES_TOWARD_GOAL` recommendation turn, `--last_n_turns 1`), with per-query groups. Each example's candidate pool is the union of that turn's multi-source retrieval output, with the recommended track labeled positive and every other pooled candidate labeled negative; this yields 24M+ (query, candidate) pairs across ~13K groups. LambdaRank scores every candidate using features:

- Rank in each source's individual list (BM25, BGE, Qwen3 dense, BPR, each I2I variant, etc.)
- Presence flag (binary) per source, including artist shortcut, album shortcut, and entity match
- Artist/album match to anchor tracks mentioned in conversation
- Entity match signals (track name, artist, album appearing in query)
- Turn number and number of anchor tracks

Training uses `--goal_filter --last_n_turns 1` to focus on recommendation turns and avoid noise from chitchat.

### 4. Response Generator (`mcrs/lm_modules/qwen3.py`)

Qwen3-8B with thinking mode enabled, routing to one of three response templates based on query specificity:
- **Discovery**: broad exploratory queries ("I want something chill")
- **Expert**: specific artist/album/genre requests
- **Conversational**: follow-up or clarification turns

Template design emphasizes natural language, avoids rigid list formatting, and includes liked-track anchoring for personalization.

---

## Experiment: Fine-Tuned Bi-Encoder (`experiments/finetune_biencoder.py`)

> **Note:** This was an experiment conducted during development. The fine-tuned bi-encoder was **not used in the final submission** — the submitted system uses only the retrieval sources listed above.

We fine-tuned `Qwen/Qwen3-Embedding-0.6B` as a query encoder against **frozen precomputed track embeddings** to explore whether a task-specific dense retriever could improve candidate recall:

- **Training data**: ~13k (query, positive track) pairs from goal-annotated training turns
- **Hard negatives**: tracks retrieved by the current system (BM25, BGE, I2I, artist shortcut) that are NOT the ground truth
- **Loss**: In-batch negatives + hard negatives via InfoNCE
- **Key implementation detail**: Qwen3 is a causal LM — must use **last-token pooling**, not CLS, for embeddings

**Result**: On a 100-session held-out mini devset, the fine-tuned retriever contributed +3 unique candidates at the pool level (additive over all other sources), raising pool recall from 0.62 → 0.64. The inference module is in `experiments/finetuned_dense.py`. Neither the training script nor the inference module is wired into `CRS_BASELINE` — they live under `experiments/` and are not part of the submitted pipeline.

To reproduce the experiment (checkpoints saved to `./cache/finetuned_biencoder/`, best to `./cache/finetuned_biencoder/best/`):

```bash
python experiments/finetune_biencoder.py \
  --epochs 5 --batch_size 16 --lr 2e-5 \
  --goal_filter --last_n_turns 1
```

---

## Datasets

All datasets are from the [TalkPlay HuggingFace collection](https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge):

| Dataset | HuggingFace ID |
|---------|---------------|
| Conversations (train/dev) | `talkpl-ai/TalkPlayData-Challenge-Dataset` |
| Track metadata | `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` |
| User metadata | `talkpl-ai/TalkPlayData-Challenge-User-Metadata` |
| Precomputed track embeddings | `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings` |
| Precomputed user embeddings | `talkpl-ai/TalkPlayData-Challenge-User-Embeddings` |

---

## Data & License

The TalkPlay challenge datasets (licensed CC-BY-NC) and the base model weights (Qwen3-8B, Qwen3-Embedding) are downloaded at runtime from their gated HuggingFace sources listed under [Datasets](#datasets); none of them are redistributed in this repository. They are used here solely for the RecSys Challenge 2026. The only model committed to this repo, `models/lambdarank_model_blindsetB.txt`, is our own trained LightGBM reranker.

---

## Repository Structure

```
mcrs/
├── crs_baseline.py                        # Main CRS class wiring all components
├── db_item/
│   └── music_catalog.py                   # Track metadata access
├── db_user/
│   └── user_profile.py                    # User profile access
├── lm_modules/
│   └── qwen3.py                           # Qwen3-8B: planner + response generator
├── reranker_modules/                      # LambdaRank is loaded directly in crs_baseline.py;
│                                           # this package is currently unused by the submitted pipeline
├── retrieval_modules/
│   ├── bm25.py                            # Weighted BM25
│   ├── bert.py                            # Dense retrieval (BGE)
│   ├── qwen3_dense.py                     # Zero-shot Qwen3-Embedding dense retrieval
│   ├── user_to_item.py                    # CF/BPR user-to-item
│   ├── item_to_item.py                    # Multimodal I2I expansion
│   ├── session_cooccurrence.py            # Session co-occurrence signals
│   ├── train_thought_bm25.py              # BM25 over training rationales
│   └── multi_source.py                    # RRF fusion of BM25+BGE+BPR
└── system_prompts/
    ├── query_planning.txt
    ├── response_generation.txt
    ├── template_discovery.txt
    ├── template_expert.txt
    ├── template_conversational.txt
    ├── personalization.txt
    └── roleplay.txt

models/
└── lambdarank_model_blindsetB.txt         # Shipped LightGBM reranker (reproduces Blind B)

scripts/
├── train_lambdarank.py                    # LambdaRank training (LightGBM)
├── precompute_planner.py                  # Cache Qwen3-8B planner outputs
└── setup_pod.sh                           # RunPod bootstrap script

experiments/                               # Explored but NOT used in the final submitted pipeline
├── finetune_biencoder.py                  # Qwen3-0.6B bi-encoder fine-tuning (experiment)
└── finetuned_dense.py                     # Fine-tuned Qwen3-0.6B retrieval inference (experiment)

dev_validation/                            # Local-only validation against the labeled dev set.
│                                           # Not part of the graded pipeline (dev has no role in
│                                           # Blind-B inference or the training reproduction check) —
│                                           # kept only so you can sanity-check a from-scratch
│                                           # LambdaRank retrain before trusting blind predictions.
├── run_inference_devset.py                # Dev set inference entry point
├── qwen3_8b_multi_source_devset.yaml      # Dev config
└── evaluate_dev_ndcg.py                   # Dev set nDCG evaluation

config/                                    # YAML inference + training configs
├── qwen3_8b_multi_source_blindset_A.yaml  # Blind A config
├── qwen3_8b_multi_source_blindset_B.yaml  # Blind B config
├── lambdarank_training.yaml               # LambdaRank training config
└── ...                                    # Additional configs

run_inference_blindset.py                  # Blind set inference entry point
pyproject.toml                            # Package definition and dependencies
```

---

## Challenge

- **Challenge**: [ACM RecSys 2026 Music-CRS Challenge (TalkPlay)](https://nlp4musa.github.io/music-crs-challenge/)
- **Evaluation server**: [Codabench](https://www.codabench.org/)
- **Evaluator**: [nlp4musa/music-crs-evaluator](https://github.com/nlp4musa/music-crs-evaluator)
