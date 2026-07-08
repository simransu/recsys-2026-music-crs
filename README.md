# Music Conversational Recommender System — RecSys Challenge 2026

A two-stage conversational music recommender built for the [ACM RecSys Challenge 2026 (TalkPlay)](https://nlp4musa.github.io/music-crs-challenge/). The system combines multi-source candidate retrieval with LambdaRank reranking and uses Qwen3-8B for structured query planning and natural-language response generation.

**Best results on the blind benchmarks:**

| Metric | Blind A | Blind B |
|--------|---------|---------|
| Composite score | **0.5296** | **0.4800** |
| nDCG@20 | 0.4124 | — |
| Catalog Diversity | 0.0320 | — |
| Lexical Diversity | 0.7273 | — |
| LLM-Judge | 4.30 / 5.0 | — |

Composite = 0.50×nDCG@20 + 0.10×CatalogDiversity + 0.10×LexicalDiversity + 0.30×LLM-Judge

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
│                    (topk = 400 each)                    │
│                                                         │
│  BM25 (weighted fields)         → up to 400 candidates  │
│  BGE dense retrieval            → up to 400 candidates  │
│  BPR user-to-item (CF)          → up to 400 candidates  │
│  Item-to-item embeddings        → up to 400 candidates  │
│    ├─ Image (SigLIP2)                                   │
│    ├─ Audio (LAION-CLAP)                                │
│    ├─ Lyrics (Qwen3-Embedding)                          │
│    ├─ Attributes (Qwen3-Embedding)                      │
│    └─ CF-BPR embeddings                                 │
│  Artist / album shortcut retrieval                      │
│  Entity matching                                        │
│  Session co-occurrence                                  │
│  Train-thought BM25                                     │
│                                                         │
│  All source lists unioned → ~2000+ unique candidates    │
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
          │  · source consensus count              │
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
| `item_to_item.py` | Multimodal I2I expansion: image (SigLIP2), audio (LAION-CLAP), lyrics, attributes, CF-BPR |
| `session_cooccurrence.py` | Session-level co-occurrence signals from training data |
| `train_thought_bm25.py` | BM25 over training-set rationale/thought annotations |
| `multi_source.py` | Internally fuses BM25 + BGE + BPR via RRF to produce a base candidate list; all sources are then unioned for LambdaRank |

Each source retrieves up to 400 candidates independently. All source lists are unioned into a single candidate set (~2000+ unique tracks) passed to LambdaRank. LambdaRank uses each source's per-candidate rank as a feature — there is no RRF step in the final ranking path.

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

A LightGBM LambdaRank model trained on the dev set with per-query groups. All source candidate lists are unioned into a single pool (~2000+ tracks per query) and LambdaRank scores every candidate using features:

- Rank in each source's individual list (BM25, BGE, BPR, each I2I variant, artist shortcut, entity, etc.)
- Source presence flag (binary) per source
- Source consensus count (how many sources returned this candidate)
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

## Experiment: Fine-Tuned Bi-Encoder (`scripts/finetune_biencoder.py`)

> **Note:** This was an experiment conducted during development. The fine-tuned bi-encoder was **not used in the final submission** — the submitted system uses only the retrieval sources listed above.

We fine-tuned `Qwen/Qwen3-Embedding-0.6B` as a query encoder against **frozen precomputed track embeddings** to explore whether a task-specific dense retriever could improve candidate recall:

- **Training data**: ~13k (query, positive track) pairs from goal-annotated dev set turns
- **Hard negatives**: tracks retrieved by the current system (BM25, BGE, I2I, artist shortcut) that are NOT the ground truth
- **Loss**: In-batch negatives + hard negatives via InfoNCE
- **Key implementation detail**: Qwen3 is a causal LM — must use **last-token pooling**, not CLS, for embeddings

**Result**: On a 100-session held-out mini devset, the fine-tuned retriever contributed +3 unique candidates at the pool level (additive over all other sources), raising pool recall from 0.62 → 0.64. The inference module is in `mcrs/retrieval_modules/finetuned_dense.py`.

---

## Results

All scores are on the blind evaluation sets (Codabench).

| Configuration | Composite | nDCG@20 | CatDiv | LexDiv | LLM-Judge |
|--------------|-----------|---------|--------|--------|-----------|
| **This repo — multi-source + LambdaRank + Qwen3-8B gen (Blind A)** | **0.5296** | **0.4124** | 0.0320 | 0.7273 | **4.30** |
| Multi-source + planner cache + Qwen3-8B gen | 0.5261 | 0.3258 | 0.0314 | 0.7504 | 4.80 |
| LLM planner + metadata filter + flash_attn | 0.4886 | 0.2833 | 0.0314 | 0.7385 | 4.60 |
| Multi-source + Qwen3 dense + I2I + shortcuts | 0.4270 | 0.2291 | 0.0315 | 0.7304 | 4.15 |
| Qwen3-8B thinking (baseline) | 0.4147 | 0.1689 | 0.1064 | 0.7584 | 4.25 |

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

## Setup

```bash
uv venv .venv --python=3.10
source .venv/bin/activate
uv pip install -e .

# On GPU machines
pip install flash-attn --no-build-isolation
```

For GPU (RunPod):

```bash
export HF_HOME=/workspace/.cache/huggingface
huggingface-cli login
pip install -e .
pip install --force-reinstall lm-format-enforcer
```

---

## Running Inference

```bash
# Dev set
python run_inference_devset.py --tid qwen3_8b_multi_source_devset --batch_size 10

# Blind set
python run_inference_blindset.py --tid qwen3_8b_multi_source_blindset_B --batch_size 10

# Verify no empty responses
python -c "
import json
d = json.load(open('exp/inference/blindset_B/qwen3_8b_multi_source_blindset_B.json'))
empty = [e['session_id'][:8] for e in d if not e['predicted_response'].strip()]
print(f'{len(empty)} empty responses:', empty)
"
```

---

## Training

### LambdaRank Reranker

```bash
# Step 1: precompute planner queries (one-time)
python scripts/precompute_planner.py --config config/lambdarank_training.yaml

# Step 2: train LambdaRank
python scripts/train_lambdarank.py \
  --config config/lambdarank_training.yaml \
  --goal_filter \
  --last_n_turns 1
```

### Two-Tower Reranker (baseline)

```bash
python train_two_tower_reranker.py \
  --output_path ./cache/two_tower_reranker.pt \
  --retrieval_device cpu \
  --batch_size 256 \
  --epochs 5
```

### Fine-Tune Bi-Encoder (experiment)

```bash
python scripts/finetune_biencoder.py \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-5 \
  --goal_filter \
  --last_n_turns 1
```

Checkpoints saved to `./cache/finetuned_biencoder/`, best checkpoint to `./cache/finetuned_biencoder/best/`.

---

## Repository Structure

```
mcrs/
├── crs_baseline.py            # Main CRS class wiring all components
├── retrieval_modules/
│   ├── bm25.py                # Weighted BM25
│   ├── bert.py                # Dense retrieval (BGE)
│   ├── qwen3_dense.py         # Zero-shot Qwen3 dense retrieval
│   ├── finetuned_dense.py     # Fine-tuned Qwen3-0.6B retrieval (experiment)
│   ├── user_to_item.py        # CF/BPR user-to-item
│   ├── item_to_item.py        # Multimodal I2I expansion
│   ├── session_cooccurrence.py
│   ├── train_thought_bm25.py
│   └── multi_source.py        # Base RRF fusion (BM25+BGE+BPR)
├── lm_modules/
│   ├── qwen3.py               # Qwen3-8B: planner + response generator
│   └── llama.py               # Llama-3.2-1B (baseline)
├── reranker_modules/
│   ├── two_tower.py           # Two-tower DCN reranker
│   └── embedding.py
└── db_item / db_user          # Catalog and user profile access

scripts/
├── train_lambdarank.py        # LambdaRank training (LightGBM)
├── finetune_biencoder.py      # Qwen3-0.6B bi-encoder fine-tuning (experiment)
├── precompute_planner.py      # Cache Qwen3-8B planner outputs
└── evaluate_dev_ndcg.py       # Dev set evaluation

config/                        # YAML inference + training configs
run_inference_devset.py        # Dev set inference entry point
run_inference_blindset.py      # Blind set inference entry point
train_two_tower_reranker.py    # Two-tower training entry point
```

---

## Challenge

- **Challenge**: [ACM RecSys 2026 Music-CRS Challenge (TalkPlay)](https://nlp4musa.github.io/music-crs-challenge/)
- **Evaluation server**: [Codabench](https://www.codabench.org/)
- **Evaluator**: [nlp4musa/music-crs-evaluator](https://github.com/nlp4musa/music-crs-evaluator)
