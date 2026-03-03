<img width="953" height="383" alt="image" src="https://github.com/user-attachments/assets/db808f6c-47fa-4d5b-855c-468b3282cacf" />


# LiquidBrain

> *"What if language was not a matrix of frozen weights, but a living graph where connections fatigue, recover, and compete — like a brain?"*

[![Rust](https://img.shields.io/badge/Rust-1.85%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Genesis-V115-brightgreen)]()
[![Status](https://img.shields.io/badge/status-experimental-yellow)]()

**LiquidBrain** is an experimental language modeling engine written entirely in Rust.  Rather than encoding language as static weight matrices (as in Transformers or classical n-gram models), it builds a **dynamic semantic graph** whose connections actively fatigue and recover during generation — drawing inspiration from biological synaptic plasticity.

---

## Abstract

Classical n-gram language models are fast and interpretable but suffer from two fundamental problems: *repetitive loops* (the model gets stuck in high-frequency cycles) and *catastrophic forgetting* (batch-only training). LiquidBrain addresses both by treating every learned association as a **synapse** with two properties: a long-term `weight` (accumulated reinforcement) and a short-term `health` (depleted on use, passively recovered).

The result is a system that:
- **Naturally avoids repetition** without post-hoc penalties, through biological fatigue.
- **Learns continuously** from every user interaction, with no gradient computation.
- **Anchors generation** at semantically rich tokens using an attention-like focus mechanism.
- **Runs entirely in RAM**, with a compact binary persistence format.

This is a research prototype — not a production language model.  Its goal is to explore whether **biologically-inspired dynamics** can compensate for the statistical limitations of small, incrementally-trained models.

---

## Key Features

| Feature | Description |
|---|---|
| **Synaptic Fatigue** | Each connection loses health when it fires, forcing the model to explore alternative paths (anti-loop mechanism). |
| **Passive Recovery** | Health regenerates over time, allowing previously silenced paths to become viable again. |
| **Focus Point** | Generation starts at the most semantically salient token of the prompt (attention × rarity score). |
| **Lateral Inhibition** | When a high-confidence fact is learned, competing connections are weakened (winner-take-all). |
| **Online Learning** | Every user input updates the graph in real time — no offline training phase. |
| **Markdown Parser** | Structural Markdown elements (tables, code blocks, lists) are translated into semantic punctuation rather than stripped. |
| **Auto-Pruning** | Weak synapses are periodically removed to maintain a compact memory footprint. |
| **Binary Persistence** | The full graph is serialised with `bincode` for fast, compact save/load. |

---

## Architecture

```
LiquidBrain
└── neurons : HashMap<Vec<u32>, Neuron>
                  │                 │
        context key (n-gram)        └── connections : HashMap<u32, Synapse>
        1 to MAX_CONTEXT_SIZE                               │          │
        token IDs                                       token_id    weight
                                                                    health
```

### Conceptual flow

```
User prompt
    │
    ▼
[Markdown Parser] ── translates structure into punctuation tokens
    │
    ▼
[Tokenizer] ── word → u32 ID mapping (incremental vocabulary)
    │
    ├──▶ [learn_live] ── online reinforcement at FACT_LEARNING_RATE
    │
    └──▶ [find_focus_point] ── attention × rarity → anchor token
              │
              ▼
        [Generation Loop]
              │
              ├── backoff context lookup (trigram → bigram → unigram)
              ├── passive health recovery for all synapses in neuron
              ├── fact-recall mode (filter noise when confidence is high)
              ├── temperature sampling: score = (weight × health)^(1/T)
              ├── roulette-wheel selection
              └── synaptic fatigue: chosen synapse.health -= SYNAPSE_COST
```

### How it differs from a Transformer

| Dimension | Transformer | LiquidBrain |
|---|---|---|
| Storage | Dense weight matrices (GB) | Sparse hash-map graph (MB) |
| Training | Offline, gradient descent, GPU | Online, Hebbian-like, CPU |
| Memory | Fixed context window | Dynamic n-gram backoff |
| Anti-repetition | Repetition penalty (post-hoc) | Synaptic fatigue (emergent) |
| Attention | Scaled dot-product | Focus-point + fact-recall |
| Interpretability | Low (opaque embeddings) | High (inspect any neuron) |

---

## Installation

### Prerequisites
- Rust 1.85+ (2024 edition) — install via [rustup.rs](https://rustup.rs)

### Clone and build

```bash
git clone https://github.com/infinition/LiquidBrain.git
cd LiquidBrain
cargo build --release
```

### Run

```bash
cargo run --release
```

Or after building:
```bash
./target/release/liquidbrain      # Linux / macOS
target\release\liquidbrain.exe    # Windows
```

---

## Usage

LiquidBrain starts an interactive REPL.  On the first launch, the brain is empty.  Teach it by training on files or typing facts directly.

### Quickstart

```
# 1. Start the engine
cargo run --release

# 2. Train on a folder of Markdown / text files
VOUS > /train ./data

# 3. Save the trained brain
VOUS > /save

# 4. Chat
VOUS > Rust est un langage système rapide et sûr.

[Focus: langage]
GENESIS > langage système rapide et sûr , Rust garantit la sécurité mémoire .
```

### CLI Command Reference

| Command | Description |
|---|---|
| `/train <folder>` | Recursively ingest all `.txt` and `.md` files from `<folder>`. |
| `/save` | Persist the current brain to `genesis.brain`. |
| `/load` | Reload the brain from `genesis.brain` at runtime. |
| `/stats` | Display memory statistics (vocab size, neuron/synapse counts). |
| `/gen <n>` | Set the maximum generation length (default: 60 tokens). |
| `/prune` | Manually trigger weak-synapse pruning. |
| `/help` | Show the command reference. |
| `/quit` or `/q` | Exit cleanly. |
| `<any text>` | Chat: learn the input and generate a response. |

### Learning behaviour

- **Declarative statements** (e.g., `"The sky is blue."`) are learned at `FACT_LEARNING_RATE = 50.0` — strong reinforcement.
- **Questions** (ending with `?`) are learned at `QUESTION_LEARNING_RATE = 0.0` — the model does not memorise interrogative phrasing.
- **Corpus files** (`/train`) are ingested at `CORPUS_LEARNING_RATE = 3.0` — moderate reinforcement, many competing phrasings.

---

## Algorithm Deep-Dive

### 1. Synaptic Fatigue (Anti-Loop Mechanism)

In a standard Markov chain, the highest-probability transition is always chosen (or sampled with high probability), creating repetitive loops.  LiquidBrain breaks this with **health**:

```
effective_weight = weight × health

# Each time a synapse fires:
health -= SYNAPSE_COST          # fatigue

# Each generation step (passive):
health += SYNAPSE_RECOVERY      # recovery
health = min(health, MAX_HEALTH)
```

When health reaches 0, the synapse is **silent** — even a heavily reinforced association becomes unavailable, forcing the model to explore the next-best path.  After a few steps without firing, health recovers and the path becomes available again.

### 2. Focus Point (Attention Approximation)

Rather than starting generation at a random or fixed token, LiquidBrain computes a **focus score** for each known token in the prompt:

```
score = attention_score × rarity

attention_score = max outgoing synapse weight (how much the model knows about this word)
rarity          = 1 / ln(global_frequency)   (rare words carry more specific meaning)
                  × 5  if frequency < 10      (extra boost for very specific terms)
```

The token with the highest score anchors the generation.  This approximates TF-IDF × attention: start from the word that is both *salient in the prompt* and *well-represented in memory*.

### 3. Lateral Inhibition (Fact Sharpening)

When learning with `weight_bonus > STRONG_FACT_THRESHOLD` (i.e., a direct user statement), the model applies **lateral inhibition**:

```
For every competitor synapse in the same neuron (where competitor ≠ target):
    competitor.weight *= INHIBITION_FACTOR   # e.g., 0.85 → -15%
```

This sharpens the model's certainty about high-confidence associations, mimicking **winner-take-all dynamics** in biological cortical columns.

### 4. Multi-Scale Context (n-gram Backoff)

Every sequence is learned simultaneously at context sizes 1, 2, and 3:

```
"the cat sat" → learns:
  [the]        → cat    (unigram context)
  [the, cat]   → sat    (bigram context)
  [the,cat,sat]→ END    (trigram context)
```

During generation, the model tries the longest matching context first and falls back to shorter ones — guaranteeing coverage even for unseen trigrams.

---

## Configuration

All hyperparameters live in `src/config.rs`.  No recompilation tricks needed — just edit the file and `cargo run --release`.

| Constant | Default | Description |
|---|---|---|
| `MAX_CONTEXT_SIZE` | `3` | Longest n-gram context (trigram) |
| `BASE_TEMPERATURE` | `1.0` | Sampling temperature (0=greedy, >1=random) |
| `SYNAPSE_COST` | `1.5` | Health lost per synapse firing |
| `SYNAPSE_RECOVERY` | `0.10` | Health gained per idle step |
| `FACT_LEARNING_RATE` | `50.0` | Reinforcement for live user statements |
| `CORPUS_LEARNING_RATE` | `3.0` | Reinforcement for `/train` ingestion |
| `INHIBITION_FACTOR` | `0.85` | Competitor weakening during strong-fact learning |
| `TRUTH_THRESHOLD` | `10.0` | Weight above which fact-recall mode activates |
| `PRUNING_THRESHOLD` | `0.5` | Minimum weight to survive pruning |
| `DEFAULT_GENERATION_LENGTH` | `60` | Default tokens per response |

---

## Project Structure

```
LiquidBrain/
├── Cargo.toml          Project manifest
├── Cargo.lock          Reproducible dependency lock
├── README.md           This file
├── ROADMAP.md          Development plan and feature backlog
├── .gitignore
└── src/
    ├── main.rs         CLI entry point and REPL
    ├── config.rs       All hyperparameters (one place to tune)
    ├── tokenizer.rs    Word ↔ ID mapping + Markdown parser
    └── brain.rs        LiquidBrain engine (learning, generation, persistence)
```

---

## Running Tests

```bash
cargo test
```

Tests cover:
- `Tokenizer`: vocabulary management, case-insensitive lookup, Markdown parser.
- `LiquidBrain`: learning, lateral inhibition, weight capping, pruning, statistics.
- `sample_token`: roulette-wheel sampling, edge cases (empty, NaN).

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan, including upcoming features, testing phases, and research directions.

---

## Limitations

- **No grammar** — generation is purely statistical; it does not model syntax.
- **Vocabulary-bound** — the model cannot handle words it has never seen.
- **No long-range coherence** — context window is max 3 tokens; long-distance dependencies are not captured.
- **Language-neutral** — works on any tokenisable language but has no built-in morphological awareness.

These are known constraints of the n-gram foundation.  See ROADMAP.md for planned mitigations.

---

## Contributing

Contributions, ideas, and forks are welcome.

1. Fork the repo.
2. Create a branch: `git checkout -b feature/my-idea`.
3. Make your changes with tests.
4. Open a pull request against `main`.

Please keep PRs focused (one feature / fix per PR) and run `cargo test` before submitting.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## References & Inspiration

- Hebb, D.O. (1949) — *The Organisation of Behaviour* — synaptic plasticity rule.
- Elman, J.L. (1990) — *Finding Structure in Time* — simple recurrent networks.
- Jurafsky, D. & Martin, J.H. (2023) — *Speech and Language Processing*, Ch. 3 — n-gram language models and backoff smoothing.
- Wilson, M.A. & McNaughton, B.L. (1994) — *Reactivation of hippocampal ensemble memories during sleep* — pruning / consolidation during rest.

---

*LiquidBrain: exploring language at the intersection of graph theory, thermodynamics, and cellular biology.*
