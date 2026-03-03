// =============================================================================
// LiquidBrain — Core Engine
// =============================================================================
//
// This module implements the bio-inspired language model at the heart of
// LiquidBrain.  The key idea: model language not as static weight matrices
// (à la Transformers) but as a *living graph* of synaptic connections that
// fatigue, recover, and compete — just like biological neurons.
//
// Architecture overview:
//
//   LiquidBrain
//   └── neurons : HashMap<Vec<u32>, Neuron>
//                     │                 │
//           context key                 └── connections : HashMap<u32, Synapse>
//       (n-gram, 1 to 3 IDs)                                  │          │
//                                                          next_token  weight + health
//
// A "neuron" is really a context node: given that the last 1–3 tokens were
// [A, B, C], which token should come next?  Each candidate is connected via a
// "synapse" that stores:
//   • weight  — accumulated reinforcement (how strongly this pairing was learned)
//   • health  — current energy (depleted on use, recovered over time)
//
// Generation uses health-weighted temperature sampling, which means a very
// frequently fired synapse becomes temporarily "tired" and the model explores
// alternative paths — preventing the repetitive loops that plague naive Markov
// chains.
//
// References / inspiration:
//   • Hebb, D.O. (1949) — "The Organisation of Behaviour" (synaptic plasticity)
//   • Elman, J.L. (1990) — Simple Recurrent Networks (context window)
//   • Jurafsky & Martin (2023) — Language Models chapter (n-gram backoff)
// =============================================================================

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::{fmt, fs};

use rand::Rng;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::config::*;
use crate::tokenizer::Tokenizer;

// =============================================================================
// Data structures
// =============================================================================

/// A directed, weighted, health-tracked connection between two tokens.
///
/// The **weight** measures how often `context → next_token` was reinforced
/// during learning.  The **health** measures how "fresh" this connection
/// currently is during generation — it decays on use and recovers passively.
///
/// ```text
/// effective_weight = weight × health
/// ```
///
/// When `health = 0`, the synapse is silent even if its weight is high,
/// forcing the model to explore alternative next tokens.
#[derive(Serialize, Deserialize, Clone)]
pub struct Synapse {
    /// Accumulated learning signal.  Bounded to [0, MAX_SYNAPSE_WEIGHT].
    pub weight: f32,

    /// Current health in [0, MAX_HEALTH].  Depleted by SYNAPSE_COST on each
    /// firing, recovered by SYNAPSE_RECOVERY passively each generation step.
    pub health: f32,
}

/// A node in the semantic graph, keyed by a context n-gram.
///
/// Holds all candidate next tokens for the context this neuron represents,
/// each as a `Synapse` storing the learned weight and current health.
#[derive(Serialize, Deserialize, Clone)]
pub struct Neuron {
    /// Maps candidate next-token ID → synaptic connection data.
    pub connections: HashMap<u32, Synapse>,
}

impl Neuron {
    pub fn new() -> Self {
        Neuron { connections: HashMap::new() }
    }

    /// Removes all synapses whose weight has fallen below `PRUNING_THRESHOLD`.
    ///
    /// Called by `LiquidBrain::sleep_and_prune`.  Think of this as synaptic
    /// elimination during sleep — the brain discards under-used connections to
    /// sharpen signal-to-noise ratio and free resources.
    pub fn prune_weak_connections(&mut self) {
        self.connections.retain(|_, syn| syn.weight > PRUNING_THRESHOLD);
    }
}

// =============================================================================
// Persistence shim
// =============================================================================

/// Serialisable mirror of `LiquidBrain` for disk storage via `bincode`.
///
/// # Why not derive Serialize directly on LiquidBrain?
/// `HashMap<Vec<u32>, Neuron>` doesn't serialise as cleanly across `bincode`
/// versions as `Vec<(Vec<u32>, Neuron)>`.  This intermediate struct converts
/// the maps to sorted vectors on save and back on load.
#[derive(Serialize, Deserialize)]
pub struct LiquidBrainPersist {
    pub neurons: Vec<(Vec<u32>, Neuron)>,
    pub tokenizer: Tokenizer,
    pub global_counts: Vec<(u32, u32)>,
}

// =============================================================================
// Statistics snapshot
// =============================================================================

/// A read-only snapshot of the brain's current state, returned by `stats()`.
pub struct BrainStats {
    pub vocab_size: usize,
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub avg_connections: f32,
}

impl fmt::Display for BrainStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Vocab: {} tokens | Neurons: {} | Synapses: {} | Avg connections/neuron: {:.1}",
            self.vocab_size, self.neuron_count, self.synapse_count, self.avg_connections
        )
    }
}

// =============================================================================
// LiquidBrain — main engine
// =============================================================================

/// The LiquidBrain semantic graph engine.
///
/// All knowledge is stored as a `HashMap` from context n-grams (`Vec<u32>`)
/// to `Neuron` objects.  This structure supports both online (live) learning
/// and batch ingestion of text files, with the same underlying mechanism.
pub struct LiquidBrain {
    /// Semantic graph: context → candidate next tokens.
    pub neurons: HashMap<Vec<u32>, Neuron>,

    /// Vocabulary and frequency tracker.
    pub tokenizer: Tokenizer,
}

impl LiquidBrain {
    // ─────────────────────────────────────────────────────────────────────────
    // Construction & statistics
    // ─────────────────────────────────────────────────────────────────────────

    /// Creates an empty brain with no vocabulary and no connections.
    pub fn new() -> Self {
        LiquidBrain {
            neurons: HashMap::new(),
            tokenizer: Tokenizer::new(),
        }
    }

    /// Returns a statistics snapshot of the current state.
    pub fn stats(&self) -> BrainStats {
        let neuron_count = self.neurons.len();
        let synapse_count: usize = self.neurons.values().map(|n| n.connections.len()).sum();
        let avg_connections = if neuron_count > 0 {
            synapse_count as f32 / neuron_count as f32
        } else {
            0.0
        };
        BrainStats {
            vocab_size: self.tokenizer.vocab_size(),
            neuron_count,
            synapse_count,
            avg_connections,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Learning
    // ─────────────────────────────────────────────────────────────────────────

    /// Core learning function: reinforces every n-gram → next-token association
    /// present in `tokens` at all context sizes from MIN to MAX simultaneously.
    ///
    /// # Algorithm
    /// For each context size `s` in [MIN_CONTEXT_SIZE, MAX_CONTEXT_SIZE]:
    ///   For each position `i` in tokens:
    ///     context = tokens[i .. i+s]
    ///     next    = tokens[i+s]
    ///     Reinforce: neuron[context].synapse[next].weight += weight_bonus
    ///
    /// # Lateral inhibition (strong facts only)
    /// When `weight_bonus > STRONG_FACT_THRESHOLD`, the target synapse's
    /// *competitors* in the same neuron are weakened by `INHIBITION_FACTOR`.
    /// This sharpens the model's certainty by suppressing alternative
    /// predictions — mimicking winner-take-all dynamics in cortical columns.
    ///
    /// # Weight cap
    /// Synapse weight is capped at `MAX_SYNAPSE_WEIGHT` to prevent any single
    /// association from completely dominating generation.
    pub fn learn_sequence(&mut self, tokens: &[u32], weight_bonus: f32) {
        // A "strong fact" is any input learned with a high-confidence rate
        // (e.g., a direct user statement vs. a corpus passage).
        let is_strong_fact = weight_bonus > STRONG_FACT_THRESHOLD;

        for size in MIN_CONTEXT_SIZE..=MAX_CONTEXT_SIZE {
            if tokens.len() <= size {
                continue; // Not enough tokens for this context size.
            }
            for i in 0..tokens.len() - size {
                let context   = tokens[i..i + size].to_vec();
                let next_token = tokens[i + size];

                let neuron = self.neurons.entry(context).or_insert_with(Neuron::new);

                // ── Lateral inhibition ───────────────────────────────────────
                // Weaken competing connections before reinforcing the winner.
                // This makes the model more "certain" about high-confidence facts.
                if is_strong_fact {
                    for (token, synapse) in neuron.connections.iter_mut() {
                        if *token != next_token {
                            synapse.weight *= INHIBITION_FACTOR;
                        }
                    }
                }

                // ── Reinforce target synapse ─────────────────────────────────
                let synapse = neuron.connections.entry(next_token).or_insert(Synapse {
                    weight: 0.0,
                    health: MAX_HEALTH,
                });
                synapse.weight = (synapse.weight + weight_bonus).min(MAX_SYNAPSE_WEIGHT);
                // Learning refreshes health: the connection is reactivated.
                synapse.health = MAX_HEALTH;
            }
        }
    }

    /// Learns from a raw text string, segmenting it into sentences.
    ///
    /// Returns the number of logical sentence segments processed.
    /// This is the shared implementation used by both `ingest_folder` and
    /// any future batch learning APIs.
    fn ingest_text(&mut self, text: &str, rate: f32) -> usize {
        let words = Tokenizer::clean_text(text);
        let mut sentence_buffer: Vec<u32> = Vec::new();
        let mut sentence_count = 0;

        for w in words {
            let id = self.tokenizer.get_or_create_id(&w);
            sentence_buffer.push(id);

            if self.tokenizer.is_sentence_ender(&w) {
                // Sentence boundary: append END token and learn the full segment.
                sentence_buffer.push(END_TOKEN_ID);
                self.learn_sequence(&sentence_buffer, rate);
                sentence_buffer.clear();
                sentence_count += 1;
            }
        }
        // Learn any trailing fragment that didn't end with a sentence ender.
        if !sentence_buffer.is_empty() {
            sentence_buffer.push(END_TOKEN_ID);
            self.learn_sequence(&sentence_buffer, rate);
        }
        sentence_count
    }

    /// Ingests all `.txt` and `.md` files found recursively under `folder_path`.
    ///
    /// Each file is:
    ///   1. Parsed by the Markdown semantic parser (structure → punctuation).
    ///   2. Segmented into logical sentences at `.`, `!`, `?` boundaries.
    ///   3. Fed to `learn_sequence` with `CORPUS_LEARNING_RATE`.
    ///
    /// After all files are processed, `sleep_and_prune` removes weak synapses
    /// automatically.
    ///
    /// # UTF-8 handling
    /// Files that cannot be read as valid UTF-8 are skipped with a warning.
    pub fn ingest_folder(&mut self, folder_path: &str) {
        println!(
            "\n>>> DÉBUT ENTRAÎNEMENT | Markdown Parser | Dossier: {}\n",
            folder_path
        );
        let mut total_files = 0;

        for entry in WalkDir::new(folder_path).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            // Only process plain text and Markdown files.
            let is_valid = path
                .extension()
                .map_or(false, |ext| ext == "txt" || ext == "md");
            if !is_valid {
                continue;
            }

            print!("  Lecture : {:?} ... ", entry.file_name());
            io::stdout().flush().unwrap();

            match fs::read_to_string(path) {
                Ok(text) => {
                    let count = self.ingest_text(&text, CORPUS_LEARNING_RATE);
                    println!("OK  ({} séquences logiques)", count);
                    total_files += 1;
                }
                Err(_) => println!("IGNORÉ  (encodage non-UTF-8)"),
            }
        }

        println!("\n>>> NETTOYAGE MÉMOIRE (élagage des synapses faibles)...");
        self.sleep_and_prune();
        println!(">>> Entraînement terminé. {} fichier(s) intégré(s).\n", total_files);
    }

    /// Learns from a single live user input in real time.
    ///
    /// Applies `FACT_LEARNING_RATE` for statements and `QUESTION_LEARNING_RATE`
    /// for questions (input ending with '?').
    ///
    /// # Why 0.0 for questions?
    /// Questions are interrogative — they don't assert facts.  We want the
    /// model to answer questions, not to memorise them as things to reproduce.
    /// If you want the model to learn question phrasings (e.g., for FAQ
    /// corpora), set `QUESTION_LEARNING_RATE` to a positive value in config.rs.
    pub fn learn_live(&mut self, input: &str) {
        let words = Tokenizer::clean_text(input);
        let tokens: Vec<u32> = words
            .iter()
            .map(|w| self.tokenizer.get_or_create_id(w))
            .collect();

        let is_question = input.trim().ends_with('?');
        let rate = if is_question { QUESTION_LEARNING_RATE } else { FACT_LEARNING_RATE };

        if rate > 0.0 {
            self.learn_sequence(&tokens, rate);
            // Also learn the sentence with its terminating END token so the model
            // knows this sequence has a natural stopping point.
            let mut seq_with_end = tokens;
            seq_with_end.push(END_TOKEN_ID);
            self.learn_sequence(&seq_with_end, rate);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Memory maintenance
    // ─────────────────────────────────────────────────────────────────────────

    /// Prunes weak synapses from all neurons, then removes empty neurons.
    ///
    /// This is analogous to synaptic elimination during sleep in biological
    /// brains: rarely-used connections are discarded.
    ///
    /// Prints a before/after neuron count to the terminal.
    pub fn sleep_and_prune(&mut self) {
        let before = self.neurons.len();
        for neuron in self.neurons.values_mut() {
            neuron.prune_weak_connections();
        }
        // Remove neurons that have no remaining connections.
        self.neurons.retain(|_, n| !n.connections.is_empty());
        let after = self.neurons.len();
        println!("   Élagage : {} → {} neurones actifs.", before, after);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Attention / Focus point
    // ─────────────────────────────────────────────────────────────────────────

    /// Scores a token by the maximum outgoing synapse weight from its unigram
    /// neuron — a proxy for "how much the model knows about this token".
    ///
    /// Tokens with many strong outgoing connections score high: the model has
    /// rich knowledge anchored to them and will produce coherent continuations.
    fn calculate_attention_score(&self, token_id: u32) -> f32 {
        let ctx = vec![token_id];
        self.neurons
            .get(&ctx)
            .map(|neuron| {
                neuron.connections.values()
                    .map(|s| s.weight)
                    .fold(0.0_f32, f32::max)
            })
            .unwrap_or(0.0)
    }

    /// Selects the most semantically significant token from the prompt as the
    /// starting point for generation.
    ///
    /// # Scoring formula
    /// `score = attention_score × rarity`
    ///
    /// - **attention_score**: maximum outgoing synapse weight from this token's
    ///   unigram neuron (richness of the model's knowledge about this word).
    /// - **rarity**: `1 / ln(frequency)` — rare words carry more specific
    ///   meaning.  Tokens seen fewer than 10 times get a ×5 bonus to ensure
    ///   very specific terms anchor generation even when their attention score
    ///   is moderate.
    ///
    /// This approximates TF-IDF × attention: start from the word that is both
    /// *salient in the prompt* and *well-represented in memory*.
    pub fn find_focus_point(&self, tokens: &[u32]) -> u32 {
        let mut best_id = tokens[0]; // Safe: caller guarantees non-empty.
        let mut max_score: f32 = -1.0;

        for &id in tokens {
            let word = self.tokenizer.decode(id);

            // Skip punctuation (no semantic content) and very short tokens
            // (articles, prepositions) that are too generic to anchor on.
            if self.tokenizer.is_punctuation(&word) || word.len() < 2 {
                continue;
            }

            let count = self.tokenizer.global_counts.get(&id).copied().unwrap_or(1);

            // Rarity: inverse of log-frequency.  Saturates for very common words.
            // ln() grows slowly so even moderately common words keep a useful score.
            let mut rarity = 1.0 / (count as f32).ln().max(1.0);

            // Extra boost for highly specific / rare vocabulary.
            if count < 10 {
                rarity *= 5.0;
            }

            let attention = self.calculate_attention_score(id);
            let score = attention * rarity;

            if score > max_score {
                max_score = score;
                best_id = id;
            }
        }
        best_id
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Generation
    // ─────────────────────────────────────────────────────────────────────────

    /// Generates a response to `prompt` of at most `length` tokens and prints
    /// it to stdout in real time.
    ///
    /// # Pipeline
    /// 1. **Online learning** — `learn_live(prompt)` incorporates the prompt.
    /// 2. **Tokenise** the prompt; keep only tokens already in the vocabulary.
    /// 3. **Focus selection** — pick the most salient known token as the
    ///    generation anchor (`find_focus_point`).
    /// 4. **Autoregressive loop** (up to `length` steps):
    ///    a. **Backoff context lookup** — try longest context first, fall back
    ///       to shorter n-grams until a matching neuron is found.
    ///    b. **Passive health recovery** — all synapses in the active neuron
    ///       recover `SYNAPSE_RECOVERY` health points.
    ///    c. **Fact recall mode** — if the strongest synapse exceeds
    ///       `TRUTH_THRESHOLD`, suppress low-confidence alternatives.
    ///    d. **Temperature sampling** — build a candidate list weighted by
    ///       `effective_weight ^ (1/temperature)`, then roulette-wheel sample.
    ///    e. **Synaptic fatigue** — the chosen synapse loses `SYNAPSE_COST`
    ///       health, reducing its probability for the next few steps.
    ///    f. **Emit token** — print the decoded word and advance the context.
    pub fn chat(&mut self, prompt: &str, length: usize) {
        let mut rng = rand::thread_rng();

        // ── Step 1: Online learning ──────────────────────────────────────────
        // The prompt itself is incorporated into the graph before generation.
        // This means the model immediately "knows" what you just said.
        self.learn_live(prompt);

        // ── Step 2: Tokenise prompt & extract known tokens ───────────────────
        let words = Tokenizer::clean_text(prompt);
        let known_tokens: Vec<u32> = words
            .iter()
            .filter_map(|w| self.tokenizer.find_id(w))
            .collect();

        if known_tokens.is_empty() {
            println!(
                "\n[Aucun token connu dans ce prompt — entraîne-moi d'abord \
                avec /train <dossier> ou en me donnant des faits.]\n"
            );
            return;
        }

        // ── Step 3: Focus selection ──────────────────────────────────────────
        let focus_id   = self.find_focus_point(&known_tokens);
        let focus_word = self.tokenizer.decode(focus_id);

        print!("\n[Focus: {}]\nGENESIS > {}", focus_word, focus_word);
        io::stdout().flush().unwrap();

        // The context window starts at the focus token and grows as we generate.
        let mut context_ids = vec![focus_id];

        // ── Step 4: Autoregressive generation loop ───────────────────────────
        'generation: for step in 0..length {

            // ── 4a. Backoff context lookup ────────────────────────────────────
            // Try the longest matching context first (trigram → bigram → unigram).
            // This "backoff" strategy guarantees we always find *some* neuron
            // as long as the focus token was learned.
            //
            // Implementation note: we store the matching key and then call
            // `self.neurons.get_mut` separately.  This keeps the borrow of
            // `self.neurons` as a *field borrow* (not a full `&mut self` method
            // call), allowing `self.tokenizer.decode` to be called later in the
            // same loop body without conflicting — Rust's NLL field-splitting.
            let ctx_key = {
                let mut found: Option<Vec<u32>> = None;
                for size in (MIN_CONTEXT_SIZE..=MAX_CONTEXT_SIZE).rev() {
                    if context_ids.len() >= size {
                        let start = context_ids.len() - size;
                        let key   = context_ids[start..].to_vec();
                        if self.neurons.contains_key(&key) {
                            found = Some(key);
                            break;
                        }
                    }
                }
                match found {
                    Some(k) => k,
                    None    => break 'generation, // Dead end: no matching context.
                }
            };

            // Borrow the neuron mutably for the rest of this iteration.
            // Field borrow: `self.neurons` is borrowed, `self.tokenizer` is free.
            let neuron = self.neurons.get_mut(&ctx_key).unwrap();

            // ── 4b. Passive health recovery ───────────────────────────────────
            // Every synapse in the active neuron recovers a small amount of health
            // per step, regardless of whether it fires.  Previously fatigued paths
            // become viable again after a few steps of non-use.
            for syn in neuron.connections.values_mut() {
                syn.health = (syn.health + SYNAPSE_RECOVERY).min(MAX_HEALTH);
            }

            // ── 4c. Fact recall mode ──────────────────────────────────────────
            // If the strongest connection exceeds TRUTH_THRESHOLD, we are in a
            // high-confidence region of the graph.  Filter out low-weight noise
            // to produce more deterministic, factual output.
            let max_weight = neuron
                .connections
                .values()
                .map(|s| s.weight)
                .fold(0.0_f32, f32::max);
            let is_fact_recall = max_weight >= TRUTH_THRESHOLD;

            // Use a near-zero temperature for the first 3 tokens: produces a
            // coherent, deterministic opening phrase.
            let temp = if step < 3 { 0.1 } else { BASE_TEMPERATURE };

            // ── 4d. Build candidate list ──────────────────────────────────────
            let mut candidates: Vec<(u32, f32)> = Vec::new();
            let mut total_energy: f32 = 0.0;

            for (&token_id, synapse) in &neuron.connections {
                // In fact-recall mode, suppress uncertain alternatives.
                if is_fact_recall && synapse.weight < (TRUTH_THRESHOLD / 2.0) {
                    continue;
                }
                // Don't emit END before reaching the minimum generation length.
                if step < MIN_GENERATION_LENGTH && token_id == END_TOKEN_ID {
                    continue;
                }
                // Effective weight: base strength × current health (fatigue factor).
                let effective = synapse.weight * synapse.health;
                if effective > 0.0 {
                    // Temperature scaling:
                    //   low temp  → sharp distribution (picks top candidate)
                    //   high temp → flat distribution (more creative)
                    let score = effective.powf(1.0 / temp);
                    candidates.push((token_id, score));
                    total_energy += score;
                }
            }

            // ── 4d. Weighted random sampling (roulette-wheel) ─────────────────
            let selected_id = sample_token(&mut rng, &candidates, total_energy);

            // Handle termination conditions.
            if selected_id == END_TOKEN_ID {
                print!(".");
                io::stdout().flush().unwrap();
                break;
            }
            if selected_id == 0 {
                break; // No valid candidate found.
            }

            // ── 4e. Synaptic fatigue ──────────────────────────────────────────
            // The chosen synapse loses health — it has just "fired".
            // This is the core anti-repetition mechanism: a heavily used path
            // becomes temporarily less probable, forcing exploration.
            if let Some(syn) = neuron.connections.get_mut(&selected_id) {
                syn.health = (syn.health - SYNAPSE_COST).max(0.0);
            }

            // ── 4f. Emit token ────────────────────────────────────────────────
            // `self.tokenizer` is a different field from `self.neurons`,
            // so this borrow is valid while `neuron` (&mut Neuron from
            // `self.neurons`) is still in scope — Rust NLL field splitting.
            let word = self.tokenizer.decode(selected_id);

            // Punctuation prints without a leading space.
            if [".", ",", "!", "?", ";", ":", "'"].contains(&word.as_str()) {
                print!("{}", word);
            } else {
                print!(" {}", word);
            }
            io::stdout().flush().unwrap();

            context_ids.push(selected_id);
        }

        println!("\n");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Persistence
    // ─────────────────────────────────────────────────────────────────────────

    /// Serialises the full brain state to a compact binary file using `bincode`.
    ///
    /// The file is a flat binary encoding of `LiquidBrainPersist`.  It is fast
    /// to read and write but is **not** human-readable and is tightly coupled
    /// to the current struct layout.  If you change `Synapse` or `Neuron`, you
    /// will need to clear existing `.brain` files.
    pub fn save_to_file(&self, filename: &str) {
        println!("Sauvegarde vers '{}' ...", filename);
        let persist = LiquidBrainPersist {
            neurons:      self.neurons.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            tokenizer:    self.tokenizer.clone(),
            global_counts: self.tokenizer.global_counts.iter().map(|(&k, &v)| (k, v)).collect(),
        };
        match File::create(filename) {
            Ok(file) => {
                let writer = BufWriter::new(file);
                match bincode::serialize_into(writer, &persist) {
                    Ok(_) => {
                        let s = self.stats();
                        println!(
                            "Cerveau sauvegardé — {} neurones, {} synapses.",
                            s.neuron_count, s.synapse_count
                        );
                    }
                    Err(e) => eprintln!("Erreur sérialisation bincode : {}", e),
                }
            }
            Err(e) => eprintln!("Impossible de créer '{}' : {}", filename, e),
        }
    }

    /// Deserialises a brain from a binary file created by `save_to_file`.
    ///
    /// Returns `None` if:
    /// - The file does not exist (first launch — expected behaviour).
    /// - The file is corrupted or was written by an incompatible version.
    ///   In that case, delete the `.brain` file and start fresh.
    pub fn load_from_file(filename: &str) -> Option<Self> {
        println!("Chargement de '{}' ...", filename);
        let file = match File::open(filename) {
            Ok(f)  => f,
            Err(_) => {
                println!(
                    "Fichier '{}' non trouvé — démarrage avec un cerveau vide.",
                    filename
                );
                return None;
            }
        };

        let reader = BufReader::new(file);
        match bincode::deserialize_from::<_, LiquidBrainPersist>(reader) {
            Ok(persist) => {
                let neurons: HashMap<Vec<u32>, Neuron> =
                    persist.neurons.into_iter().collect();
                let mut brain = LiquidBrain {
                    neurons,
                    tokenizer: persist.tokenizer,
                };
                // Restore global frequency counts (stored separately for
                // bincode compatibility with older versions).
                for (k, v) in persist.global_counts {
                    brain.tokenizer.global_counts.insert(k, v);
                }
                let s = brain.stats();
                println!(
                    "Cerveau chargé — {} tokens, {} neurones, {} synapses.",
                    s.vocab_size, s.neuron_count, s.synapse_count
                );
                Some(brain)
            }
            Err(e) => {
                eprintln!(
                    "Erreur de chargement (fichier incompatible ou corrompu) : {}\n\
                     → Supprime '{}' et relance pour repartir de zéro.",
                    e, filename
                );
                None
            }
        }
    }
}

// =============================================================================
// Generation helpers (free functions to avoid borrow-checker conflicts)
// =============================================================================

/// Weighted random sampling (roulette-wheel / fitness-proportionate selection).
///
/// Samples one token from `candidates` proportionally to its score.
///
/// # Arguments
/// - `rng`          — mutable random number generator
/// - `candidates`   — `(token_id, score)` pairs; scores must be ≥ 0
/// - `total_energy` — sum of all scores (pre-computed by the caller)
///
/// # Fallback
/// If `total_energy` is not finite (e.g., NaN from extreme temperature values),
/// returns the highest-scoring candidate deterministically (argmax).
fn sample_token(
    rng: &mut impl Rng,
    candidates: &[(u32, f32)],
    total_energy: f32,
) -> u32 {
    if candidates.is_empty() {
        return 0; // No candidates → dead end.
    }

    if total_energy.is_finite() && total_energy > 0.0 {
        // Standard roulette-wheel selection.
        let mut pick = rng.gen_range(0.0..total_energy);
        for &(token_id, score) in candidates {
            pick -= score;
            if pick <= 0.0 {
                return token_id;
            }
        }
        // Floating-point rounding edge case: return the last candidate.
        candidates.last().map(|&(id, _)| id).unwrap_or(0)
    } else {
        // Degenerate case (extreme temperatures): pure argmax.
        candidates
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(id, _)| id)
            .unwrap_or(0)
    }
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Builds a small brain that knows "hello world ."
    fn brain_with_hello_world() -> (LiquidBrain, u32, u32) {
        let mut brain = LiquidBrain::new();
        let id_hello = brain.tokenizer.get_or_create_id("hello");
        let id_world = brain.tokenizer.get_or_create_id("world");
        let tokens   = vec![id_hello, id_world, END_TOKEN_ID];
        brain.learn_sequence(&tokens, FACT_LEARNING_RATE);
        (brain, id_hello, id_world)
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_brain_is_empty() {
        let brain = LiquidBrain::new();
        assert!(brain.neurons.is_empty());
        assert_eq!(brain.tokenizer.vocab_size(), 1); // only <END>
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    #[test]
    fn test_learn_sequence_creates_neuron() {
        let (brain, id_hello, id_world) = brain_with_hello_world();
        let ctx = vec![id_hello];
        assert!(brain.neurons.contains_key(&ctx), "unigram neuron for 'hello' should exist");
        let neuron = brain.neurons.get(&ctx).unwrap();
        assert!(neuron.connections.contains_key(&id_world), "'world' should follow 'hello'");
    }

    #[test]
    fn test_learn_sequence_weight_accumulates() {
        let (mut brain, id_hello, id_world) = brain_with_hello_world();
        let tokens = vec![id_hello, id_world, END_TOKEN_ID];
        brain.learn_sequence(&tokens, FACT_LEARNING_RATE);
        // After two reinforcements, weight should be 2× (or capped at MAX).
        let ctx    = vec![id_hello];
        let neuron = brain.neurons.get(&ctx).unwrap();
        let weight = neuron.connections[&id_world].weight;
        assert!(weight >= FACT_LEARNING_RATE, "weight should have accumulated");
    }

    #[test]
    fn test_learn_sequence_multi_context_sizes() {
        let mut brain = LiquidBrain::new();
        let a = brain.tokenizer.get_or_create_id("a");
        let b = brain.tokenizer.get_or_create_id("b");
        let c = brain.tokenizer.get_or_create_id("c");
        brain.learn_sequence(&[a, b, c, END_TOKEN_ID], 1.0);

        // Unigram context [a] → b
        assert!(brain.neurons.contains_key(&vec![a]));
        // Bigram context [a, b] → c
        assert!(brain.neurons.contains_key(&vec![a, b]));
        // Trigram context [a, b, c] → END
        assert!(brain.neurons.contains_key(&vec![a, b, c]));
    }

    #[test]
    fn test_strong_fact_triggers_lateral_inhibition() {
        let mut brain = LiquidBrain::new();
        let ctx_id = brain.tokenizer.get_or_create_id("sky");
        let blue   = brain.tokenizer.get_or_create_id("blue");
        let red    = brain.tokenizer.get_or_create_id("red");

        // First teach "sky → red" weakly.
        brain.learn_sequence(&[ctx_id, red, END_TOKEN_ID], 2.0);
        let weight_before = brain.neurons[&vec![ctx_id]].connections[&red].weight;

        // Then teach "sky → blue" as a strong fact — should inhibit "red".
        brain.learn_sequence(&[ctx_id, blue, END_TOKEN_ID], FACT_LEARNING_RATE);
        let weight_after = brain.neurons[&vec![ctx_id]].connections[&red].weight;

        assert!(
            weight_after < weight_before,
            "lateral inhibition should weaken competing synapse (red): {} → {}",
            weight_before, weight_after
        );
    }

    #[test]
    fn test_weight_is_capped() {
        let (mut brain, id_hello, id_world) = brain_with_hello_world();
        // Learn 1000 times — weight should never exceed MAX_SYNAPSE_WEIGHT.
        let tokens = vec![id_hello, id_world, END_TOKEN_ID];
        for _ in 0..1000 {
            brain.learn_sequence(&tokens, FACT_LEARNING_RATE);
        }
        let weight = brain.neurons[&vec![id_hello]].connections[&id_world].weight;
        assert!(weight <= MAX_SYNAPSE_WEIGHT, "weight exceeded cap: {}", weight);
    }

    // ── Pruning ───────────────────────────────────────────────────────────────

    #[test]
    fn test_pruning_removes_weak_connections() {
        let mut brain = LiquidBrain::new();
        let a = brain.tokenizer.get_or_create_id("a");
        let b = brain.tokenizer.get_or_create_id("b");
        let ctx = vec![a];
        brain.neurons.insert(ctx.clone(), Neuron::new());
        brain.neurons.get_mut(&ctx).unwrap().connections.insert(
            b,
            Synapse { weight: PRUNING_THRESHOLD - 0.1, health: MAX_HEALTH },
        );

        brain.sleep_and_prune();

        // The neuron should be completely removed (all connections pruned).
        assert!(
            !brain.neurons.contains_key(&ctx),
            "neuron with only weak connections should be removed"
        );
    }

    #[test]
    fn test_pruning_keeps_strong_connections() {
        let mut brain = LiquidBrain::new();
        let a = brain.tokenizer.get_or_create_id("a");
        let b = brain.tokenizer.get_or_create_id("b");
        let ctx = vec![a];
        brain.neurons.insert(ctx.clone(), Neuron::new());
        brain.neurons.get_mut(&ctx).unwrap().connections.insert(
            b,
            Synapse { weight: PRUNING_THRESHOLD + 10.0, health: MAX_HEALTH },
        );

        brain.sleep_and_prune();

        assert!(brain.neurons.contains_key(&ctx), "strong neuron should survive pruning");
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_counts() {
        let (brain, _, _) = brain_with_hello_world();
        let s = brain.stats();
        assert!(s.vocab_size >= 2, "should have at least hello and world");
        assert!(s.neuron_count > 0);
        assert!(s.synapse_count > 0);
        assert!(s.avg_connections > 0.0);
    }

    // ── Sample token ──────────────────────────────────────────────────────────

    #[test]
    fn test_sample_token_empty_returns_zero() {
        let mut rng = rand::thread_rng();
        assert_eq!(sample_token(&mut rng, &[], 0.0), 0);
    }

    #[test]
    fn test_sample_token_single_candidate() {
        let mut rng = rand::thread_rng();
        let candidates = vec![(42u32, 1.0_f32)];
        // With a single candidate, must always return it.
        for _ in 0..20 {
            assert_eq!(sample_token(&mut rng, &candidates, 1.0), 42);
        }
    }

    #[test]
    fn test_sample_token_degenerate_fallback() {
        let mut rng = rand::thread_rng();
        // Non-finite total_energy should trigger argmax fallback.
        let candidates = vec![(1u32, 3.0_f32), (2u32, 10.0_f32), (3u32, 1.0_f32)];
        let result = sample_token(&mut rng, &candidates, f32::NAN);
        assert_eq!(result, 2, "argmax fallback should pick the highest-score candidate");
    }
}
