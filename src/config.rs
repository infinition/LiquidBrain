// =============================================================================
// LiquidBrain — Configuration & Hyperparameters
// =============================================================================
//
// All tuneable constants are centralised here so you never have to grep the
// codebase to find a magic number.  Each constant is documented with:
//   • What it controls
//   • Typical range / effect of increasing or decreasing it
//
// The naming convention mirrors neuroscience vocabulary on purpose: this makes
// the mapping between code and biological metaphor immediately obvious.
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// CONTEXT WINDOW
// ─────────────────────────────────────────────────────────────────────────────

/// Largest n-gram used as a context key in the neuron graph.
///
/// Example with MAX = 3: the model can look at ["le", "chat", "mange"] to
/// predict the next token.  Larger values capture more syntax but cost more
/// memory.  Values above 4 usually degrade quality on small corpora.
pub const MAX_CONTEXT_SIZE: usize = 3;

/// Smallest n-gram used during both learning and generation backoff.
///
/// The model *always* learns uni-gram, bi-gram, … up to MAX_CONTEXT_SIZE
/// associations simultaneously, giving it multiple levels of granularity.
pub const MIN_CONTEXT_SIZE: usize = 1;

// ─────────────────────────────────────────────────────────────────────────────
// GENERATION SAMPLING
// ─────────────────────────────────────────────────────────────────────────────

/// Softmax temperature applied during token sampling.
///
/// • → 0.0 : fully greedy (always picks the highest-weight synapse)
/// • = 1.0 : proportional to learned weights (default, balanced)
/// • > 1.5 : very creative / random, likely incoherent
///
/// Note: the first 3 generation steps always use temperature 0.1 regardless
/// of this value, to produce a coherent opening phrase.
pub const BASE_TEMPERATURE: f32 = 1.0;

/// Default number of tokens generated per response when the user does not
/// specify a custom length with `/gen <n>`.
pub const DEFAULT_GENERATION_LENGTH: usize = 60;

/// The model refuses to emit the END token before this many tokens have been
/// generated.  Prevents degenerate one-word answers on sparse graphs.
pub const MIN_GENERATION_LENGTH: usize = 5;

// ─────────────────────────────────────────────────────────────────────────────
// SYNAPTIC FATIGUE (ANTI-LOOP MECHANISM)
// ─────────────────────────────────────────────────────────────────────────────

/// Health points subtracted from a synapse each time it fires during generation.
///
/// This is the core anti-repetition mechanism:
///   effective_weight = base_weight × health
///
/// When health reaches 0.0 the synapse is effectively silent, and the model
/// must choose an alternative path — preventing the infinite-loop failure mode
/// of classic Markov chains.
///
/// Higher value → faster fatigue → more path diversity, but shorter coherent spans.
pub const SYNAPSE_COST: f32 = 1.5;

/// Health points recovered per generation step for every synapse in the
/// currently active neuron.
///
/// Recovery is passive (it happens regardless of whether the synapse fired),
/// so heavily-used synapses can become available again after a few steps.
/// This mimics the refractory period / recovery of biological neurons.
///
/// Lower value → slower recovery → stronger diversity pressure.
pub const SYNAPSE_RECOVERY: f32 = 0.10;

/// Maximum health value a synapse can hold (normalised to 1.0).
/// Learning resets health to this value, "refreshing" the connection.
pub const MAX_HEALTH: f32 = 1.0;

// ─────────────────────────────────────────────────────────────────────────────
// LEARNING RATES
// ─────────────────────────────────────────────────────────────────────────────

/// Weight bonus applied when the user types a declarative statement (non-question).
///
/// 50.0 is deliberately high so that a single live statement strongly
/// reinforces the relevant synapses — useful for teaching facts interactively.
pub const FACT_LEARNING_RATE: f32 = 50.0;

/// Weight bonus applied when the user types a question (ends with '?').
///
/// Set to 0.0 because questions are interrogative, not declarative: we do not
/// want the model to "memorise" questions as facts to be reproduced.
/// Increasing this value would make the model remember the question phrasing,
/// which may be desirable for FAQ-style corpora.
pub const QUESTION_LEARNING_RATE: f32 = 0.0;

/// Weight bonus applied when ingesting files via `/train <folder>`.
///
/// Lower than FACT_LEARNING_RATE: corpus knowledge is less "certain" than a
/// direct user statement, and many competing phrasings in a corpus should not
/// all be equally forced.
pub const CORPUS_LEARNING_RATE: f32 = 3.0;

/// Hard cap on synapse weight.  Prevents runaway reinforcement after many
/// repeated exposures to the same sequence.
pub const MAX_SYNAPSE_WEIGHT: f32 = 100.0;

// ─────────────────────────────────────────────────────────────────────────────
// LATERAL INHIBITION & FACT SHARPENING
// ─────────────────────────────────────────────────────────────────────────────

/// Weight bonus threshold above which a learning event is considered a
/// "strong fact" and triggers lateral inhibition of competing synapses.
///
/// When learning with `weight_bonus > STRONG_FACT_THRESHOLD`, every *other*
/// outgoing synapse in the same neuron has its weight multiplied by
/// INHIBITION_FACTOR.  This sharpens the model's certainty about the
/// association, mimicking winner-take-all dynamics in cortical columns.
pub const STRONG_FACT_THRESHOLD: f32 = 10.0;

/// Multiplicative penalty applied to competing synapses during lateral
/// inhibition.  0.85 = 15% reduction per strong-fact learning event.
///
/// • → 1.0 : no inhibition (pure additive reinforcement)
/// • → 0.0 : maximum inhibition (competing synapses immediately silenced)
pub const INHIBITION_FACTOR: f32 = 0.85;

/// During generation, if the strongest synapse in the current neuron exceeds
/// this weight, the model enters "fact recall" mode: low-weight candidates
/// (weight < TRUTH_THRESHOLD / 2) are filtered out, making the output more
/// deterministic and factual.
pub const TRUTH_THRESHOLD: f32 = 10.0;

// ─────────────────────────────────────────────────────────────────────────────
// MEMORY MANAGEMENT
// ─────────────────────────────────────────────────────────────────────────────

/// Synapses with weight ≤ this value are removed during `sleep_and_prune`.
///
/// Pruning is called automatically after `/train` and can be triggered
/// manually.  It frees RAM and sharpens the graph by discarding noisy,
/// barely-reinforced connections.
///
/// Higher value → more aggressive pruning → smaller, sharper graph.
pub const PRUNING_THRESHOLD: f32 = 0.5;

// ─────────────────────────────────────────────────────────────────────────────
// SPECIAL TOKENS
// ─────────────────────────────────────────────────────────────────────────────

/// Reserved token ID representing the end of a sentence / sequence boundary.
/// Always mapped to the string "<END>" and never reassigned.
pub const END_TOKEN_ID: u32 = 0;
