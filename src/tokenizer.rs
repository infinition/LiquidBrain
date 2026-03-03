// =============================================================================
// LiquidBrain — Tokenizer
// =============================================================================
//
// Responsibilities:
//   • Bidirectional word ↔ integer ID mapping (incremental vocabulary)
//   • Global frequency tracking (used for TF-IDF-like rarity scoring)
//   • Case-insensitive lookup fallback (handles sentence-initial capitalisation)
//   • Semantic Markdown parser: translates formatting into logical punctuation
//
// Design choice — why integer IDs?
//   HashMap keys and Vec indices based on u32 are compared in O(1) with minimal
//   cache pressure.  Storing the full word strings as keys in the neuron graph
//   would make context lookups ~10× slower on large vocabularies.
// =============================================================================

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::config::END_TOKEN_ID;

// =============================================================================
// Tokenizer struct
// =============================================================================

/// Bidirectional word ↔ ID mapping with global frequency tracking.
///
/// # Vocabulary layout
/// - ID 0 is always `<END>` (end-of-sentence sentinel, pre-registered on `new()`).
/// - IDs 1..n are assigned incrementally as new words are encountered.
///
/// # Thread safety
/// Not thread-safe by design — LiquidBrain runs single-threaded.
#[derive(Serialize, Deserialize, Clone)]
pub struct Tokenizer {
    /// Word → integer ID lookup.  Primary direction used during learning.
    pub word_to_id: HashMap<String, u32>,

    /// Integer ID → word string.  Used when decoding generated token IDs
    /// back into human-readable text.
    pub id_to_word: HashMap<u32, String>,

    /// Global occurrence count per token ID.
    ///
    /// Incremented every time `get_or_create_id` is called for a token,
    /// whether during corpus ingestion or live interaction.  Used by
    /// `find_focus_point` to compute inverse-frequency (rarity) scores:
    /// rarer words carry more specific semantic weight.
    pub global_counts: HashMap<u32, u32>,

    /// Monotonic counter for assigning new IDs.  Starts at 1; 0 is reserved.
    next_id: u32,
}

impl Tokenizer {
    // ─────────────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────────────

    /// Creates a fresh tokenizer with only the END sentinel pre-registered.
    pub fn new() -> Self {
        let mut t = Tokenizer {
            word_to_id: HashMap::new(),
            id_to_word: HashMap::new(),
            global_counts: HashMap::new(),
            next_id: 1, // 0 is reserved for END_TOKEN_ID
        };
        // Pre-register the end-of-sequence sentinel.
        t.id_to_word.insert(END_TOKEN_ID, "<END>".to_string());
        t.word_to_id.insert("<END>".to_string(), END_TOKEN_ID);
        t
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Vocabulary management
    // ─────────────────────────────────────────────────────────────────────────

    /// Returns the ID for `word`, creating a new entry if it doesn't exist yet.
    ///
    /// **Side-effect:** always increments the global frequency count for the
    /// returned ID, so this should only be called during learning/ingestion —
    /// never during generation (use `find_id` there instead).
    pub fn get_or_create_id(&mut self, word: &str) -> u32 {
        let id = if let Some(&id) = self.word_to_id.get(word) {
            id
        } else {
            // New word: assign the next available ID.
            let id = self.next_id;
            self.word_to_id.insert(word.to_string(), id);
            self.id_to_word.insert(id, word.to_string());
            self.next_id += 1;
            id
        };
        // Track frequency regardless of whether this was a new word.
        *self.global_counts.entry(id).or_insert(0) += 1;
        id
    }

    /// Looks up the ID for `word` **without** creating a new entry.
    ///
    /// Used during generation to check whether a prompt word exists in the
    /// vocabulary.  Returns `None` for completely unknown words.
    ///
    /// # Case-insensitive fallback chain
    /// 1. Exact match                    → "rust" maps to "rust"
    /// 2. Capitalised form               → "rust" tries "Rust"
    /// 3. All-uppercase                  → "rust" tries "RUST"
    /// 4. All-lowercase                  → "Rust" tries "rust"
    ///
    /// This handles the common case where a word was learned mid-sentence
    /// (lowercase) but appears capitalised at the start of a user prompt.
    pub fn find_id(&self, word: &str) -> Option<u32> {
        // 1. Exact match.
        if let Some(&id) = self.word_to_id.get(word) {
            return Some(id);
        }
        // 2. Capitalised (first letter upper, rest unchanged).
        let mut chars = word.chars();
        if let Some(first) = chars.next() {
            let capitalised = first.to_uppercase().to_string() + chars.as_str();
            if let Some(&id) = self.word_to_id.get(&capitalised) {
                return Some(id);
            }
        }
        // 3. All-uppercase.
        if let Some(&id) = self.word_to_id.get(&word.to_uppercase()) {
            return Some(id);
        }
        // 4. All-lowercase.
        if let Some(&id) = self.word_to_id.get(&word.to_lowercase()) {
            return Some(id);
        }
        None
    }

    /// Decodes a token ID back to its string representation.
    ///
    /// Returns `<?>` for unknown IDs — this should never happen under normal
    /// operation but prevents a panic if the persistence file is corrupted.
    pub fn decode(&self, id: u32) -> String {
        self.id_to_word
            .get(&id)
            .cloned()
            .unwrap_or_else(|| "<?>".to_string())
    }

    /// Returns the total number of unique tokens including the END sentinel.
    pub fn vocab_size(&self) -> usize {
        self.word_to_id.len()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Markdown Semantic Parser
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Philosophy: don't *strip* Markdown — *understand* it.
    //
    // Classic NLP pipelines discard all formatting.  LiquidBrain instead
    // translates structural elements into punctuation tokens that carry
    // semantic rhythm: a list item is a new idea (period), a table column is
    // a related concept (comma), a paragraph break is a sentence boundary.
    //
    // This means feeding the model a Git cheatsheet, a Markdown tutorial, or
    // technical documentation produces a graph that captures the *structure*
    // of the knowledge — not just the raw word co-occurrences.
    //
    // Transformation table (applied in order to avoid double-replacing):
    //
    //  Markdown element        │ Becomes │ Rationale
    //  ────────────────────────┼─────────┼──────────────────────────────────
    //  \r\n (Windows CRLF)     │ \n      │ Normalise line endings first
    //  | (table cell separator)│  ,      │ Columns = related ideas
    //  ----- (table divider)   │ (space) │ Pure horizontal rule, no content
    //  ``` (code fence)        │  .      │ Code block = distinct logical unit
    //  \n- (list item)         │  .      │ Each bullet = new independent idea
    //  \n\n (paragraph break)  │  .      │ New paragraph = sentence boundary
    //  \n (soft line break)    │  ,      │ Line break = light pause
    //  # * ` [ ] ( ) > =      │ (space) │ Decorators stripped
    //  . , ! ? ; : '           │ padded  │ Become distinct tokens (not fused)
    // ─────────────────────────────────────────────────────────────────────────

    /// Cleans and tokenises raw text (plain text or Markdown) into a word list.
    ///
    /// This is a **static** method because it does not need vocabulary state —
    /// it purely transforms a string into a sequence of token strings.
    /// The caller then passes each token through `get_or_create_id` or `find_id`.
    ///
    /// # Returns
    /// A `Vec<String>` of whitespace-separated tokens, ready for the brain.
    pub fn clean_text(text: &str) -> Vec<String> {
        let cleaned = text
            // ── 0. NORMALISE LINE ENDINGS ────────────────────────────────────
            .replace("\r\n", "\n")

            // ── 1. TABLE CELLS ───────────────────────────────────────────────
            // Pipe becomes comma: columns are related but distinct concepts.
            // "git clone | récupère le repo" → "git clone , récupère le repo"
            .replace('|', " , ")
            // Drop horizontal separator lines (---|---|--- style).
            .replace("-----", " ")

            // ── 2. CODE FENCES ───────────────────────────────────────────────
            // Each ``` marks a sentence boundary — code is a logical unit.
            .replace("```", " . ")

            // ── 3. LIST ITEMS ────────────────────────────────────────────────
            // "\n-" = start of a list bullet → new idea = period.
            // Must come before the generic "\n" replacement.
            .replace("\n-", " . ")

            // ── 4. PARAGRAPH BREAKS ──────────────────────────────────────────
            // Double newline = hard paragraph boundary = sentence end.
            .replace("\n\n", " . ")

            // ── 5. SOFT LINE BREAKS ──────────────────────────────────────────
            // Single newline = light pause = comma.
            // This preserves the rhythm of command lists and short-line docs.
            .replace('\n', " , ")

            // ── 6. MARKDOWN DECORATORS ───────────────────────────────────────
            // Strip visual formatting that carries no semantic content.
            .replace('#', " ")   // ATX headers
            .replace('*', " ")   // Bold / italic / horizontal rule
            .replace('`', " ")   // Inline code → plain text
            .replace('[', " ")   // Link text open
            .replace(']', " ")   // Link text close
            .replace('(', " ")   // URL open
            .replace(')', " ")   // URL close
            .replace('>', " ")   // Block quotes
            .replace('=', " ")   // Setext headers / assignment operators

            // ── 7. PUNCTUATION TOKENISATION ──────────────────────────────────
            // Add spaces around every punctuation mark so they become separate
            // tokens rather than being fused with adjacent words.
            // "hello." → "hello . "   (3 tokens, not 1)
            .replace('.', " . ")
            .replace(',', " , ")
            .replace('!', " ! ")
            .replace('?', " ? ")
            .replace(';', " ; ")
            .replace(':', " : ")
            .replace('\'', " ' ")  // ASCII apostrophe
            .replace('\u{2019}', " ' ")  // Unicode right single quotation mark '
            .replace('«', " ")     // French opening guillemet (stripped)
            .replace('»', " ");    // French closing guillemet (stripped)

        // Split on any whitespace and collect — this handles runs of spaces
        // produced by the replacements above.
        cleaned.split_whitespace().map(|s| s.to_string()).collect()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Token classification helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Returns `true` if `word` is a punctuation token (no space before it
    /// when printing generated text).
    pub fn is_punctuation(&self, word: &str) -> bool {
        matches!(word, "." | "," | "!" | "?" | ";" | ":" | "'" | "<END>")
    }

    /// Returns `true` if `word` marks a hard sentence boundary.
    ///
    /// Used during corpus ingestion to split the text into learning segments.
    pub fn is_sentence_ender(&self, word: &str) -> bool {
        matches!(word, "." | "!" | "?")
    }
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tokenizer_has_end_token() {
        let tok = Tokenizer::new();
        // END token must always be ID 0.
        assert_eq!(tok.word_to_id.get("<END>"), Some(&END_TOKEN_ID));
        assert_eq!(tok.vocab_size(), 1); // only <END>
    }

    #[test]
    fn test_get_or_create_id_incremental() {
        let mut tok = Tokenizer::new();
        let id_hello = tok.get_or_create_id("hello");
        let id_world = tok.get_or_create_id("world");
        let id_hello2 = tok.get_or_create_id("hello"); // should reuse

        // Same word → same ID.
        assert_eq!(id_hello, id_hello2);
        // Different words → different IDs.
        assert_ne!(id_hello, id_world);
        // Neither word should collide with END token.
        assert_ne!(id_hello, END_TOKEN_ID);
        assert_ne!(id_world, END_TOKEN_ID);
        // Vocabulary grew by 2.
        assert_eq!(tok.vocab_size(), 3); // <END> + hello + world
    }

    #[test]
    fn test_global_counts_increment() {
        let mut tok = Tokenizer::new();
        let id = tok.get_or_create_id("rust");
        tok.get_or_create_id("rust");
        tok.get_or_create_id("rust");
        // "rust" was seen 3 times.
        assert_eq!(tok.global_counts[&id], 3);
    }

    #[test]
    fn test_find_id_exact_match() {
        let mut tok = Tokenizer::new();
        tok.get_or_create_id("rust");
        assert!(tok.find_id("rust").is_some());
    }

    #[test]
    fn test_find_id_case_fallback() {
        let mut tok = Tokenizer::new();
        // Learn the word in its capitalised form (as it appears mid-sentence).
        tok.get_or_create_id("Rust");

        // Should find via lowercase, uppercase, and exact.
        assert!(tok.find_id("rust").is_some(),  "lowercase fallback failed");
        assert!(tok.find_id("RUST").is_some(),  "uppercase fallback failed");
        assert!(tok.find_id("Rust").is_some(),  "exact match failed");
    }

    #[test]
    fn test_find_id_unknown_returns_none() {
        let tok = Tokenizer::new();
        assert!(tok.find_id("unknownword").is_none());
    }

    #[test]
    fn test_decode_known_id() {
        let mut tok = Tokenizer::new();
        let id = tok.get_or_create_id("bonjour");
        assert_eq!(tok.decode(id), "bonjour");
    }

    #[test]
    fn test_decode_unknown_id_returns_placeholder() {
        let tok = Tokenizer::new();
        assert_eq!(tok.decode(9999), "<?>");
    }

    #[test]
    fn test_clean_text_basic_words() {
        let tokens = Tokenizer::clean_text("hello world");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_clean_text_strips_markdown_headers() {
        // "# Title" should produce "Title", not "# Title"
        let tokens = Tokenizer::clean_text("# Title");
        assert!(!tokens.iter().any(|t| t.contains('#')));
        assert!(tokens.contains(&"Title".to_string()));
    }

    #[test]
    fn test_clean_text_list_becomes_period() {
        // "\n- item" → " . item"
        let tokens = Tokenizer::clean_text("intro\n- item");
        assert!(tokens.contains(&".".to_string()), "list item should produce a period token");
    }

    #[test]
    fn test_clean_text_paragraph_break_becomes_period() {
        let tokens = Tokenizer::clean_text("para1\n\npara2");
        assert!(tokens.contains(&".".to_string()));
    }

    #[test]
    fn test_clean_text_table_pipe_becomes_comma() {
        let tokens = Tokenizer::clean_text("col1 | col2");
        assert!(tokens.contains(&",".to_string()));
    }

    #[test]
    fn test_clean_text_punctuation_tokenised() {
        // "end." should produce separate tokens "end" and "."
        let tokens = Tokenizer::clean_text("end.");
        assert!(tokens.contains(&"end".to_string()));
        assert!(tokens.contains(&".".to_string()));
    }

    #[test]
    fn test_is_punctuation() {
        let tok = Tokenizer::new();
        assert!(tok.is_punctuation("."));
        assert!(tok.is_punctuation("?"));
        assert!(tok.is_punctuation("<END>"));
        assert!(!tok.is_punctuation("hello"));
        assert!(!tok.is_punctuation("a"));
    }

    #[test]
    fn test_is_sentence_ender() {
        let tok = Tokenizer::new();
        assert!(tok.is_sentence_ender("."));
        assert!(tok.is_sentence_ender("!"));
        assert!(tok.is_sentence_ender("?"));
        assert!(!tok.is_sentence_ender(","));
        assert!(!tok.is_sentence_ender("end"));
    }
}
