// =============================================================================
// LiquidBrain — Interactive CLI
// =============================================================================
//
// Entry point for the LiquidBrain engine.  Provides a simple REPL (Read–
// Eval–Print Loop) that lets you:
//   • Chat with the model and watch it learn in real time.
//   • Batch-train it on a folder of .txt / .md files.
//   • Save and load the brain to/from disk.
//   • Inspect memory statistics and tune generation length.
//
// All heavy lifting is delegated to the `brain` module; this file is purely
// I/O and dispatch.
// =============================================================================

// Declare modules — Rust will look for src/config.rs, src/tokenizer.rs, etc.
mod config;
mod tokenizer;
mod brain;

use std::io::{self, Write};

use brain::LiquidBrain;
use config::DEFAULT_GENERATION_LENGTH;

// Default filename for the serialised brain.
// Keeping it in the project root makes it easy to inspect with `ls`.
const BRAIN_FILE: &str = "genesis.brain";

// =============================================================================
// Entry point
// =============================================================================

fn main() {
    print_banner();

    // Try to load an existing brain from disk; start fresh if none is found.
    let mut brain = LiquidBrain::load_from_file(BRAIN_FILE)
        .unwrap_or_else(LiquidBrain::new);

    // Generation length is runtime-configurable via `/gen <n>`.
    let mut gen_length = DEFAULT_GENERATION_LENGTH;

    // ── REPL ─────────────────────────────────────────────────────────────────
    loop {
        // Prompt.
        print!("\nVOUS > ");
        io::stdout().flush().unwrap();

        // Read a line of input.
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            // Broken pipe / EOF (e.g., piped input finished).
            break;
        }
        let prompt = input.trim();
        if prompt.is_empty() {
            continue;
        }

        // ── Command dispatch ─────────────────────────────────────────────────
        match prompt {

            // ── /quit ────────────────────────────────────────────────────────
            "/quit" | "/exit" | "/q" => {
                println!("Au revoir !");
                break;
            }

            // ── /save ────────────────────────────────────────────────────────
            "/save" => {
                brain.save_to_file(BRAIN_FILE);
            }

            // ── /load ────────────────────────────────────────────────────────
            // BUG FIX: this command was documented but never implemented in
            // earlier versions.  It now reloads the brain from disk at runtime,
            // useful after external training runs.
            "/load" => {
                match LiquidBrain::load_from_file(BRAIN_FILE) {
                    Some(loaded) => {
                        brain = loaded;
                    }
                    None => {
                        println!(
                            "Fichier '{}' non trouvé.  Lance /train d'abord, \
                             puis /save.",
                            BRAIN_FILE
                        );
                    }
                }
            }

            // ── /stats ────────────────────────────────────────────────────────
            "/stats" => {
                print_stats(&brain, gen_length);
            }

            // ── /prune ───────────────────────────────────────────────────────
            // Manual pruning — useful after heavy live learning sessions.
            "/prune" => {
                println!("Élagage manuel...");
                brain.sleep_and_prune();
            }

            // ── /help ────────────────────────────────────────────────────────
            "/help" | "/h" => {
                print_help();
            }

            // ── /gen <n> ─────────────────────────────────────────────────────
            // Adjust the maximum number of tokens generated per response.
            cmd if cmd.starts_with("/gen ") => {
                let arg = cmd.strip_prefix("/gen ").unwrap_or("").trim();
                match arg.parse::<usize>() {
                    Ok(n) if n > 0 => {
                        gen_length = n;
                        println!("Longueur de génération : {} tokens.", gen_length);
                    }
                    _ => println!("Usage : /gen <nombre>  (ex: /gen 80)"),
                }
            }

            // ── /train <folder> ───────────────────────────────────────────────
            // Recursively ingest all .txt and .md files from a folder.
            cmd if cmd.starts_with("/train ") => {
                let folder = cmd.strip_prefix("/train ").unwrap_or("").trim();
                if folder.is_empty() {
                    println!("Usage : /train <dossier>  (ex: /train ./data)");
                } else {
                    brain.ingest_folder(folder);
                }
            }

            // ── Unknown command ───────────────────────────────────────────────
            cmd if cmd.starts_with('/') => {
                println!(
                    "Commande inconnue : '{}'.  Tape /help pour la liste.",
                    cmd
                );
            }

            // ── Free text → chat ──────────────────────────────────────────────
            // Any non-command input is treated as a prompt: the model learns
            // from it and generates a response.
            _ => {
                brain.chat(prompt, gen_length);
            }
        }
    }
}

// =============================================================================
// UI helpers
// =============================================================================

/// Prints the startup banner with version and quick command reference.
fn print_banner() {
    // ASCII art generated from "LiquidBrain" via figlet (slant font), trimmed.
    println!(
        r#"
  _      _             _     _ ____             _
 | |    (_)           (_)   | |  _ \           (_)
 | |     _  __ _ _   _ _  __| | |_) |_ __ __ _ _ _ __
 | |    | |/ _` | | | | |/ _` |  _ <| '__/ _` | | '_ \
 | |____| | (_| | |_| | | (_| | |_) | | | (_| | | | | |
 |______|_|\__, |\__,_|_|\__,_|____/|_|  \__,_|_|_| |_|
              | |
              |_|   Genesis V115 — Moteur Sémantique Bio-Inspiré
"#
    );
    println!("  Tape /help pour la liste des commandes.\n");
}

/// Prints the formatted command reference.
fn print_help() {
    println!(
        r#"
╔══════════════════════════════════════════════════════════════╗
║                   LIQUIDBRAIN — COMMANDES                    ║
╠══════════════════════════════════════════════════════════════╣
║  /train <dossier>   Ingérer tous les .txt/.md récursivement  ║
║  /save              Sauvegarder le cerveau (genesis.brain)   ║
║  /load              Recharger le cerveau depuis le disque    ║
║  /stats             Afficher les statistiques mémoire        ║
║  /gen <n>           Régler la longueur de génération         ║
║  /prune             Élaguer manuellement les synapses faibles ║
║  /help              Afficher cette aide                      ║
║  /quit  /q          Quitter                                  ║
╠══════════════════════════════════════════════════════════════╣
║  <texte>            Chatter : apprendre + générer            ║
╚══════════════════════════════════════════════════════════════╝
Astuce : les phrases déclaratives renforcent la mémoire.
         Les questions (?) n'apprennent pas (rate = 0).
"#
    );
}

/// Prints a formatted statistics panel for the current brain state.
fn print_stats(brain: &LiquidBrain, gen_length: usize) {
    let s = brain.stats();
    println!(
        r#"
╔══════════════════════════════════════╗
║        LIQUIDBRAIN — STATS           ║
╠══════════════════════════════════════╣
║  Vocabulaire    : {:>8} tokens    ║
║  Neurones       : {:>8}           ║
║  Synapses       : {:>8}           ║
║  Connexions/N   : {:>8.1}           ║
║  Longueur /gen  : {:>8} tokens    ║
╚══════════════════════════════════════╝"#,
        s.vocab_size,
        s.neuron_count,
        s.synapse_count,
        s.avg_connections,
        gen_length
    );
}
