# LiquidBrain — Roadmap

Plan de développement, backlog de features, et directions de recherche.

---

## Légende

| Symbole | Signification |
|---|---|
| ✅ | Terminé |
| 🔄 | En cours |
| 📋 | Planifié |
| 💡 | Idée / à évaluer |
| 🔬 | Recherche / expérimental |

---

## Phase 0 — Stabilisation (V115) ✅

*Objectif : nettoyer la base de code et la rendre publiable.*

- ✅ Refactoring modulaire (`config.rs`, `tokenizer.rs`, `brain.rs`, `main.rs`)
- ✅ Correction du bug `/load` (commande documentée mais non implémentée)
- ✅ Ajout de la commande `/stats` (vocabulaire, neurones, synapses)
- ✅ Ajout de la commande `/prune` (élagage manuel)
- ✅ Ajout de la commande `/gen <n>` (longueur de génération configurable)
- ✅ Ajout de la commande `/help`
- ✅ Commentaires complets sur tout le code (doc-comments Rust)
- ✅ Constante nommée `STRONG_FACT_THRESHOLD` (remplace le magic number `10.0`)
- ✅ Constante nommée `CORPUS_LEARNING_RATE` (remplace la valeur inline `3.0`)
- ✅ `Cargo.toml` corrigé : nom `liquidbrain`, bins orphelins supprimés, `serde_json` retiré
- ✅ Suppression de `Cargo copy.toml`
- ✅ `.gitignore` complet (target/, *.brain, /data/, artefacts éditeur)
- ✅ Tests unitaires : `Tokenizer` (12 tests), `LiquidBrain` (9 tests), `sample_token` (3 tests)
- ✅ README professionnel (architecture, algorithme, comparaison Transformer, références)
- ✅ Dépôt GitHub initialisé : `infinition/LiquidBrain`

---

## Phase 1 — Tests & Benchmarks 📋

*Objectif : valider le comportement du moteur sur des corpus réels et mesurer les performances.*

### 1.1 Tests d'intégration

- 📋 Test end-to-end : ingestion d'un fichier → génération → résultat non vide
- 📋 Test de persistance : `save` → `load` → vérifier l'identité du graphe
- 📋 Test de déterminisme à température 0 : même prompt → même output
- 📋 Test de non-régression : golden outputs sur un corpus fixe

### 1.2 Tests de robustesse

- 📋 Entrée vide / whitespace only → pas de crash
- 📋 Prompt avec 100% de tokens inconnus → message clair
- 📋 Fichier UTF-8 invalide dans `/train` → skip propre sans crash
- 📋 Fichier `genesis.brain` corrompu → message d'erreur clair + fallback

### 1.3 Benchmarks

- 📋 Benchmark d'ingestion : tokens/seconde selon la taille du corpus
- 📋 Benchmark de génération : tokens/seconde selon la taille du graphe
- 📋 Profil mémoire : taille du graphe (neurones, synapses) vs. corpus ingéré
- 📋 Évolution du temps de pruning selon le nombre de synapses

**Outil recommandé :** [`criterion`](https://docs.rs/criterion) pour les benchmarks Rust.

---

## Phase 2 — Améliorations du Cœur 📋

*Objectif : améliorer la qualité de génération sans sacrifier la lisibilité du code.*

### 2.1 Génération

- 📋 **Beam search** — explorer k chemins en parallèle et retenir le meilleur
  *Complexité estimée : ~150 lignes*
- 📋 **Nucleus sampling (top-p)** — filtrer les candidats dont la probabilité cumulée dépasse p
  *Alternative plus simple à beam search pour la diversité*
- 📋 **Repetition ngram penalty** — pénaliser les tokens déjà générés dans la fenêtre courante
  *Complémentaire à la fatigue synaptique pour les séquences longues*
- 📋 **Génération multi-thread** — paralléliser les k meilleures continuations
  *Nécessite de rendre LiquidBrain `Send` ou d'utiliser des clones*

### 2.2 Focus Point

- 📋 Pondérer le score de focus par la **position dans le prompt** (les mots récents comptent plus)
- 📋 Supporter **plusieurs points de focus** simultanés pour les prompts longs
- 💡 Utiliser un **vrai TF-IDF** avec corpus de référence (nécessite un second passage sur le corpus)

### 2.3 Apprentissage

- 📋 **Decay temporel** : réduire légèrement le poids de toutes les synapses à chaque session
  *Simule l'oubli naturel — empêche les anciennes associations de dominer indéfiniment*
- 📋 **Learning rate scheduling** : réduire `FACT_LEARNING_RATE` après N répétitions du même fait
  *Évite la sur-fixation sur les faits répétés*
- 💡 **Negative learning** : commande `/forget <fait>` pour affaiblir explicitement une association
- 📋 **Ingestion incrémentale** : mémoriser quels fichiers ont déjà été ingérés (`/train` idempotent)

### 2.4 Tokenisation

- 📋 **Stemming basique** : `apprendre` = `apprend` = `apprentissage` → même racine
  *Réduirait le vocabulaire de 20–40% sur les corpus français*
- 📋 **Stop-words configurables** : exclure les mots trop fréquents du focus point
  *Actuellement géré partiellement par le filtre `len < 2`*
- 💡 **BPE (Byte-Pair Encoding)** : tokenisation sous-mot pour une meilleure couverture
  *Grosse refonte — Phase 4 territory*

---

## Phase 3 — Nouvelles Fonctionnalités 📋

*Objectif : rendre LiquidBrain utilisable comme outil autonome.*

### 3.1 Interface

- 📋 **TUI interactive** (terminal UI) avec [ratatui](https://ratatui.rs/) :
  - Affichage temps réel du graphe (neurones actifs pendant génération)
  - Barre de progression pendant `/train`
  - Historique de conversation scrollable
- 📋 **Coloration syntaxique** de la génération : mots factuels (haute confiance) vs. probabilistes
- 💡 **Web UI minimal** via une page HTML servie localement (sans JS framework)

### 3.2 API

- 📋 **Mode pipe** : `echo "prompt" | liquidbrain` → sortie sur stdout (scripts / CI)
- 📋 **REST API** avec [axum](https://github.com/tokio-rs/axum) :
  ```
  POST /chat   { "prompt": "...", "length": 80 }
  GET  /stats
  POST /train  { "folder": "..." }
  ```
- 📋 **Arguments CLI** avec [clap](https://docs.rs/clap) :
  ```
  liquidbrain --brain custom.brain --gen 100
  liquidbrain train ./docs
  ```

### 3.3 Multi-cerveau

- 💡 **Cerveau spécialisé** : charger plusieurs `.brain` et les fusionner à la demande
- 💡 **Profils** : `liquidbrain --profile rust` → charge `rust.brain` automatiquement

### 3.4 Export / Visualisation

- 📋 **Export GraphViz** : `liquidbrain export-dot > graph.dot` → visualise le graphe avec Graphviz
- 📋 **Export JSON** : format lisible pour inspection / débogage
- 💡 **Heatmap** des neurones les plus actifs lors d'une session de génération

---

## Phase 4 — Recherche 🔬

*Objectif : publier les résultats et explorer les directions théoriques.*

### 4.1 Évaluation formelle

- 🔬 **Perplexité** : mesurer la perplexité sur un corpus de test réservé
- 🔬 **BLEU / ROUGE** : comparer les réponses générées à des références
- 🔬 **Cohérence thématique** : mesurer la dérive du topic sur de longues générations

### 4.2 Comparaisons

- 🔬 Comparer LiquidBrain à un n-gram classique (NLTK) à taille de corpus égale
- 🔬 Comparer la diversité des outputs avec/sans fatigue synaptique
- 🔬 Mesurer l'impact de chaque hyperparamètre sur la perplexité (`SYNAPSE_COST`, `INHIBITION_FACTOR`, etc.)

### 4.3 Extensions théoriques

- 🔬 **Inhibition récurrente** : les neurones fortement activés inhibent leurs voisins (pas seulement leurs synapses concurrentes)
- 🔬 **Mémoire de travail** : fenêtre d'activation courte (~7 tokens) influençant le focus point
- 🔬 **Consolidation nocturne améliorée** : algorithme de pruning plus sophistiqué (LRU, frequence × ancienneté)
- 🔬 **Graphe hiérarchique** : niveau sémantique (phrases) + niveau syntaxique (mots)
- 🔬 **Multi-modalité** : intégrer des embeddings d'images dans le graphe (vision-langage)

### 4.4 Publication

- 🔬 Article technique : *"LiquidBrain: Synaptic Fatigue as an Emergent Anti-Repetition Mechanism in Graph-Based Language Models"*
- 🔬 Notebook Jupyter démontrant les propriétés du modèle sur un corpus benchmark (WikiText-2 FR)

---

## Backlog (sans priorité fixée)

- 💡 Plugin Obsidian : utiliser LiquidBrain comme moteur de suggestion dans les notes
- 💡 Mode "dictée" : transcription vocale → apprentissage en continu
- 💡 Intégration avec un LLM externe (Claude / GPT) comme "oracle" pour corriger les sorties de LiquidBrain
- 💡 Versionning des cerveaux : `genesis.brain.v1`, `.v2`, diff entre versions
- 💡 `--dry-run` pour `/train` : montrer combien de tokens/séquences seraient appris sans les apprendre

---

## Historique des versions

| Version | Date | Changements majeurs |
|---|---|---|
| V43 | Dec 2024 | Prototype initial, n-grammes simples |
| V61 | Dec 2024 | Ajout de la santé synaptique |
| V90 | Jan 2025 | Focus point, inhibition latérale |
| V115 | Jan 2025 | Parser Markdown sémantique, persistance binaire |
| V115.1 | Mar 2026 | Refactoring modulaire, tests, README, bugfixes CLI |

---

*Ce roadmap est un document vivant. Les priorités évoluent avec les expériences.*
*Ouvre une issue sur GitHub pour proposer une feature ou discuter d'une direction.*
