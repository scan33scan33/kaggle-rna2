# kaggle-rna2 Planning

## Objective
Build a strong, reproducible solution for the Kaggle RNA 2 competition with a disciplined experimentation loop, leakage-safe validation, and a final submission pipeline that is robust rather than leaderboard-lucky.

## Principles
- Start with the simplest end-to-end baseline that produces a valid submission
- Treat validation design as a first-class problem
- Prefer controlled ablations over bundled changes
- Optimize for reproducibility, not just a single lucky score
- Keep inference and submission generation clean from the start

## What success looks like
- A fully reproducible training and inference pipeline
- A local validation setup that is directionally predictive of leaderboard movement
- At least one strong single model and one ensemble candidate
- Clear documentation of what helped, what did not, and where uncertainty remains

---

## Phase 0: Understand the competition exactly

### Tasks
- Read the competition overview, metric, and evaluation details carefully
- Document:
  - input schema
  - output schema
  - sequence length / structure assumptions
  - any public baseline or organizer hints
  - inference-time constraints
- Identify what the model is actually predicting and at what granularity
- List all obvious leakage vectors

### Deliverable
`docs/problem.md`
- problem statement
- metric definition
- submission format
- data fields
- leakage notes
- initial hypotheses

---

## Phase 1: Build a minimal end-to-end pipeline

### Goal
Get to the first valid submission as fast as possible.

### Tasks
- Add deterministic data loading
- Add preprocessing pipeline
- Create train / validation split
- Train a minimal baseline
- Run inference on validation and test
- Generate a valid submission file
- Verify exact submission schema

### Baseline preference
Start with the lowest-complexity model that still respects the data structure:
- simple sequence model
- small transformer / encoder
- shallow GNN or structure-aware baseline only if structure is central to the task
- avoid heavy architecture work before the pipeline is stable

### Deliverable
- first valid submission
- first local CV score
- one-page note on baseline weaknesses

---

## Phase 2: Validation and scoring strategy

This is likely one of the highest leverage parts of the project.

### Questions to settle
- What split best matches test conditions?
- Are there homologous / near-duplicate / correlated examples that can leak across folds?
- Does public leaderboard correlate with local CV?
- Is seed variance large enough to distort conclusions?

### Tasks
- Implement a few candidate validation schemes
- Compare score stability across:
  - random split
  - grouped split
  - time-aware split if relevant
  - structure-family split if relevant
- Estimate variance from multiple seeds
- Create a “trust ranking” for local evaluation schemes

### Deliverable
`docs/validation.md`
- recommended CV design
- known limitations
- confidence level

---

## Phase 3: Feature and representation strategy

### Representation buckets to evaluate
- raw sequence tokens
- secondary / structural signals if available
- positional encodings
- engineered numeric features
- pretrained biological embeddings
- pairwise or relational features if the target depends on interactions

### Plan
1. establish sequence-only baseline
2. add structure-aware information
3. test pretrained embeddings
4. test hybrid models
5. compare complexity versus gain

### Key question
Does the competition reward:
- local token-level pattern recognition,
- global structural reasoning,
- or prior biological pretraining?

Do not assume. Measure.

---

## Phase 4: Model roadmap

### Track A: Baseline models
Purpose: stable references.

Candidate models:
- simple MLP or shallow model on engineered features
- small sequence encoder
- compact transformer

### Track B: Structure-aware models
Purpose: exploit RNA-specific inductive bias.

Candidate models:
- graph neural network
- transformer with structural edges or biases
- sequence + structure dual-tower encoder

### Track C: Pretrained adaptation
Purpose: import biological prior knowledge.

Candidate models:
- frozen pretrained encoder + small head
- finetuned pretrained encoder
- embedding extraction + lightweight downstream model

### Track D: Ensemble track
Purpose: improve robustness and capture complementary signal.

Candidate ensembles:
- seed ensemble
- architecture ensemble
- fold ensemble
- weighted blend based on CV

---

## Phase 5: Experiment discipline

### Rules
Every experiment should answer one question.

Good:
- “Does structure feature X help over sequence-only baseline?”
- “Does grouped CV change ranking of models?”
- “Does finetuning beat frozen embeddings?”

Bad:
- “Changed architecture, features, loss, augmentation, and optimizer together.”

### Experiment log template
For each run, store:
- run id
- commit hash
- data split version
- model config
- seed
- training duration
- local metric
- notes
- conclusion

### Minimum folders
- `configs/`
- `experiments/`
- `logs/`
- `submissions/`

---

## Phase 6: Error analysis

### Goal
Understand where the model fails before adding complexity.

### Tasks
- Slice validation by:
  - sequence length
  - structural complexity
  - target distribution
  - rare vs common patterns
- Compare top good cases vs top bad cases
- Inspect whether failures come from:
  - preprocessing bugs
  - split mismatch
  - underfitting
  - overfitting
  - missing structural bias
  - noisy labels

### Deliverable
`docs/error_analysis.md`

---

## Phase 7: Robustness and generalization

### Tasks
- run multiple seeds on promising configs
- check fold consistency
- compare local CV against public LB
- test whether gains survive small data perturbations
- estimate whether leaderboard movement is likely real or noise

### Rule
Do not trust a model that wins only once.

---

## Phase 8: Inference and submission pipeline

### Tasks
- create clean inference script
- support fold-wise prediction
- support ensembling
- verify deterministic output format
- add safety checks:
  - row count
  - column names
  - NaN checks
  - value range checks
  - ordering checks

### Deliverable
One command to generate a final submission from saved checkpoints.

---

## Prioritized roadmap

### Milestone 1: foundation
- [ ] write `docs/problem.md`
- [ ] implement data loading
- [ ] implement baseline split
- [ ] train minimal baseline
- [ ] submit first valid file

### Milestone 2: trustworthy evaluation
- [ ] compare 2–4 split strategies
- [ ] measure seed variance
- [ ] settle on primary CV
- [ ] document leakage risks

### Milestone 3: model expansion
- [ ] sequence-only stronger model
- [ ] structure-aware model
- [ ] pretrained embedding model
- [ ] compare all under same CV

### Milestone 4: refinement
- [ ] focused ablations
- [ ] regularization and augmentation tuning
- [ ] checkpoint averaging or fold ensembling
- [ ] error analysis pass

### Milestone 5: final push
- [ ] choose final model family
- [ ] run multi-seed / multi-fold training
- [ ] build final ensemble
- [ ] verify reproducibility
- [ ] submit final candidates

---

## Repo structure

```text
kaggle-rna2/
├── planning.md
├── README.md
├── docs/
│   ├── problem.md
│   ├── validation.md
│   └── error_analysis.md
├── configs/
│   ├── baseline/
│   ├── structure/
│   ├── pretrained/
│   └── ensemble/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   └── utils/
├── notebooks/
├── experiments/
├── logs/
├── checkpoints/
└── submissions/