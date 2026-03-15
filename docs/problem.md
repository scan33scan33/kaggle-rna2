# Stanford RNA 3D Folding Part 2 — Problem Statement

## Competition
- **URL**: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2
- **Prize**: $75,000 (top 3 teams)
- **Entry deadline**: March 18, 2026
- **Submission deadline**: March 25, 2026
- **Type**: Code competition (notebooks run on Kaggle servers, no internet access)

## Task
Predict the **3D structure** of RNA molecules from sequence alone.
For each RNA sequence in the test set, submit **5 candidate structures**.

## Input
- RNA sequence (nucleotides: A, U, G, C + modified residues)
- Optional: Multiple Sequence Alignments (MSA) from RNAcentral, Rfam, NCBI Nucleotide

## Output
- **C1' atom (x, y, z) coordinates** for every residue, for each of 5 predicted structures
- Submission format: one row per (target, structure_index, residue_index) with columns x, y, z

## Metric
**TM-score** (Template Modelling score), range [0, 1], higher is better.
- Scores > 0.45 indicate a correct global fold
- Final score = mean of best-of-5 TM-scores across all targets
- Only residues aligned by numbering are rewarded (no partial credit for misaligned fragments)

## Data Fields
| Field | Description |
|-------|-------------|
| `target_id` | Unique identifier for each RNA |
| `sequence` | RNA nucleotide sequence |
| `temporal_cutoff` | Date after which structures are hidden |
| `description` | Free-text description |

## Baseline Models (provided by organizers)
| Model | Score | Notes |
|-------|-------|-------|
| AlphaFold 3 | ~0.45 | Run with rMSA output |
| Vfold (human expert) | Higher than AF3 | Led CASP16 |

## Key Differences from Part 1
- Harder targets: novel RNA folds with **no available PDB templates**
- RNA–protein complexes and RNA–small molecule complexes
- Assemblies up to **6,000 nucleotides**
- Stricter scoring: no partial credit for template-reuse shortcuts

## Leakage Vectors
1. **Template leakage**: PDB structures deposited after training cutoff may match test targets
   - Mitigation: strict temporal cutoff on training data
2. **MSA leakage**: homologous sequences in public databases
   - Mitigation: use only databases current at training cutoff
3. **Sequence similarity**: near-duplicate RNAs across train/test
   - Mitigation: cluster-based CV split

## Lessons from Part 1
- Top strategy: **template-based modelling (TBM)** — no deep learning needed for 95% of targets
- 19 of 20 Part 1 targets had usable PDB templates (TM-align > 0.45)
- Part 2 deliberately introduces template-free cases to force learned modelling

## Recommended Strategy
1. **Template search first** (BLAST/Infernal against PDB)
2. **If good template found** (TM-align > 0.45): use template modelling
3. **If no template**: use fine-tuned deep learning model
   - Primary: fine-tuned RhoFold+ or RibonanzaNet2
   - Fallback: Boltz-1 / NVIDIA RNAPro
4. **Ensemble** top-k predictions, pick best 5

## Initial Hypotheses
- Template search will solve the easy subset; the competition is decided on template-free targets
- Fine-tuning RhoFold+ on competition training data should outperform zero-shot inference
- RibonanzaNet2 embeddings may improve structure prediction when used as encoder features
- Ensembling multiple models on difficult targets will reduce variance
