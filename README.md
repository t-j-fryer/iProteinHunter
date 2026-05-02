# iProteinHunter

iProteinHunter is a local iterative protein binder-design pipeline for Apple Silicon.
It combines structure prediction, sequence redesign, optional post-prediction scoring,
and motif-scaffolding utilities in one reproducible command-line workflow.

The main runner is:

- `iproteinhunter_run.sh`

Core backends:

- design prediction with `boltz`, `intellifold`, or `openfold-3-mlx`
- iterative sequence redesign with `LigandMPNN`
- optional post-prediction with one or more secondary predictors
- optional motif scaffolding and partial redesign modes

---

## Requirements

- macOS, with Apple Silicon recommended
- Python `3.11`
- `git`
- internet access for package installs, model downloads, and MSA server use

The installer creates local virtual environments under `venvs/` and clones external
tool source trees under `src/`. Those folders are intentionally ignored by Git.

---

## Install

```bash
git clone https://github.com/t-j-fryer/iProteinHunter.git
cd iProteinHunter
bash install_iProteinHunter.sh
```

Installer actions:

- creates venvs for Boltz, LigandMPNN, IntelliFold, and OpenFold3-MLX
- creates a notebook venv and registers Jupyter kernel `iProteinHunter Notebook`
- clones required repos into `src/`
- installs dependencies for each tool
- downloads the OpenFold checkpoint to `${OPENFOLD_CACHE:-$HOME/.openfold3}`
- makes `iproteinhunter_run.sh` executable

---

## Notebook Kernel

The installer creates:

- venv: `venvs/iProteinHunter_notebook`
- Jupyter kernel: `iProteinHunter Notebook`

This is intended for VS Code notebooks.

```bash
code .
```

Then open `notebooks/iproteinhunter_pipeline_control.ipynb` and select kernel
`iProteinHunter Notebook`.

Recommended VS Code extensions:

- `ms-python.python`
- `ms-toolsai.jupyter`

---

## Quick Start

Run 10 Boltz-guided designs from the aCbx example:

```bash
caffeinate -dims ./iproteinhunter_run.sh \
  --predictor boltz \
  --template-yaml ./examples/aCbx_bind.yaml \
  --run-name acbx_demo_10 \
  --num-runs 10 \
  --num-opt-cycles 5 \
  --binder-min-len 65 \
  --binder-max-len 150 \
  --max-parallel auto
```

Outputs are written under:

- `output/<run-name>/`

Useful result files include:

- `summary_all_runs.csv`
- `run_*/rows.csv`
- exported structures from cycles that pass the iPTM threshold
- post-prediction summaries when `--post-predictor` is used

---

## Input YAMLs

Example inputs live under `examples/`.

Common starting points:

- `examples/aCbx_bind.yaml`
- `examples/aCbx_bind_contacts.yaml`
- `examples/Benchmark.yaml`
- `examples/Unconditional.yaml`
- `examples/3RGK.yaml`
- `examples/PBP3_contacts.yaml`
- `examples/SUMO.yaml`
- `examples/pTF205.yaml`
- `examples/Fluorescein.yaml`
- `examples/JF549.yaml`
- `examples/iFluor800.yaml`

Contact benchmark variants live under:

- `examples/aCbx_bind_contacts_benchmark_variants/`
- `examples/aCbx_bind_contacts_benchmark_variants_v2/`

---

## CLI Reference

Run:

```bash
./iproteinhunter_run.sh --help
```

### Core

- `--predictor TOOL`: `boltz`, `intellifold`, or `openfold-3-mlx`
- `--run-name NAME`
- `--num-runs N`
- `--num-opt-cycles N`: optimization cycles after `cycle_00`
- `--num-cycles N`: alias for `--num-opt-cycles`
- `--model NAME`: IntelliFold model, default `v2-flash`
- `--template-yaml PATH`
- `--out-root PATH`
- `--cpu-only`

### Binder Seeding

- `--binder-min-len N`
- `--binder-max-len N`
- `--binder-percent-x P`

For Boltz and IntelliFold, seed sequences use `X` at the requested rate. For
OpenFold3-MLX, ambiguous positions are represented with amino-acid spikes because
OpenFold does not use `X` in the same way.

### Motif Scaffolding

Motif scaffolding seeds a binder with fixed motif residues copied from a source
sequence, randomly places those motifs within the binder length range, and asks
LigandMPNN to redesign the non-fixed positions.

This mode currently supports Boltz design runs only.

```bash
caffeinate -dims ./iproteinhunter_run.sh \
  --predictor boltz \
  --template-yaml ./examples/SUMO.yaml \
  --run-name motif_demo \
  --num-runs 10 \
  --num-opt-cycles 5 \
  --binder-min-len 65 \
  --binder-max-len 120 \
  --motif-scaffolding \
  --motif-source-seq "PASTE_SOURCE_SEQUENCE_HERE" \
  --motif-positions "31-45,63-106" \
  --gap-between-motifs 8
```

Motif options:

- `--motif-scaffolding`
- `--motif-positions STR`: JSON or 1-based ranges, for example `31-45,63-106`
- `--motif-source-seq STR`
- `--motif-fixed-positions STR`: optional 1-based subset to fix
- `--gap-between-motifs N`: default `8`

Motif runs write:

- `motif_positions_by_cycle.csv`

### Partial Redesign

Partial redesign keeps an existing binder sequence fixed except for selected
1-based ranges. The template YAML must contain a concrete chain `A` sequence.

```bash
caffeinate -dims ./iproteinhunter_run.sh \
  --predictor boltz \
  --template-yaml ./examples/SUMO.yaml \
  --run-name partial_redesign_demo \
  --num-runs 10 \
  --num-opt-cycles 5 \
  --partial-redesign \
  --partial-redesign-ranges "25-50,70-75"
```

Partial redesign and motif scaffolding are mutually exclusive.

### Design Controls

- `--helix-kill`
- `--negative-helix-constant X`: `0..1`, default `0.5`
- `--loopkill X`: `0..1`, default `0`
- `--unk-patch-mode MODE`: `auto`, `ala`, `ala_gly`, or `ala_gly_ser`
- `--ligand-temp-cycle1 T`
- `--ligand-temp-cycle01 T`: alias for `--ligand-temp-cycle1`
- `--ligand-temp-other T`
- `--ligand-temp T`: alias for `--ligand-temp-other`
- `--mpnn-bias-aa-cycle1 STR`
- `--mpnn-bias-aa-other STR`
- `--boltz-use-potentials`
- `--boltz-no-potentials`

### Filtering

- `--iptm-threshold T`: controls which design structures are exported

### Post-Prediction

Post-prediction evaluates designed sequences with one or more secondary predictors
after design runs finish.

- `--post-predictor TOOL[,TOOL]`: `none`, `boltz`, `intellifold`, `openfold-3-mlx`
- `--post-mode MODE`: `none`, `all`, or `iptm`
- `--post-iptm-threshold T`
- `--post-include-cycle00`

Examples:

```bash
./iproteinhunter_run.sh \
  --predictor boltz \
  --post-predictor intellifold,openfold-3-mlx \
  --post-mode iptm \
  --post-iptm-threshold 0.80 \
  --template-yaml ./examples/aCbx_bind.yaml \
  --run-name acbx_crosscheck
```

### Parallelism

- `--no-parallel`
- `--max-parallel N|auto`
- `--mem-budget-gb X|auto`
- `--mem-safety S`
- `--mps-aware`
- `--no-mps-aware`
- `--mps-max-parallel N|auto`
- `--mps-mem-fraction F`
- `--mps-cpu-cap N`

With `--max-parallel auto`, the runner performs a calibration prediction and uses
observed memory use plus CPU availability to choose a conservative parallelism.

### Extra Backend Flags

- `--boltz-extra "ARGS"`
- `--intellifold-extra "ARGS"`
- `--openfold-extra "ARGS"`
- `--ligand-extra "ARGS"`

---

## Motif Utilities

`motif_scaffolding_helper.py` is used internally by the runner, but it can also be
called directly:

```bash
python3 motif_scaffolding_helper.py validate \
  --motif-positions "31-45,63-106" \
  --source-sequence "PASTE_SOURCE_SEQUENCE_HERE" \
  --min-length 65 \
  --max-length 120
```

`score_motif_scaffolding.py` scores motif preservation after a motif-scaffolding
run by aligning motif residues against a reference structure.

```bash
python3 score_motif_scaffolding.py \
  --run-root output/motif_demo \
  --reference-structure path/to/reference.cif \
  --reference-chain A \
  --design-chain A \
  --save-aligned-structures
```

The scorer writes `motif_scaffold_quality.csv` by default.

---

## Repository Layout

Shared source files are kept in Git. Local generated data and personal campaign
scripts are ignored.

Tracked:

- `iproteinhunter_run.sh`: main pipeline runner
- `install_iProteinHunter.sh`: installer/bootstrap script
- `motif_scaffolding_helper.py`: motif placement helper
- `score_motif_scaffolding.py`: motif structural scoring helper
- `backfill_openfold_metrics.py`: utility for filling OpenFold metrics in outputs
- `examples/`: reusable input templates
- `notebooks/`: shared notebook workflows

Ignored local-only paths include:

- `output/`
- `protein_hunter_runs/`
- `venvs/`
- `src/`
- `bin/`
- one-off queue, comparison, benchmark, plotting, and recovery scripts

---

## Notes

- By default, post-prediction starts after all design runs finish.
- `--post-mode all` evaluates every designed cycle except `cycle_00` unless
  `--post-include-cycle00` is set.
- MSA caching is handled automatically for supported predictors.
- OpenFold checkpoint setup is non-interactive; the runner ensures the checkpoint
  exists before OpenFold prediction.
- Keep personal campaign scripts local-only unless they are reusable enough to
  document and support for other users.
