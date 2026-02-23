# iProteinHunter

iProteinHunter is a local iterative binder-design pipeline for Apple Silicon that combines:

- structure prediction (`boltz`, `intellifold`, or `openfold-3-mlx`)
- sequence redesign (`LigandMPNN`)
- optional post-prediction scoring with one or more secondary predictors

The main runner is:

- `iproteinhunter_run.sh`

---

## What is committed vs local-only

This repo is set up so GitHub can stay clean:

- committed: core runner, installer, source integration, example YAMLs
- local-only (gitignored): `output/`, `venvs/`, RAM test assets, benchmarking/campaign helper scripts, and other local experiment artifacts

If you run `git add -A`, ignored folders/files are not staged.

---

## Requirements

- macOS (Apple Silicon recommended)
- Python `3.11`
- `git`
- internet access (MSA server + model/package downloads)

---

## Install

```bash
git clone https://github.com/t-j-fryer/iProteinHunter.git
cd iProteinHunter
bash install_iProteinHunter.sh
```

Installer actions:

- creates venvs for Boltz, LigandMPNN, IntelliFold, and OpenFold3-MLX
- clones required repos into `src/`
- installs dependencies for each tool
- makes `iproteinhunter_run.sh` executable

---

## Quick run example (requested settings)

Example using `aCbx_bind.yaml`, auto parallelization, 10 designs, binder length 65-150:

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

---

## CLI arguments (`iproteinhunter_run.sh`)

### Core

- `--predictor TOOL` (`boltz | intellifold | openfold-3-mlx`)
- `--run-name NAME`
- `--num-runs N`
- `--num-opt-cycles N` (cycles after `cycle_00`)
- `--template-yaml PATH`
- `--out-root PATH`

### Binder seed (`cycle_00`)

- `--binder-min-len N`
- `--binder-max-len N`
- `--binder-percent-x P`

### Anti-helix seed controls

- `--helix-kill`
- `--negative-helix-constant X`
- `--unk-patch-mode ala|ala_gly`

### LigandMPNN temperature

- `--ligand-temp-cycle1 T`
- `--ligand-temp-other T`

### Filtering / export

- `--iptm-threshold T`

### Post-prediction

- `--post-predictor TOOL[,TOOL]` (`none | boltz | intellifold | openfold-3-mlx`)
- `--post-mode MODE` (`all | iptm`)
- `--post-iptm-threshold T`
- `--post-include-cycle00`

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

### Extra backend flags passthrough

- `--boltz-extra "ARGS"`
- `--intellifold-extra "ARGS"`
- `--openfold-extra "ARGS"`
- `--ligand-extra "ARGS"`

### Help

- `-h`, `--help`

---

## Example YAMLs

Example inputs live in:

- `examples/aCbx_bind.yaml`
- `examples/Boltz_Cbx.yaml`
- `examples/Benchmark.yaml`
- `examples/iFluor800.yaml`

---

## Notes

- By default, post-prediction starts after all design runs finish.
- `--post-mode all` evaluates every designed cycle (except `cycle_00` unless `--post-include-cycle00` is set).
- MSA caching is handled automatically in the runner for supported predictors.
