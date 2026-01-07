iProteinHunter

iProteinHunter is a local, reproducible pipeline for iterative protein binder design using
Boltz (structure prediction & confidence scoring) and LigandMPNN (sequence redesign).

It is designed to:

run locally on Apple Silicon (macOS)

avoid CUDA and PATH pollution

use explicit virtual environments

be driven by a single orchestration script

This makes it suitable for long-running design sweeps, unattended runs, and reproducible methods.

Features

Iterative binder design loop (Boltz → LigandMPNN → Boltz)

Per-run and per-cycle tracking of iPTM scores

Automatic reuse of target MSAs across cycles

Fully local installation (no Docker, no Conda required)

Deterministic environment setup via a single installer

One command to run full experiments

Repository structure
iProteinHunter/
├── install_iProteinHunter.sh     # One-time installer
├── iProteinHunter_run.sh         # Main pipeline runner
├── examples/                     # Example Boltz YAML templates
├── src/
│   └── LigandMPNN/               # Cloned by installer
├── venvs/                        # Python virtual environments (gitignored)
├── output/                       # Run outputs (gitignored)
└── README.md

Installation
Requirements

macOS (Apple Silicon)

Python 3.11

git

Internet access (for model weights + MSAs)

Install
git clone https://github.com/t-j-fryer/iProteinHunter.git
cd iProteinHunter
bash install_iProteinHunter.sh


This will:

create two Python virtual environments

install Boltz and LigandMPNN with compatible dependencies

download LigandMPNN model weights

make iProteinHunter_run.sh executable

No PATH changes are made.

Quick start

Run the pipeline using the default example YAML:

./iProteinHunter_run.sh \
  --run-name test_run \
  --num-runs 1 \
  --num-cycles 3


Outputs will be written to:

output/test_run/

Example: unattended run
caffeinate -dims ./iProteinHunter_run.sh \
  --template-yaml examples/TEM1.yaml \
  --run-name binder_sweep \
  --num-runs 3 \
  --num-cycles 5

Command-line options
--run-name                  Name of the experiment
--num-runs                  Number of independent runs
--num-cycles                Cycles per run
--template-yaml              Boltz YAML template
--out-root                   Output directory (default: ./output)
--binder-min-len              Minimum binder length
--binder-max-len              Maximum binder length
--binder-percent-x            Percent of X positions in binder
--boltz-extra                 Extra flags passed to Boltz
--ligand-extra                Extra flags passed to LigandMPNN


For the full list:

./iProteinHunter_run.sh --help

Design philosophy

iProteinHunter intentionally avoids:

global PATH modification

implicit environment activation

hidden background processes

Instead:

all environments are activated explicitly inside the pipeline

all paths are deterministic

everything needed for a run is visible in one script

This makes runs easy to debug, reproduce, and document.

Outputs

For each run:

per-cycle Boltz structures and confidence JSONs

redesigned sequences from LigandMPNN

CSVs tracking iPTM over cycles

summary plots:

iPTM vs cycle

iPTM vs run (all cycles)

These are written under output/<run-name>/.

Citation

If you use this pipeline in published work, please cite the original tools:

Boltz

LigandMPNN

(Exact citation text coming soon.)

Status

This repository is under active development and currently optimized for:

Apple Silicon macOS

local exploratory and large-sweep binder design workflows
