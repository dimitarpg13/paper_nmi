# paper_nmi — Companion repository

Companion code, data, and derivations for the manuscript

> **Attention and Intrinsic Geometry are Structurally Incompatible:
> A Prescriptive Lagrangian Alternative for Language Modelling**
> Dimitar P. Gueorguiev (Independent Researcher)
> *Prepared for submission to Nature Machine Intelligence.*

This is the focused, geometry-first companion to the broader **Semantic
Simulation** framework. The full framework paper and its experiments live in
the [`semsimula-paper`](https://github.com/dimitarpg13/semsimula-paper)
repository; this repository hosts only the material needed to follow and
reproduce the results in the NMI manuscript.

## What the paper claims

1. **Conservative Obstruction Theorem.** No `C²` scalar potential on the token
   states can reproduce three defining properties of scaled dot-product
   attention — asymmetric coupling (P1), coupling–content decoupling (P2), and a
   sequence-length-independent influence budget (P3) — *regardless of the order
   of the dynamics*. Consequently, **attention cannot host an intrinsic
   Riemannian metric**; metrics extracted from trained transformers are
   *descriptive* overlays, not the law of motion.

2. **Prescriptive geometric architecture.** A second-order Lagrangian
   scalar-potential language model (SPLM) whose inference is a damped
   Euler–Lagrange flow on a single learned energy field `V_θ`. It carries an
   intrinsic **Jacobi metric** `g̃ = 2(E − V_θ) g`, is conservative by
   construction, and runs in **O(1) memory** per step (no KV cache).

3. **Attention Optimality Conjecture.** Within the design class of smooth,
   content-decoupled, row-normalised routing with zero auxiliary state, softmax
   attention is uniquely Pareto-optimal — pinning attention and its
   intrinsic-geometry alternative to opposite corners of one design lattice.

## Repository layout

```
paper_nmi/
├── README.md
├── requirements.txt                 # pinned runtime deps
├── pyproject.toml                    # machine-readable dependency spec
├── LICENSE, LICENSE-CC-BY-4.0
└── notebooks/conservative_arch/
    ├── data_module.py                # TinyStories loading (HF) + batching
    ├── matched_baseline_model.py     # matched-parameter attention GPT baseline
    ├── parf/                         # conservative + Fock-augmented models
    │   ├── model_parf.py
    │   ├── model_parf_sparse.py
    │   ├── model_parf_multixi.py
    │   ├── model_fock_parf.py
    │   ├── model_fock_parf_v2.py
    │   ├── model_fock_parf_multixi.py   # the register-augmented LM (headline result)
    │   └── causal_probe_multixi.py      # causality assertions
    ├── multixi/model_multixi.py      # multi-ξ K-channel SPLM core
    ├── energetic_minima/model_ln.py  # LayerNorm SPLM variant
    ├── sarf_mass_variant/model_sarf_mass.py
    └── scaleup/
        ├── colab_fock_multixi_h128.ipynb            # ← run this in Colab
        ├── train_fock_multixi_scaleup.py            # Fock-PARFLM trainer
        ├── train_matched_baseline_scaleup.py        # attention baseline trainer
        └── compute_unigram_frequencies_tinystories.py
```

This is a self-contained subset of the framework repository
[`semsimula-paper`](https://github.com/dimitarpg13/semsimula-paper), pruned to
exactly the modules needed to reproduce the NMI results. The LaTeX source of the
manuscript is mirrored here on submission.

## Reproducing the headline result (constant-memory LM, Colab)

The central empirical claim of the paper — a conservative, **O(1)-memory**
language model reaching **12.0 perplexity** on TinyStories against **7.81** for
matched-parameter attention — is reproduced end-to-end by a single notebook:

1. Open
   [`notebooks/conservative_arch/scaleup/colab_fock_multixi_h128.ipynb`](notebooks/conservative_arch/scaleup/colab_fock_multixi_h128.ipynb)
   in Google Colab (GPU runtime recommended: A100/H100).
2. Run all cells. The setup cell clones **this** repository
   (`https://github.com/dimitarpg13/paper_nmi.git`), installs dependencies,
   precomputes the one-time unigram-surprisal file, and launches training.

The notebook is self-contained: every Python module it imports lives in this
repository under `notebooks/conservative_arch/`, and the import paths resolve
without modification.

### Matched-attention baseline (7.81 PPL)

The comparison point is produced by the attention baseline trainer. After the
repository is cloned in the notebook environment (or locally), run:

```bash
cd notebooks/conservative_arch/scaleup
python train_matched_baseline_scaleup.py   # see --help for the schedule flags
```

### Shared-potential diagnostic

The Jacobian-symmetry / shared-potential separator that operationalises the
Conservative Obstruction Theorem (SPLM $R^2 = 0.957$ vs. attention
$R^2 = 0.46$–$0.54$) is part of the broader framework programme; the full
diagnostic harness lives in the
[`semsimula-paper`](https://github.com/dimitarpg13/semsimula-paper) repository
under `notebooks/conservative_arch/`. The causality probe used by the trainer
here (`parf/causal_probe_multixi.py`) is included.

## Citation

A `CITATION.cff` and BibTeX entry will be added upon submission. Until then,
please cite the framework paper from the `semsimula-paper` repository.

## License

To be finalised before submission (intended: CC-BY 4.0 for text, MIT/Apache-2.0
for code), consistent with the framework repository.
