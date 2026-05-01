# self-model-numerics

Python toolkit for closed-loop self-models: predictive-equivalence quotients (KT, KT-CL), world-dependent opacity, and structural quantities ($\alpha_t$, $\sigma^{impl}$, $\eta$, $\Delta_{WD}$). Companion to Suzuki (2026).

## Paper

> Suzuki, K. (2026). *From World Models to Self-Models: Closed-Loop Computational Mechanics of Selfhood.* Draft.

LaTeX source, rigorous-proofs companion (FOUNDATIONS.md, STRESS_TEST_KT.md), and author review notes are at the manuscript repository: <https://github.com/ksk-S/self-model-closed-loop>.

## Contents

The four figure scripts are self-contained. Running `python fig_*.py` writes both PDF (paper-ready) and PNG (preview) outputs.

| Script | Paper section | What it shows |
|---|---|---|
| [`fig_alpha_efe.py`](fig_alpha_efe.py) | §5.4, §6.4 | Identity $\alpha_t = E_Y[\mathrm{BS}(Y)]$ (Proposition `prop:alpha-bs`) verified across the temperature $\beta$ of a Boltzmann active-inference policy. Maximum residual $2.2 \times 10^{-16}$ (machine precision). |
| [`fig_dissolution.py`](fig_dissolution.py) | §5.4, §6.4 | Limit transitions of Corollary `prop:dissolution` on the $(\lambda, \gamma)$ plane (self-memory $\lambda$, world informativeness $\gamma$), with the three corners corresponding to ego dissolution, pure awareness, and normal mode. |
| [`fig_hierarchy.py`](fig_hierarchy.py) | §7.2 | Schematic of the experience hierarchy on the $(\alpha_t, \|\varepsilon^{CL}_{self}\|)$ plane, locating normal / ED / PA / DP. |
| [`fig_ca_wd.py`](fig_ca_wd.py) | §6.3 (Example C) | Theorem WD verified across the 256 elementary cellular automata. WD gap $\Delta_{WD}$ vs reachable joint-state count $\|\mathcal{R}\|$, with Wolfram class indicated; histogram of $\Delta_{WD}$ across all rules. |

## Reproducing the figures

```bash
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate            # Windows
pip install -r requirements.txt
python fig_alpha_efe.py
python fig_dissolution.py
python fig_hierarchy.py
python fig_ca_wd.py
```

Each script writes `fig_*.pdf` (paper-ready) and `fig_*.png` (preview) into the working directory.

## Dependencies

- Python 3.9+
- NumPy
- Matplotlib

See `requirements.txt`.

## Roadmap (planned)

The paper proposes three distinctively testable predictions (Suzuki 2026, §7.4); reference implementations are planned here:

- **P1: Self-minimisation phase transition** — ε-machine simulations showing $\|\varepsilon^{CL}_{self}\|$ contraction with critical scaling $\eta_c \sim \log\|\mathcal{S}\|/\|\mathcal{R}\|$ as $\eta = 1 - d_{CL}/d_{OL}$ is varied.
- **P2: Architecture convergence under transparent worlds** — opacity-graded T-maze variants comparing action-aware vs action-unaware agents as $\Omega_t \to 0$, extending Torresan et al. (2025) quantitatively.
- **P3: Structural marker for deceptive alignment** — $\Delta = \|\mathcal{T}\| - \|\varepsilon^{CL}_{self}\|$ measurement on sleeper-agent benchmarks (Hubinger et al. 2019).

A general-purpose minimum-state estimation utility (Givan-Dean-Greig partition refinement applied to closed-loop $\varepsilon$-transducers; see Appendix I.4 of the paper) is also planned.

## Citation

```bibtex
@unpublished{suzuki2026selfhood,
  author = {Suzuki, Keisuke},
  title  = {From World Models to Self-Models: Closed-Loop Computational Mechanics of Selfhood},
  year   = {2026},
  note   = {Draft}
}
```

## Author

Keisuke Suzuki — Center for Human Nature, Artificial Intelligence, and Neuroscience (CHAIN), Hokkaido University.
Contact: ksk@chain.hokudai.ac.jp

## License

TBD (will be set before paper submission).
