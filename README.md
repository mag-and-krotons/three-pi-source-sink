# Three-π singularity structure: closed, self-recharging source and sink

Code for A. Singh, *The Three-π Singularity Structure as a Closed, Self-Recharging Source and Sink: Exact Persistent
Flow, Conservation, Recharge and Observer-Availability Theorems for a Twisted 28-State Band* (2026).

Both scripts work from Definitions 1–6 of the paper in the twisted frame (Eq. (1)), with exact propagation through a
full diagonalisation. Nothing is fitted.

| script | output | reproduces |
|---|---|---|
| `analysis.py` | `results/analysis.json` | Theorem 1 (levels, J₀, gaps), Corollary 1, Table I and the Sec. VIII ledgers, Proposition 1, Lemma 2, Table II, Theorem 4 bounds, Proposition 2, Fig. 6 end values, the 0.99 mT field of Sec. IX |
| `figs.py` | `figs/fig1.pdf` … `fig6.pdf` | Figs. 1–6 |

```
pip install -r requirements.txt
python3 analysis.py        # about 2 minutes; SMOKE=1 for a quick reduced run
python3 figs.py            # about 1 minute
```

Licence: CC BY 4.0.
