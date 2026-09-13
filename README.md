# Exhaustion Reversals in Crypto Perpetual Futures

**A reproducibility-first study of an open-interest / volume-delta signal — and a
systematic, quantitative demonstration of why its apparent edge is *real* as a
market phenomenon but *not tradable* under any causal entry rule.**

**TL;DR** 
- The pattern “directional move + falling open interest” reliably marks a
short-horizon exhaustion-reversal event (~66% of the following hours reverse, mean
≈ +0.34%, stable 2021–2026). 

- Yet every *causal* specification (1h passive, 1htrigger, 5m trigger, causal episode-start) 
yields ≈ 0 edge. 

- 100% of the apparent multi-timeframe edge is hindsight selection, and the
true effect is an amplification (~6×) of baseline intraday mean-reversion that is
concentrated on *terminal* signals. 

- The correct object for future work is
**episode-termination prediction**, not the raw signal.

---

## 1. The pattern

On a 1h bar `t`, a signal fires when price, signed volume and open interest agree on
exhaustion:

| Side  | Condition |
|-------|-----------|
| Long  | `return < 0` **and** `volume_delta < 0` **and** `ΔOI < 0` |
| Short | `return > 0` **and** `volume_delta > 0` **and** `ΔOI < 0` |

Consecutive signal bars form **episodes** (~35% of signals cluster), so bars inside an
episode are not independent — all episode-level statistics use episode aggregation and
non-overlapping sampling.

`volume_delta` (buy−sell volume estimate) is computed by Bulk Volume Classification on
1m data and aggregated; see [`data/README.md`](data/README.md) and the linked BVC
tooling.

---

## 2. What this repository is (and is not)

-  **It is** a fully reproducible empirical pipeline (data → signals → events →
  statistics → figures) with a *feasibility ladder* that separates oracle (look-ahead)
  from causal specifications, plus controls and a quantitative bias decomposition.
-  **It is** a methodological cautionary note: standard event-study and
  multi-timeframe backtests systematically overstate this signal.
-  **It is not** a trading system. No causal specification produces a tradable edge;
  costs are not modeled; nothing here is investment advice.

---

## 3. Repository layout

```
oi-volume-signal-research/
├── README.md
├── LICENSE                     # MIT
├── requirements.txt
├── pyproject.toml              # installable package: oivdc_research
├── .gitignore
├── data/
│   └── README.md               # schema + how to build YOUR OWN dataset           
├── src/oivdc_research/
│   ├── config.py               # ResearchConfig (paths, horizons, tolerances, theme)
│   ├── data.py                 # 1h load / validate / clean (ValidationReport)
│   ├── signals.py              # long/short signal + episode ids
│   ├── forward_returns.py      # entry open[t+1], fwd returns, direction-adjusted PnL
│   ├── clustering.py           # run detection, episode labelling
│   ├── event_selection.py      # raw / episode / oracle / confirmation events
│   ├── statistics.py           # winrate+Wilson CI, t-test, bootstrap, PF, MFE/MAE symmetry
│   ├── plotting.py             # two themes: light & dark (ice nine research)
│   ├── intrabar/               # 5m analysis (S1 decomposition, S4 triggers + controls)
│   └── causal/                 # causal episode starts, raw-signal intrabar (S5)
├── notebooks/                  # linear, reproducible pipeline (shipped with outputs)
│   ├── 01_data_validation.ipynb
│   ├── 02_signal_generation.ipynb
│   ├── 03_forward_returns.ipynb
│   ├── 04_event_selection.ipynb
│   ├── 05_s1_oracle.ipynb
│   ├── 06_s2_passive.ipynb
│   ├── 07_s3_trigger.ipynb
│   ├── 08_s4_multitimeframe.ipynb
│   └── 09_s5_terminal.ipynb
└── reports/
    │── figures/                # per-stage figures (data/, s1/…s5/)
```

---

## 4. The notebook pipeline

Each notebook is self-contained, reads only artifacts produced earlier, and saves its
own figures/tables. Notebooks ship **with outputs** for reading; they also run clean
from scratch (`Restart & Run All`) once you build the data.

| # | Notebook | Question / hypothesis | Key output |
|---|----------|----------------------|------------|
| 01 | `data_validation`   | Are 1h **and** 5m data clean & continuous? | validated 1h + 5m parquet |
| 02 | `signal_generation` | How frequent / clustered are signals? | signals parquet, clustering figs |
| 03 | `forward_returns`   | Raw baseline (existence check) | forward-returns parquet |
| 04 | `event_selection`   | Raw vs episode aggregation | events parquet, methodology fig |
| 05 | `s1_oracle`         | Does the pattern mark a reversal (upper bound)? | S1 stats, Test-0 path, yearly stability |
| 06 | `s2_passive`        | Causal entry after the gap bar | S2 stats + MFE/MAE symmetry |
| 07 | `s3_trigger`        | Does a directed 1h confirmation help? | S3 stats + symmetry |
| 08 | `s4_multitimeframe` | Can a 5m trigger catch the impulse? | S4-A/B + controls + bias |
| 09 | `s5_terminal`       | Where does the edge live? causal starts | signal/next/gap, isolated vs multi |

---

## 5. Methodology: the feasibility ladder

The core idea: evaluate the *same* phenomenon under progressively **causal** entry
rules. If the effect survives only under oracle (look-ahead) rules, it is a
description, not a strategy.

| Stage | Entry rule | Causal? | Result (BTC 1h, 2021–2026) |
|-------|-----------|:-------:|----------------------------|
| **S1 Oracle** | open of the gap bar (known ex-post) | **No** | ~66% of gap hours reverse; mean ≈ +0.34% |
| **S2 Passive** | open after the gap bar closes | Yes | ~51% WR, **mean ≈ 0** |
| **S3 Trigger** | after a directed 1h confirmation | Yes | ~50% WR, **MFE/MAE ≈ 1.0** |
| **S4-A Oracle-5m** | 5m directed trigger inside the *known* gap hour | **No** | 58–60% WR, t ≤ 17.6 |
| **S4-B Causal-5m** | monitor after *every* 1h signal (reset on new signal) | Yes | ~52% WR, **t ≈ 0** |
| **S5 Causal start** | open after every episode start | Yes | ≈ 0 aggregate **(see §6)** |

**Controls (S4):**
- `CTRL-A` random hour, right-side ≈ 50% → the 1h state is required.
- `CTRL-B` gap hour, **wrong-side** ≈ 60% WR → the mechanism is **intraday
  mean-reversion (a bounce)**, not directional continuation.
- `CTRL-C` random hour, wrong-side ≈ 51% → baseline intraday mean-reversion.

**Bias decomposition:** `S4-A − S4-B ≈ S4-A` → **100% of the apparent 5m edge is
hindsight selection** of the reversal hour.

**Diagnostic tools introduced:** feasibility ladder; MFE/MAE touch-probability symmetry
ratio (≈1.0 ⇒ symmetric walk, >~1.2 ⇒ directed impulse); oracle−causal bias
decomposition; the A/B/C control triple.

---


## 6. Key results 

| Finding | Number |
|---------|--------|
| Gap hours that reverse (S1 oracle) | ~66% (mean ≈ +0.34%) |
| Yearly stability of `share_reversed` | 64.1–67.4% (range 3.3pp, std 1.3%) |
| Reversal structure inside gap hour | 50% by 5m-bar 3, 80% by bar 5 (not a single jump) |
| Causal 1h edge (S2/S3) | ≈ 0; MFE/MAE ≈ 1.0 |
| Causal 5m edge (S4-B) | ≈ 0; apparent S4-A edge = selection bias |
| Isolated (terminal) episode starts | +0.10% mean, 57.7% WR |
| Multi-bar (continuation) starts | −0.40% mean, 31.8% WR (toxic) |
| Intraday mean-reversion amplification in gap hours | ~6× baseline |

**Interpretation.** The signal marks a genuine exhaustion state, but the profitable
component is *conditional on the episode terminating* — which is only known ex-post.
A causal entry on episode starts mixes profitable isolated starts with toxic
continuation starts, netting to zero. Hence the correct object for future work is
**predicting episode termination**, not trading the raw signal.

---

## 7. Installation

```bash
git clone https://github.com/Zettnq/oi-volume-signal-research && cd oi-volume-signal-research
pip install -r requirements.txt
pip install -e .            # installs the oivdc_research package
```

Requires `pandas ≥ 2`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `pyarrow`.

---

## 8. Data

The repository is **schema-driven**: it does not depend on a specific symbol or exchange.
Bring your own dataset matching the schema in
[`data/README.md`](data/README.md), then validate it in notebook 01.

If critical checks pass, notebooks 02–09 run unchanged. Raw/processed data are
git-ignored; only the schema, the validator and the build guide ship with the repo.

---

## 9. Reproducibility notes

- Entry convention is `open[t+1]` (signal known at close of `t`); all causal stages
  respect this. Oracle stages are explicitly labelled and never used for trading claims.
- Statistics use direction-adjusted PnL, Wilson CIs for winrate, one-sided t-tests and
  bootstrap CIs for the mean; episode-level sampling handles signal clustering.
- Primary horizon `h=3` was fixed before testing; other horizons are exploratory.
- Notebooks are numbered to mirror the paper sections (S1→S5).

---

## 10. Connect

- (RU) Telegram: https://t.me/icenineresearch
- (RU) Substack: https://substack.com/@icenineresearch 

## 11. License & disclaimer

MIT — see [`LICENSE`](LICENSE).
**Research only. Not investment advice.** Nothing here constitutes a recommendation to
buy or sell any asset.

*© 2026 Zettnq · oi-volume-signal-research · MIT License*