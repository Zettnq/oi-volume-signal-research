# Data

The repository is **schema-driven**: it does not depend on a specific symbol 
or exchange. Bring your own dataset that matches the schema below.

## Required schema (highier timeframe)

| column                | type  | meaning                                  |
|-----------------------|-------|------------------------------------------|
| timestamp             | dt64  | bar open time (UTC, regular cadence)     |
| open/high/low/close   | f8    | OHLC                                     | 
| volume                | f8    | base volume                              |
| volume_delta          | f8    | signed (buy−sell) volume estimate        |
| open_interest         | f8    | open interest                            |
| return                | f8    | close-to-close return                    |
| delta_oi              | f8    | change of open interest                  |

Lower timeframe used to break down price movement within a higher-timeframe bar. Lower timeframe dataset needs the same OHLCV + `volume_delta` (without OI).

## How the author built the data

- `volume_delta` and OHLCV: Bulk Volume Classification on 1m trades/candles, aggregated to 1h/5m
  (see the author's BVC repository: https://github.com/Zettnq/Volume-Delta).
- `open_interest` / `delta_oi`: exchange OI series, diffed per bar.
  (see the authors's Open Interest fetcher script: https://github.com/Zettnq/Bybit-Open-Interest-Fetcher)

## Build your own

1. Fetch OHLCV + OI for any symbol/exchange/period you like.
2. Estimate `volume_delta` by any method (BVC, tick-rule, trade classification)
   or use the linked BVC tooling.
3. Aggregate to your timeframe and save as CSV/parquet with the schema above.
4. Place files at `data/raw/{asset}_1h.csv` and `data/raw/{asset}_5m.csv`
   (set `ASSET`/`TIMEFRAME` in notebooks to match your names).
5. Validate before running in notebook 01

If critical checks pass, the pipeline (notebooks 02–09) runs unchanged.