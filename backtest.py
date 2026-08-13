"""Insider Cluster-Buy Backtester (entry point).

Daily-rebalanced backtest of the strategies registered in
backtest/strategies.py against the SEC EDGAR Form 4 insider-cluster signal.
Each strategy runs as an independent portfolio with its own starting capital
across a set of exit methods (fixed 30/90/180/365 trading-day holds, plus
365-day-capped trailing-stop exits at 10/20/30%, per FIFO lot).
SPY is the buy-and-hold benchmark.

Run with backtest.bat (creates .venv + installs deps), or:
    python backtest.py
Env vars:
    BT_MONTHS, BT_CAPITAL, BT_RF (zero|tbills), BT_OFFLINE (0|1)
    BT_STRATEGIES (comma list or 'all'), BT_EXITS (comma list or 'all')
    BT_SLIPPAGE_BPS, BT_LIQUIDITY_FLOOR, BT_MIN_PRICE
    BT_COST_SWEEP (none|low_med_high), BT_VERBOSE (0|1)
    BT_FIT (0|1), BT_FIT_HORIZON (30|90|180|365), BT_WRITE_WEIGHTS (0|1)
    BT_EVENTS_FROM (path to an events parquet, or 'latest') — skips Phase 1
    BT_MODEL_SCORES (path to an OOF scores parquet, or 'latest') — model ranking
    BT_AS_OF (YYYY-MM-DD) — pins the window end so a run is reproducible
    BT_DROP_TICKER_REUSE (0|1, default 1) — drop Form 4 events whose ticker
        later changed hands to an unrelated company (see ticker_reuse.py);
        matches build_research_dataset's own drop_ticker_reuse default
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import webbrowser
from datetime import date, datetime, timedelta

import pandas as pd

import insider_cluster_buys as ics
from backtest import history, prices, state as state_mod, strategies, engine, metrics, report, signal_fit, model_scores

OUTPUT_ROOT = "out"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("backtest")


def _prompt_with_default(prompt: str, default: str, env_var: str | None = None) -> str:
    if env_var and os.environ.get(env_var):
        val = os.environ[env_var].strip()
        log.info("Using %s=%s from env", env_var, val)
        return val
    raw = input(f"{prompt} [default {default}]: ").strip()
    return raw or default


def _parse_subset(raw: str, full: list, kind: str) -> list:
    """Parse a comma-separated subset string, validating against `full`.

    `kind` is 'strategy' or 'exit' (controls lookup)."""
    s = raw.strip().lower()
    if not s or s == "all":
        return list(full)
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if kind == "exit":
        known = {e.label for e in full}
        unknown = [t for t in tokens if t not in known]
        if unknown:
            raise SystemExit(
                f"Unknown exit methods {unknown}; choose from "
                f"{sorted(known)} or 'all'"
            )
        label_to_exit = {e.label: e for e in full}
        # preserve user order
        return [label_to_exit[t] for t in tokens]
    # strategy
    known = {s.name for s in full}
    unknown = [t for t in tokens if t not in known]
    if unknown:
        raise SystemExit(
            f"Unknown strategies {unknown}; choose from "
            f"{sorted(known)} or 'all'"
        )
    name_to_strat = {s.name: s for s in full}
    return [name_to_strat[t] for t in tokens]


def _prompt_inputs() -> dict:
    print("Insider Cluster-Buy Backtester")
    print("Re-runs reuse parse_cache/, clusters_history/, and price_cache/.")
    print()
    months = int(_prompt_with_default("Months of history", "36", "BT_MONTHS"))
    capital = float(_prompt_with_default(
        "Starting capital per strategy ($)", "100000", "BT_CAPITAL",
    ))
    rf = _prompt_with_default("Risk-free model {zero,tbills}", "zero", "BT_RF").lower()
    offline_raw = _prompt_with_default(
        "Generate offline (inline plotly.js) HTML? [y/N]", "N", "BT_OFFLINE",
    )
    offline = offline_raw.lower() in ("y", "yes", "1", "true")

    strategies_raw = _prompt_with_default(
        "Strategies (comma list or 'all')", "all", "BT_STRATEGIES",
    )
    exits_raw = _prompt_with_default(
        "Exit methods (comma list of labels, or 'all')", "all", "BT_EXITS",
    )
    slippage_bps = int(_prompt_with_default(
        "Slippage per side (bps)", "10", "BT_SLIPPAGE_BPS",
    ))
    liquidity_floor = int(_prompt_with_default(
        "Liquidity floor (min 20-day median $-vol)", "500000", "BT_LIQUIDITY_FLOOR",
    ))
    # Default mirrors engine.MIN_PRICE_FLOOR. This prompt's own default used
    # to be "0" (no floor at all) and, because main() always passes
    # cfg["min_price"] through explicitly, that "0" silently overrode
    # engine.py's default on every run regardless of what engine.py set it
    # to -- the min_price kwarg only helps if the value actually reaching it
    # is non-zero. Keep the two in sync; 0 remains a valid opt-out.
    min_price = float(_prompt_with_default(
        "Minimum entry price ($, 0 = no floor)", str(engine.MIN_PRICE_FLOOR), "BT_MIN_PRICE",
    ))
    cost_sweep = _prompt_with_default(
        "Cost-sensitivity sweep {none,low_med_high}", "none", "BT_COST_SWEEP",
    ).lower()
    if cost_sweep not in ("none", "low_med_high"):
        raise SystemExit(f"Invalid BT_COST_SWEEP {cost_sweep!r}")
    verbose_raw = _prompt_with_default(
        "Verbose (DEBUG) logging? [0/1]", "0", "BT_VERBOSE",
    )
    verbose = verbose_raw.strip() in ("1", "y", "yes", "true")

    fit_signal_raw = _prompt_with_default(
        "Fit signal weights from history? [1/0]", "1", "BT_FIT",
    )
    fit_signal = fit_signal_raw.strip() in ("1", "y", "yes", "true")
    fit_horizon = int(_prompt_with_default(
        "Signal-fit horizon (trading days) {10,30,90,180,365}", "90", "BT_FIT_HORIZON",
    ))
    if fit_horizon not in (10, 30, 90, 180, 365):
        raise SystemExit(f"Invalid BT_FIT_HORIZON {fit_horizon!r}; choose from 10, 30, 90, 180, 365")
    write_weights_raw = _prompt_with_default(
        "Update root signal_weights.json for the live scanner? [0/1]", "0", "BT_WRITE_WEIGHTS",
    )
    write_weights = write_weights_raw.strip() in ("1", "y", "yes", "true")

    # Reuse a previous scrape instead of re-walking SEC EDGAR. Empty = scrape.
    events_from = _prompt_with_default(
        "Reuse events parquet (path, 'latest', or blank to scrape)", "", "BT_EVENTS_FROM",
    ).strip()
    # OOF ranking-model scores. Blank = do not attach any, which leaves
    # model_score None everywhere. Any strategy whose rank_fn is
    # rank_by_model_score then ranks every candidate at -inf, and
    # _rank_capacity_order's (-score, ticker) sort degrades to alphabetical
    # order by ticker. That is worse than the arrival order it replaced, and
    # it fails silently, so main() refuses that combination outright.
    model_scores_path = _prompt_with_default(
        "Model OOF scores parquet (path, 'latest', or blank for none)",
        "latest", "BT_MODEL_SCORES",
    ).strip()
    as_of_raw = _prompt_with_default(
        "As-of date YYYY-MM-DD (blank = today)", "", "BT_AS_OF",
    ).strip()
    as_of = None
    if as_of_raw:
        try:
            as_of = datetime.strptime(as_of_raw, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit(f"Invalid BT_AS_OF {as_of_raw!r}; expected YYYY-MM-DD")

    # Drop Form 4 events whose ticker later changed hands to an unrelated
    # company (see ticker_reuse.py) -- the same guard build_research_dataset
    # applies by default. On by default because pricing a delisted company's
    # insider buys against today's occupant of its old ticker is a data bug,
    # not a modeling choice; BT_DROP_TICKER_REUSE=0 reproduces the
    # unfiltered, pre-fix backtest exactly.
    drop_ticker_reuse_raw = _prompt_with_default(
        "Drop unsafe ticker-reuse events? [1/0]", "1", "BT_DROP_TICKER_REUSE",
    )
    drop_ticker_reuse = drop_ticker_reuse_raw.strip() in ("1", "y", "yes", "true")

    return {
        "months": months,
        "capital": capital,
        "rf_model": rf,
        "offline": offline,
        "strategies_raw": strategies_raw,
        "exits_raw": exits_raw,
        "slippage_bps": slippage_bps,
        "liquidity_floor": liquidity_floor,
        "min_price": min_price,
        "cost_sweep": cost_sweep,
        "verbose": verbose,
        "fit_signal": fit_signal,
        "fit_horizon": fit_horizon,
        "write_weights": write_weights,
        "events_from": events_from,
        "model_scores": model_scores_path,
        "as_of": as_of.isoformat() if as_of else None,
        "drop_ticker_reuse": drop_ticker_reuse,
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }


def _cost_sweep_levels(mode: str) -> list[int]:
    if mode == "low_med_high":
        return [5, 10, 20]
    return []


def main() -> None:
    cfg = _prompt_inputs()
    if cfg["verbose"]:
        logging.getLogger().setLevel(logging.DEBUG)
        log.info("Verbose logging enabled")
    log.info("Config: %s", cfg)

    # Apply engine overrides up front so every run sees them.
    engine.LIQUIDITY_FLOOR = int(cfg["liquidity_floor"])
    base_slippage = float(cfg["slippage_bps"]) / 10_000.0

    # Resolve strategy + exit-method subsets (validates against the registry).
    chosen_strategies = _parse_subset(
        cfg["strategies_raw"], strategies.STRATEGIES, "strategy",
    )
    chosen_exits = _parse_subset(
        cfg["exits_raw"], strategies.EXIT_METHODS, "exit",
    )
    # Mutate EXIT_METHODS in place so report.py (which imported it at module
    # load) picks up the subset.
    strategies.EXIT_METHODS[:] = chosen_exits

    if ics.USER_AGENT.startswith("Your Name"):
        log.warning(
            "SEC_USER_AGENT is the placeholder — SEC may rate-limit or block. "
            "Set SEC_USER_AGENT in .env to a real name + email."
        )

    # ----- 1. Scrape + build event timeline -----
    as_of = date.fromisoformat(cfg["as_of"]) if cfg["as_of"] else date.today()
    if cfg["events_from"]:
        log.info("=== Phase 1: load cached events (skipping scrape) ===")
    else:
        log.info("=== Phase 1: scrape filings ===")
    events_df, win_start, win_end, parse_errors, n_dropped_ticker_reuse = history.build_history(
        cfg["months"], as_of=as_of, events_from=cfg["events_from"] or None,
        drop_ticker_reuse=cfg["drop_ticker_reuse"],
    )
    if events_df.empty:
        log.error("No qualifying events in the window — aborting.")
        sys.exit(1)
    if parse_errors:
        sample = [f"{e.get('adsh','?')}: {e.get('error','?')}" for e in parse_errors[:3]]
        log.warning("Discarded %d parse errors (sample: %s)", len(parse_errors), sample)
    log.info("Events: %d rows, %d unique tickers", len(events_df), events_df["ticker"].nunique())

    # ----- 2. Build daily-state index -----
    log.info("=== Phase 2: build daily state index ===")
    state_builder = state_mod.DailyStateBuilder(events_df)

    # Attach ranking-model scores, and refuse the silent-failure combination.
    # A model-ranked strategy with no scores does not error and does not
    # produce a null result. It produces a confident-looking alphabetical
    # ranking, which would be read as evidence about the model. Catch it here
    # rather than after the run, because a run costs about a day per year of
    # history.
    needs_scores = [s.name for s in chosen_strategies
                    if s.rank_fn is strategies.rank_by_model_score]
    scores_path = model_scores.resolve_model_scores_path(cfg["model_scores"])
    # Record what 'latest' (or an explicit path) actually resolved to, not just
    # the spec string. config.json used to store only "latest", so a finished
    # run gave no way to tell which of several score files it had ranked on.
    cfg["model_scores_resolved"] = scores_path
    cfg["model_scores_column"] = model_scores.DEFAULT_SCORE_COL if scores_path else None
    if scores_path:
        scores = model_scores.load_model_scores(scores_path)
        state_builder.set_model_scores(scores)
        log.info("Attached %d model scores from %s", len(scores), scores_path)
        # Sanity-check the score file against this run's universe. We cannot
        # key-match here: scores are keyed on the cluster event_day (the
        # decision day a cluster fires), and events_df holds raw Form 4 rows
        # keyed on transaction_date. Those are different days by construction.
        # Ticker and date-range overlap still catch the failure that matters,
        # which is a score file built for some other window or ticker set.
        ev_tickers = set(events_df["ticker"].astype(str).str.upper())
        sc_tickers = {t for t, _ in scores}
        overlap = len(ev_tickers & sc_tickers)
        sc_days = [d for _, d in scores]
        sc_first, sc_last = min(sc_days), max(sc_days)
        cfg["model_scores_first_day"] = sc_first.isoformat()
        cfg["model_scores_last_day"] = sc_last.isoformat()
        log.info(
            "Model-score universe: %d/%d run tickers have scores (%.1f%%); "
            "score dates %s..%s",
            overlap, len(ev_tickers), 100.0 * overlap / max(len(ev_tickers), 1),
            sc_first, sc_last,
        )
        if overlap == 0:
            log.warning(
                "NO ticker in this run has a model score. Every model-ranked "
                "strategy will rank alphabetically by ticker. Check that the "
                "score file matches this run's window and ticker set."
            )
        # Coverage gap check. rank_by_model_score maps an unscored candidate to
        # -inf, which sorts it LAST but leaves it eligible, so a day outside the
        # score window still fills its book -- by _rank_capacity_order's ticker
        # tie-break, i.e. alphabetically. In backtest_20260812_234954 that was
        # 17 months of a 96-month window and about a sixth of every
        # model_ranked_* strategy's lots. It is invisible in the output unless
        # something says so here.
        if needs_scores:
            uncovered_before = (sc_first - win_start).days
            uncovered_after = (as_of - sc_last).days
            if uncovered_before > 0 or uncovered_after > 0:
                total_days = max((as_of - win_start).days, 1)
                log.warning(
                    "MODEL-SCORE COVERAGE GAP: run window is %s..%s but scores only "
                    "span %s..%s -- %d day(s) before and %d day(s) after are unscored "
                    "(%.1f%% of the window). Strategies %s stay ELIGIBLE on those days "
                    "and rank alphabetically by ticker there, because "
                    "rank_by_model_score sorts an unscored candidate last rather than "
                    "excluding it. Clamp BT_MONTHS/BT_AS_OF to the score window, or "
                    "read those strategies' early lots as unranked.",
                    win_start, as_of, sc_first, sc_last,
                    max(uncovered_before, 0), max(uncovered_after, 0),
                    100.0 * (max(uncovered_before, 0) + max(uncovered_after, 0)) / total_days,
                    needs_scores,
                )
    elif needs_scores:
        raise SystemExit(
            f"Strategies {needs_scores} rank by model score, but no score file "
            f"was given. Set BT_MODEL_SCORES to an OOF parquet, or drop those "
            f"strategies. Running without scores would rank them alphabetically "
            f"by ticker and the result would look like a real ranking."
        )

    # ----- 3. Fetch price data (universe = all event tickers + SPY) -----
    log.info("=== Phase 3: fetch price data ===")
    tickers = sorted(set(events_df["ticker"].astype(str).str.upper())) + ["SPY"]
    price_start = win_start - timedelta(days=30)
    price_end = as_of + timedelta(days=400)  # cushion for 365-day lots near end
    pu = prices.PriceUniverse()
    pu.ensure(tickers, price_start, price_end)
    pu.finalize()
    if "SPY" not in pu.frames:
        log.error("SPY price data could not be loaded — aborting.")
        sys.exit(1)

    # ----- 4. Build trading calendar from SPY -----
    calendar = pu.trading_calendar(win_start, min(as_of, price_end))
    if not calendar:
        log.error("Empty trading calendar — aborting.")
        sys.exit(1)
    log.info("Backtest calendar: %d trading days (%s .. %s)",
             len(calendar), calendar[0], calendar[-1])

    # ----- 4.5. Fit signal weights (train period only) -----
    events_ds: pd.DataFrame | None = None
    fit_result = None
    tail_fit_result = None
    fit_cfg = signal_fit.FitConfig(primary_horizon=cfg["fit_horizon"])
    if cfg["fit_signal"]:
        log.info("=== Phase 4.5: fit signal weights (horizon %dd) ===", cfg["fit_horizon"])
        # build_event_dataset() reads state_builder.state_for_day() for every
        # calendar day, which populates the builder's per-day memoization
        # cache with learned_score=None/tail_score=None (no learned/tail
        # weights are attached yet). set_learned_weights() / set_tail_weights()
        # below clear that cache so every subsequent engine run (Phase 5)
        # re-scores each day with the fitted weights instead of serving the
        # stale None-scored cached states.
        events_ds = signal_fit.build_event_dataset(state_builder, pu, calendar, fit_cfg)
        fit_result = signal_fit.fit_weights(events_ds, fit_cfg)
        # Independent of fit_weights' success — the tail-probability score
        # is a separate research-triage output (see signal_fit.fit_tail_score).
        tail_fit_result = signal_fit.fit_tail_score(events_ds, fit_cfg)
        if tail_fit_result is not None:
            state_builder.set_tail_weights(tail_fit_result.weights_int)
            log.info(
                "Tail-score fit: split %s, train %d / test %d events, base rate "
                "P(adj_%dd > %.0f%%)=%.3f, tail weights=%s",
                tail_fit_result.split_date, tail_fit_result.train_events,
                tail_fit_result.test_events, tail_fit_result.tail_horizon,
                tail_fit_result.moonshot_thresh * 100, tail_fit_result.base_rate,
                tail_fit_result.weights_int,
            )
        else:
            log.warning(
                "Tail-score fit failed (too few train events with a valid "
                "target) — learned_tail_concentrated stays flat this run "
                "(learned_tpo_gated is unaffected; it only needs the mean fit)."
            )
        if fit_result is not None:
            state_builder.set_learned_weights(fit_result.weights_int)
            log.info(
                "Signal fit: split %s, train %d events / test %d events, "
                "lambda=%s, learned weights=%s",
                fit_result.split_date, fit_result.train_events, fit_result.test_events,
                fit_result.lambda_used, fit_result.weights_int,
            )
        else:
            log.warning(
                "Signal fit failed (too few train events) — dropping learned_* "
                "strategies from this run."
            )
            chosen_strategies = [s for s in chosen_strategies if not s.name.startswith("learned_")]
    else:
        log.info("Signal fit disabled (BT_FIT=0) — dropping learned_* strategies from this run.")
        chosen_strategies = [s for s in chosen_strategies if not s.name.startswith("learned_")]

    # ----- 5. Run strategies × exit methods (×slippage levels if sweep on) -----
    sweep_bps = _cost_sweep_levels(cfg["cost_sweep"])
    if sweep_bps:
        slippage_levels = [(bps / 10_000.0, f"_{bps}bps") for bps in sweep_bps]
        log.info("=== Phase 5: simulate %d strategies × %d exit methods × %d slippage levels ===",
                 len(chosen_strategies), len(chosen_exits), len(slippage_levels))
    else:
        slippage_levels = [(base_slippage, "")]
        log.info("=== Phase 5: simulate %d strategies × %d exit methods ===",
                 len(chosen_strategies), len(chosen_exits))

    results_by_exit: dict[str, list[engine.RunResult]] = {ex.label: [] for ex in chosen_exits}
    strategy_order: list[str] = []
    for slip, suffix in slippage_levels:
        engine.SLIPPAGE = slip
        for s in chosen_strategies:
            run_strat = s if not suffix else dataclasses.replace(s, name=f"{s.name}{suffix}")
            if run_strat.name not in strategy_order:
                strategy_order.append(run_strat.name)
            for ex in chosen_exits:
                log.info("  Running %s @ exit %s (slippage %.0f bps) …",
                         run_strat.name, ex.label, slip * 10_000)
                r = engine.run_strategy(
                    strategy=run_strat, exit_method=ex,
                    starting_capital=cfg["capital"],
                    calendar=calendar, prices=pu, states=state_builder,
                    rf_model=cfg["rf_model"],
                    min_price=cfg["min_price"],
                )
                results_by_exit[ex.label].append(r)
                log.info("    -> NAV %.0f, %d lots, %d rebal, skips=%s",
                         r.equity_curve.iloc[-1], sum(1 for _ in r.trades),
                         r.n_rebalances, dict(r.skips))

    # SPY baseline uses the configured slippage (or base, if sweep is on).
    engine.SLIPPAGE = base_slippage
    log.info("  Running spy_buy_and_hold baseline …")
    spy_result = engine.run_spy_baseline(cfg["capital"], calendar, pu)
    log.info("    -> NAV %.0f", spy_result.equity_curve.iloc[-1])

    # ----- 6. Metrics + summary -----
    log.info("=== Phase 6: compute metrics ===")
    all_results = [r for runs in results_by_exit.values() for r in runs]
    summary_df = metrics.summary_table(
        all_results, spy_result, rf_model=cfg["rf_model"],
        fit_train_end=fit_result.train_end if fit_result is not None else None,
        tail_train_end=tail_fit_result.train_end if tail_fit_result is not None else None,
    )

    # ----- 7. Write outputs -----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, f"backtest_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    rf_descr = "0% on idle cash"
    if cfg["rf_model"] == "tbills":
        rf_descr = f"{engine.TBILL_ANNUAL_RATE*100:.1f}% APR on idle cash (T-bills proxy)"
    sweep_descr = "off" if not sweep_bps else f"on ({', '.join(f'{b}bps' for b in sweep_bps)})"
    if fit_result is not None:
        fit_descr = (
            f"horizon {fit_result.primary_horizon}d, train through {fit_result.split_date} "
            f"({fit_result.train_events} events), OOS after ({fit_result.test_events} events); "
            f"lambda={fit_result.lambda_used}"
        )
    elif cfg["fit_signal"]:
        fit_descr = "attempted, failed (too few train events)"
    else:
        fit_descr = "off"
    config_for_report = {
        "Generated": cfg["run_at"],
        "Months of history": cfg["months"],
        "Starting capital / strategy": f"${cfg['capital']:,.0f}",
        "Backtest window": f"{win_start} .. {calendar[-1]}",
        "Trading days simulated": len(calendar),
        "Exit methods": ", ".join(ex.label for ex in chosen_exits),
        "Strategies": ", ".join(strategy_order + ["spy_buy_and_hold"]),
        "Slippage per side": f"{cfg['slippage_bps']} bps",
        "Liquidity floor (20-day median $-vol)": f"${engine.LIQUIDITY_FLOOR:,}",
        "Min entry price": "off" if cfg["min_price"] <= 0 else f"${cfg['min_price']:.2f}",
        "Participation cap (share of 20d median share vol)": f"{engine.MAX_PARTICIPATION_PCT:.0%}",
        "Cost-sensitivity sweep": sweep_descr,
        "Risk-free model": rf_descr,
        "Signal fit": fit_descr,
    }

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({**cfg,
                   "window_start": str(win_start),
                   "window_end": str(calendar[-1]),
                   "trading_days": len(calendar),
                   "tickers_loaded": len(pu.frames),
                   "tickers_missing": len(pu.missing),
                   "events_dropped_ticker_reuse": n_dropped_ticker_reuse,
                   "strategies": strategy_order,
                   "exit_methods": [ex.label for ex in chosen_exits]}, fh, indent=2)

    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    for r in all_results:
        engine.trades_to_dataframe(r.trades).to_csv(
            os.path.join(out_dir, f"trades_{r.strategy}_{r.exit_label}.csv"),
            index=False,
        )
        r.equity_curve.to_csv(
            os.path.join(out_dir, f"equity_{r.strategy}_{r.exit_label}.csv"),
            header=["nav"],
        )
    engine.trades_to_dataframe(spy_result.trades).to_csv(
        os.path.join(out_dir, "trades_spy_buy_and_hold.csv"), index=False,
    )
    spy_result.equity_curve.to_csv(
        os.path.join(out_dir, "equity_spy_buy_and_hold.csv"), header=["nav"],
    )

    fit_summary = None
    if fit_result is not None:
        payload_dict = signal_fit.weights_payload(fit_result, fit_cfg, tail_fit=tail_fit_result)
        signal_fit.save_weights(payload_dict, os.path.join(out_dir, "signal_weights.json"))
        if events_ds is not None:
            events_ds.to_csv(os.path.join(out_dir, "signal_fit_events.csv"), index=False)
        fit_result.feature_stats.to_csv(os.path.join(out_dir, "signal_fit_stats.csv"))
        if tail_fit_result is not None:
            tail_fit_result.tail_stats.to_csv(os.path.join(out_dir, "signal_fit_tail_stats.csv"))

        if cfg["write_weights"]:
            root_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signal_weights.json")
            if tail_fit_result is not None:
                # Research-triage (P(moonshot)) sorting is the dashboard's
                # purpose, so the TAIL weights become the live "weights";
                # the mean-return-based fit is preserved under "mean_weights".
                root_payload = dict(payload_dict)
                root_payload["mean_weights"] = payload_dict["weights"]
                root_payload["weights"] = tail_fit_result.weights_int
                signal_fit.save_weights(root_payload, root_weights_path)
                log.warning(
                    "*** Wrote learned signal weights to %s — the LIVE SCANNER will now use the "
                    "TAIL-PROBABILITY (P(moonshot)) weights as \"weights\"; the mean-return-based "
                    "fit is preserved under \"mean_weights\". ***",
                    root_weights_path,
                )
            else:
                signal_fit.save_weights(payload_dict, root_weights_path)
                log.warning(
                    "*** Wrote learned signal weights to %s — the LIVE SCANNER will now "
                    "use these (mean-return-based) weights instead of "
                    "insider_cluster_buys.DEFAULT_WEIGHTS (tail-score fit failed or was "
                    "skipped, so mean weights remain the root \"weights\"). ***",
                    root_weights_path,
                )

        oos_df = signal_fit.oos_strategy_stats(all_results, spy_result, fit_result.split_date)
        fit_summary = {"fit": fit_result, "oos_df": oos_df, "tail_fit": tail_fit_result}

    html = report.render_html(
        summary_df=summary_df,
        results_by_exit=results_by_exit,
        spy_result=spy_result,
        strategy_order=strategy_order,
        config=config_for_report,
        offline=cfg["offline"],
        fit_summary=fit_summary,
    )
    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    log.info("Wrote outputs to %s", out_dir)
    print()
    print(f"Done. Report: {html_path}")

    try:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
