"""Backtest engine: daily loop with FIFO tax-lot accounting.

For each (strategy, exit_method) pair we run an independent portfolio with its
own starting capital. Orders decided at end of day D execute at open of day
D+1 (next trading day) with per-side slippage. Each new buy creates a lot
with a fixed H-trading-day life (exit_method.exit_days); trims sell oldest
lots first. Trailing-stop exit methods additionally close a lot early once
its price falls trailing_stop_pct off its running peak close.

When strategy.hold_days is set, its lots use min(hold_days, exit_method.
exit_days) as the life instead, and the trim path (order execution's
delta < 0 branch) refuses to touch them: they only close via that expiry,
a stop-loss, or a trailing stop, never because the signal decayed.

When strategy.rank_fn is set and demand for capacity slots outstrips
max_concurrent_tickers, step 2 grants slots to the highest-scoring buy
orders first instead of the queue (arrival) order. When rank_fn is None,
the queue order is untouched, so every existing strategy keeps its old,
first-come-first-served capacity behavior exactly.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Optional

import pandas as pd

from . import strategies as strat_mod
from .prices import PriceUniverse
from .state import DailyStateBuilder, empty_state

log = logging.getLogger(__name__)

SLIPPAGE = 0.001          # 10 bps per side
REBALANCE_TOLERANCE = 1.0 # ignore sub-dollar rebalances
LIQUIDITY_FLOOR = 500_000 # min 20-day median $ volume for buys
TBILL_ANNUAL_RATE = 0.045 # ~4.5% APR — proxy for cash earnings under 'tbills' rf mode

# Floor on the close price a ticker must clear before a BUY is even
# considered (sells are never screened). LIQUIDITY_FLOOR alone is not
# enough: a sub-penny name can pass a 20-day median $-volume screen on the
# strength of a few pre-collapse days and then trade a single wick the
# engine happily fills at, position size unbounded, because "buy $X of a
# $0.0002 stock" is Y/0.0002 shares no matter how small $X is. SMFL entered
# at $0.0002 on 2024-09-20 for 34.9M shares and a $2.4M lot; a $1.00 floor
# would have rejected it outright. Overridable by the caller (run_strategy's
# min_price kwarg) -- see backtest.py for how the CLI plumbs it through.
MIN_PRICE_FLOOR = 1.00

# Cap on order SHARES as a fraction of the ticker's own recent median DAILY
# share volume (see PriceUniverse.median_share_volume). LIQUIDITY_FLOOR only
# screens dollar volume, which a name can clear while trading almost no
# shares if its price is high enough (or, per MIN_PRICE_FLOOR above, low
# enough that a handful of dollars still fabricates a huge share order).
# SMFL is the motivating case: it cleared LIQUIDITY_FLOOR on 2024-09-20 (its
# 20-day median $-volume still carried $108M from 2024-09-12), but that
# day's ENTIRE trading volume was 316,699 shares, and the engine bought
# 34,881,540 -- about 110x the day's whole tape -- for a single lot that
# alone produced $17.04M of P&L across the grid (95.4% of one strategy's
# total return). No real order absorbs 10x a stock's daily volume without
# moving the tape far past any price this engine models, so an order sized
# above this fraction is truncated rather than filled in full; a fraction
# small enough to still bind on names that clear LIQUIDITY_FLOOR but trade
# thin volume, without touching a normal liquid name's order size.
MAX_PARTICIPATION_PCT = 0.10


def rf_daily_rate(rf_model: str) -> float:
    """Per-trading-day risk-free rate. 'zero' → 0; 'tbills' → ~4.5% APR."""
    if rf_model == "tbills":
        return (1 + TBILL_ANNUAL_RATE) ** (1 / 252) - 1
    return 0.0


def _rank_score(rank_fn, state: dict) -> float:
    """Turn a Strategy.rank_fn call into a sortable score.

    A rank_fn must never crash a run and must never let a bad score win a
    capacity slot. So a raised exception, a None result, or a NaN result all
    map to negative infinity, which sorts last in descending order. NaN is
    truthy in Python and equals nothing, including itself, so it must be
    checked with `score != score`, never with a bare truthiness test.
    """
    try:
        score = rank_fn(state)
    except Exception:
        log.warning("rank_fn raised for ticker %s; ranking it last.",
                    state.get("ticker"), exc_info=True)
        return float("-inf")
    if score is None:
        return float("-inf")
    try:
        score = float(score)
    except (TypeError, ValueError):
        log.warning("rank_fn returned a non-numeric score for ticker %s; "
                    "ranking it last.", state.get("ticker"))
        return float("-inf")
    if score != score:  # NaN check. NaN never equals itself.
        return float("-inf")
    return score


def _rank_capacity_order(
    pending: list["Order"],
    rank_fn,
    prices: "PriceUniverse",
    day: date,
    portfolio: "Portfolio",
) -> list["Order"]:
    """Reassign buy slots in pending by descending rank_fn score.

    Only buy orders (target above current holding) move. A sell or trim
    order, and a buy order that cannot be priced today, stay at their
    original index, so the relative timing between a sell that frees a slot
    and a buy that wants one is preserved exactly. Only WHICH ticker fills
    each buy slot changes. Ties break on ticker so the order is the same on
    every run.

    This mirrors, but does not replace, the current-vs-target computation
    that step 2 repeats when it later executes each order. That repeat is
    still authoritative. This pass only decides priority among orders that
    already look like a buy against the portfolio as it stands before step 2
    starts executing anything, and each ticker appears at most once in
    pending, so no earlier order in this same pass can change the reading
    for a later one.
    """
    buy_positions: list[int] = []
    buy_orders: list["Order"] = []
    for i, order in enumerate(pending):
        px = prices.open(order.ticker, day)
        if px is None:
            continue  # Unpriceable today; step 2 skips it regardless of order.
        current = sum(l.shares * px for l in portfolio.lots.get(order.ticker, []))
        if order.target_dollars - current > REBALANCE_TOLERANCE:
            buy_positions.append(i)
            buy_orders.append(order)

    if len(buy_orders) <= 1:
        return pending  # Nothing to reorder.

    buy_orders.sort(key=lambda o: (-_rank_score(rank_fn, o.state), o.ticker))

    ranked = list(pending)
    for pos, order in zip(buy_positions, buy_orders):
        ranked[pos] = order
    return ranked


@dataclass
class Lot:
    ticker: str
    entry_date: date
    entry_idx: int
    expiry_idx: int
    shares: float
    entry_price: float       # post-slippage
    cost_basis: float        # = shares * entry_price at open
    score_at_entry: int
    decision_date: date
    peak_price: float = 0.0  # running high-water-mark close, for trailing stops


@dataclass
class Trade:
    strategy: str
    exit_label: str
    ticker: str
    entry_date: date
    exit_date: date
    shares: float
    entry_price: float
    exit_price: float
    cost_basis: float
    proceeds: float
    pnl: float
    return_pct: float
    score_at_entry: int
    days_held: int               # calendar days
    trading_days_held: int       # SPY trading days (matches exit_days units)
    exit_reason: str             # 'expiry' | 'trim' | 'final_liquidation' | 'stop_loss' | 'trailing_stop'
    delisted: bool = False


@dataclass
class Order:
    ticker: str
    target_dollars: float
    decided_on: date
    state: dict


@dataclass
class RunResult:
    strategy: str
    exit_label: str
    equity_curve: pd.Series             # indexed by date, in dollars
    trades: list[Trade]
    skips: Counter
    n_rebalances: int
    exposure_curve: pd.Series           # daily invested $ / NAV
    n_tickers_curve: pd.Series          # distinct tickers held


class Portfolio:
    def __init__(self, starting_capital: float) -> None:
        self.cash = float(starting_capital)
        self.starting_cash = float(starting_capital)
        self.lots: dict[str, list[Lot]] = defaultdict(list)

    def open_tickers(self) -> set[str]:
        return {t for t, ls in self.lots.items() if ls}

    def total_shares(self, ticker: str) -> float:
        return sum(l.shares for l in self.lots.get(ticker, []))

    def buy(self, *, ticker: str, dollars: float, price_open: float,
            today: date, today_idx: int, expiry_days: int,
            score: int, decision_date: date) -> None:
        exec_price = price_open * (1 + SLIPPAGE)
        shares = dollars / exec_price
        self.cash -= dollars
        self.lots[ticker].append(Lot(
            ticker=ticker,
            entry_date=today,
            entry_idx=today_idx,
            expiry_idx=today_idx + expiry_days,
            shares=shares,
            entry_price=exec_price,
            cost_basis=dollars,
            score_at_entry=score,
            decision_date=decision_date,
            peak_price=exec_price,
        ))

    def sell_fifo(self, *, ticker: str, dollars: float, price_open: float,
                  today: date, today_idx: int, strategy: str, exit_label: str,
                  exit_reason: str = "trim",
                  delisted: bool = False) -> list[Trade]:
        if not self.lots.get(ticker):
            return []
        exec_price = price_open * (1 - SLIPPAGE)
        trades: list[Trade] = []
        remaining = dollars
        i = 0
        while remaining > REBALANCE_TOLERANCE and i < len(self.lots[ticker]):
            lot = self.lots[ticker][i]
            lot_value = lot.shares * exec_price
            if lot_value <= remaining + REBALANCE_TOLERANCE:
                # Sell entire lot
                proceeds = lot_value
                self.cash += proceeds
                trades.append(self._make_trade(
                    lot, exec_price, today, today_idx, proceeds, lot.shares,
                    lot.cost_basis, strategy, exit_label, exit_reason, delisted,
                ))
                del self.lots[ticker][i]
                remaining -= proceeds
            else:
                shares_to_sell = remaining / exec_price
                proceeds = shares_to_sell * exec_price
                self.cash += proceeds
                proportion = shares_to_sell / lot.shares
                sold_cb = lot.cost_basis * proportion
                trades.append(self._make_trade(
                    lot, exec_price, today, today_idx, proceeds, shares_to_sell,
                    sold_cb, strategy, exit_label, exit_reason, delisted,
                ))
                lot.shares -= shares_to_sell
                lot.cost_basis -= sold_cb
                remaining = 0
        if not self.lots[ticker]:
            self.lots.pop(ticker, None)
        return trades

    def force_close_lot(self, *, ticker: str, lot: Lot, price_open: float,
                        today: date, today_idx: int, strategy: str, exit_label: str,
                        exit_reason: str, delisted: bool = False) -> Trade:
        exec_price = price_open * (1 - SLIPPAGE)
        proceeds = lot.shares * exec_price
        self.cash += proceeds
        trade = self._make_trade(
            lot, exec_price, today, today_idx, proceeds, lot.shares,
            lot.cost_basis, strategy, exit_label, exit_reason, delisted,
        )
        self.lots[ticker].remove(lot)
        if not self.lots[ticker]:
            self.lots.pop(ticker, None)
        return trade

    @staticmethod
    def _make_trade(lot: Lot, exit_price: float, exit_date: date, exit_idx: int,
                    proceeds: float, shares_sold: float, cost_basis: float,
                    strategy: str, exit_label: str, exit_reason: str,
                    delisted: bool) -> Trade:
        pnl = proceeds - cost_basis
        return_pct = (pnl / cost_basis) if cost_basis > 0 else 0.0
        return Trade(
            strategy=strategy,
            exit_label=exit_label,
            ticker=lot.ticker,
            entry_date=lot.entry_date,
            exit_date=exit_date,
            shares=shares_sold,
            entry_price=lot.entry_price,
            exit_price=exit_price,
            cost_basis=cost_basis,
            proceeds=proceeds,
            pnl=pnl,
            return_pct=return_pct,
            score_at_entry=lot.score_at_entry,
            days_held=(exit_date - lot.entry_date).days,
            trading_days_held=max(0, exit_idx - lot.entry_idx),
            exit_reason=exit_reason,
            delisted=delisted,
        )


def run_strategy(
    *,
    strategy: strat_mod.Strategy,
    exit_method: strat_mod.ExitMethod,
    starting_capital: float,
    calendar: list[date],
    prices: PriceUniverse,
    states: DailyStateBuilder,
    rf_model: str = "zero",
    min_price: float = MIN_PRICE_FLOOR,
) -> RunResult:
    portfolio = Portfolio(starting_capital)
    pending: list[Order] = []
    trades: list[Trade] = []
    skips: Counter = Counter()
    equity = []
    exposure = []
    n_tickers = []
    n_rebalances = 0
    rf_daily = rf_daily_rate(rf_model)

    # A fixed-hold strategy wants every lot to live hold_days trading days no
    # matter what the signal does. exit_method.exit_days still applies as an
    # upper safety cap, so a researcher who picks a short exit method (e.g.
    # "30d") gets that ceiling honored across every strategy in the grid,
    # fixed-hold included, rather than a strategy silently overriding it.
    # The trade-off: a hold_days strategy only realizes its full intended
    # hold when paired with an exit method whose exit_days is at least as
    # long (see backtest/strategies.py EXIT_METHODS — everything except
    # "30d" clears hold_days=63).
    if strategy.hold_days is not None and exit_method.exit_days < strategy.hold_days:
        # The clamp below is intentional (some sweeps legitimately want a
        # short exit even for a fixed-hold strategy), but it is also easy
        # to trigger by accident -- e.g. pairing a hold_days=63 strategy
        # meant to let a multi-month drift signal play out against the
        # "30d" entry in EXIT_METHODS silently cuts that hold to 30 days
        # and the effect the strategy exists to measure never gets a
        # chance to show up. Loud, not fatal: log and keep going.
        log.warning(
            "Strategy %r requested hold_days=%d but exit_method %r caps lots "
            "at exit_days=%d; the clamp min(hold_days, exit_days) will hold "
            "every lot for only %d trading days, not the requested %d.",
            strategy.name, strategy.hold_days, exit_method.label,
            exit_method.exit_days, exit_method.exit_days, strategy.hold_days,
        )

    lot_expiry_days = (
        min(strategy.hold_days, exit_method.exit_days)
        if strategy.hold_days is not None
        else exit_method.exit_days
    )

    for idx, D in enumerate(calendar):
        # 0. Accrue risk-free interest on idle cash overnight (skip first day)
        if idx > 0 and rf_daily > 0:
            portfolio.cash *= 1 + rf_daily

        # 1a. Stop-loss exits (checked before time-expiry so a same-day expiry
        # that breaches the stop is recorded as stop_loss, not expiry).
        if strategy.stop_loss_pct is not None and idx > 0:
            stop_pct = float(strategy.stop_loss_pct)
            prev_d = calendar[idx - 1]
            for ticker in list(portfolio.lots.keys()):
                ref_px = prices.close(ticker, prev_d)
                if ref_px is None:
                    continue
                for lot in list(portfolio.lots[ticker]):
                    if lot.entry_price <= 0:
                        continue
                    if ref_px / lot.entry_price - 1.0 <= -stop_pct:
                        px = prices.open(ticker, D)
                        delisted = False
                        if px is None:
                            px = prices.last_close_on_or_before(ticker, D)
                            delisted = px is not None and prices.is_past_last_bar(ticker, D)
                        if px is None:
                            # No price at all: write off at zero (rare).
                            trades.append(Portfolio._make_trade(
                                lot, 0.0, D, idx, 0.0, lot.shares, lot.cost_basis,
                                strategy.name, exit_method.label, "stop_loss", True,
                            ))
                            portfolio.lots[ticker].remove(lot)
                            if not portfolio.lots[ticker]:
                                portfolio.lots.pop(ticker, None)
                        else:
                            trades.append(portfolio.force_close_lot(
                                ticker=ticker, lot=lot, price_open=px, today=D,
                                today_idx=idx, strategy=strategy.name,
                                exit_label=exit_method.label,
                                exit_reason="stop_loss", delisted=delisted,
                            ))

        # 1b. Trailing-stop exits (independent of 1a's fixed stop-loss; re-reads
        # portfolio.lots so lots already closed in 1a aren't double-processed).
        if exit_method.trailing_stop_pct is not None and idx > 0:
            trail_pct = float(exit_method.trailing_stop_pct)
            prev_d = calendar[idx - 1]
            for ticker in list(portfolio.lots.keys()):
                ref_px = prices.close(ticker, prev_d)
                if ref_px is None:
                    continue
                for lot in list(portfolio.lots[ticker]):
                    lot.peak_price = max(lot.peak_price, ref_px)
                    if lot.peak_price <= 0:
                        continue
                    if ref_px / lot.peak_price - 1.0 <= -trail_pct:
                        px = prices.open(ticker, D)
                        delisted = False
                        if px is None:
                            px = prices.last_close_on_or_before(ticker, D)
                            delisted = px is not None and prices.is_past_last_bar(ticker, D)
                        if px is None:
                            # No price at all: write off at zero (rare).
                            trades.append(Portfolio._make_trade(
                                lot, 0.0, D, idx, 0.0, lot.shares, lot.cost_basis,
                                strategy.name, exit_method.label, "trailing_stop", True,
                            ))
                            portfolio.lots[ticker].remove(lot)
                            if not portfolio.lots[ticker]:
                                portfolio.lots.pop(ticker, None)
                        else:
                            trades.append(portfolio.force_close_lot(
                                ticker=ticker, lot=lot, price_open=px, today=D,
                                today_idx=idx, strategy=strategy.name,
                                exit_label=exit_method.label,
                                exit_reason="trailing_stop", delisted=delisted,
                            ))

        # 1. Expire lots that hit their H-day clock today
        for ticker in list(portfolio.lots.keys()):
            for lot in list(portfolio.lots[ticker]):
                if idx >= lot.expiry_idx:
                    px = prices.open(ticker, D)
                    delisted = False
                    if px is None:
                        px = prices.last_close_on_or_before(ticker, D)
                        # Only flag delisted if we're actually past the ticker's
                        # last available bar — not just a single-day gap/halt.
                        delisted = px is not None and prices.is_past_last_bar(ticker, D)
                    if px is None:
                        # Truly unpriceable; write off at zero
                        portfolio.cash += 0.0
                        trades.append(Portfolio._make_trade(
                            lot, 0.0, D, idx, 0.0, lot.shares, lot.cost_basis,
                            strategy.name, exit_method.label, "expiry", True,
                        ))
                        portfolio.lots[ticker].remove(lot)
                        if not portfolio.lots[ticker]:
                            portfolio.lots.pop(ticker, None)
                    else:
                        trades.append(portfolio.force_close_lot(
                            ticker=ticker, lot=lot, price_open=px, today=D,
                            today_idx=idx, strategy=strategy.name,
                            exit_label=exit_method.label,
                            exit_reason="expiry", delisted=delisted,
                        ))

        # 2. Execute pending orders decided yesterday at today's open. When
        # strategy.rank_fn is None, pending is executed in exactly its
        # queued (arrival) order, unchanged, so every strategy without a
        # rank_fn stays bit-for-bit identical to before rank_fn existed.
        if strategy.rank_fn is not None:
            pending = _rank_capacity_order(pending, strategy.rank_fn, prices, D, portfolio)
        for order in pending:
            px = prices.open(order.ticker, D)
            if px is None:
                skips["no_price"] += 1
                continue
            # Recompute current vs target at execution time (some lots may have
            # expired in step 1; the target stays as decided yesterday).
            current = sum(l.shares * px for l in portfolio.lots.get(order.ticker, []))
            delta = order.target_dollars - current
            if abs(delta) <= REBALANCE_TOLERANCE:
                continue
            if delta > 0:
                if (order.ticker not in portfolio.lots
                        and strategy.max_concurrent_tickers is not None
                        and len(portfolio.lots) >= strategy.max_concurrent_tickers):
                    skips["capacity"] += 1
                    if strategy.rank_fn is not None:
                        # Diagnostics for "what did we turn away, and was it
                        # better than what we held": accumulate sum/count/max
                        # of the rank score of every rejected candidate, kept
                        # in the same skips Counter as the other skip
                        # reasons. Report code divides sum by n for the mean.
                        # A candidate whose score failed (see _rank_score)
                        # is excluded from sum/max so one bad score cannot
                        # drag the mean to negative infinity.
                        score = _rank_score(strategy.rank_fn, order.state)
                        if score != float("-inf"):
                            skips["capacity_rank_score_sum"] += score
                            skips["capacity_rank_score_n"] += 1
                            skips["capacity_rank_score_max"] = max(
                                skips.get("capacity_rank_score_max", float("-inf")),
                                score,
                            )
                    continue
                buy_amount = min(delta, portfolio.cash)
                if buy_amount < REBALANCE_TOLERANCE:
                    skips["cash"] += 1
                    continue
                # Participation cap: truncate (never reject outright) an
                # order whose share count would exceed MAX_PARTICIPATION_PCT
                # of the ticker's own recent median daily share volume. This
                # is a check LIQUIDITY_FLOOR cannot do, since that screens
                # dollar volume decided the day before execution -- see the
                # SMFL case documented at MAX_PARTICIPATION_PCT's definition
                # above. exec_price mirrors Portfolio.buy's own slippage math
                # so the share count checked here is the share count that
                # would actually be booked.
                exec_price = px * (1 + SLIPPAGE)
                msv = prices.median_share_volume(order.ticker, D)
                if msv is not None:
                    max_shares = msv * MAX_PARTICIPATION_PCT
                    wanted_shares = buy_amount / exec_price
                    if wanted_shares > max_shares:
                        capped_amount = max_shares * exec_price
                        if capped_amount < REBALANCE_TOLERANCE:
                            # Capped down to next to nothing -- not worth
                            # booking a lot over, so skip it entirely rather
                            # than execute a dust-sized trade.
                            skips["participation"] += 1
                            continue
                        skips["participation_capped"] += 1
                        buy_amount = capped_amount
                portfolio.buy(
                    ticker=order.ticker, dollars=buy_amount, price_open=px,
                    today=D, today_idx=idx, expiry_days=lot_expiry_days,
                    score=int(order.state.get("conviction_score", 0)),
                    decision_date=order.decided_on,
                )
                n_rebalances += 1
            else:
                if strategy.hold_days is not None:
                    # Fixed-hold strategies never trim on a target decrease,
                    # signal decay or otherwise. sell_fifo has no other
                    # caller, so this is the one place a "trim" trade can be
                    # created; skipping it here is enough to guarantee a
                    # locked lot only closes via its own expiry (step 1) or
                    # a stop-loss / trailing stop (steps 1a/1b), both of
                    # which run earlier in the day and are untouched by this.
                    continue
                sold = portfolio.sell_fifo(
                    ticker=order.ticker, dollars=abs(delta), price_open=px,
                    today=D, today_idx=idx, strategy=strategy.name,
                    exit_label=exit_method.label,
                    exit_reason="trim",
                )
                trades.extend(sold)
                if sold:
                    n_rebalances += 1
        pending = []

        # 3. Compute today's state and queue tomorrow's orders
        if idx < len(calendar) - 1:
            day_states = dict(states.state_for_day(D))  # copy; we may augment
            # Held tickers whose signal has decayed off today's window must still
            # be evaluated — otherwise the target stays at the last firing value
            # and positions never trim until lot expiry.
            for held_ticker in portfolio.lots:
                if held_ticker not in day_states:
                    day_states[held_ticker] = empty_state(held_ticker, D)

            # 3a. Inject price-derived signals into every state up front — this
            # must happen before any target_fn runs, including the equal-weight
            # divisor pass below. No-lookahead: price_signals uses only closes
            # strictly before D.
            for ticker, st in day_states.items():
                sig = prices.price_signals(ticker, D)
                st["momentum_20d"] = sig["momentum_20d"]
                st["vol_30d"] = sig["vol_30d"]
                st["dist_from_high_90d"] = sig["dist_from_high_90d"]

            # 3b. Raw per-ticker targets, then the equal-weight divisor
            # (count of qualifying tickers today).
            raw_targets = {
                ticker: strategy.target_fn(st, starting_capital)
                for ticker, st in day_states.items()
            }
            n_qualifying = 0
            if strategy.equal_weight:
                n_qualifying = sum(1 for v in raw_targets.values() if v > 0)

            # 3c. Queue tomorrow's orders using the (possibly divided) target.
            for ticker, st in day_states.items():
                raw = raw_targets[ticker]
                if strategy.equal_weight:
                    target = (raw / n_qualifying) if n_qualifying > 0 else 0.0
                else:
                    target = raw
                cur_close = prices.close(ticker, D)
                if cur_close is None and target > 0:
                    # Can't even price it today — skip new buys
                    if ticker not in portfolio.lots:
                        skips["no_price"] += 1
                    continue
                price_for_current = cur_close or 0.0
                current = sum(l.shares * price_for_current
                              for l in portfolio.lots.get(ticker, []))
                delta = target - current
                if abs(delta) <= REBALANCE_TOLERANCE:
                    continue
                if delta < 0 and strategy.hold_days is not None and ticker in portfolio.lots:
                    # The signal decayed (or an equal-weight divisor shift)
                    # pulled the target below what is locked in. Don't even
                    # queue the order: step 2 would refuse to act on it
                    # anyway, and skipping it here keeps skip counters (e.g.
                    # no_price) from counting a no-op trim that was never
                    # going to execute.
                    continue
                if delta > 0:
                    if min_price > 0 and cur_close is not None and cur_close < min_price:
                        if ticker not in portfolio.lots:
                            skips["price_floor"] += 1
                        continue
                    mdv = prices.median_dollar_volume(ticker, D)
                    if mdv is not None and mdv < LIQUIDITY_FLOOR:
                        skips["liquidity"] += 1
                        continue
                pending.append(Order(
                    ticker=ticker, target_dollars=target,
                    decided_on=D, state=st,
                ))

        # 4. Mark to market at close
        mtm = 0.0
        for ticker, lot_list in portfolio.lots.items():
            px = prices.close(ticker, D) or prices.last_close_on_or_before(ticker, D) or 0.0
            for lot in lot_list:
                mtm += lot.shares * px
        nav = portfolio.cash + mtm
        equity.append((D, nav))
        exposure.append((D, mtm / nav if nav > 0 else 0.0))
        n_tickers.append((D, len(portfolio.lots)))

    # End: liquidate any remaining lots at last available price
    last_day = calendar[-1]
    last_idx = len(calendar) - 1
    for ticker in list(portfolio.lots.keys()):
        for lot in list(portfolio.lots[ticker]):
            px = prices.last_close_on_or_before(ticker, last_day)
            delisted = px is not None and prices.is_past_last_bar(ticker, last_day)
            if px is None:
                trades.append(Portfolio._make_trade(
                    lot, 0.0, last_day, last_idx, 0.0, lot.shares, lot.cost_basis,
                    strategy.name, exit_method.label, "final_liquidation", True,
                ))
                portfolio.lots[ticker].remove(lot)
            else:
                trades.append(portfolio.force_close_lot(
                    ticker=ticker, lot=lot, price_open=px, today=last_day,
                    today_idx=last_idx, strategy=strategy.name,
                    exit_label=exit_method.label,
                    exit_reason="final_liquidation", delisted=delisted,
                ))
        if not portfolio.lots.get(ticker):
            portfolio.lots.pop(ticker, None)

    # Replace last equity value with post-liquidation NAV
    if equity:
        equity[-1] = (equity[-1][0], portfolio.cash)

    eq_series = pd.Series(
        [v for _, v in equity],
        index=pd.DatetimeIndex([d for d, _ in equity]),
        name=f"{strategy.name}_{exit_method.label}",
    )
    ex_series = pd.Series(
        [v for _, v in exposure],
        index=pd.DatetimeIndex([d for d, _ in exposure]),
        name="exposure",
    )
    nt_series = pd.Series(
        [v for _, v in n_tickers],
        index=pd.DatetimeIndex([d for d, _ in n_tickers]),
        name="n_tickers",
    )
    return RunResult(
        strategy=strategy.name,
        exit_label=exit_method.label,
        equity_curve=eq_series,
        trades=trades,
        skips=skips,
        n_rebalances=n_rebalances,
        exposure_curve=ex_series,
        n_tickers_curve=nt_series,
    )


def run_spy_baseline(
    starting_capital: float, calendar: list[date], prices: PriceUniverse
) -> RunResult:
    """Buy SPY at calendar[0]'s open, hold to calendar[-1]'s close."""
    first = calendar[0]
    last = calendar[-1]
    px_in = prices.open("SPY", first)
    if px_in is None:
        px_in = prices.close("SPY", first)
    exec_in = px_in * (1 + SLIPPAGE)
    shares = starting_capital / exec_in
    cost_basis = starting_capital

    equity = []
    for D in calendar:
        px = prices.close("SPY", D) or prices.last_close_on_or_before("SPY", D) or exec_in
        equity.append((D, shares * px))

    px_out = prices.close("SPY", last) or prices.last_close_on_or_before("SPY", last)
    exec_out = px_out * (1 - SLIPPAGE)
    proceeds = shares * exec_out
    final_trade = Trade(
        strategy="spy_buy_and_hold",
        exit_label="n/a",
        ticker="SPY",
        entry_date=first,
        exit_date=last,
        shares=shares,
        entry_price=exec_in,
        exit_price=exec_out,
        cost_basis=cost_basis,
        proceeds=proceeds,
        pnl=proceeds - cost_basis,
        return_pct=(proceeds - cost_basis) / cost_basis if cost_basis > 0 else 0.0,
        score_at_entry=0,
        days_held=(last - first).days,
        trading_days_held=len(calendar) - 1,
        exit_reason="final_liquidation",
        delisted=False,
    )
    equity[-1] = (last, proceeds)
    eq = pd.Series(
        [v for _, v in equity],
        index=pd.DatetimeIndex([d for d, _ in equity]),
        name="spy_buy_and_hold",
    )
    ex = pd.Series(1.0, index=eq.index, name="exposure")
    nt = pd.Series(1, index=eq.index, name="n_tickers")
    return RunResult(
        strategy="spy_buy_and_hold",
        exit_label="n/a",
        equity_curve=eq,
        trades=[final_trade],
        skips=Counter(),
        n_rebalances=0,
        exposure_curve=ex,
        n_tickers_curve=nt,
    )


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([asdict(t) for t in trades])
