"""Strategy registry. Each strategy has a name, target_fn, and max ticker cap.

target_fn(state, starting_capital) -> desired $ exposure in this ticker today.
The engine compares this to current $ exposure and buys/sells the delta on
the next trading day's open. Lots have a per-strategy fixed H-day life.

State fields target_fn can read (see backtest/state.py):
  num_insiders, conviction_score, includes_director, includes_officer,
  includes_ten_percent_owner, is_recent_ipo, total_value,
  max_pct_of_prior_stake, any_10b5_1, role_mix,
  momentum_20d, vol_30d, dist_from_high_90d  (injected by engine; may be None)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

State = dict
TargetFn = Callable[[State, float], float]


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    max_concurrent_tickers: Optional[int]
    target_fn: TargetFn
    stop_loss_pct: Optional[float] = None
    equal_weight: bool = False
    # When set, a lot entered by this strategy is held for exactly hold_days
    # TRADING days, no matter what the signal does after entry. The rolling
    # cluster window can decay to nothing and the position still is not
    # trimmed. Stops (stop_loss_pct and exit_method's trailing_stop_pct)
    # still apply and can close the lot early: locking only blocks the
    # signal-decay trim path, never a risk exit. None preserves the old
    # behavior exactly: exit_method.exit_days is the lot's whole life.
    hold_days: Optional[int] = None
    # Optional priority score for capacity allocation. A higher score wins a
    # slot first when qualifying candidates outnumber max_concurrent_tickers
    # (see backtest/engine.py step 2). Takes the per-ticker state dict used
    # to decide the order, the same dict target_fn reads. None preserves the
    # old behavior exactly: capacity is granted in queue (arrival) order, so
    # every strategy defined above this field was added stays untouched.
    rank_fn: Optional[Callable[[State], float]] = None


@dataclass(frozen=True)
class ExitMethod:
    label: str
    exit_days: int
    trailing_stop_pct: Optional[float] = None


# ---------------------------------------------------------------------------
# Original strategies
# ---------------------------------------------------------------------------
def _s1_target(state: State, cap: float) -> float:
    # Sanity check: any ≥2-insider window qualifies.
    if state.get("num_insiders", 0) >= 2:
        return 0.02 * cap
    return 0.0


def _s2_target(state: State, cap: float) -> float:
    # Trust the scorer's "conviction" tier; skip recent IPOs (noisy lockup).
    if state.get("conviction_score", 0) >= 3 and not state.get("is_recent_ipo"):
        return 0.05 * cap
    return 0.0


def _s3_target(state: State, cap: float) -> float:
    # 10% holder signals are rare; size up.
    if state.get("includes_ten_percent_owner") and state.get("conviction_score", 0) >= 0:
        return 0.10 * cap
    return 0.0


def _s_model_top_target(state: State, cap: float) -> float:
    # Same >=2-insider gate as _s1_target, but sized for a 10-slot cap
    # instead of a 50-slot one. Every hold63 strategy in this registry holds
    # to cap * weight == 100% deployed: tpo_gated_hold63 is 10 x 10%,
    # all_clusters_hold63 is 50 x 2%, conviction_only_hold63 is 20 x 5%.
    # Reusing _s1_target's 2% at a 10-slot cap would deploy only 20% and
    # leave 80% in cash, so the strategy would show about a fifth of the
    # return of model_ranked_hold63 for a reason that has nothing to do
    # with ranking skill. That would read as "tighter ranking hurts", which
    # is the exact opposite of what this strategy exists to test.
    if state.get("num_insiders", 0) >= 2:
        return 0.10 * cap
    return 0.0


def _s4_target(state: State, cap: float) -> float:
    if state.get("num_insiders", 0) >= 3 and state.get("conviction_score", 0) > 0:
        return 0.04 * cap
    return 0.0


def _s5_target(state: State, cap: float) -> float:
    # Continuous: size scales with score, capped at 8%.
    s = max(0, state.get("conviction_score", 0))
    if s <= 0:
        return 0.0
    return min(0.08 * cap, 0.015 * cap * s)


def _s6_target(state: State, cap: float) -> float:
    if (
        state.get("total_value", 0) >= 500_000
        and state.get("includes_director")
        and state.get("conviction_score", 0) >= 0
    ):
        return 0.05 * cap
    return 0.0


# ---------------------------------------------------------------------------
# New: state-only strategies (require only fields already in state)
# ---------------------------------------------------------------------------
def _officer_director_combo(state: State, cap: float) -> float:
    if (
        state.get("includes_officer")
        and state.get("includes_director")
        and state.get("num_insiders", 0) >= 2
    ):
        return 0.06 * cap
    return 0.0


def _big_stake_increase(state: State, cap: float) -> float:
    max_pct = state.get("max_pct_of_prior_stake")
    if (
        max_pct is not None
        and max_pct >= 25
        and state.get("conviction_score", 0) >= 1
    ):
        return 0.07 * cap
    return 0.0


def _non_10b5_1_only(state: State, cap: float) -> float:
    if (
        state.get("num_insiders", 0) >= 3
        and state.get("conviction_score", 0) > 0
        and not state.get("any_10b5_1")
    ):
        return 0.04 * cap
    return 0.0


# ---------------------------------------------------------------------------
# New: price-derived strategies (need engine-injected momentum/vol fields)
# ---------------------------------------------------------------------------
def _momentum_confirmed_cluster(state: State, cap: float) -> float:
    mom = state.get("momentum_20d")
    if (
        state.get("conviction_score", 0) >= 2
        and mom is not None
        and mom > 0
    ):
        return 0.05 * cap
    return 0.0


def _oversold_cluster_reversion(state: State, cap: float) -> float:
    dist = state.get("dist_from_high_90d")
    mom = state.get("momentum_20d")
    if (
        state.get("conviction_score", 0) >= 2
        and dist is not None
        and dist <= -0.20
        and mom is not None
        and mom > -0.10
    ):
        return 0.05 * cap
    return 0.0


def _low_vol_conviction(state: State, cap: float) -> float:
    vol = state.get("vol_30d")
    if (
        state.get("conviction_score", 0) >= 3
        and vol is not None
        and vol < 0.40
    ):
        return 0.06 * cap
    return 0.0


def _vol_scaled_conviction(state: State, cap: float) -> float:
    s = max(0, state.get("conviction_score", 0))
    if s <= 0:
        return 0.0
    vol = state.get("vol_30d")
    if vol is None:
        return 0.0
    scale = 0.30 / max(vol, 0.30)
    return min(0.08 * cap, 0.02 * cap * s * scale)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
STRATEGIES: list[Strategy] = [
    Strategy(
        name="all_clusters_equal_weight",
        description="Sanity check: 2% on any ≥2-insider rolling cluster.",
        max_concurrent_tickers=50,
        target_fn=_s1_target,
    ),
    Strategy(
        name="conviction_only",
        description="5% when conviction_score ≥ +3 (skip recent IPOs).",
        max_concurrent_tickers=20,
        target_fn=_s2_target,
    ),
    Strategy(
        name="ten_percent_owner_gated",
        description="10% when a 10% holder participates and score ≥ 0.",
        max_concurrent_tickers=10,
        target_fn=_s3_target,
    ),
    Strategy(
        name="multi_insider_positive",
        description="4% when ≥3 distinct insiders and score > 0.",
        max_concurrent_tickers=25,
        target_fn=_s4_target,
    ),
    Strategy(
        name="score_weighted",
        description="Continuous: 1.5% × max(0, score), capped at 8%.",
        max_concurrent_tickers=30,
        target_fn=_s5_target,
    ),
    Strategy(
        name="big_money_director",
        description="5% when rolling $ value ≥ $500k AND a director is involved.",
        max_concurrent_tickers=20,
        target_fn=_s6_target,
    ),
    # ---- New: state-only ----
    Strategy(
        name="officer_director_combo",
        description="6% when an officer AND a director both buy in the window.",
        max_concurrent_tickers=20,
        target_fn=_officer_director_combo,
    ),
    Strategy(
        name="big_stake_increase",
        description="7% when some insider grew prior holding ≥25% and score ≥ 1.",
        max_concurrent_tickers=15,
        target_fn=_big_stake_increase,
    ),
    Strategy(
        name="non_10b5_1_only",
        description="4% when multi-insider positive AND no Rule 10b5-1 plan txs.",
        max_concurrent_tickers=25,
        target_fn=_non_10b5_1_only,
    ),
    # ---- New: price-derived ----
    Strategy(
        name="momentum_confirmed_cluster",
        description="5% when score ≥ +2 AND trailing 20-day return > 0.",
        max_concurrent_tickers=20,
        target_fn=_momentum_confirmed_cluster,
    ),
    Strategy(
        name="oversold_cluster_reversion",
        description="5% when score ≥ +2, price ≥ 20% off 90-day high, 20d > -10%.",
        max_concurrent_tickers=20,
        target_fn=_oversold_cluster_reversion,
    ),
    Strategy(
        name="low_vol_conviction",
        description="6% when score ≥ +3 AND 30-day annualized vol < 40%.",
        max_concurrent_tickers=15,
        target_fn=_low_vol_conviction,
    ),
    Strategy(
        name="vol_scaled_conviction",
        description="Continuous: 2% × score × (0.30 / max(vol_30d, 0.30)), cap 8%.",
        max_concurrent_tickers=30,
        target_fn=_vol_scaled_conviction,
    ),
    # ---- New: stop-loss variants ----
    Strategy(
        name="conviction_only_stopped_15",
        description="conviction_only with a -15% stop-loss per lot.",
        max_concurrent_tickers=20,
        target_fn=_s2_target,
        stop_loss_pct=0.15,
    ),
    Strategy(
        name="multi_insider_stopped_20",
        description="multi_insider_positive with a -20% stop-loss per lot.",
        max_concurrent_tickers=25,
        target_fn=_s4_target,
        stop_loss_pct=0.20,
    ),
    Strategy(
        name="score_weighted_stopped_15",
        description="score_weighted with a -15% stop-loss per lot.",
        max_concurrent_tickers=30,
        target_fn=_s5_target,
        stop_loss_pct=0.15,
    ),
    # ---- New: fixed-hold variants ----
    # The rolling 14-day cluster window decays fast, so the exit-day grid
    # rarely binds: positions get trimmed to 0 within ~7-10 days almost
    # regardless of the exit method, well short of the 90-day forward
    # horizon the signal is actually scored on. These three mirror existing
    # strategies' target_fn 1:1 and add hold_days=63 (~3 trading months) so
    # the position survives the signal's decay and the documented multi-
    # month drift has a chance to show up in the exit-grid results.
    Strategy(
        name="tpo_gated_hold63",
        description="ten_percent_owner_gated (10% when a 10% holder participates "
                     "and score >= 0), held a fixed 63 trading days per lot.",
        max_concurrent_tickers=10,
        target_fn=_s3_target,
        hold_days=63,
    ),
    Strategy(
        name="all_clusters_hold63",
        description="all_clusters_equal_weight (2% on any >=2-insider rolling "
                     "cluster), held a fixed 63 trading days per lot.",
        max_concurrent_tickers=50,
        target_fn=_s1_target,
        hold_days=63,
    ),
    Strategy(
        name="conviction_only_hold63",
        description="conviction_only (5% when conviction_score >= +3, skip recent "
                     "IPOs), held a fixed 63 trading days per lot.",
        max_concurrent_tickers=20,
        target_fn=_s2_target,
        hold_days=63,
    ),
]

# ---------------------------------------------------------------------------
# New: signal-threshold strategies (equal-weight across qualifying tickers)
# ---------------------------------------------------------------------------
THRESHOLDS = list(range(-5, 14))


def _make_threshold_target(threshold: int) -> TargetFn:
    def _target(state: State, cap: float) -> float:
        # num_insiders gate: empty_state (decayed held ticker) has score 0,
        # which would otherwise qualify at negative thresholds and never trim.
        if state.get("num_insiders", 0) >= 2 and state.get("conviction_score", 0) > threshold:
            return cap
        return 0.0
    return _target


def _threshold_label(threshold: int) -> str:
    sign = "m" if threshold < 0 else "p"
    return f"thr_gt_{sign}{abs(threshold):02d}"


for _threshold in THRESHOLDS:
    _label = _threshold_label(_threshold)
    STRATEGIES.append(Strategy(
        name=_label,
        description=f"Equal-weight across all tickers with conviction_score > {_threshold} "
                     f"(and ≥2 insiders).",
        max_concurrent_tickers=None,
        target_fn=_make_threshold_target(_threshold),
        equal_weight=True,
    ))
del _threshold, _label

# ---------------------------------------------------------------------------
# New: learned-signal strategies. `learned_score` is populated by the
# backtest's signal-fit phase (backtest/signal_fit.py) from regression on
# forward returns; it is None whenever that phase did not run, in which case
# these strategies simply stay flat.
# ---------------------------------------------------------------------------
def _make_learned_threshold_target(threshold: int) -> TargetFn:
    def _target(state: State, cap: float) -> float:
        s = state.get("learned_score")
        if s is None:
            return 0.0
        # num_insiders gate: empty_state (decayed held ticker) has score 0,
        # which would otherwise qualify at negative thresholds and never trim.
        if state.get("num_insiders", 0) >= 2 and s > threshold:
            return cap
        return 0.0
    return _target


STRATEGIES.append(Strategy(
    name="learned_gt_m01",
    description="Equal-weight across all tickers with learned_score > -1, i.e. no "
                 "net red flags (and ≥2 insiders); actionable even when the fit "
                 "learns only negative weights.",
    max_concurrent_tickers=None,
    target_fn=_make_learned_threshold_target(-1),
    equal_weight=True,
))
STRATEGIES.append(Strategy(
    name="learned_gt_p00",
    description="Equal-weight across all tickers with learned_score > 0 "
                 "(and ≥2 insiders); weights fit by the backtest's signal-fit phase.",
    max_concurrent_tickers=None,
    target_fn=_make_learned_threshold_target(0),
    equal_weight=True,
))
STRATEGIES.append(Strategy(
    name="learned_gt_p03",
    description="Equal-weight across all tickers with learned_score > 3 "
                 "(and ≥2 insiders); weights fit by the backtest's signal-fit phase.",
    max_concurrent_tickers=None,
    target_fn=_make_learned_threshold_target(3),
    equal_weight=True,
))


def _learned_score_weighted(state: State, cap: float) -> float:
    # Continuous: size scales with the learned score, capped at 8%.
    s = state.get("learned_score")
    if s is None or s <= 0:
        return 0.0
    return min(0.08 * cap, 0.015 * cap * s)


STRATEGIES.append(Strategy(
    name="learned_score_weighted",
    description="Continuous: 1.5% × max(0, learned_score), capped at 8%; "
                 "weights fit by the backtest's signal-fit phase.",
    max_concurrent_tickers=30,
    target_fn=_learned_score_weighted,
))

# ---------------------------------------------------------------------------
# New: apples-to-apples comparisons against ten_percent_owner_gated. That
# strategy dominates the matrix largely because of its concentrated 10%/
# 10-slot fully-invested policy shape, not necessarily because of its
# hand-tuned gate — the learned-score strategies above use diluted
# equal-weight or underinvested sizing, confounding the comparison. These
# two strategies reuse ten_percent_owner_gated's exact policy shape (10% of
# capital, up to 10 concurrent tickers) and swap in a learned gate, so any
# performance delta vs. ten_percent_owner_gated is attributable to the gate
# alone.
# ---------------------------------------------------------------------------
def _learned_tpo_gated(state: State, cap: float) -> float:
    s = state.get("learned_score")
    if (
        state.get("includes_ten_percent_owner")
        and s is not None
        and s >= 0
    ):
        return 0.10 * cap
    return 0.0


STRATEGIES.append(Strategy(
    name="learned_tpo_gated",
    description="10% when a 10% holder participates and the learned mean-fit "
                 "score >= 0 (no learned red flags) — ten_percent_owner_gated "
                 "with the hand-score gate replaced by the learned avoid gate.",
    max_concurrent_tickers=10,
    target_fn=_learned_tpo_gated,
))


def _learned_tail_concentrated(state: State, cap: float) -> float:
    s = state.get("tail_score")
    if (
        state.get("num_insiders", 0) >= 2
        and s is not None
        and s >= 3
    ):
        return 0.10 * cap
    return 0.0


STRATEGIES.append(Strategy(
    name="learned_tail_concentrated",
    description="10% across up to 10 tickers with tail (P(moonshot)) score "
                 ">= 3 — ten_percent_owner_gated's policy shape driven by the "
                 "learned tail score (threshold matched to the tpo flag's "
                 "~1 qualifying cluster/day selectivity).",
    max_concurrent_tickers=10,
    target_fn=_learned_tail_concentrated,
))

# ---------------------------------------------------------------------------
# Rank functions: optional Strategy.rank_fn implementations. Each function
# takes the per-ticker state dict and returns a priority score, where a
# higher score wins a capacity slot first (see backtest/engine.py step 2,
# _rank_score). A strategy opts in by setting rank_fn to one of these, or to
# any other callable with the same signature.
# ---------------------------------------------------------------------------
def rank_by_conviction(state: State) -> float:
    """Rank candidates by conviction_score, highest score first."""
    return float(state.get("conviction_score", 0))


def rank_by_model_score(state: State) -> float:
    """Rank candidates by model_score, highest score first.

    A candidate with no model_score sorts last, not first, so it never wins
    a slot over a scored candidate. This lets a ranking model populate
    model_score for only part of the candidate set while it is under
    development, and still get a well-defined order.
    """
    score = state.get("model_score")
    if score is None:
        return float("-inf")
    score = float(score)
    if score != score:  # NaN never equals itself. Treat it as unscored.
        return float("-inf")
    return score


# ---------------------------------------------------------------------------
# New: model-ranked hold63 variants. Reuse all_clusters_hold63's target_fn
# (_s1_target) and hold_days=63 exactly, adding rank_fn=rank_by_model_score
# so capacity contention is decided by the OOF ranking model's score
# instead of arrival order. model_score is None on every candidate until a
# caller populates DailyStateBuilder via set_model_scores() (see
# backtest/model_scores.py's loader) -- until then, rank_by_model_score
# maps every candidate to the same -inf, so _rank_capacity_order's ticker
# tie-break decides (deterministic, not a crash, just not yet doing
# anything useful).
#
# Defined here (after rank_by_model_score) rather than inside the STRATEGIES
# list literal above, because rank_by_model_score doesn't exist yet at that
# point in the file -- same reason the learned-signal and apples-to-apples
# strategies above are appended near their own target_fns instead of living
# in the literal.
# ---------------------------------------------------------------------------
STRATEGIES.append(Strategy(
    name="model_ranked_hold63",
    description="all_clusters_equal_weight's target_fn (2% on any >=2-insider "
                 "rolling cluster), held a fixed 63 trading days per lot, same "
                 "as all_clusters_hold63, except capacity is granted to the "
                 "highest OOF model score first instead of arrival order.",
    max_concurrent_tickers=50,
    target_fn=_s1_target,
    hold_days=63,
    rank_fn=rank_by_model_score,
))
STRATEGIES.append(Strategy(
    name="model_ranked_top_hold63",
    description="model_ranked_hold63 with max_concurrent_tickers cut from 50 "
                 "to 10 so ranking actually binds: at 50 slots, "
                 "all_clusters_equal_weight's loose >=2-insider gate rarely "
                 "produces more qualifying buys in a day than there are open "
                 "slots, so rank_fn would have nothing to break ties over. 10 "
                 "matches ten_percent_owner_gated's concentrated cap, the "
                 "tightest cap already proven to bind elsewhere in this "
                 "registry (see RESEARCH_NOTES.md, 'Capacity is the hidden "
                 "selector').",
    max_concurrent_tickers=10,
    target_fn=_s_model_top_target,
    hold_days=63,
    rank_fn=rank_by_model_score,
))


# ---------------------------------------------------------------------------
# New: model_ranked_n{NN}_hold63 -- a slot-count sweep between the two data
# points above. model_ranked_top_hold63 (10 slots, 10% weight) beats SPY and
# beats unranked all_clusters_hold63; model_ranked_hold63 (50 slots, 2%
# weight) is WORSE than unranked all_clusters_hold63 despite using the exact
# same rank_fn. That is a claim about where the model's edge lives, not just
# a claim about 10 vs. 50, and a two-point comparison can't tell a real peak
# from a knife-edge coincidence at n=10. This family fills in the curve at
# 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100 slots so the peak (if any) is
# actually visible.
#
# 50, 60, 75, 100 extend the original 5-40 sweep to test breadth on its own
# terms: the pool-wide payoff this registry ranks into is lottery-shaped
# (63-day SPY-adjusted return mean +0.76% but median -2.34%, 43.9% win rate,
# and the skew gets worse with horizon -- 252-day mean +2.58%, median
# -10.53%, win 38.5%; no horizon has a positive median). A handful of slots
# means a handful of chances to land one of the rare large winners; widening
# to 50-100 slots is the direct test of whether more, cheaper draws on that
# same lottery beat a small, concentrated bet on the model's top-ranked
# names. 100 slots is also the point where the sizing rule below implies a
# 1% weight per position -- see VERIFY note below.
#
# SIZING: every hold63 strategy in this registry holds to slots * weight ==
# 100% deployed -- see _s_model_top_target's comment above for the full
# argument. Concretely: if every variant instead reused a fixed weight (say
# _s_model_top_target's 10%), the 40-slot variant would try to deploy 400%
# of capital (capped by cash, so in practice it would just always be
# cash-constrained and full), while the 5-slot variant would deploy only
# 50% and sit half in cash. Either way, moving along the sweep would change
# how much of the portfolio is invested at all, and a Sharpe/return
# difference between two slot counts would be a mix of "ranking quality at
# this depth" and "cash drag," with no way to separate them after the fact.
# Setting weight = 1.0 / slots for every variant holds total deployment
# fixed at 100% across the whole sweep, so the ONLY thing that changes
# between adjacent points is how many (and, via rank_fn, which) candidates
# the model's ranking gets to pick -- exactly the one variable this sweep
# exists to isolate. Reuses the same >=2-insider gate as _s_model_top_target
# and _s1_target; only the sizing constant differs, and it differs solely
# to hold slots * weight == 1.0.
#
# VERIFY (100-slot / 1% weight): checked backtest/engine.py's three buy-side
# screens against a shrinking per-slot weight and none of them break --
# if anything a smaller weight makes every one of them LESS likely to bind,
# not more, because they all gate on the ticker's own price/liquidity or on
# a dollar amount that only gets smaller as slots grow:
#   - MIN_PRICE_FLOOR ($1.00, engine.py ~L52): screens the ticker's own
#     close price, independent of order size. Unaffected by slot count.
#   - MAX_PARTICIPATION_PCT (10% of median daily share volume, engine.py
#     ~L69): truncates orders that are TOO LARGE relative to a name's own
#     volume. A 1% order is smaller than a 10% order, so this caps fewer of
#     them, not more.
#   - REBALANCE_TOLERANCE ($1.00, engine.py ~L39, the closest thing to a
#     minimum-order-size rule this engine has): an order is skipped only if
#     its dollar delta is under $1. At the default BT_CAPITAL of $100,000
#     (backtest.py), a 100-slot/1%-weight order is ~$1,000 -- three orders
#     of magnitude above the $1 tolerance. This would only bite at a
#     starting capital under roughly $100 for a 100-slot strategy, well
#     outside any capital size this project runs with.
# No minimum-order-size rule exists in backtest/engine.py beyond
# REBALANCE_TOLERANCE, and no screen requires a minimum position weight.
# ---------------------------------------------------------------------------
MODEL_RANKED_SLOT_COUNTS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]


def _make_model_ranked_slot_target(weight: float) -> TargetFn:
    def _target(state: State, cap: float) -> float:
        if state.get("num_insiders", 0) >= 2:
            return weight * cap
        return 0.0
    return _target


for _slots in MODEL_RANKED_SLOT_COUNTS:
    _weight = 1.0 / _slots
    _name = f"model_ranked_n{_slots:02d}_hold63"
    STRATEGIES.append(Strategy(
        name=_name,
        description=f"model_ranked_top_hold63's policy shape (>=2-insider gate, "
                     f"63-trading-day fixed hold, capacity ranked by OOF model "
                     f"score) at {_slots} concurrent slots instead of 10, sized "
                     f"at {_weight:.4%} per slot so {_slots} x {_weight:.4%} == "
                     f"100% deployed -- part of the model_ranked_n{{NN}}_hold63 "
                     f"slot-count sweep (see comment above) that fills in the "
                     f"curve between model_ranked_top_hold63 (10 slots, wins) "
                     f"and model_ranked_hold63 (50 slots, loses) to find where "
                     f"the ranking model's edge actually peaks.",
        max_concurrent_tickers=_slots,
        target_fn=_make_model_ranked_slot_target(_weight),
        hold_days=63,
        rank_fn=rank_by_model_score,
    ))
del _slots, _weight, _name

# ---------------------------------------------------------------------------
# New: model_ranked_n{NN}_hold{H} -- short-hold variants. Pool-wide, the
# skew of the SPY-adjusted forward return gets WORSE with horizon (63-day:
# mean +0.76%, median -2.34%, win rate 43.9%; 252-day: mean +2.58%, median
# -10.53%, win rate 38.5%; no horizon has a positive median), but at 10 days
# the payoff is far more symmetric (48.2% win rate). If the 63-day hold in
# the sweep above is itself a large part of why these strategies look
# lottery-shaped -- forcing a lot to sit through the worst of the decay
# before it can exit -- a materially shorter hold should show up as a
# higher win rate and a less negative median at the same slot count.
#
# This family reuses model_ranked_n{NN}_hold63's exact policy shape (the
# same >=2-insider gate via _make_model_ranked_slot_target, the same
# rank_fn=rank_by_model_score, the same slots * weight == 1.0 sizing rule)
# and changes ONLY hold_days, at a representative subset of slot counts (10,
# 25, 50 -- one concentrated, one mid, one broad) rather than the full
# MODEL_RANKED_SLOT_COUNTS sweep, so that slot count (breadth) and hold_days
# (horizon) can each be read as an independent axis instead of a 2D grid
# nobody asked for. hold_days=10 and hold_days=21 are both well under every
# EXIT_METHODS.exit_days value except "30d" (>= 10 and >= 21 in all cases),
# so -- unlike the hold63 family -- none of these ever trip the
# min(hold_days, exit_days) clamp warning in backtest/engine.py.
# ---------------------------------------------------------------------------
MODEL_RANKED_SHORT_HOLD_DAYS = [10, 21]
MODEL_RANKED_SHORT_HOLD_SLOT_COUNTS = [10, 25, 50]

for _hold_days in MODEL_RANKED_SHORT_HOLD_DAYS:
    for _slots in MODEL_RANKED_SHORT_HOLD_SLOT_COUNTS:
        _weight = 1.0 / _slots
        _name = f"model_ranked_n{_slots:02d}_hold{_hold_days}"
        STRATEGIES.append(Strategy(
            name=_name,
            description=f"model_ranked_n{_slots:02d}_hold63's policy shape "
                         f"(>=2-insider gate, capacity ranked by OOF model "
                         f"score, {_weight:.4%} per slot so {_slots} x "
                         f"{_weight:.4%} == 100% deployed) held a fixed "
                         f"{_hold_days} trading days per lot instead of 63 -- "
                         f"part of the short-hold family (see comment above) "
                         f"testing whether a shorter horizon trades better "
                         f"against a payoff whose median only turns solidly "
                         f"negative at longer holds.",
            max_concurrent_tickers=_slots,
            target_fn=_make_model_ranked_slot_target(_weight),
            hold_days=_hold_days,
            rank_fn=rank_by_model_score,
        ))
del _hold_days, _slots, _weight, _name

# ---------------------------------------------------------------------------
# Score-floor variants (model_floor_p50/p75/p90_hold63) were scoped alongside
# the slot-count sweep above but are DELIBERATELY NOT IMPLEMENTED here.
#
# The ask was a percentile floor on model_score computed from data available
# at decision time. target_fn's signature is (state, cap) -- one ticker's
# own state, no view of what any other candidate looks like today and no
# history object. That leaves exactly two ways to get a percentile number,
# and both are broken:
#   1. Precompute one fixed cutoff from the full oof_scores parquet (e.g.
#      np.percentile of every score in the file) and bake it in as a
#      constant, the way THRESHOLDS does for conviction_score. This is NOT
#      analogous to THRESHOLDS: conviction_score's cutoffs are hand-picked
#      integers with no dependency on this run's data, whereas a
#      model-score percentile computed from the whole parquet is a function
#      of every event in the backtest window, including events dated after
#      the decision day being evaluated. A candidate on day 1 would be
#      compared against a threshold informed by scores from day 1000 --
#      whole-history lookahead baked straight into a "fixed" constant.
#   2. Compute an expanding-window percentile (today's candidate vs. only
#      scores dated on-or-before today) inline in target_fn via a mutable
#      closure that accumulates scores as the engine calls it. This
#      requires call-order to equal chronological order, which is only
#      true within a single run_strategy pass -- it breaks the moment two
#      strategy variants (e.g. p50 and p75) or two exit_method reruns over
#      the same strategy share a DailyStateBuilder, and it turns every
#      target_fn in this module from a pure, order-independent function
#      into one with hidden run-to-run and variant-to-variant state, which
#      is a correctness landmine for a file everything else in this
#      registry deliberately keeps stateless.
# A clean version of (2) is buildable -- precompute a per-(ticker,event_day)
# expanding percentile once in backtest/model_scores.py or backtest/state.py
# (the same place model_score itself is forward-filled with an explicit
# no-lookahead bisect, see state.py's _build_state) and expose it as a new
# state field alongside model_score. That is real plumbing work outside
# strategies.py, not a target_fn trick, so it is left undone rather than
# shipped leaky or guessed at.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Exit methods: fixed holding horizons + trailing-stop-only exits (capped at
# 365 trading days as a safety valve).
# ---------------------------------------------------------------------------
EXIT_METHODS: list[ExitMethod] = [
    ExitMethod("30d", 30), ExitMethod("90d", 90),
    ExitMethod("180d", 180), ExitMethod("365d", 365),
    ExitMethod("trail10", 365, 0.10), ExitMethod("trail20", 365, 0.20),
    ExitMethod("trail30", 365, 0.30),
]
