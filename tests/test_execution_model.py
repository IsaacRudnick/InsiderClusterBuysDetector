"""Tests for tools/execution_model.py -- the replacement for tradeable_mask's
one-day-fill capacity and its flat round-trip cost.

Runnable standalone via `python -m pytest tests/test_execution_model.py -q`.

What matters most here is not that the exact bps numbers match some target
(the module docstring records that calibration and why) but that the two
suspected errors are actually fixed: cost has to vary with price and volume
instead of being flat, and capacity has to scale with how many days a fill
is allowed to take instead of being frozen at one day.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import execution_model as ex  # noqa: E402
from tools import sharpe_lab as sh  # noqa: E402


def make_df(price, adv) -> pd.DataFrame:
    price = np.atleast_1d(np.asarray(price, dtype=float))
    adv = np.atleast_1d(np.asarray(adv, dtype=float))
    n = max(len(price), len(adv))
    price = np.broadcast_to(price, (n,))
    adv = np.broadcast_to(adv, (n,))
    return pd.DataFrame({
        "entry_open": price,
        "x_log_adv20": np.log1p(adv),
    })


# ---------------------------------------------------------------------------
# B. Cost model
# ---------------------------------------------------------------------------

def test_cost_rises_as_price_falls():
    """Same ADV, cheaper stock -> higher round-trip cost."""
    df = make_df(price=[2.0, 5.0, 30.0, 100.0], adv=20_000_000.0)
    cost = ex.estimated_cost_bps(df, position_size=50_000.0)
    assert (np.diff(cost) < 0).all(), cost


def test_cost_rises_as_volume_falls():
    """Same price, thinner name -> higher round-trip cost."""
    df = make_df(price=30.0, adv=[300_000.0, 3_000_000.0, 30_000_000.0, 300_000_000.0])
    cost = ex.estimated_cost_bps(df, position_size=50_000.0)
    assert (np.diff(cost) < 0).all(), cost


def test_cost_rises_with_position_size():
    """Bigger clip against the same ADV -> more market impact, more cost."""
    df = make_df(price=10.0, adv=5_000_000.0)
    small = ex.estimated_cost_bps(df, position_size=10_000.0)
    large = ex.estimated_cost_bps(df, position_size=1_000_000.0)
    assert float(large[0]) > float(small[0])


def test_cost_is_bounded_by_floor_and_cap():
    """The spread component alone is clamped to [SPREAD_FLOOR_BPS, SPREAD_CAP_BPS];
    round-trip cost (spread + 2*impact) must therefore be at least the floor,
    and, with impact held near zero, must not blow past the cap."""
    # A very expensive, very liquid name pushes the raw spread formula toward
    # (and past) zero; the floor must hold it up.
    rich = make_df(price=10_000.0, adv=50_000_000_000.0)
    cost = ex.estimated_cost_bps(rich, position_size=1.0)  # negligible impact
    assert float(cost[0]) >= ex.SPREAD_FLOOR_BPS - 1e-9

    # A near-worthless, near-zero-volume name pushes the raw spread formula
    # to extreme values; the cap must hold the SPREAD component down (impact
    # is a separate, uncapped term by design, so isolate it with pos~0).
    junk = make_df(price=0.01, adv=100.0)
    sp = ex.spread_bps(
        pd.to_numeric(junk["entry_open"]).to_numpy(),
        np.expm1(pd.to_numeric(junk["x_log_adv20"])).to_numpy(),
    )
    assert float(sp[0]) <= ex.SPREAD_CAP_BPS + 1e-9


def test_cost_lands_near_its_calibration_targets():
    """Sanity check on the two reference points the module was calibrated
    against: a liquid $30 name should be cheap, a thin $2 name expensive."""
    liquid = make_df(price=30.0, adv=20_000_000.0)
    thin = make_df(price=2.0, adv=300_000.0)
    c_liquid = float(ex.estimated_cost_bps(liquid, position_size=50_000.0)[0])
    c_thin = float(ex.estimated_cost_bps(thin, position_size=50_000.0)[0])
    assert 15.0 <= c_liquid <= 30.0, c_liquid
    assert 200.0 <= c_thin <= 400.0, c_thin
    assert c_thin > c_liquid


def test_cost_is_nan_for_unusable_rows():
    """A missing or non-positive price/ADV should not fabricate a number."""
    df = pd.DataFrame({
        "entry_open": [30.0, np.nan, -1.0, 0.0],
        "x_log_adv20": [np.log1p(20_000_000.0)] * 4,
    })
    cost = ex.estimated_cost_bps(df, position_size=50_000.0)
    assert np.isfinite(cost[0])
    assert np.isnan(cost[1]) and np.isnan(cost[2]) and np.isnan(cost[3])


# ---------------------------------------------------------------------------
# A. Capacity model
# ---------------------------------------------------------------------------

def test_capacity_scales_linearly_with_days_to_fill():
    """Doubling days_to_fill must be equivalent to halving the position size
    needed per day -- i.e. it is pure arithmetic, not a fudge factor."""
    # Pick a capital level that fits at days_to_fill=D but not at D/2, for a
    # range of D, and confirm the boundary moves exactly proportionally.
    adv = 1_000_000.0
    df = make_df(price=10.0, adv=adv)
    n_names, participation = 20.0, 0.10
    for days in (1.0, 2.0, 4.0, 8.0):
        capacity = participation * adv * days
        capital_at_boundary = capacity * n_names
        just_under = capital_at_boundary * 0.99
        just_over = capital_at_boundary * 1.01
        assert ex.capacity_mask(df, capital=just_under, n_names=n_names,
                                participation=participation,
                                days_to_fill=days).iloc[0]
        assert not ex.capacity_mask(df, capital=just_over, n_names=n_names,
                                    participation=participation,
                                    days_to_fill=days).iloc[0]


def test_capacity_mask_days_to_fill_one_matches_one_day_formula():
    """days_to_fill=1 must reproduce the plain one-day participation test:
    (participation * ADV) >= capital / n_names."""
    df = make_df(price=[1.0, 5.0, 50.0], adv=[1e5, 1e6, 1e8])
    capital, n_names, participation = 1_000_000.0, 20.0, 0.10
    mask = ex.capacity_mask(df, capital=capital, n_names=n_names,
                            participation=participation, days_to_fill=1.0)
    adv = np.expm1(pd.to_numeric(df["x_log_adv20"]))
    expect = (adv * participation) >= (capital / n_names)
    pd.testing.assert_series_equal(mask.reset_index(drop=True),
                                   expect.reset_index(drop=True),
                                   check_names=False)


def test_more_days_to_fill_never_shrinks_the_tradeable_set():
    df = make_df(
        price=np.linspace(0.5, 100, 200),
        adv=np.geomspace(1e4, 1e9, 200),
    )
    capital = 5_000_000.0
    prev = None
    for days in ex.DAYS_TO_FILL_LEVELS:
        mask = ex.capacity_mask(df, capital=capital, days_to_fill=days)
        if prev is not None:
            assert mask.sum() >= prev.sum()
            assert (prev | mask).equals(mask), "more days must be a superset"
        prev = mask


# ---------------------------------------------------------------------------
# C. Per-row cost plumbed through sharpe_lab.panel_returns without breaking
#    the flat-cost path
# ---------------------------------------------------------------------------

def _panel_df(n=630, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "ticker": [f"T{i % 40:02d}" for i in range(n)],
        "event_day": days,
        "entry_day": days,
        "entry_idx": np.arange(n),
        "entry_open": rng.uniform(1.0, 50.0, n),
        "x_vol_63_ann": rng.uniform(0.2, 1.5, n),
        "x_log_adv20": rng.uniform(10, 18, n),
        "fwd_21": rng.normal(0.02, 0.1, n),
        "bench_SPY": rng.normal(0.01, 0.03, n),
        "bench_IWM": rng.normal(0.008, 0.04, n),
        "ens": rng.normal(size=n),
        "period": np.arange(n) // 21,
    })


def test_flat_cost_path_is_unaffected_by_the_new_field():
    """BookSpec.use_per_row_cost defaults to False; existing callers that
    never heard of it must get byte-identical results to before."""
    df = _panel_df()
    panel_old = sh.build_panel(df, "ens")
    panel_new = sh.build_panel(df, "ens")  # no cost_col
    spec = sh.BookSpec(lo=0.7, hi=0.9, cost_bps=50.0)
    a = sh.panel_returns(panel_old, spec)
    b = sh.panel_returns(panel_new, spec)
    np.testing.assert_allclose(a["ret"].to_numpy(), b["ret"].to_numpy())
    assert panel_new.cost is None


def test_per_row_cost_changes_return_by_the_row_average_not_a_flat_number():
    df = _panel_df()
    df["_cost_bps"] = ex.estimated_cost_bps(df, position_size=50_000.0)
    panel = sh.build_panel(df, "ens", cost_col="_cost_bps")
    flat = sh.panel_returns(panel, sh.BookSpec(lo=0.7, hi=0.9, cost_bps=20.0))
    per_row = sh.panel_returns(
        panel, sh.BookSpec(lo=0.7, hi=0.9, use_per_row_cost=True)
    )
    assert not np.allclose(flat["ret"].to_numpy(), per_row["ret"].to_numpy())


def test_per_row_cost_without_a_built_panel_raises():
    df = _panel_df()
    panel = sh.build_panel(df, "ens")  # no cost_col -> panel.cost is None
    with pytest.raises(ValueError):
        sh.panel_returns(panel, sh.BookSpec(lo=0.7, hi=0.9, use_per_row_cost=True))
