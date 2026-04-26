"""
pricing_app.py
B2B Pricing Analytics & Demand Forecasting Platform
XGBoost elasticity model · Streamlit interface
"""

from __future__ import annotations

# ── Page config — must be the first Streamlit call ────────────────────────
import streamlit as st

st.set_page_config(
    page_title="Pricing Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Standard imports ──────────────────────────────────────────────────────
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Project imports ───────────────────────────────────────────────────────
from combined_data_loader import (
    get_customers,
    get_parts_for_customer,
    load_combined_dataset,
    get_model_ready_df,
)
from elasticity_model import (
    _merge_lookups_and_derive,
    build_lookup_tables,
    train_model,
    price_sweep,
)


# ══════════════════════════════════════════════════════════════════════════
# Cached resource: load data + train model (runs ONCE per server process)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading data and training XGBoost model — please wait…")
def _load_and_train(data_path: str):
    """
    Full pipeline:
      xlsx → clean DataFrame → model-ready → lookup tables
           → feature enrichment → train model

    Returns a 5-tuple cached for the lifetime of the server process.
    Items:
      df          — raw clean DataFrame (all spec columns, snake_case)
      enriched    — post-lookup DataFrame (has all fields price_sweep needs)
      model       — trained XGBRegressor (median / quantile model)
      feature_cols — list[str] of feature column names
      lookups     — dict of lookup DataFrames (for re-enriching new rows)
    """
    df = load_combined_dataset(data_path)
    mdf = get_model_ready_df(df)
    lookups = build_lookup_tables(mdf)
    enriched = _merge_lookups_and_derive(mdf, lookups)
    _, model, feature_cols, _ = train_model(enriched)
    return df, enriched, model, feature_cols, lookups


_DATA_PATH = str(Path(__file__).parent / "synthetic_data.xlsx")
df, enriched, model, feature_cols, lookups = _load_and_train(_DATA_PATH)


# ══════════════════════════════════════════════════════════════════════════
# Module-level constants shared across tabs
# ══════════════════════════════════════════════════════════════════════════

# Chart dark theme
_CHART_BG      = "#0e1117"
_CHART_TMPL    = "plotly_dark"
_CHART_HEIGHT  = 360
_AXIS_FONT     = 11
_TICK_FONT     = 10
_VLINE_BASE    = "#9ca3af"   # grey — current negotiated price
_VLINE_SIM     = "#f97316"   # orange — simulated price
_DOT_BASE      = "#9ca3af"
_DOT_SIM       = "#f97316"

# Scenario table row background colours (dark-theme safe, white text)
_TABLE_ROW_BG = {
    "current":  "#1e293b",   # neutral slate
    "revenue":  "#052e16",   # dark emerald
    "margin":   "#78350f",   # dark amber
    "green":    "#052e16",
    "blue":     "#172554",
    "yellow":   "#422006",
    "red":      "#450a0a",
}

_LAYOUT_COMMON = dict(
    template=_CHART_TMPL,
    height=_CHART_HEIGHT,
    margin=dict(l=60, r=20, t=55, b=60),
    showlegend=False,
    paper_bgcolor=_CHART_BG,
    plot_bgcolor=_CHART_BG,
)


# ══════════════════════════════════════════════════════════════════════════
# Tab 1 — cached price sweep helper
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _get_sweep_tab1(
    customer_id: str, part_id: str, floor_p: float, ceiling_p: float
) -> pd.DataFrame:
    """
    Run price_sweep() over the full slider range [floor_p, ceiling_p] for the
    given customer-part pair. Cached per (customer_id, part_id, floor_p,
    ceiling_p) — recomputed only when the pair selection changes.

    Accesses module-level globals: enriched, model, feature_cols.
    """
    enr_mask = (enriched["CustID"] == customer_id) & (enriched["PartID"] == part_id)
    enr_rows = enriched[enr_mask].sort_values("invoice_date", ascending=False)
    if enr_rows.empty:
        return pd.DataFrame()
    base_row = enr_rows.iloc[0]
    base_px  = float(base_row["unit_retail"])
    if base_px <= 0:
        return pd.DataFrame()
    lo = floor_p   / base_px
    hi = ceiling_p / base_px
    return price_sweep(model, feature_cols, base_row, price_range=(lo, hi), n_points=120)


# ══════════════════════════════════════════════════════════════════════════
# Shared chart helper — vertical reference lines
# ══════════════════════════════════════════════════════════════════════════

def _add_price_vlines(
    fig: go.Figure,
    baseline: float,
    sim_price: float,
    is_sim: bool,
) -> None:
    """Add dashed vertical reference lines for current price and simulated price."""
    fig.add_vline(
        x=baseline,
        line_dash="dash",
        line_color=_VLINE_BASE,
        line_width=1.5,
        annotation_text="Current Negotiated Price",
        annotation_position="top right",
        annotation_font_size=9,
        annotation_font_color=_VLINE_BASE,
    )
    if is_sim:
        fig.add_vline(
            x=sim_price,
            line_dash="dash",
            line_color=_VLINE_SIM,
            line_width=1.5,
            annotation_text="Simulated Price",
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color=_VLINE_SIM,
        )


# ══════════════════════════════════════════════════════════════════════════
# Pair-info helper
# ══════════════════════════════════════════════════════════════════════════

def _get_pair_info(customer_id: str, part_id: str) -> dict:
    """
    Compute display anchors, slider bounds, and the enriched base_row for
    a customer-part pair.

    Uses the most recent transaction in the enriched DataFrame so that
    price_sweep() receives all pre-computed lookup fields it requires.
    """
    raw_mask = (df["customer_id"] == customer_id) & (df["part_id"] == part_id)
    raw_rows = df[raw_mask].sort_values("invoice_date", ascending=False)

    enr_mask = (enriched["CustID"] == customer_id) & (enriched["PartID"] == part_id)
    enr_rows = enriched[enr_mask].sort_values("invoice_date", ascending=False)

    if raw_rows.empty or enr_rows.empty:
        return {}

    r = raw_rows.iloc[0]   # most recent raw row — display columns
    e = enr_rows.iloc[0]   # most recent enriched row — model base_row

    unit_cost = float(r["unit_cost"])
    unit_price_catalogue = float(r["unit_price_catalogue"])
    pair_avg_price_paid = float(r["pair_avg_price_paid"])

    # Slider bounds anchored to training price range for this part
    # Floor: cost floor (can't price below cost)
    # Ceiling: just above the max price the model has ever seen for this part
    part_min = float(e.get("part_min_price", unit_cost))
    part_max = float(e.get("part_max_price", unit_price_catalogue))
    slider_floor = round(max(unit_cost, part_min * 0.90), 2)
    slider_ceiling = round(part_max * 1.10, 2)

    # Safety: ensure baseline is within [floor, ceiling]
    baseline_price = float(np.clip(round(pair_avg_price_paid, 2), slider_floor, slider_ceiling))

    # Competitor: "ever present" means competitor_present == 1 on at least
    # one historical transaction for this pair.
    comp_ever_present = bool((raw_rows["competitor_present"] == 1).any())

    # Show competitor price from the baseline row only if ever present.
    # Use the most recent transaction where competitor was observed.
    comp_rows = raw_rows[raw_rows["competitor_present"] == 1]
    comp_price_display = float(comp_rows.iloc[0]["competitor_price"]) if not comp_rows.empty else None

    return {
        "unit_cost":            unit_cost,
        "unit_price_catalogue": unit_price_catalogue,
        "pair_avg_price_paid":  pair_avg_price_paid,
        "baseline_price":       baseline_price,
        "slider_floor":         slider_floor,
        "slider_ceiling":       slider_ceiling,
        "comp_ever_present":    comp_ever_present,
        "comp_price_display":   comp_price_display,
        "enriched_row":         e,   # pd.Series — passed to price_sweep as base_row
    }


# ══════════════════════════════════════════════════════════════════════════
# Pair history + competitor helpers (used by Tab 2 and Tab 5)
# ══════════════════════════════════════════════════════════════════════════

def _get_pair_history(
    df: pd.DataFrame, customer_id: str, part_id: str
) -> pd.DataFrame:
    """
    Return all historical rows for the selected customer-part pair, sorted by
    invoice_date ascending.  Used by Tab 2 (competitive viability, basket
    context, loyalty) and later by Tab 5 (customer behaviour profile).
    """
    mask = (df["customer_id"] == customer_id) & (df["part_id"] == part_id)
    rows = df[mask].copy()
    rows = rows.sort_values("invoice_date", ascending=True).reset_index(drop=True)
    return rows


def _get_competitor_status(
    df: pd.DataFrame, customer_id: str, part_id: str
) -> dict:
    """
    Summarise competitor presence for this pair.

    Returns:
      has_competitor      — True if any historical row has competitor_present == 1
      comp_price_avg      — mean competitor_price over rows with competitor present, or None
      comp_presence_rate  — percentage of rows with competitor present, 0-100
      latest_comp_price   — most recent competitor_price on a row with competitor present, or None
    """
    hist = _get_pair_history(df, customer_id, part_id)
    if hist.empty:
        return {
            "has_competitor":     False,
            "comp_price_avg":     None,
            "comp_presence_rate": 0.0,
            "latest_comp_price":  None,
        }

    presence = hist["competitor_present"].fillna(0).astype(int)
    has_competitor = bool((presence == 1).any())
    comp_rows = hist[presence == 1]

    comp_price_avg = (
        float(comp_rows["competitor_price"].mean()) if not comp_rows.empty else None
    )
    # ascending sort → last row is most recent
    latest_comp_price = (
        float(comp_rows.iloc[-1]["competitor_price"]) if not comp_rows.empty else None
    )
    comp_presence_rate = float((presence == 1).mean() * 100.0)

    return {
        "has_competitor":     has_competitor,
        "comp_price_avg":     comp_price_avg,
        "comp_presence_rate": comp_presence_rate,
        "latest_comp_price":  latest_comp_price,
    }


# ══════════════════════════════════════════════════════════════════════════
# Tab 3 — portfolio helpers
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _get_all_pairs_for_customer(customer_id: str) -> pd.DataFrame:
    """
    Aggregate one summary row per part this customer has purchased.

    Columns returned:
      PartID, oem_flag, unit_cost, pair_avg_price_paid, unit_price_catalogue,
      competitor_present, latest_comp_price, units_per_month,
      pair_transaction_count, price_percentile_in_part
    """
    cust = df[df["customer_id"] == customer_id].copy()
    if cust.empty:
        return pd.DataFrame()

    # Latest competitor price per part (most recent row where competitor present)
    comp_only   = cust[cust["competitor_present"] == 1].sort_values(
        "invoice_date", ascending=False
    )
    latest_comp = (
        comp_only.groupby("part_id")["competitor_price"]
        .first()
        .rename("latest_comp_price")
    )

    # Latest price_percentile_in_part per part (most recent transaction)
    latest_pct = (
        cust.sort_values("invoice_date", ascending=False)
        .groupby("part_id")["price_percentile_in_part"]
        .first()
        .rename("price_percentile_in_part")
    )

    # Units per month: mean of monthly sums per part
    cust["_ym"] = cust["invoice_date"].dt.to_period("M")
    units_pm = (
        cust.groupby(["part_id", "_ym"])["units_sold"]
        .sum()
        .groupby("part_id")
        .mean()
        .rename("units_per_month")
    )

    # Core aggregation
    def _mode_int(x: pd.Series) -> int:
        m = x.mode()
        return int(m.iloc[0]) if not m.empty else 0

    agg = (
        cust.groupby("part_id")
        .agg(
            oem_flag              = ("is_oem_part",         _mode_int),
            unit_cost             = ("unit_cost",           "mean"),
            pair_avg_price_paid   = ("pair_avg_price_paid", "mean"),
            unit_price_catalogue  = ("unit_price_catalogue","mean"),
            competitor_present    = ("competitor_present",  "max"),
            pair_transaction_count= ("customer_id",         "count"),
        )
        .reset_index()
    )

    agg = agg.merge(latest_comp.reset_index(), on="part_id", how="left")
    agg = agg.merge(units_pm.reset_index(),    on="part_id", how="left")
    agg = agg.merge(latest_pct.reset_index(),  on="part_id", how="left")
    agg = agg.rename(columns={"part_id": "PartID"})
    return agg


@st.cache_data(show_spinner=False)
def _get_sweep_for_pair(customer_id: str, part_id: str) -> pd.DataFrame:
    """
    Run price_sweep() for any customer-part pair.
    Cached per (customer_id, part_id) — same semantics as _get_sweep_tab1
    but exposed for Tab 3 which needs sweeps for all parts simultaneously.

    Accesses module-level globals: enriched, model, feature_cols, df.
    Price range: unit_cost → unit_price_catalogue × 1.5
    """
    enr_mask = (enriched["CustID"] == customer_id) & (enriched["PartID"] == part_id)
    enr_rows = enriched[enr_mask].sort_values("invoice_date", ascending=False)
    if enr_rows.empty:
        return pd.DataFrame()

    base_row = enr_rows.iloc[0]
    base_px  = float(base_row["unit_retail"])
    if base_px <= 0:
        return pd.DataFrame()

    raw_mask = (df["customer_id"] == customer_id) & (df["part_id"] == part_id)
    raw_rows = df[raw_mask].sort_values("invoice_date", ascending=False)
    if raw_rows.empty:
        return pd.DataFrame()

    floor_p   = float(raw_rows.iloc[0]["unit_cost"])
    ceiling_p = float(raw_rows.iloc[0]["unit_price_catalogue"]) * 1.5

    if floor_p <= 0 or ceiling_p <= floor_p:
        lo, hi = 0.80, 1.30
    else:
        lo = max(0.01, floor_p   / base_px)
        hi = max(lo + 0.01, ceiling_p / base_px)

    return price_sweep(model, feature_cols, base_row,
                       price_range=(lo, hi), n_points=80)


def _compute_band_metrics(
    sweep_df: pd.DataFrame,
    unit_cost: float,
    current_price: float,
) -> dict:
    """
    Derive key price band metrics from a sweep DataFrame.

    Parameters
    ----------
    sweep_df      : output of price_sweep()
    unit_cost     : cost per unit for this pair
    current_price : pair_avg_price_paid (historical negotiated average)

    Returns dict with keys:
      floor, min_viable, rev_optimal_price, rev_optimal_revenue,
      margin_optimal_price, current_price, current_revenue, upside
    """
    min_viable = unit_cost / 0.85 if unit_cost > 0 else 0.0

    if sweep_df.empty or unit_cost <= 0:
        return dict(
            floor=unit_cost, min_viable=min_viable,
            rev_optimal_price=current_price, rev_optimal_revenue=0.0,
            margin_optimal_price=current_price,
            current_price=current_price, current_revenue=0.0, upside=0.0,
        )

    prices    = sweep_df["unit_retail"].values
    revenues  = sweep_df["revenue"].values
    gm_dollar = sweep_df["gross_margin"].values

    rv_idx = int(np.argmax(revenues))
    mg_idx = int(np.argmax(gm_dollar))
    rev_optimal_price    = float(prices[rv_idx])
    rev_optimal_revenue  = float(revenues[rv_idx])
    margin_optimal_price = float(prices[mg_idx])

    cp_clamped      = float(np.clip(current_price, prices.min(), prices.max()))
    current_revenue = float(np.interp(cp_clamped, prices, revenues))
    upside          = max(0.0, rev_optimal_revenue - current_revenue)

    return dict(
        floor=unit_cost,
        min_viable=min_viable,
        rev_optimal_price=rev_optimal_price,
        rev_optimal_revenue=rev_optimal_revenue,
        margin_optimal_price=margin_optimal_price,
        current_price=current_price,
        current_revenue=current_revenue,
        upside=upside,
    )


@st.cache_data(show_spinner=False)
def _get_basket_history(customer_id: str, part_id: str, months: int = 12) -> pd.DataFrame:
    """Return last `months` months of transactions for this customer-part pair, ascending."""
    hist = _get_pair_history(df, customer_id, part_id)
    if hist.empty:
        return hist
    cutoff = hist["invoice_date"].max() - pd.DateOffset(months=months)
    return hist[hist["invoice_date"] >= cutoff].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
# Session state helpers
# ══════════════════════════════════════════════════════════════════════════

def _init_pair_state(pair_info: dict) -> None:
    """
    Reset all simulation-related session state keys to the baseline for the
    current pair.  Called on first load and on every customer/part change.
    """
    bp = pair_info["baseline_price"]
    fl = pair_info["slider_floor"]
    cl = pair_info["slider_ceiling"]

    # Clamp to guard against stale out-of-range values
    bp_clamped = float(np.clip(bp, fl, cl))

    st.session_state["sim_price"]       = bp_clamped
    st.session_state["_baseline_price"] = bp_clamped
    st.session_state["_pending_reset"]  = True
    st.session_state["is_simulating"]   = False


def _compute_status(
    sim_price: float, unit_cost: float, comp_ever_present: bool
) -> tuple[str, str]:
    """
    Return (decision_label, color_key) for the scenario status badge.

    Precedence (hard rules first):
      1. Red  — Cannot Compete Profitably: sim_price <= unit_cost
      2. Green — Uncontested — Raise With Confidence: no competitor ever observed for this pair
      3. Yellow — Tight Margin:            gross margin at sim_price < 15 %
      4. Blue  — Viable — Compete and Hold: everything else

    color_key maps to the CSS hex pairs in the badge renderer below.
    """
    if sim_price <= unit_cost:
        return "Cannot Compete Profitably", "red"

    gm = (sim_price - unit_cost) / sim_price if sim_price > 0 else 0.0

    if not comp_ever_present:
        return "Uncontested — Raise With Confidence", "green"
    if gm < 0.15:
        return "Tight Margin — Review Basket", "yellow"
    return "Viable — Compete and Hold", "blue"


# ══════════════════════════════════════════════════════════════════════════
# Bidirectional slider ↔ number_input sync callbacks
# ══════════════════════════════════════════════════════════════════════════

def _on_slider_change() -> None:
    """Slider moved → push new value into the number_input key."""
    new_val = float(st.session_state["_sl_val"])
    st.session_state["_ni_val"]  = new_val
    st.session_state["sim_price"] = new_val


def _on_input_change() -> None:
    """Number input typed → clamp and push into the slider key."""
    info  = st.session_state.get("_pair_info", {})
    fl    = float(info.get("slider_floor", 0.0))
    cl    = float(info.get("slider_ceiling", 9999.0))
    raw   = float(st.session_state["_ni_val"])
    clamped = float(np.clip(round(raw, 2), fl, cl))

    st.session_state["_sl_val"]   = clamped
    st.session_state["_ni_val"]   = clamped   # snap input back to clamped value
    st.session_state["sim_price"] = clamped


# ══════════════════════════════════════════════════════════════════════════
# Simulation banner (rendered at the top of every tab body)
# ══════════════════════════════════════════════════════════════════════════

def render_simulation_banner() -> None:
    """
    Displays a blue info banner when the slider is off its baseline position.
    Hidden when at baseline.  Call this as the first element inside each tab.
    """
    if st.session_state.get("is_simulating", False):
        sim  = st.session_state.get("sim_price", 0.0)
        base = st.session_state.get("_baseline_price", sim)
        st.info(
            f"Simulating **\\${sim:.2f}** — actual negotiated price **\\${base:.2f}**",
            icon="📊",
        )


# ══════════════════════════════════════════════════════════════════════════
# Global Sidebar
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:

    st.markdown("## Pricing Analytics Platform")
    st.caption("Powered by XGBoost elasticity model")
    st.divider()

    # ── 1. Customer dropdown ──────────────────────────────────────────────
    customers    = get_customers(df)
    prev_customer = st.session_state.get("_prev_customer")

    selected_customer = st.selectbox(
        "Customer",
        options=customers,
        key="_cust_sel",
    )

    customer_changed = selected_customer != prev_customer
    if customer_changed:
        st.session_state["_prev_customer"] = selected_customer
        # Wipe previous part tracking so part dropdown resets to index 0
        st.session_state.pop("_prev_part", None)

    # ── 2. Part dropdown (filtered to this customer's purchase history) ───
    parts = get_parts_for_customer(df, selected_customer)

    # Use a customer-scoped key so the widget gets a fresh state on
    # every customer change, guaranteeing the index resets to 0.
    part_key = f"_part_sel_{selected_customer}"
    selected_part = st.selectbox(
        "Part",
        options=parts,
        key=part_key,
    )

    prev_part    = st.session_state.get("_prev_part")
    part_changed = customer_changed or (selected_part != prev_part)
    st.session_state["_prev_part"] = selected_part

    # ── 3. Compute pair info and (re)init state on pair change ────────────
    pair_info = _get_pair_info(selected_customer, selected_part)
    st.session_state["_pair_info"]      = pair_info
    st.session_state["_baseline_price"] = pair_info["baseline_price"]

    if part_changed or "sim_price" not in st.session_state:
        _init_pair_state(pair_info)

    fl = pair_info["slider_floor"]
    cl = pair_info["slider_ceiling"]

    # Safety clamp: ensure widget keys are within this pair's bounds before
    # widgets render (guards against stale values from previous pair).
    for wkey in ("_sl_val", "_ni_val", "sim_price"):
        if wkey in st.session_state:
            st.session_state[wkey] = float(
                np.clip(round(float(st.session_state[wkey]), 2), fl, cl)
            )

    st.divider()

    # ── 4. Price slider ───────────────────────────────────────────────────
    if st.session_state.pop("_pending_reset", False):
        st.session_state["_sl_val"] = st.session_state.get("sim_price", pair_info["baseline_price"])
        st.session_state["_ni_val"] = st.session_state["_sl_val"]

    st.markdown("**Simulated Price**")

    st.slider(
        label="Simulated Price",
        min_value=fl,
        max_value=cl,
        step=0.01,
        format="$%.2f",
        key="_sl_val",
        on_change=_on_slider_change,
        label_visibility="collapsed",
    )

    # ── 5. Manual price input (synced bidirectionally with slider) ────────
    st.number_input(
        label="Enter price manually",
        min_value=fl,
        max_value=cl,
        step=0.01,
        format="%.2f",
        key="_ni_val",
        on_change=_on_input_change,
        label_visibility="collapsed",
    )

    sim_price = float(st.session_state["sim_price"])

    # ── 6. Static reference anchors (muted text) ──────────────────────────
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    comp_price_val = pair_info.get("comp_price_display")
    comp_text = f"${comp_price_val:.2f}" if comp_price_val is not None else "Not Available"

    st.markdown(
        f"""
        <div style="color:#888;font-size:0.84rem;line-height:2.0;padding:2px 0;">
            Our Cost per Unit:&ensp;<strong>${pair_info['unit_cost']:.2f}</strong><br>
            Customer's Historical Price Point:&ensp;<strong>${pair_info['pair_avg_price_paid']:.2f}</strong><br>
            Competitor Price:&ensp;<strong>{comp_text}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 7. Scenario status badge ──────────────────────────────────────────
    status_label, status_color = _compute_status(
        sim_price,
        pair_info["unit_cost"],
        pair_info["comp_ever_present"],
    )

    _BADGE_COLORS: dict[str, tuple[str, str]] = {
        "green":  ("#d1f5d3", "#1a6b2a"),
        "blue":   ("#d0e8ff", "#0a4d8c"),
        "yellow": ("#fff8d6", "#7a5f00"),
        "red":    ("#fde0e0", "#8b1a1a"),
    }
    badge_bg, badge_fg = _BADGE_COLORS.get(status_color, ("#e2e3e5", "#383d41"))

    st.markdown(
        f"""
        <div style="
            background:{badge_bg};
            color:{badge_fg};
            border:1.5px solid {badge_fg}55;
            border-radius:6px;
            padding:8px 12px;
            font-size:0.88rem;
            font-weight:700;
            text-align:center;
            letter-spacing:0.01em;
        ">{status_label}</div>
        """,
        unsafe_allow_html=True,
    )

    # Gross margin at simulated price — informational, below badge
    gm_pct = (
        (sim_price - pair_info["unit_cost"]) / sim_price if sim_price > 0 else 0.0
    )
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.caption(f"Gross margin at simulated price: {gm_pct:.1%}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── 8. Reset button ───────────────────────────────────────────────────
    if st.button("Reset to current negotiated price", use_container_width=True):
        _init_pair_state(pair_info)
        st.rerun()

    # ── 9. Update simulation banner flag ──────────────────────────────────
    is_simulating = abs(sim_price - pair_info["baseline_price"]) > 0.005
    st.session_state["is_simulating"] = is_simulating
    st.session_state["sim_price"]     = sim_price   # persist final value


# ══════════════════════════════════════════════════════════════════════════
# Tab placeholders (Phase 3–7 will fill these)
# ══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Deal Simulator",
    "Competitive Viability",
    "Pricing Bands per SKU",
    "Basket Intelligence",
    "Customer Behavior Profile",
])

with tab1:
    # ── 1. Simulation banner ──────────────────────────────────────────────
    render_simulation_banner()

    # ── Pull live state from session state ───────────────────────────────
    sim_price  = float(st.session_state.get("sim_price", 0.0))
    pair_info  = st.session_state.get("_pair_info", {})
    baseline   = float(st.session_state.get("_baseline_price", sim_price))
    is_sim     = abs(sim_price - baseline) > 0.005

    if not pair_info:
        st.info("Select a customer and part from the sidebar to begin.")
    else:
        unit_cost  = pair_info["unit_cost"]
        floor_p    = pair_info["slider_floor"]
        ceiling_p  = pair_info["slider_ceiling"]

        # ── 2. Price sweep (cached per pair, not per price) ───────────────
        sweep = _get_sweep_tab1(selected_customer, selected_part, floor_p, ceiling_p)

        if sweep.empty:
            st.warning("Price sweep could not be computed for this pair.")
        else:
            prices_arr    = sweep["unit_retail"].values
            qty_arr       = sweep["predicted_quantity"].values
            rev_arr       = sweep["revenue"].values
            gm_pct_arr    = sweep["gross_margin_pct_sweep"].values * 100  # decimal → %
            gm_dollar_arr = sweep["gross_margin"].values

            # Interpolated values at the key prices
            q_sim    = float(np.interp(sim_price, prices_arr, qty_arr))
            q_base   = float(np.interp(baseline,  prices_arr, qty_arr))
            rev_sim  = sim_price * q_sim
            rev_base = baseline  * q_base
            gm_pct_sim  = (sim_price - unit_cost) / sim_price * 100 if sim_price > 0 else 0.0
            gm_pct_base = (baseline  - unit_cost) / baseline  * 100 if baseline  > 0 else 0.0

            # Revenue-optimal — argmax of dollar revenue
            rev_opt_idx   = int(np.argmax(rev_arr))
            rev_opt_price = float(prices_arr[rev_opt_idx])
            rev_opt_qty   = float(qty_arr[rev_opt_idx])
            rev_opt_rev   = float(rev_arr[rev_opt_idx])
            rev_opt_gm    = float(gm_pct_arr[rev_opt_idx])

            # Margin-optimal — argmax of dollar gross margin (accounts for demand loss)
            mgn_opt_idx   = int(np.argmax(gm_dollar_arr))
            mgn_opt_price = float(prices_arr[mgn_opt_idx])
            mgn_opt_qty   = float(qty_arr[mgn_opt_idx])
            mgn_opt_rev   = float(mgn_opt_price * qty_arr[mgn_opt_idx])
            mgn_opt_gm    = float(gm_pct_arr[mgn_opt_idx])
            # GM % at the margin-optimal price (for chart dot position)
            mgn_opt_gm_pct = (mgn_opt_price - unit_cost) / mgn_opt_price * 100 if mgn_opt_price > 0 else 0.0

            # Sidebar status for simulated price (used in table + recommendation)
            status_label, status_color = _compute_status(
                sim_price, unit_cost, pair_info["comp_ever_present"]
            )

            # ── 3. KPI header row ─────────────────────────────────────────
            kc1, kc2, kc3, kc4, kc5 = st.columns(5)
            with kc1:
                st.metric("Current Negotiated Price", f"${baseline:.2f}")
            with kc2:
                delta_val = sim_price - baseline
                st.metric(
                    "Simulated Price",
                    f"${sim_price:.2f}",
                    delta=f"${delta_val:+.2f}" if is_sim else None,
                    delta_color="normal",
                )
            with kc3:
                st.metric("Predicted Demand", f"{q_sim:.1f} units")
                st.caption("XGBoost model estimate")
            with kc4:
                st.metric("Projected Monthly Revenue", f"${rev_sim:.2f}")
            with kc5:
                st.metric("Gross Margin", f"{gm_pct_sim:.1f}%")

            st.divider()

            # ── 4. Single centered revenue curve ──────────────────────────
            # Cost floor revenue: unit_cost × quantity at that price point
            cost_floor_rev_arr = unit_cost * qty_arr

            fig_r = go.Figure()

            # — Cost floor band (shaded region below unit_cost revenue line) —
            fig_r.add_trace(go.Scatter(
                x=prices_arr, y=cost_floor_rev_arr,
                mode="lines",
                line=dict(color="#ef4444", width=1.5, dash="dot"),
                name="Cost Floor",
                hovertemplate=(
                    "Price: $%{x:.2f}<br>"
                    "Revenue at Cost: $%{y:.2f}<br>"
                    "<i>Pricing below this line loses money</i>"
                    "<extra></extra>"
                ),
            ))

            # Shaded no-go zone between zero and cost floor line
            fig_r.add_trace(go.Scatter(
                x=list(prices_arr) + list(prices_arr[::-1]),
                y=list(cost_floor_rev_arr) + [0] * len(prices_arr),
                fill="toself",
                fillcolor="rgba(239,68,68,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))

            # — Revenue curve —
            fig_r.add_trace(go.Scatter(
                x=prices_arr, y=rev_arr,
                mode="lines",
                line=dict(color="#22c55e", width=2.5),
                name="Projected Revenue",
                hovertemplate=(
                    "Price: $%{x:.2f}<br>"
                    "Projected Monthly Revenue: $%{y:.2f}"
                    "<extra></extra>"
                ),
            ))

            # — Revenue-optimal diamond (no annotation box — kept clean) —
            fig_r.add_trace(go.Scatter(
                x=[rev_opt_price], y=[rev_opt_rev],
                mode="markers",
                name=f"Revenue Optimal: ${rev_opt_price:.2f}",
                marker=dict(color="#22c55e", size=13, symbol="diamond",
                            line=dict(color="white", width=1.5)),
                hovertemplate=(
                    f"Revenue-Optimal Price: ${rev_opt_price:.2f}<br>"
                    f"Projected Monthly Revenue: ${rev_opt_rev:.2f}"
                    "<extra></extra>"
                ),
            ))

            # — Current negotiated price dot —
            fig_r.add_trace(go.Scatter(
                x=[baseline], y=[rev_base],
                mode="markers",
                name=f"Current Price: ${baseline:.2f}",
                marker=dict(color=_DOT_BASE, size=10,
                            line=dict(color="white", width=1)),
                hovertemplate=(
                    f"Current Negotiated Price: ${baseline:.2f}<br>"
                    f"Projected Monthly Revenue: ${rev_base:.2f}"
                    "<extra></extra>"
                ),
            ))

            # — Simulated price dot (only when slider has moved) —
            if is_sim:
                fig_r.add_trace(go.Scatter(
                    x=[sim_price], y=[rev_sim],
                    mode="markers",
                    name=f"Simulated Price: ${sim_price:.2f}",
                    marker=dict(color=_DOT_SIM, size=10,
                                line=dict(color="white", width=1)),
                    hovertemplate=(
                        f"Simulated Price: ${sim_price:.2f}<br>"
                        f"Projected Monthly Revenue: ${rev_sim:.2f}"
                        "<extra></extra>"
                    ),
                ))

            # — Vertical dashed line for current price only (no label on chart) —
            fig_r.add_vline(
                x=baseline,
                line_dash="dash",
                line_color=_VLINE_BASE,
                line_width=1.2,
            )
            if is_sim:
                fig_r.add_vline(
                    x=sim_price,
                    line_dash="dash",
                    line_color=_VLINE_SIM,
                    line_width=1.2,
                )

            # — Demand CI band (upper then lower with fill=tonexty) on y2 —
            fig_r.add_trace(go.Scatter(
                x=prices_arr,
                y=qty_arr * 1.15,
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                name="Demand Uncertainty Range (±15%)",
                yaxis="y2",
                showlegend=True,
                hoverinfo="skip",
            ))
            fig_r.add_trace(go.Scatter(
                x=prices_arr,
                y=qty_arr * 0.85,
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                fill="tonexty",
                fillcolor="rgba(59,130,246,0.10)",
                name="Demand Uncertainty Range (±15%)",
                yaxis="y2",
                showlegend=False,
                hoverinfo="skip",
            ))

            # — Demand curve on y2 —
            fig_r.add_trace(go.Scatter(
                x=prices_arr,
                y=qty_arr,
                mode="lines",
                line=dict(color="#3b82f6", width=2),
                name="Predicted Demand",
                yaxis="y2",
                hovertemplate="Price: $%{x:.2f}<br>Predicted Demand: %{y:.1f} units<extra></extra>",
            ))

            # — Equilibrium annotation at revenue peak —
            fig_r.add_annotation(
                x=rev_opt_price,
                y=rev_opt_rev,
                text="Equilibrium — Revenue Peak",
                arrowhead=2,
                arrowcolor="#22c55e",
                font=dict(color="#22c55e", size=10),
                ax=0,
                ay=-40,
                yref="y",
            )

            # — Amber danger zone to the right of rev_opt_price —
            fig_r.add_vrect(
                x0=rev_opt_price, x1=ceiling_p,
                fillcolor="rgba(234,179,8,0.08)",
                line_width=0,
                annotation_text="Danger Zone — Diminishing Returns",
                annotation_position="top right",
                annotation_font_size=9,
                annotation_font_color="#ca8a04",
            )

            fig_r.update_layout(
                **{**_LAYOUT_COMMON, "height": 400, "showlegend": True,
                   "margin": dict(l=65, r=80, t=70, b=120)},
                title=dict(text="Monthly Revenue by Price Point", font=dict(size=15), x=0),
                xaxis_title="Negotiated Price per Unit ($)",
                yaxis_title="Projected Monthly Revenue ($)",
                yaxis2=dict(
                    title="Predicted Quantity (units)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    tickfont=dict(size=10, color="#3b82f6"),
                    titlefont=dict(size=11, color="#3b82f6"),
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.25,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=9),
                    bgcolor="rgba(0,0,0,0)",
                    itemwidth=80,
                ),
            )
            fig_r.update_xaxes(
                title_font_size=_AXIS_FONT, tickfont_size=_TICK_FONT,
                tickprefix="$",
            )
            fig_r.update_yaxes(
                title_font_size=_AXIS_FONT, tickfont_size=_TICK_FONT,
                tickprefix="$",
            )

            # Center the chart with padding columns
            _rc_pad, _rc_mid, _rc_pad2 = st.columns([1, 6, 1])
            with _rc_mid:
                st.plotly_chart(fig_r, use_container_width=True,
                                config={"displayModeBar": False})

            st.markdown(
                "<div style='font-size:11px;color:#6b7280;font-style:italic;"
                "margin-top:-4px;'>Price sweep anchored to the most recent catalogue "
                "list price for this pair. Negotiated prices may differ.</div>",
                unsafe_allow_html=True,
            )

            # ── 4b. Impact tiles row ───────────────────────────────────────
            # Four tiles: Revenue (current → simulated), Demand shift,
            # Gross Margin at simulated price, Revenue-Optimal call-out.
            rev_delta   = rev_sim - rev_base
            q_delta     = q_sim   - q_base
            _rev_sign   = "+" if rev_delta >= 0 else ""
            _q_sign     = "+" if q_delta   >= 0 else ""
            _rev_color  = "#22c55e" if rev_delta >= 0 else "#ef4444"
            _q_color    = "#22c55e" if q_delta   >= 0 else "#ef4444"
            _gm_color   = "#22c55e" if gm_pct_sim >= 20 else ("#f59e0b" if gm_pct_sim >= 15 else "#ef4444")
            _opt_delta  = rev_opt_price - baseline
            _opt_sign   = "+" if _opt_delta >= 0 else ""

            st.markdown(
                f"""
                <div style="display:flex;gap:12px;margin-top:4px;margin-bottom:8px;">

                  <div style="flex:1;background:#111827;border:1px solid #1f2937;
                              border-radius:8px;padding:14px 16px;">
                    <div style="font-size:11px;color:#9ca3af;margin-bottom:4px;
                                text-transform:uppercase;letter-spacing:0.05em;">
                      Monthly Revenue
                    </div>
                    <div style="font-size:20px;font-weight:700;color:#f9fafb;">
                      ${rev_sim:.2f}
                    </div>
                    <div style="font-size:12px;color:{_rev_color};margin-top:2px;">
                      {_rev_sign}${abs(rev_delta):.2f} vs current
                    </div>
                  </div>

                  <div style="flex:1;background:#111827;border:1px solid #1f2937;
                              border-radius:8px;padding:14px 16px;">
                    <div style="font-size:11px;color:#9ca3af;margin-bottom:4px;
                                text-transform:uppercase;letter-spacing:0.05em;">
                      Predicted Demand
                    </div>
                    <div style="font-size:20px;font-weight:700;color:#f9fafb;">
                      {q_sim:.1f} units
                    </div>
                    <div style="font-size:12px;color:{_q_color};margin-top:2px;">
                      {_q_sign}{q_delta:.1f} units vs current
                    </div>
                  </div>

                  <div style="flex:1;background:#111827;border:1px solid #1f2937;
                              border-radius:8px;padding:14px 16px;">
                    <div style="font-size:11px;color:#9ca3af;margin-bottom:4px;
                                text-transform:uppercase;letter-spacing:0.05em;">
                      Gross Margin
                    </div>
                    <div style="font-size:20px;font-weight:700;color:{_gm_color};">
                      {gm_pct_sim:.1f}%
                    </div>
                    <div style="font-size:12px;color:#9ca3af;margin-top:2px;">
                      at simulated price
                    </div>
                  </div>

                  <div style="flex:1;background:#0f2d1a;border:1px solid #166534;
                              border-radius:8px;padding:14px 16px;">
                    <div style="font-size:11px;color:#86efac;margin-bottom:4px;
                                text-transform:uppercase;letter-spacing:0.05em;">
                      Revenue-Optimal Price
                    </div>
                    <div style="font-size:20px;font-weight:700;color:#4ade80;">
                      ${rev_opt_price:.2f}
                    </div>
                    <div style="font-size:12px;color:#86efac;margin-top:2px;">
                      {_opt_sign}${abs(_opt_delta):.2f} vs current negotiated
                    </div>
                  </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.divider()

            # ── 5. Scenario comparison table ──────────────────────────────
            st.subheader("Scenario Comparison")

            # Revenue-optimal status
            _rv_status, _ = _compute_status(
                rev_opt_price, unit_cost, pair_info["comp_ever_present"]
            )
            rev_status_text = (
                "Recommended" if rev_opt_price > unit_cost else "Below Cost Floor"
            )

            table_scenarios = [
                "Current",
                "Price That Maximizes Revenue",
                "Price That Maximizes Margin",
                "Simulated Price",
            ]
            table_prices = [
                f"${baseline:.2f}",
                f"${rev_opt_price:.2f}",
                f"${mgn_opt_price:.2f}",
                f"${sim_price:.2f}",
            ]
            table_units = [
                f"{q_base:.1f}",
                f"{rev_opt_qty:.1f}",
                f"{mgn_opt_qty:.1f}",
                f"{q_sim:.1f}",
            ]
            table_revenues = [
                f"${rev_base:.2f}",
                f"${rev_opt_rev:.2f}",
                f"${mgn_opt_rev:.2f}",
                f"${rev_sim:.2f}",
            ]
            table_margins = [
                f"{gm_pct_base:.1f}%",
                f"{rev_opt_gm:.1f}%",
                f"{mgn_opt_gm:.1f}%",
                f"{gm_pct_sim:.1f}%",
            ]
            table_statuses = [
                "—",
                rev_status_text,
                "Margin Maximizer",
                status_label,
            ]

            scenario_df = pd.DataFrame({
                "Scenario":        table_scenarios,
                "Negotiated Price": table_prices,
                "Units per Month": table_units,
                "Monthly Revenue": table_revenues,
                "Gross Margin":    table_margins,
                "Status":          table_statuses,
            })

            # Row background colours — Row 3 tracks sidebar badge live
            _row_bgs = [
                _TABLE_ROW_BG["current"],
                _TABLE_ROW_BG["revenue"],
                _TABLE_ROW_BG["margin"],
                _TABLE_ROW_BG.get(status_color, _TABLE_ROW_BG["current"]),
            ]

            def _style_rows(df: pd.DataFrame) -> pd.DataFrame:
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                for i, bg in enumerate(_row_bgs):
                    styles.iloc[i] = f"background-color:{bg};color:white;"
                return styles

            st.dataframe(
                scenario_df.style.apply(_style_rows, axis=None),
                hide_index=True,
                use_container_width=True,
            )

            # ── 6. Recommendation sentence ────────────────────────────────
            q_rev_opt = float(np.interp(rev_opt_price, prices_arr, qty_arr))
            q_delta       = q_sim  - q_base
            rev_delta     = rev_sim - rev_base
            rev_opt_delta_price = rev_opt_price - baseline
            rev_opt_q_delta     = q_rev_opt - q_base

            if sim_price <= unit_cost:
                rec_text = (
                    f"Simulated price of **\\${sim_price:.2f}** is below your cost per unit of "
                    f"**\\${unit_cost:.2f}**. This price loses money on every unit sold."
                )
            elif not is_sim:
                _direction = "increase" if rev_opt_delta_price >= 0 else "decrease"
                _delta_abs = abs(rev_opt_delta_price)
                _q_sign = "+" if rev_opt_q_delta >= 0 else ""
                rec_text = (
                    f"Current negotiated price of **\\${baseline:.2f}** is within the viable "
                    f"pricing band. Revenue-optimal price is **\\${rev_opt_price:.2f}** — "
                    f"a **\\${_delta_abs:.2f} {_direction}** that the XGBoost model estimates "
                    f"would change monthly demand by **{_q_sign}{rev_opt_q_delta:.1f} units**."
                )
            elif gm_pct_sim < 15.0:
                rec_text = (
                    f"Simulated price of **\\${sim_price:.2f}** produces a gross margin of "
                    f"**{gm_pct_sim:.1f}%**, which is below the 15% minimum viable threshold. "
                    f"Consider the basket context in Tab 4 before proceeding."
                )
            else:
                _q_word = "increase" if q_delta >= 0 else "decrease"
                _q_abs  = abs(q_delta)
                _r_word = "increase" if rev_delta >= 0 else "decrease"
                _r_abs  = abs(rev_delta)
                rec_text = (
                    f"At a simulated price of **\\${sim_price:.2f}**, the XGBoost model "
                    f"estimates monthly demand of **{q_sim:.1f} units** — an estimated "
                    f"**{_q_word} of {_q_abs:.1f} units** versus current. "
                    f"Projected monthly revenue is **\\${rev_sim:.2f}**, "
                    f"a **{_r_word} of \\${_r_abs:.2f}** versus current price."
                )

            st.markdown(rec_text)

with tab2:
    # ── 1. Simulation banner ──────────────────────────────────────────────
    render_simulation_banner()

    # ── Pull live state from session state ───────────────────────────────
    sim_price = float(st.session_state.get("sim_price", 0.0))
    pair_info = st.session_state.get("_pair_info", {})
    baseline  = float(st.session_state.get("_baseline_price", sim_price))
    is_sim    = abs(sim_price - baseline) > 0.005

    if not pair_info:
        st.info("Select a customer and part from the sidebar to begin.")
    else:
        unit_cost            = pair_info["unit_cost"]
        unit_price_catalogue = pair_info["unit_price_catalogue"]
        pair_avg_price_paid  = pair_info["pair_avg_price_paid"]
        floor_p              = pair_info["slider_floor"]
        ceiling_p            = pair_info["slider_ceiling"]

        # ── 2. History + competitor summary ──────────────────────────────
        history     = _get_pair_history(df, selected_customer, selected_part)
        comp_status = _get_competitor_status(df, selected_customer, selected_part)
        has_competitor     = comp_status["has_competitor"]
        latest_comp_price  = comp_status["latest_comp_price"]
        comp_presence_rate = comp_status["comp_presence_rate"]

        # Gross margin at simulated price
        gm_pct_sim = ((sim_price - unit_cost) / sim_price * 100.0
                      if sim_price > 0 else 0.0)

        # Revenue-optimal price (reuse Tab 1 cache — same pair, same bounds)
        sweep = _get_sweep_tab1(selected_customer, selected_part, floor_p, ceiling_p)
        if sweep.empty:
            rev_opt_price = pair_avg_price_paid
        else:
            _rv_idx        = int(np.argmax(sweep["revenue"].values))
            rev_opt_price  = float(sweep["unit_retail"].values[_rv_idx])

        # Minimum viable price = the price at which margin is exactly 15 %
        min_viable_price = unit_cost / (1.0 - 0.15) if unit_cost > 0 else 0.0

        # ── 3. Viability verdict state ────────────────────────────────────
        v_label, v_state = _compute_status(sim_price, unit_cost, has_competitor)
        if v_state == "red":
            v_text  = (
                f"Competitor price is at or below your cost per unit of "
                f"${unit_cost:.2f}. No profitable price exists at this level. "
                f"Review basket context in Tab 4 before walking away."
            )
        elif v_state == "yellow":
            v_text  = (
                f"Gross margin at simulated price is {gm_pct_sim:.1f}% which is "
                f"below the 15% minimum viable threshold. A competitor is present. "
                f"Check basket context in Tab 4 before making a final decision."
            )
        elif v_state == "green":
            v_text  = (
                "No competitor has been observed on this customer-part pair across "
                "all historical transactions. You have pricing flexibility here."
            )
        else:
            v_text  = (
                f"Competitor is present on {comp_presence_rate:.1f}% of transactions. "
                f"Current margin at simulated price is {gm_pct_sim:.1f}% which is above "
                f"the minimum viable threshold. Hold price and compete on availability "
                f"and relationship."
            )

        _VERDICT_COLORS: dict[str, tuple[str, str]] = {
            "red":    ("#fde0e0", "#8b1a1a"),
            "yellow": ("#fff8d6", "#7a5f00"),
            "green":  ("#d1f5d3", "#1a6b2a"),
            "blue":   ("#d0e8ff", "#0a4d8c"),
        }
        v_bg, v_fg = _VERDICT_COLORS[v_state]

        st.markdown(
            f"""
            <div style="
                background:{v_bg};
                color:{v_fg};
                border:1.5px solid {v_fg}55;
                border-radius:8px;
                padding:18px 22px;
                text-align:center;
                margin-bottom:6px;
            ">
                <div style="font-size:1.55rem;font-weight:800;letter-spacing:0.02em;">{v_label}</div>
                <div style="font-size:0.95rem;font-weight:500;margin-top:8px;line-height:1.5;">{v_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── 4. Revenue Curve — Price Viability ──────────────────────────
        if sweep.empty:
            st.warning("Price sweep could not be computed — revenue curve unavailable.")
        else:
            _t2_prices  = sweep["unit_retail"].values
            _t2_qty     = sweep["predicted_quantity"].values
            _t2_rev     = sweep["revenue"].values

            _t2_cost_floor = unit_cost * _t2_qty

            fig_t2 = go.Figure()

            # Red dotted cost floor line
            fig_t2.add_trace(go.Scatter(
                x=_t2_prices, y=_t2_cost_floor,
                mode="lines",
                line=dict(color="#ef4444", width=1.5, dash="dot"),
                name="Cost Floor",
                hovertemplate="Price: $%{x:.2f}<br>Cost Floor Revenue: $%{y:.2f}<extra></extra>",
            ))

            # Red shaded no-go zone below cost floor
            fig_t2.add_trace(go.Scatter(
                x=list(_t2_prices) + list(_t2_prices[::-1]),
                y=list(_t2_cost_floor) + [0] * len(_t2_prices),
                fill="toself",
                fillcolor="rgba(239,68,68,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))

            # Green revenue curve
            fig_t2.add_trace(go.Scatter(
                x=_t2_prices, y=_t2_rev,
                mode="lines",
                line=dict(color="#22c55e", width=2.5),
                name="Projected Revenue",
                hovertemplate="Price: $%{x:.2f}<br>Revenue: $%{y:.2f}<extra></extra>",
            ))

            # Amber vrect — danger zone
            fig_t2.add_vrect(
                x0=rev_opt_price, x1=ceiling_p,
                fillcolor="rgba(234,179,8,0.08)",
                line_width=0,
                annotation_text="Danger Zone — Diminishing Returns",
                annotation_position="top right",
                annotation_font_size=9,
                annotation_font_color="#ca8a04",
            )

            # Vertical dashed lines — no annotations (labels shown in markdown below)
            fig_t2.add_vline(x=unit_cost, line_dash="dash", line_color="#ef4444", line_width=1.2)
            fig_t2.add_vline(x=min_viable_price, line_dash="dash", line_color="#f97316", line_width=1.2)
            fig_t2.add_vline(x=pair_avg_price_paid, line_dash="dash", line_color="#9ca3af", line_width=1.2)
            fig_t2.add_vline(x=rev_opt_price, line_dash="dash", line_color="#22c55e", line_width=1.2)
            if is_sim:
                fig_t2.add_vline(x=sim_price, line_dash="dash", line_color="#f97316", line_width=1.2)
            if has_competitor and latest_comp_price is not None:
                fig_t2.add_vline(x=latest_comp_price, line_dash="dash", line_color="#3b82f6", line_width=1.2)

            fig_t2.update_layout(
                **{**_LAYOUT_COMMON, "height": 400,
                   "margin": dict(l=65, r=30, t=70, b=120)},
                title=dict(text="Revenue Curve — Price Viability", font=dict(size=15), x=0),
                xaxis_title="Negotiated Price per Unit ($)",
                yaxis_title="Projected Monthly Revenue ($)",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.25,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=9),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            fig_t2.update_xaxes(
                title_font_size=_AXIS_FONT, tickfont_size=_TICK_FONT, tickprefix="$",
            )
            fig_t2.update_yaxes(
                title_font_size=_AXIS_FONT, tickfont_size=_TICK_FONT, tickprefix="$",
            )

            st.plotly_chart(fig_t2, use_container_width=True, config={"displayModeBar": False})

            st.markdown(
                "<div style='font-size:11px;color:#6b7280;font-style:italic;"
                "margin-top:-4px;'>Price sweep anchored to the most recent catalogue "
                "list price for this pair. Negotiated prices may differ.</div>",
                unsafe_allow_html=True,
            )

            st.markdown("""
<div style="display:flex;gap:20px;flex-wrap:wrap;font-size:0.78rem;
            color:#9ca3af;margin-top:-8px;padding-left:4px;">
  <span><span style="color:#ef4444;font-weight:700;">— </span>Cost Floor: ${cost_floor:.2f}</span>
  <span><span style="color:#f97316;font-weight:700;">— </span>Min Viable (15% margin): ${min_viable:.2f}</span>
  <span><span style="color:#9ca3af;font-weight:700;">— </span>Current Price: ${current_price:.2f}</span>
  <span><span style="color:#22c55e;font-weight:700;">— </span>Revenue Optimal: ${rev_opt:.2f}</span>
  <span><span style="color:#f97316;font-weight:700;">-- </span>Simulated Price: ${sim_price:.2f}</span>
  {comp_entry}
</div>
""".format(
                cost_floor=unit_cost,
                min_viable=min_viable_price,
                current_price=pair_avg_price_paid,
                rev_opt=rev_opt_price,
                sim_price=sim_price,
                comp_entry=(
                    f'<span><span style="color:#3b82f6;font-weight:700;">-- </span>'
                    f'Competitor: ${latest_comp_price:.2f}</span>'
                ) if has_competitor and latest_comp_price is not None else ""
            ), unsafe_allow_html=True)

        st.divider()

        # ── 5. Loyalty + basket metrics shared by both lower columns ─────
        if not history.empty:
            # Total orders for this pair
            n_orders = int(len(history))

            # Price stability (need at least 2 rows for std)
            price_std = (
                float(history["unit_price_actual"].std())
                if len(history) > 1 else 0.0
            )
            price_max = float(history["unit_price_actual"].max())
            price_min = float(history["unit_price_actual"].min())
            price_range = price_max - price_min

            # Month span between first and last invoice
            if history["invoice_date"].notna().any():
                d_first = history["invoice_date"].min()
                d_last  = history["invoice_date"].max()
                month_span = max(1, int(round((d_last - d_first).days / 30.44)))
            else:
                month_span = 0

            # Historical bundling rate (% of orders that were multi-item)
            bundling_rate = float(
                (history["is_multi_item_order"].fillna(0).astype(int) == 1).mean() * 100.0
            )
        else:
            n_orders      = 0
            price_std     = 0.0
            price_range   = 0.0
            month_span    = 0
            bundling_rate = 0.0

        # Loyalty classification (strict spec rules; fall back to Emerging
        # for edge cases where the explicit rules do not match)
        if n_orders <= 5:
            loyalty_label = "Emerging Relationship"
            loyalty_color = "#6b7280"   # slate
        elif n_orders > 10 and price_std < 2.00:
            loyalty_label = "Relationship-Driven Buyer"
            loyalty_color = "#0a4d8c"   # blue
        elif n_orders > 5 and price_std >= 2.00:
            loyalty_label = "Price-Driven Buyer"
            loyalty_color = "#7a5f00"   # amber
        else:
            loyalty_label = "Emerging Relationship"
            loyalty_color = "#6b7280"

        # ── 6. Three-column row: Basket / Loyalty / Recommendation ───────
        col_b, col_l, col_r = st.columns(3)

        # — Left: Basket Context —————————————————————————————————
        with col_b:
            st.subheader("Basket Context")
            if history.empty:
                st.markdown("No transaction history available for this pair.")
            else:
                most_recent = history.iloc[-1]
                is_multi = int(most_recent.get("is_multi_item_order", 0) or 0) == 1

                if is_multi:
                    bi_count = int(most_recent.get("basket_item_count", 0) or 0)
                    bt_value = float(most_recent.get("basket_total_value", 0) or 0)
                    bs_raw   = float(most_recent.get("basket_share_pct", 0) or 0)
                    # Column may be stored as a fraction (0-1) or as a percent (0-100);
                    # detect by magnitude so display is always on a % scale.
                    bs_display = bs_raw * 100.0 if bs_raw <= 1.0 else bs_raw

                    st.markdown(
                        f"This part was purchased as part of an "
                        f"**{bi_count}-item order** worth **\\${bt_value:.2f}** total. "
                        f"It represents **{bs_display:.1f}%** of that order value."
                    )

                    # Basket Gross Margin — use the item's gross margin at the
                    # relevant price (only this item's price is changing).
                    basket_gm_current = (
                        (pair_avg_price_paid - unit_cost) / pair_avg_price_paid * 100.0
                        if pair_avg_price_paid > 0 else 0.0
                    )
                    st.markdown(
                        f"Basket Gross Margin at current price: "
                        f"**{basket_gm_current:.1f}%**"
                    )
                    if is_sim:
                        basket_gm_sim = (
                            (sim_price - unit_cost) / sim_price * 100.0
                            if sim_price > 0 else 0.0
                        )
                        st.markdown(
                            f"At simulated price of **\\${sim_price:.2f}**, basket "
                            f"gross margin changes to **{basket_gm_sim:.1f}%**."
                        )
                else:
                    st.markdown(
                        "This part was purchased as a standalone order — "
                        "basket context does not apply to this transaction."
                    )

                st.markdown(
                    f"<div style='color:#9ca3af;font-size:0.88rem;"
                    f"margin-top:14px;line-height:1.5;'>"
                    f"This customer-part pair is bundled in "
                    f"<strong>{bundling_rate:.1f}%</strong> of historical orders."
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # — Middle: Customer Loyalty Signal ——————————————————————————
        with col_l:
            st.subheader("Customer Loyalty Signal")
            if history.empty:
                st.markdown("No transaction history available for this pair.")
            else:
                month_word = "month" if month_span == 1 else "months"
                st.markdown(
                    f"**Purchase History for This Part:** {n_orders} orders "
                    f"over {month_span} {month_word}"
                )

                deviation = sim_price - pair_avg_price_paid
                direction = "above" if deviation >= 0 else "below"
                st.markdown(
                    f"**Deviation from Customer's Historical Price Point:** "
                    f"Simulated price is **\\${abs(deviation):.2f} {direction}** "
                    f"this customer's historical price point of "
                    f"**\\${pair_avg_price_paid:.2f}**."
                )

                if price_std < 2.00:
                    st.markdown(
                        f"**Price Stability:** Consistent Pricer — price has been "
                        f"stable within a **\\${price_range:.2f}** range across all "
                        f"orders."
                    )
                else:
                    st.markdown(
                        f"**Price Stability:** Variable Pricer — price has varied "
                        f"by up to **\\${price_range:.2f}** across orders."
                    )

                st.markdown(
                    f"""
                    <div style="
                        background:{loyalty_color}22;
                        color:#ffffff;
                        border:1.5px solid {loyalty_color};
                        border-radius:6px;
                        padding:8px 12px;
                        margin-top:14px;
                        font-size:0.9rem;
                        font-weight:700;
                        text-align:center;
                        letter-spacing:0.01em;
                    ">{loyalty_label}</div>
                    """,
                    unsafe_allow_html=True,
                )

        # — Right: Recommendation ——————————————————————————————————
        with col_r:
            st.subheader("Recommendation")
            if history.empty:
                rec_text = "No transaction history available to generate a recommendation."
            elif v_state == "red":
                loss_per_unit = abs(unit_cost - sim_price)
                rec_text = (
                    f"Simulated price of **\\${sim_price:.2f}** is below your cost "
                    f"per unit. Every unit sold at this price loses "
                    f"**\\${loss_per_unit:.2f}**. Before walking away check whether "
                    f"this part is a significant share of a bundled order in Tab 4."
                )
            elif v_state == "yellow" and loyalty_label == "Relationship-Driven Buyer":
                rec_text = (
                    f"Margin is tight at **{gm_pct_sim:.1f}%** but this customer "
                    f"has a strong purchase history of **{n_orders} orders**. "
                    f"Consider whether the relationship justifies a short-term "
                    f"thin margin. Check basket context in Tab 4."
                )
            elif v_state == "yellow" and loyalty_label == "Price-Driven Buyer":
                rec_text = (
                    f"Margin is tight at **{gm_pct_sim:.1f}%** and this customer "
                    f"shows price-sensitive buying patterns. Risk of losing this "
                    f"order is elevated. Check basket context before deciding."
                )
            elif v_state == "yellow":   # Emerging Relationship
                rec_text = (
                    f"Margin is tight at **{gm_pct_sim:.1f}%** and this customer "
                    f"has only **{n_orders} orders** of history — an emerging "
                    f"relationship. Check basket context in Tab 4 before committing."
                )
            elif v_state == "green":
                rec_text = (
                    f"No competitor has been observed on this pair. Current "
                    f"pricing may be conservative — the XGBoost model estimates "
                    f"revenue-optimal price is **\\${rev_opt_price:.2f}**. Consider "
                    f"a controlled price increase."
                )
            elif v_state == "blue" and loyalty_label == "Relationship-Driven Buyer":
                rec_text = (
                    f"Margin is healthy at **{gm_pct_sim:.1f}%** and this customer "
                    f"has **{n_orders} orders** of purchase history. Hold price "
                    f"firm. Price matching the competitor is unlikely to increase "
                    f"volume and will reduce margin."
                )
            elif v_state == "blue" and loyalty_label == "Price-Driven Buyer":
                if latest_comp_price is not None:
                    gap = sim_price - latest_comp_price
                    gap_dir = "above" if gap >= 0 else "below"
                    gap_sentence = (
                        f"Current gap of **\\${abs(gap):.2f} {gap_dir} competitor** "
                        f"is within acceptable range."
                    )
                else:
                    gap_sentence = "Competitor price reference is Not Available."
                rec_text = (
                    f"Margin is healthy at **{gm_pct_sim:.1f}%** but this customer "
                    f"shows price sensitivity. Monitor competitor gap. {gap_sentence}"
                )
            else:   # Blue + Emerging Relationship
                rec_text = (
                    f"Margin is healthy at **{gm_pct_sim:.1f}%** and this customer "
                    f"has **{n_orders} orders** of history — an emerging "
                    f"relationship. Maintain steady pricing and continue to build "
                    f"the account."
                )

            st.markdown(rec_text)

with tab3:
    # ── 1. Simulation banner ──────────────────────────────────────────────
    render_simulation_banner()

    sim_price = float(st.session_state.get("sim_price", 0.0))
    pair_info = st.session_state.get("_pair_info", {})
    baseline  = float(st.session_state.get("_baseline_price", sim_price))
    is_sim    = abs(sim_price - baseline) > 0.005

    if not pair_info:
        st.info("Select a customer and part from the sidebar to begin.")
    else:
        # ── 2. Load portfolio data ────────────────────────────────────────
        all_pairs = _get_all_pairs_for_customer(selected_customer)

        if all_pairs.empty:
            st.warning("No portfolio data found for this customer.")
        else:
            # Run sweeps + compute metrics for every part in the portfolio
            metrics_by_part: dict[str, dict] = {}
            for _, _pr in all_pairs.iterrows():
                _pid = _pr["PartID"]
                _sw  = _get_sweep_for_pair(selected_customer, _pid)
                metrics_by_part[_pid] = _compute_band_metrics(
                    _sw, float(_pr["unit_cost"]), float(_pr["pair_avg_price_paid"])
                )

            n_parts = len(all_pairs)

            # ── 3. Portfolio summary cards ────────────────────────────────
            pc1, pc2, pc3, pc4 = st.columns(4)

            below_cost_count = sum(
                1 for _, _pr in all_pairs.iterrows()
                if float(_pr["pair_avg_price_paid"]) < float(_pr["unit_cost"])
            )
            headroom_count = sum(
                1 for _, _pr in all_pairs.iterrows()
                if (metrics_by_part[_pr["PartID"]]["rev_optimal_price"]
                    - float(_pr["pair_avg_price_paid"])) > 0.50
            )
            comp_count   = int((all_pairs["competitor_present"].fillna(0) == 1).sum())
            total_upside = sum(m["upside"] for m in metrics_by_part.values())

            with pc1:
                st.metric("SKUs Priced Below Cost Floor", str(below_cost_count))
                _bc_color = "#ef4444" if below_cost_count > 0 else "#22c55e"
                _bc_msg   = "⚠ Requires review" if below_cost_count > 0 else "✓ All clear"
                st.markdown(
                    f"<div style='color:{_bc_color};font-size:0.82rem;"
                    f"margin-top:-14px;'>{_bc_msg}</div>",
                    unsafe_allow_html=True,
                )
            with pc2:
                st.metric("SKUs With Pricing Headroom", str(headroom_count))
            with pc3:
                st.metric("SKUs With Active Competitor", str(comp_count))
            with pc4:
                st.metric("Total Monthly Revenue Upside", f"${total_upside:.2f}")

            st.divider()
            st.subheader("Revenue Upside by SKU")
            st.caption(
                "How much additional monthly revenue is available if each part "
                "were priced at its revenue-optimal point."
            )

            # ── 4. Horizontal upside bar chart ────────────────────────────
            # Build one row per SKU, sorted by upside descending
            _bar_rows = []
            for _, _pr in all_pairs.iterrows():
                _pid    = _pr["PartID"]
                _uc     = float(_pr["unit_cost"])
                _papp   = float(_pr["pair_avg_price_paid"])
                _m      = metrics_by_part[_pid]
                _upside = _m["upside"]
                _rop    = _m["rev_optimal_price"]
                _cur_rev = _m["current_revenue"]
                _opt_rev = _m["rev_optimal_revenue"]
                _comp_ever = int(_pr.get("competitor_present", 0) or 0) == 1
                _is_sel = (_pid == selected_part)

                # Status drives bar color
                _p_check = sim_price if _is_sel else _papp
                _s_label, _s_color = _compute_status(_p_check, _uc, _comp_ever)

                _bar_color = {
                    "red":    "#ef4444",
                    "yellow": "#f59e0b",
                    "green":  "#22c55e",
                    "blue":   "#3b82f6",
                }.get(_s_color, "#3b82f6")

                _oem_tag = "OEM" if int(_pr["oem_flag"]) == 1 else "AEM"
                _label   = f"{_pid[-8:]} | {_oem_tag}"

                _bar_rows.append({
                    "label":     _label,
                    "upside":    _upside,
                    "color":     _bar_color,
                    "status":    _s_label,
                    "cur_rev":   _cur_rev,
                    "opt_rev":   _opt_rev,
                    "rop":       _rop,
                    "cur_price": _papp,
                    "is_sel":    _is_sel,
                })

            # Sort by upside descending so biggest opportunity is at top
            _bar_rows.sort(key=lambda r: r["upside"], reverse=True)

            _bar_labels  = [r["label"]   for r in _bar_rows]
            _bar_values  = [r["upside"]  for r in _bar_rows]
            _bar_colors  = [r["color"]   for r in _bar_rows]
            _bar_hover   = [
                (
                    f"<b>{r['label']}</b><br>"
                    f"Current Price: ${r['cur_price']:.2f}<br>"
                    f"Revenue-Optimal Price: ${r['rop']:.2f}<br>"
                    f"Monthly Revenue Now: ${r['cur_rev']:.2f}<br>"
                    f"Monthly Revenue at Optimal: ${r['opt_rev']:.2f}<br>"
                    f"Upside: ${r['upside']:.2f}<br>"
                    f"Status: {r['status']}"
                )
                for r in _bar_rows
            ]

            # Highlight selected part with a white border
            _bar_line_colors = [
                "white" if r["is_sel"] else "rgba(0,0,0,0)"
                for r in _bar_rows
            ]
            _bar_line_widths = [2 if r["is_sel"] else 0 for r in _bar_rows]

            # — Scatter / bubble plot: Current Price vs Revenue-Optimal Price —
            _scatter_x    = [r["cur_price"] for r in _bar_rows]
            _scatter_y    = [r["rop"]       for r in _bar_rows]
            _scatter_size = [max(8, min(40, r["upside"] * 2)) for r in _bar_rows]
            _scatter_col  = [r["color"]     for r in _bar_rows]
            _scatter_text = [r["label"]     for r in _bar_rows]
            _scatter_hover = [
                (
                    f"<b>{r['label']}</b><br>"
                    f"Current Price: ${r['cur_price']:.2f}<br>"
                    f"Optimal Price: ${r['rop']:.2f}<br>"
                    f"Revenue Impact: ${r['upside']:.2f}<br>"
                    f"Status: {r['status']}"
                )
                for r in _bar_rows
            ]

            _all_prices = _scatter_x + _scatter_y
            _diag_min = min(_all_prices) if _all_prices else 0
            _diag_max = max(_all_prices) if _all_prices else 1

            fig_scatter = go.Figure()

            # Diagonal reference line (y = x)
            fig_scatter.add_trace(go.Scatter(
                x=[_diag_min, _diag_max],
                y=[_diag_min, _diag_max],
                mode="lines",
                line=dict(color="#4b5563", width=1, dash="dot"),
                name="Current = Optimal",
                showlegend=False,
            ))

            # SKU bubbles
            fig_scatter.add_trace(go.Scatter(
                x=_scatter_x,
                y=_scatter_y,
                mode="markers+text",
                marker=dict(
                    color=_scatter_col,
                    size=_scatter_size,
                    line=dict(
                        color=["white" if r["is_sel"] else "rgba(0,0,0,0)" for r in _bar_rows],
                        width=[2 if r["is_sel"] else 0 for r in _bar_rows],
                    ),
                ),
                text=_scatter_text,
                textposition="top center",
                textfont=dict(size=9, color="#9ca3af"),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=_scatter_hover,
                showlegend=False,
            ))

            # Annotations for quadrant labels
            fig_scatter.add_annotation(
                x=_diag_min + (_diag_max - _diag_min) * 0.15,
                y=_diag_max - (_diag_max - _diag_min) * 0.05,
                text="Underpriced — Room to Raise",
                showarrow=False,
                font=dict(size=10, color="#22c55e"),
                xanchor="left",
            )
            fig_scatter.add_annotation(
                x=_diag_max - (_diag_max - _diag_min) * 0.05,
                y=_diag_min + (_diag_max - _diag_min) * 0.05,
                text="Overpriced — Demand Risk",
                showarrow=False,
                font=dict(size=10, color="#ef4444"),
                xanchor="right",
            )

            fig_scatter.update_layout(
                **{**_LAYOUT_COMMON, "height": 420},
                title=dict(text="Current Price vs. Revenue-Optimal Price by SKU",
                           font=dict(size=15), x=0),
                xaxis=dict(
                    title="Current Negotiated Price ($)",
                    titlefont_size=_AXIS_FONT,
                    tickfont_size=_TICK_FONT,
                    tickprefix="$",
                ),
                yaxis=dict(
                    title="Revenue-Optimal Price ($)",
                    titlefont_size=_AXIS_FONT,
                    tickfont_size=_TICK_FONT,
                    tickprefix="$",
                ),
            )

            st.plotly_chart(fig_scatter, use_container_width=True,
                            config={"displayModeBar": False})

            # Color legend
            st.markdown(
                "<div style='color:#9ca3af;font-size:0.82rem;margin-top:-4px;"
                "display:flex;gap:20px;flex-wrap:wrap;'>"
                "<span><span style='color:#22c55e;font-weight:700;'>●</span>"
                " Uncontested — Raise With Confidence</span>"
                "<span><span style='color:#3b82f6;font-weight:700;'>●</span>"
                " Viable — Compete and Hold</span>"
                "<span><span style='color:#f59e0b;font-weight:700;'>●</span>"
                " Tight Margin — Review Basket</span>"
                "<span><span style='color:#ef4444;font-weight:700;'>●</span>"
                " Cannot Compete Profitably</span>"
                "<span style='color:#ffffff;'>○ = currently selected part (white border)</span>"
                "</div>",
                unsafe_allow_html=True,
            )

            st.divider()

            # ── 5. SKU summary table ──────────────────────────────────────
            def _price_tier_pctl(pct_val) -> str:
                try:
                    v = float(pct_val)
                    if pd.isna(v):
                        return "—"
                    n = int(round(v))
                    suffix = "th"
                    if n % 100 not in (11, 12, 13):
                        if n % 10 == 1: suffix = "st"
                        elif n % 10 == 2: suffix = "nd"
                        elif n % 10 == 3: suffix = "rd"
                    return f"{n}{suffix} pctl"
                except:
                    return "—"

            _table_rows = []
            for _, _pr in all_pairs.iterrows():
                _pid    = _pr["PartID"]
                _uc     = float(_pr["unit_cost"])
                _papp   = float(_pr["pair_avg_price_paid"])
                _oem    = int(_pr["oem_flag"]) == 1
                _m      = metrics_by_part[_pid]

                _table_rows.append({
                    "Part":           _pid[-8:],
                    "Type":           "OEM" if _oem else "AEM",
                    "Current Price":  f"${_papp:.2f}",
                    "Optimal Price":  f"${_m['rev_optimal_price']:.2f}",
                    "Market Tier":    _price_tier_pctl(_pr.get("price_percentile_in_part")),
                    "Revenue Impact": f"${_m['upside']:.2f}",
                    "_upside_sort":   _m["upside"],
                    "_part_full":     _pid,
                })

            _tbl_df = (
                pd.DataFrame(_table_rows)
                .sort_values("_upside_sort", ascending=False)
                .reset_index(drop=True)
            )
            _display_df = _tbl_df.drop(columns=["_upside_sort", "_part_full"])

            st.dataframe(_display_df, hide_index=True, use_container_width=True)

            st.markdown(
                "<div style='font-size:11px;color:#6b7280;font-style:italic;"
                "margin-top:4px;'>Revenue-optimal prices are derived from a sweep anchored "
                "to the most recent catalogue list price per part. Negotiated prices may differ.</div>",
                unsafe_allow_html=True,
            )

            # ── 6. Drill-in selector (on_select not available in 1.32.0) ─
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            _part_display = _tbl_df["Part"].tolist()
            _full_pid_map = dict(zip(_tbl_df["Part"], _tbl_df["_part_full"]))

            _dc1, _dc2 = st.columns([3, 1])
            with _dc1:
                _drill_sel = st.selectbox(
                    "Select a part to drill into",
                    options=_part_display,
                    key="_tab3_drill_sel",
                )
            with _dc2:
                st.markdown(
                    "<div style='height:28px'></div>", unsafe_allow_html=True
                )
                if st.button(
                    "Go to Tab 1 for this part",
                    key="_tab3_drill_btn",
                    use_container_width=True,
                ):
                    _full_pid  = _full_pid_map.get(_drill_sel, _drill_sel)
                    _part_key  = f"_part_sel_{selected_customer}"
                    _cust_parts = get_parts_for_customer(df, selected_customer)
                    if _full_pid in _cust_parts:
                        st.session_state[_part_key] = _full_pid
                        st.rerun()

with tab4:
    render_simulation_banner()

    # ── Pull basket data from most-recent row for this pair ────────────────
    _t4_hist = _get_basket_history(selected_customer, selected_part, months=12)
    _t4_all  = _get_pair_history(df, selected_customer, selected_part)

    if _t4_all.empty:
        st.info("No transaction history found for this customer-part pair.")
    else:
        _t4_latest = _t4_all.iloc[-1]

        # ── Extract basket fields ──────────────────────────────────────────
        _t4_basket_total    = float(_t4_latest.get("basket_total_value", 0.0) or 0.0)
        _t4_basket_items    = int(_t4_latest.get("basket_item_count", 1) or 1)
        _t4_actual_revenue  = float(_t4_latest.get("actual_revenue", 0.0) or 0.0)
        _t4_units_sold      = float(_t4_latest.get("units_sold", 0.0) or 0.0)

        # basket_share_pct: detect fraction vs percent
        _t4_bs_raw = float(_t4_latest.get("basket_share_pct", 0.0) or 0.0)
        _t4_basket_share_display = _t4_bs_raw * 100.0 if _t4_bs_raw <= 1.0 else _t4_bs_raw

        # gross_margin_pct: detect decimal vs percent for basket cost calculation
        _t4_gm_raw = float(_t4_latest.get("gross_margin_pct", 0.0) or 0.0)
        _t4_gm_decimal = _t4_gm_raw if _t4_gm_raw <= 1.0 else _t4_gm_raw / 100.0

        _t4_is_multi = _t4_basket_items > 1

        # ── Basket cost (fixed — doesn't change with price) ────────────────
        _t4_basket_cost = _t4_basket_total * (1.0 - _t4_gm_decimal) if _t4_basket_total > 0 else 0.0

        # ── Simulated basket value and margin ─────────────────────────────
        _t4_sim_basket_value = (
            _t4_basket_total - _t4_actual_revenue + sim_price * _t4_units_sold
            if _t4_units_sold > 0 else _t4_basket_total
        )
        _t4_sim_basket_margin = _t4_sim_basket_value - _t4_basket_cost
        _t4_sim_basket_gm_pct = (
            (_t4_sim_basket_margin / _t4_sim_basket_value * 100.0)
            if _t4_sim_basket_value > 0 else 0.0
        )

        # ── Baseline basket margin (at pair_avg_price_paid) ───────────────
        _t4_baseline_price   = pair_info["pair_avg_price_paid"]
        _t4_base_basket_val  = (
            _t4_basket_total - _t4_actual_revenue + _t4_baseline_price * _t4_units_sold
            if _t4_units_sold > 0 else _t4_basket_total
        )
        _t4_base_basket_gm_pct = (
            ((_t4_base_basket_val - _t4_basket_cost) / _t4_base_basket_val * 100.0)
            if _t4_base_basket_val > 0 else 0.0
        )

        # ── Bundle/order statistics from 12-month history ─────────────────
        if not _t4_hist.empty:
            _t4_bundled_rate = (
                (_t4_hist["basket_item_count"] > 1).sum() / len(_t4_hist) * 100.0
            )
            _t4_avg_basket_size = float(_t4_hist["basket_item_count"].mean())
        else:
            _t4_bundled_rate    = 0.0
            _t4_avg_basket_size = float(_t4_basket_items)

        # ── Competitor basket calculation (for walk-away panel) ────────────
        _t4_comp_info = _get_competitor_status(df, selected_customer, selected_part)
        _t4_latest_comp_price = _t4_comp_info.get("latest_comp_price")
        _t4_comp_basket_value = (
            _t4_basket_total - _t4_actual_revenue + _t4_latest_comp_price * _t4_units_sold
            if (_t4_latest_comp_price is not None and _t4_units_sold > 0)
            else None
        )

        # ══════════════════════════════════════════════════════════════════
        # 1. Basket Status Card — full width
        # ══════════════════════════════════════════════════════════════════
        if _t4_is_multi:
            _t4_card_border = {
                "Cannot Compete Profitably": "#ef4444",
                "Tight Margin — Review Basket": "#f59e0b",
                "Viable — Compete and Hold": "#22c55e",
                "Uncontested — Raise With Confidence": "#3b82f6",
            }.get(status_label, "#374151")

            st.markdown(
                f"""
                <div style="border:1px solid {_t4_card_border};border-radius:8px;
                            padding:16px 20px;margin-bottom:16px;
                            background:#111827;">
                  <div style="font-size:13px;color:#9ca3af;margin-bottom:4px;">
                    Basket Context
                  </div>
                  <div style="font-size:15px;color:#f9fafb;font-weight:600;">
                    This part is purchased alongside <b>{_t4_basket_items - 1}
                    other item{'s' if _t4_basket_items - 1 != 1 else ''}</b> —
                    pricing decisions affect the full order value.
                  </div>
                  <div style="font-size:13px;color:#9ca3af;margin-top:6px;">
                    Total basket value (most recent order): <b style="color:#f9fafb;">
                    ${_t4_basket_total:,.2f}</b> &nbsp;|&nbsp;
                    This item's share: <b style="color:#f9fafb;">
                    {_t4_basket_share_display:.1f}%</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="border:1px solid #374151;border-radius:8px;
                            padding:14px 20px;margin-bottom:16px;
                            background:#111827;">
                  <div style="font-size:13px;color:#9ca3af;margin-bottom:4px;">
                    Basket Context
                  </div>
                  <div style="font-size:15px;color:#9ca3af;">
                    This part is typically ordered on its own — no basket
                    bundling effect to consider.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ══════════════════════════════════════════════════════════════════
        # 2. Two-column row
        # ══════════════════════════════════════════════════════════════════
        _t4_col_l, _t4_col_r = st.columns([1, 1], gap="large")

        # ── LEFT column ───────────────────────────────────────────────────
        with _t4_col_l:
            st.markdown("##### Basket Share & Margin Impact")

            # Large share display
            _t4_share_color = (
                "#ef4444" if _t4_basket_share_display >= 40
                else "#f59e0b" if _t4_basket_share_display >= 20
                else "#22c55e"
            )
            st.markdown(
                f"""
                <div style="text-align:center;padding:20px 0 10px;">
                  <div style="font-size:52px;font-weight:700;
                              color:{_t4_share_color};line-height:1;">
                    {_t4_basket_share_display:.1f}%
                  </div>
                  <div style="font-size:13px;color:#9ca3af;margin-top:4px;">
                    of basket revenue
                  </div>
                  <div style="font-size:13px;color:#9ca3af;margin-top:2px;">
                    {_t4_basket_items} item{'s' if _t4_basket_items != 1 else ''} in basket
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Basket margin impact — live simulation
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown("**Basket Gross Margin — Live Simulation**")
            st.markdown(
                "<div style='font-size:11px;color:#9ca3af;margin-top:-4px;"
                "margin-bottom:6px;'>Assumes same order quantity — reflects price "
                "effect only, not demand shift.</div>",
                unsafe_allow_html=True,
            )

            _t4_delta_pct = _t4_sim_basket_gm_pct - _t4_base_basket_gm_pct
            _t4_delta_color = "#22c55e" if _t4_delta_pct >= 0 else "#ef4444"
            _t4_delta_sign  = "+" if _t4_delta_pct >= 0 else ""

            _t4_m1, _t4_m2, _t4_m3 = st.columns(3)
            with _t4_m1:
                st.metric(
                    "At Historical Price",
                    f"{_t4_base_basket_gm_pct:.1f}%",
                )
            with _t4_m2:
                st.metric(
                    "At Simulated Price",
                    f"{_t4_sim_basket_gm_pct:.1f}%",
                )
            with _t4_m3:
                st.metric(
                    "Change",
                    f"{_t4_delta_sign}{_t4_delta_pct:.1f}%",
                )

            # Basket value row
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            _t4_v1, _t4_v2 = st.columns(2)
            with _t4_v1:
                st.metric("Basket Value (Historical)", f"${_t4_base_basket_val:,.2f}")
            with _t4_v2:
                st.metric("Basket Value (Simulated)", f"${_t4_sim_basket_value:,.2f}")

        # ── RIGHT column ──────────────────────────────────────────────────
        with _t4_col_r:

            # Walk-away override panel — only for "Cannot Compete Profitably"
            if status_label == "Cannot Compete Profitably":
                st.markdown("##### Walk-Away Override Analysis")

                _t4_comp_price_disp = (
                    f"${_t4_latest_comp_price:,.2f}"
                    if _t4_latest_comp_price is not None
                    else "Not Available"
                )
                _t4_comp_bv_disp = (
                    f"${_t4_comp_basket_value:,.2f}"
                    if _t4_comp_basket_value is not None
                    else "Not Available"
                )

                st.markdown(
                    f"""
                    <div style="border:1px solid #ef4444;border-radius:8px;
                                padding:14px 18px;background:#1c0a0a;
                                margin-bottom:12px;">
                      <div style="font-size:13px;color:#fca5a5;font-weight:600;
                                  margin-bottom:8px;">
                        If you walk away from this part:
                      </div>
                      <table style="width:100%;font-size:13px;color:#f9fafb;
                                    border-collapse:collapse;">
                        <tr>
                          <td style="padding:3px 0;color:#9ca3af;">
                            Competitor's price on this part
                          </td>
                          <td style="text-align:right;font-weight:600;">
                            {_t4_comp_price_disp}
                          </td>
                        </tr>
                        <tr>
                          <td style="padding:3px 0;color:#9ca3af;">
                            Basket value at competitor price
                          </td>
                          <td style="text-align:right;font-weight:600;">
                            {_t4_comp_bv_disp}
                          </td>
                        </tr>
                        <tr>
                          <td style="padding:3px 0;color:#9ca3af;">
                            Current basket value
                          </td>
                          <td style="text-align:right;font-weight:600;">
                            ${_t4_basket_total:,.2f}
                          </td>
                        </tr>
                      </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if _t4_comp_basket_value is not None:
                    _t4_at_risk = _t4_basket_total - _t4_comp_basket_value
                    _t4_risk_color = "#ef4444" if _t4_at_risk > 0 else "#22c55e"
                    _t4_risk_label = (
                        f"${abs(_t4_at_risk):,.2f} at risk if customer walks"
                        if _t4_at_risk > 0
                        else f"${abs(_t4_at_risk):,.2f} basket upside over competitor"
                    )
                    st.markdown(
                        f"<div style='font-size:13px;color:{_t4_risk_color};"
                        f"font-weight:600;margin-bottom:12px;'>"
                        f"{_t4_risk_label}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                # Neutral card for all other statuses
                st.markdown("##### Basket Pricing Context")
                st.markdown(
                    f"""
                    <div style="border:1px solid #374151;border-radius:8px;
                                padding:14px 18px;background:#111827;
                                margin-bottom:12px;">
                      <div style="font-size:13px;color:#9ca3af;margin-bottom:8px;">
                        This item contributes <b style="color:#f9fafb;">
                        {_t4_basket_share_display:.1f}%</b> of the basket value.
                        {'Consider protecting the full order when negotiating.' if _t4_basket_share_display >= 20
                         else 'Low share — pricing this item has limited basket-level impact.'}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Historical patterns (always shown)
            st.markdown("**Historical Basket Patterns (Last 12 Months)**")
            _t4_hp1, _t4_hp2, _t4_hp3 = st.columns(3)
            with _t4_hp1:
                st.metric("Bundled Order Rate", f"{_t4_bundled_rate:.1f}%")
            with _t4_hp2:
                st.metric("Avg Items per Order", f"{_t4_avg_basket_size:.1f}")
            with _t4_hp3:
                st.metric(
                    "Typical Basket Share",
                    f"{_t4_basket_share_display:.1f}%",
                )

        # ── Top-3 Basket Contributors ──────────────────────────────────────
        st.markdown("##### Top 3 Basket Contributors")

        # Use the most recent row in _t4_all to get the reference month
        if not _t4_all.empty:
            _t4_ref_date  = _t4_all["invoice_date"].max()
            _t4_ref_month = _t4_ref_date.month
            _t4_ref_year  = _t4_ref_date.year
            _t4_month_label = _t4_ref_date.strftime("%B %Y")

            # All customer purchases in that same month/year from the raw df
            _t4_same_month = df[
                (df["customer_id"] == selected_customer)
                & (df["invoice_date"].dt.month == _t4_ref_month)
                & (df["invoice_date"].dt.year  == _t4_ref_year)
            ].copy()

            if not _t4_same_month.empty:
                _t4_contrib = (
                    _t4_same_month.groupby("part_id")["actual_revenue"]
                    .sum()
                    .reset_index()
                    .sort_values("actual_revenue", ascending=False)
                    .head(3)
                )
                _t4_basket_grand_total = float(_t4_same_month["actual_revenue"].sum())
                _t4_n_parts_month = _t4_same_month["part_id"].nunique()

                _tb3_labels = [pid[-8:] for pid in _t4_contrib["part_id"]]
                _tb3_values = [float(v) for v in _t4_contrib["actual_revenue"]]
                _tb3_colors = [
                    "#3b82f6" if pid == selected_part else "#6b7280"
                    for pid in _t4_contrib["part_id"]
                ]
                _tb3_pcts = [
                    v / _t4_basket_grand_total * 100.0 if _t4_basket_grand_total > 0 else 0.0
                    for v in _tb3_values
                ]

                fig_basket_top3 = go.Figure()
                fig_basket_top3.add_trace(go.Bar(
                    x=_tb3_values,
                    y=_tb3_labels,
                    orientation="h",
                    marker=dict(color=_tb3_colors),
                    text=[f"{p:.0f}%" for p in _tb3_pcts],
                    textposition="inside",
                    textfont=dict(size=12, color="white"),
                    hovertemplate="<b>%{y}</b><br>Revenue: $%{x:,.2f}<extra></extra>",
                ))
                fig_basket_top3.update_layout(
                    **{**_LAYOUT_COMMON, "height": 180,
                       "margin": dict(l=10, r=30, t=40, b=40)},
                    title=dict(
                        text=f"Top 3 Revenue Contributors — {_t4_month_label}",
                        font=dict(size=13), x=0,
                    ),
                    xaxis=dict(
                        title="Revenue ($)",
                        tickprefix="$",
                        titlefont_size=_AXIS_FONT,
                        tickfont_size=_TICK_FONT,
                    ),
                )
                st.plotly_chart(fig_basket_top3, use_container_width=True,
                                config={"displayModeBar": False})
                st.caption(
                    f"Showing top contributors from {_t4_month_label} order. "
                    f"Total basket: ${_t4_basket_grand_total:,.2f} across {_t4_n_parts_month} parts."
                )
            else:
                st.markdown(
                    "<div style='color:#9ca3af;font-size:0.88rem;'>"
                    "No purchase data found for the reference month.</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='color:#9ca3af;font-size:0.88rem;'>"
                "No transaction history found for this pair.</div>",
                unsafe_allow_html=True,
            )

with tab5:
    render_simulation_banner()

    # ── Customer-level data (all pairs for this customer) ─────────────────
    _t5_cust_df = df[df["customer_id"] == selected_customer].copy()
    _t5_pair_df = _get_pair_history(df, selected_customer, selected_part)

    if _t5_pair_df.empty:
        st.info("No transaction history found for this customer-part pair.")
    else:
        # ── Customer-level metrics ─────────────────────────────────────────
        _t5_latest_pair = _t5_pair_df.iloc[-1]

        _t5_annual_spend_raw = float(
            _t5_latest_pair.get("customer_annual_spend", 0.0) or 0.0
        )
        _t5_spend_zscore = float(
            _t5_latest_pair.get("customer_spend_zscore", 0.0) or 0.0
        )
        _t5_active_since = _t5_cust_df["invoice_date"].min()
        _t5_distinct_parts = _t5_cust_df["part_id"].nunique()

        # Dollar threshold → account tier
        if _t5_annual_spend_raw < 500_000:
            _t5_acct_tier = "Small Account"
        elif _t5_annual_spend_raw < 2_000_000:
            _t5_acct_tier = "Mid Account"
        elif _t5_annual_spend_raw < 5_000_000:
            _t5_acct_tier = "Large Account"
        else:
            _t5_acct_tier = "Key Account"

        # ── Pair-level loyalty metrics ─────────────────────────────────────
        _t5_n_orders   = len(_t5_pair_df)
        _t5_price_std  = float(_t5_pair_df["unit_retail"].std(ddof=0)) if _t5_n_orders > 1 else 0.0

        if _t5_n_orders <= 5:
            _t5_loyalty = "Emerging Relationship"
        elif _t5_n_orders > 10 and _t5_price_std < 2.00:
            _t5_loyalty = "Relationship-Driven Buyer"
        elif _t5_n_orders > 5 and _t5_price_std >= 2.00:
            _t5_loyalty = "Price-Driven Buyer"
        else:
            _t5_loyalty = "Emerging Relationship"

        _t5_avg_price      = float(_t5_pair_df["unit_retail"].mean())
        _t5_papp           = float(_t5_latest_pair.get("pair_avg_price_paid", _t5_avg_price))
        _t5_first_order    = _t5_pair_df["invoice_date"].min()
        _t5_last_order     = _t5_pair_df["invoice_date"].max()
        _t5_months_active  = max(
            1,
            (_t5_last_order.year - _t5_first_order.year) * 12
            + (_t5_last_order.month - _t5_first_order.month),
        )
        _t5_orders_per_month = _t5_n_orders / _t5_months_active

        # ── Competitor response data ───────────────────────────────────────
        _t5_comp_info = _get_competitor_status(df, selected_customer, selected_part)
        _t5_has_comp  = _t5_comp_info["has_competitor"]

        # Units with / without competitor (≥2 obs each required)
        _t5_with_comp    = _t5_pair_df[_t5_pair_df["competitor_present"] == 1]["units_sold"]
        _t5_without_comp = _t5_pair_df[_t5_pair_df["competitor_present"] == 0]["units_sold"]
        _t5_comp_valid   = len(_t5_with_comp) >= 2 and len(_t5_without_comp) >= 2

        # Price deviation impact: above $2.00 over papp threshold
        _t5_above_papp   = _t5_pair_df[
            _t5_pair_df["unit_retail"] >= _t5_papp + 2.00
        ]["units_sold"]
        _t5_at_papp      = _t5_pair_df[
            _t5_pair_df["unit_retail"] < _t5_papp + 2.00
        ]["units_sold"]

        # ══════════════════════════════════════════════════════════════════
        # 1. Loyalty Card — 4 metrics, full width
        # ══════════════════════════════════════════════════════════════════
        _t5_loyalty_color = {
            "Relationship-Driven Buyer": "#3b82f6",
            "Price-Driven Buyer":        "#f59e0b",
            "Emerging Relationship":     "#6b7280",
        }.get(_t5_loyalty, "#6b7280")

        st.markdown(
            f"""
            <div style="border:1px solid {_t5_loyalty_color};border-radius:8px;
                        padding:14px 20px;margin-bottom:18px;background:#111827;">
              <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
                <span style="font-size:13px;color:#9ca3af;">Loyalty Profile</span>
                <span style="background:{_t5_loyalty_color};color:#fff;
                             font-size:12px;font-weight:600;padding:2px 10px;
                             border-radius:12px;">{_t5_loyalty}</span>
              </div>
              <div style="font-size:13px;color:#9ca3af;">
                {_t5_n_orders} orders &nbsp;|&nbsp;
                Price variation: ${_t5_price_std:.2f} &nbsp;|&nbsp;
                Avg price: ${_t5_avg_price:.2f} &nbsp;|&nbsp;
                {_t5_orders_per_month:.1f} orders/month
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── 4-metric row ───────────────────────────────────────────────────
        _t5_c1, _t5_c2, _t5_c3, _t5_c4 = st.columns(4)
        with _t5_c1:
            st.metric(
                "Annual Spend",
                f"${_t5_annual_spend_raw:,.0f}" if _t5_annual_spend_raw > 0 else "Not Available",
            )
        with _t5_c2:
            st.metric("Account Tier", _t5_acct_tier)
        with _t5_c3:
            st.metric(
                "Active Since",
                _t5_active_since.strftime("%b %Y") if pd.notnull(_t5_active_since) else "Not Available",
            )
        with _t5_c4:
            st.metric("Distinct Parts Purchased", str(_t5_distinct_parts))

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════
        # 2. Two-column row: Loyalty Profile detail | Behavioral Analytics
        # ══════════════════════════════════════════════════════════════════
        _t5_col_l, _t5_col_r = st.columns([1, 1], gap="large")

        # ── LEFT: Loyalty profile (expanded) ──────────────────────────────
        with _t5_col_l:
            st.markdown("##### Loyalty Profile")

            st.markdown(
                f"""
                <div style="background:#1e293b;border-radius:8px;
                            padding:14px 16px;margin-bottom:10px;">
                  <table style="width:100%;font-size:13px;color:#f9fafb;
                                border-collapse:collapse;">
                    <tr>
                      <td style="padding:4px 0;color:#9ca3af;">Classification</td>
                      <td style="text-align:right;font-weight:600;
                                 color:{_t5_loyalty_color};">{_t5_loyalty}</td>
                    </tr>
                    <tr>
                      <td style="padding:4px 0;color:#9ca3af;">Total Orders (this part)</td>
                      <td style="text-align:right;font-weight:600;">{_t5_n_orders}</td>
                    </tr>
                    <tr>
                      <td style="padding:4px 0;color:#9ca3af;">Price Variation</td>
                      <td style="text-align:right;font-weight:600;">
                        ${_t5_price_std:.2f}
                      </td>
                    </tr>
                    <tr>
                      <td style="padding:4px 0;color:#9ca3af;">Avg Selling Price</td>
                      <td style="text-align:right;font-weight:600;">
                        ${_t5_avg_price:.2f}
                      </td>
                    </tr>
                    <tr>
                      <td style="padding:4px 0;color:#9ca3af;">Customer's Historical Price Point</td>
                      <td style="text-align:right;font-weight:600;">
                        ${_t5_papp:.2f}
                      </td>
                    </tr>
                    <tr>
                      <td style="padding:4px 0;color:#9ca3af;">First Order</td>
                      <td style="text-align:right;font-weight:600;">
                        {_t5_first_order.strftime("%b %d, %Y")}
                      </td>
                    </tr>
                    <tr>
                      <td style="padding:4px 0;color:#9ca3af;">Most Recent Order</td>
                      <td style="text-align:right;font-weight:600;">
                        {_t5_last_order.strftime("%b %d, %Y")}
                      </td>
                    </tr>
                    <tr>
                      <td style="padding:4px 0;color:#9ca3af;">Order Frequency</td>
                      <td style="text-align:right;font-weight:600;">
                        {_t5_orders_per_month:.1f} per month
                      </td>
                    </tr>
                  </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── RIGHT: Behavioral analytics ────────────────────────────────────
        with _t5_col_r:
            st.markdown("##### Behavioral Analytics")

            # Scatter: Price vs Quantity (no trendline — exact title per spec)
            _t5_x_col = "unit_price_actual" if "unit_price_actual" in _t5_pair_df.columns else "pair_avg_price_paid"
            _t5_fig_scatter = go.Figure()
            _t5_fig_scatter.add_trace(
                go.Scatter(
                    x=_t5_pair_df[_t5_x_col],
                    y=_t5_pair_df["units_sold"],
                    mode="markers",
                    marker=dict(
                        color="#3b82f6",
                        size=8,
                        opacity=0.7,
                        line=dict(width=1, color="#1e40af"),
                    ),
                    hovertemplate="Negotiated Price: $%{x:.2f}<br>Units: %{y}<extra></extra>",
                )
            )
            _t5_fig_scatter.update_layout(
                **{**_LAYOUT_COMMON, "height": 260},
                title=dict(
                    text="Price vs. Quantity Sold",
                    font=dict(size=13),
                    x=0,
                ),
                xaxis=dict(
                    title="Actual Negotiated Price ($)",
                    titlefont=dict(size=_AXIS_FONT),
                    tickfont=dict(size=_TICK_FONT),
                    tickprefix="$",
                ),
                yaxis=dict(
                    title="Units Sold",
                    titlefont=dict(size=_AXIS_FONT),
                    tickfont=dict(size=_TICK_FONT),
                ),
            )
            st.plotly_chart(
                _t5_fig_scatter,
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # Interpretive caption — tight x-axis range = consistent pricing
            _t5_price_range = float(_t5_pair_df[_t5_x_col].max() - _t5_pair_df[_t5_x_col].min())
            if _t5_price_range < 2.00:
                _t5_scatter_caption = (
                    "Prices are tightly clustered — this customer has been "
                    "quoted consistently with little variation over time."
                )
            elif _t5_price_range < 5.00:
                _t5_scatter_caption = (
                    "Moderate price spread across orders — some negotiation "
                    "or discount variability is present in this account."
                )
            else:
                _t5_scatter_caption = (
                    "Wide price spread across orders — significant price "
                    "variability suggests active negotiation or changing "
                    "competitive conditions."
                )
            st.markdown(
                f"<div style='font-size:12px;color:#6b7280;margin-top:-8px;"
                f"margin-bottom:8px;line-height:1.5;'>{_t5_scatter_caption}</div>",
                unsafe_allow_html=True,
            )
            if _t5_has_comp and _t5_comp_valid:
                _t5_avg_with    = float(_t5_with_comp.mean())
                _t5_avg_without = float(_t5_without_comp.mean())
                _t5_cr1, _t5_cr2 = st.columns(2)
                with _t5_cr1:
                    st.metric(
                        "Avg Units (Competitor Present)",
                        f"{_t5_avg_with:.1f}",
                    )
                with _t5_cr2:
                    st.metric(
                        "Avg Units (No Competitor)",
                        f"{_t5_avg_without:.1f}",
                    )
            elif _t5_has_comp:
                st.markdown(
                    "<div style='font-size:13px;color:#9ca3af;margin-top:4px;'>"
                    "Insufficient data to compare units with vs. without competitor "
                    "(need at least 2 observations each).</div>",
                    unsafe_allow_html=True,
                )

            # Price deviation impact
            _t5_di1, _t5_di2 = st.columns(2)
            with _t5_di1:
                _t5_avg_at_papp_thresh = (
                    float(_t5_at_papp.mean()) if not _t5_at_papp.empty else 0.0
                )
                st.metric(
                    "Avg Units at ≤ Hist. Price + $2.00",
                    f"{_t5_avg_at_papp_thresh:.1f}",
                )
            with _t5_di2:
                _t5_avg_above_papp_thresh = (
                    float(_t5_above_papp.mean()) if not _t5_above_papp.empty else 0.0
                )
                st.metric(
                    "Avg Units at > Hist. Price + $2.00",
                    f"{_t5_avg_above_papp_thresh:.1f}",
                )

        # ══════════════════════════════════════════════════════════════════
        # 3. Recommended Stance — full width
        # ══════════════════════════════════════════════════════════════════
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("##### Recommended Stance")

        # 4 branches: Red / Emerging / Relationship+Blue+Green / Price-Driven+Blue
        _t5_disclaimer = (
            "<div style=\"font-size:11px;color:#9ca3af;margin-top:10px;"
            "font-style:italic;\">"
            "Based on historical price patterns — verify with your account knowledge."
            "</div>"
        )

        if status_label == "Cannot Compete Profitably":
            _t5_stance_color  = "#ef4444"
            _t5_stance_border = "#ef4444"
            _t5_stance_bg     = "#1c0a0a"
            _t5_stance_text   = (
                "Pricing is below cost on this part. Do not proceed without a price "
                "correction. Escalate to your manager before the next negotiation — "
                "there is no margin to work with at current levels."
            )
        elif _t5_loyalty == "Emerging Relationship":
            _t5_stance_color  = "#6b7280"
            _t5_stance_border = "#6b7280"
            _t5_stance_bg     = "#111827"
            _t5_stance_text   = (
                "This is a new account with limited history. Focus on winning trust "
                "and volume — do not push for margin yet. Hold price close to the "
                "customer's historical level and check back in 60–90 days."
            )
        elif _t5_loyalty == "Relationship-Driven Buyer" and status_label in (
            "Uncontested — Raise With Confidence", "Viable — Compete and Hold"
        ):
            _t5_stance_color  = "#3b82f6"
            _t5_stance_border = "#3b82f6"
            _t5_stance_bg     = "#0f1b2e"
            _t5_stance_text   = (
                "This customer buys consistently and is not shopping around on price. "
                "You can hold firm or nudge price up — they are unlikely to walk away "
                "over a small increase. Focus on service and reliability."
            )
        else:
            # Price-Driven Buyer — or any residual case
            _t5_stance_color  = "#f59e0b"
            _t5_stance_border = "#f59e0b"
            _t5_stance_bg     = "#1c1007"
            _t5_stance_text   = (
                "This customer is price-sensitive. Any increase above their historical "
                "price point carries real risk of losing the order. Do not move price "
                "without knowing what the competitor is offering first."
            )

        st.markdown(
            f"""
            <div style="border:1px solid {_t5_stance_border};border-radius:8px;
                        padding:16px 20px;background:{_t5_stance_bg};">
              <div style="font-size:13px;font-weight:600;color:{_t5_stance_color};
                          margin-bottom:6px;">
                Stance for {selected_customer}
              </div>
              <div style="font-size:14px;color:#f9fafb;line-height:1.6;">
                {_t5_stance_text}
              </div>
              {_t5_disclaimer}
            </div>
            """,
            unsafe_allow_html=True,
        )
