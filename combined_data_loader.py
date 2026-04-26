"""
combined_data_loader.py
=======================
Loads synthetic_data.xlsx and produces a single clean DataFrame for both:
  1. The Streamlit app  — all display columns with canonical snake_case names
  2. The elasticity model — column aliases matching elasticity_model.py internals

──────────────────────────────────────────────────────────────────────────────
MACRO COLUMN MAPPING  (confirmed by user — do not alter without explicit approval)
──────────────────────────────────────────────────────────────────────────────
  xlsx display name           elasticity_model.py name    Data series
  ──────────────────────────  ──────────────────────────  ──────────────────────────────
  Market Seasonality Index    (no model mapping)          Internal seasonality index
  Freight Shipment Index      macro_freight_idx           Freight shipment volume index
  Truck Tonnage Index         macro_truck_tonnage_idx     Truck tonnage index
  Vehicle Miles Traveled      macro_vehicle_miles_idx     Vehicle miles traveled index
  Heavy Truck Sales SAAR      macro_heavy_truck_sales_idx Heavy truck unit sales index
  Diesel Price Index          macro_diesel_price_idx      Diesel fuel price index

──────────────────────────────────────────────────────────────────────────────
CRITICAL SEMANTIC RULE — ENFORCED HERE, DO NOT CHANGE
──────────────────────────────────────────────────────────────────────────────
  'Actual Revenue' in xlsx  →  model column 'retail_sales'
  This ensures:  unit_retail = retail_sales / quantity = unit_price_actual
                 (the negotiated price per unit, NOT the catalogue list price)
  The xlsx 'Retail Sales' column (= catalogue price × qty) is renamed
  'catalogue_revenue' everywhere so it cannot be accidentally used in model
  training or in price calculations.

──────────────────────────────────────────────────────────────────────────────
NAMING DISCREPANCIES RESOLVED HERE
──────────────────────────────────────────────────────────────────────────────
  xlsx 'Price Deviation from Habit'  →  spec name 'price_vs_customer_habit'
  xlsx 'Annual Spend'                →  spec name 'customer_annual_spend'
  xlsx 'Prize Zscore within Part'    →  corrected to 'price_zscore_in_part'
                                        ('Prize' is a typo in the source file)
  xlsx 'Montly Transaction Sequence' →  corrected to 'monthly_txn_sequence'
                                        ('Montly' is a typo in the source file)
  xlsx 'Basket  Item Count'          →  'basket_item_count'
                                        (double space in source is normalised)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column rename tables
# ---------------------------------------------------------------------------

# xlsx display name (stripped, as it appears in row 1) → canonical spec name.
# Keys must match the exact string after str.strip(); double-spaces are handled
# separately via the whitespace-collapse fallback in _rename_columns().
_DISPLAY_TO_SPEC: dict[str, str] = {
    # ── Identifiers ──────────────────────────────────────────────────────────
    "Customer ID":                    "customer_id",
    "Part ID":                        "part_id",
    "Invoice Date":                   "invoice_date",
    "Invoice Month":                  "invoice_month",
    "Invoice Year":                   "invoice_year",
    # ── Customer Flags ───────────────────────────────────────────────────────
    "Is National Account":            "is_national_account",
    "Is OEM part":                    "is_oem_part",
    # ── Raw Transaction ──────────────────────────────────────────────────────
    "Units Sold":                     "units_sold",
    "Retail Sales":                   "catalogue_revenue",       # catalogue price × qty — NOT negotiated
    "Cost of Goods":                  "cost_of_goods",
    "Actual Revenue":                 "actual_revenue",
    "Gross Profit":                   "gross_profit",
    "Competitor Price":               "competitor_price",
    "Annual Spend":                   "customer_annual_spend",   # spec name normalisation
    # ── Per-Unit Pricing ─────────────────────────────────────────────────────
    "Unit Price Actual":              "unit_price_actual",
    "Unit Retail Price":              "unit_price_catalogue",
    "Unit Cost":                      "unit_cost",
    # ── Competitor Signals ───────────────────────────────────────────────────
    "Competitor Present":             "competitor_present",
    "Price Gap vs Competitor":        "price_gap_vs_competitor",
    # ── Price Position ───────────────────────────────────────────────────────
    "Discount Rate":                  "discount_rate",
    "Price Position in Part Range":   "price_percentile_in_part",
    "Prize Zscore within Part":       "price_zscore_in_part",    # 'Prize' typo corrected
    "Price Deviation from Habit":     "price_vs_customer_habit", # spec name normalisation
    "Price to Cost Ratio":            "price_to_cost_ratio",
    "Price Change vs Prior Month":    "price_change_vs_prior_month",
    # ── Customer Behavior ────────────────────────────────────────────────────
    "Customer Avg Monthly Spend":     "customer_avg_monthly_spend",
    "Customer Spend Zscore":          "customer_spend_zscore",
    "Pair Transaction Count":         "pair_transaction_count",
    "Pair Avg Price Paid":            "pair_avg_price_paid",
    "Pair Avg Order Quantity":        "pair_avg_order_quantity",
    "Customer Share of Part Volume":  "customer_share_of_part_volume",
    # ── Basket / Co-Purchase ─────────────────────────────────────────────────
    "Basket Total Value":             "basket_total_value",
    "Basket Share pct":               "basket_share_pct",
    "Basket Item Count":              "basket_item_count",        # double-space normalised below
    "Is Multi Item Order":            "is_multi_item_order",
    "Is Bulk Order":                  "is_bulk_order",
    "Montly Transaction Sequence":    "monthly_txn_sequence",    # 'Montly' typo corrected
    # ── Margin ───────────────────────────────────────────────────────────────
    "Gross Margin Pct":               "gross_margin_pct",
    # ── Macro Economics ──────────────────────────────────────────────────────
    # Market Seasonality Index has no elasticity_model.py mapping.
    "Market Seasonality Index":       "market_seasonality_index",
    "Freight Shipment Index":         "macro_freight_idx",
    "Truck Tonnage Index":            "macro_truck_tonnage_idx",
    "Vehicle Miles Traveled":         "macro_vehicle_miles_idx",
    "Heavy Truck Sales SAAR":         "macro_heavy_truck_sales_idx",
    "Diesel Price Index":             "macro_diesel_price_idx",
}

# Spec column name → elasticity_model.py internal alias.
# These are added as extra columns (not replacements) so the app can use
# spec names while model internals can use their expected names.
_SPEC_TO_MODEL: dict[str, str] = {
    "customer_id":           "CustID",
    "part_id":               "PartID",
    "units_sold":            "quantity",
    # CRITICAL — see header note: actual negotiated price, not catalogue
    "unit_price_actual":     "unit_retail",
    "unit_cost":             "unit_cogs",
    # CRITICAL — actual_revenue maps to retail_sales so unit_retail = negotiated price
    "actual_revenue":        "retail_sales",
    "gross_profit":          "posgp",
    "cost_of_goods":         "cogs",
    "competitor_price":      "comp_price",
    "is_oem_part":           "oem_flag",
    "customer_annual_spend": "annual_spend",
}

# Columns that should never be coerced to numeric.
_NON_NUMERIC: frozenset[str] = frozenset({"customer_id", "part_id", "invoice_date"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_combined_dataset(path: str | Path = "synthetic_data.xlsx") -> pd.DataFrame:
    """
    Load synthetic_data.xlsx and return a fully prepared DataFrame.

    The returned DataFrame has:
    - All 44 source columns renamed to canonical spec names (snake_case)
    - Model-compatible column aliases added (CustID, PartID, unit_retail, etc.)
    - 'month' and 'year' integer columns derived from invoice_date
    - 'actual_sales' column (= actual_revenue) required by model internals
    - Numeric types enforced on all non-identifier columns
    - Source typos in column names silently corrected (documented above)
    - Validation check: unit_retail must match unit_price_actual within $0.01

    Parameters
    ----------
    path : str or Path
        Path to synthetic_data.xlsx. Defaults to 'synthetic_data.xlsx'
        in the current working directory.

    Returns
    -------
    pd.DataFrame
        4,971 rows × 50+ columns (44 source + aliases + derived).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Ensure synthetic_data.xlsx is in the working directory."
        )

    # ── 1. Read raw Excel — two-row header ──────────────────────────────────
    # Row 0: group labels (Identifiers, Customer Flags, …) — skip
    # Row 1: display column names — use as headers
    # Rows 2+: data
    raw = pd.read_excel(path, sheet_name="Training Data", header=None)

    display_names = [str(c).strip() for c in raw.iloc[1].tolist()]
    df = raw.iloc[2:].copy()
    df.columns = display_names
    df = df.reset_index(drop=True)

    # ── 2. Rename display names → spec names ────────────────────────────────
    df = _rename_columns(df)

    # ── 3. Enforce numeric types ─────────────────────────────────────────────
    for col in df.columns:
        if col not in _NON_NUMERIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 4. Parse invoice_date; derive month and year ─────────────────────────
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["month"] = df["invoice_date"].dt.month.astype("Int64")
    df["year"] = df["invoice_date"].dt.year.astype("Int64")

    # ── 5. Add model-compatible column aliases ────────────────────────────────
    for spec_col, model_col in _SPEC_TO_MODEL.items():
        if spec_col in df.columns:
            df[model_col] = df[spec_col]

    # 'actual_sales' is required by model internals (build_lookup_tables,
    # _merge_lookups_and_derive). It equals actual_revenue.
    if "actual_sales" not in df.columns and "actual_revenue" in df.columns:
        df["actual_sales"] = df["actual_revenue"]

    # ── 6. Validation: unit_retail must match unit_price_actual ──────────────
    _validate_unit_retail(df)

    return df


def get_model_ready_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the output of load_combined_dataset(), return a filtered copy that
    the model's internal functions (build_lookup_tables, train_model, etc.)
    can consume directly.

    Filters applied:
    - quantity > 0  (model requires positive demand)
    - unit_retail > 0  (model requires positive price)
    - invoice_date is not null  (required for temporal features)

    Does NOT call load_and_preprocess() — bypasses it entirely because
    load_and_preprocess() expects a CSV path and would recompute unit_retail
    from the wrong source column.
    """
    required_model_cols = [
        "CustID", "PartID", "quantity", "unit_retail", "unit_cogs",
        "actual_sales", "posgp", "cogs", "comp_price", "oem_flag",
        "annual_spend", "month", "year", "invoice_date", "discount_rate",
    ]
    missing = [c for c in required_model_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"get_model_ready_df(): DataFrame is missing required columns: {missing}\n"
            "Ensure load_combined_dataset() was called before this function."
        )

    mdf = df.copy()
    mdf = mdf[(mdf["quantity"] > 0) & (mdf["unit_retail"] > 0)]
    mdf = mdf.dropna(subset=["invoice_date"])
    mdf = mdf.reset_index(drop=True)

    return mdf


# ---------------------------------------------------------------------------
# Convenience accessor for the app sidebar
# ---------------------------------------------------------------------------

def get_customers(df: pd.DataFrame) -> list[str]:
    """Sorted list of unique customer IDs."""
    return sorted(df["customer_id"].dropna().unique().tolist())


def get_parts_for_customer(df: pd.DataFrame, customer_id: str) -> list[str]:
    """
    Parts this specific customer has purchased historically.
    Returns only parts present in the customer's transaction history —
    not the full part catalogue.
    """
    mask = df["customer_id"] == customer_id
    return sorted(df.loc[mask, "part_id"].dropna().unique().tolist())


def get_pair_baseline(
    df: pd.DataFrame, customer_id: str, part_id: str
) -> Optional[pd.Series]:
    """
    Return a single-row Series of representative baseline values for a
    customer-part pair, used to initialise the sidebar and seed price_sweep().

    The baseline row is the most recent transaction for this pair.
    Returns None if no rows exist for the given pair.
    """
    mask = (df["customer_id"] == customer_id) & (df["part_id"] == part_id)
    subset = df[mask].copy()
    if subset.empty:
        return None
    subset = subset.sort_values("invoice_date", ascending=False)
    return subset.iloc[0]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply _DISPLAY_TO_SPEC lookup to rename all columns.
    Falls back to whitespace-collapsed lookup (handles 'Basket  Item Count')
    then to auto snake_case for any column not in the lookup table.
    """
    rename_map: dict[str, str] = {}
    for col in df.columns:
        col_stripped = col.strip()

        if col_stripped in _DISPLAY_TO_SPEC:
            rename_map[col] = _DISPLAY_TO_SPEC[col_stripped]
            continue

        # Collapse multiple internal spaces and retry
        col_collapsed = re.sub(r"\s+", " ", col_stripped)
        if col_collapsed in _DISPLAY_TO_SPEC:
            rename_map[col] = _DISPLAY_TO_SPEC[col_collapsed]
            continue

        # Auto snake_case fallback — should not be reached for known columns
        auto = _auto_snake(col_stripped)
        rename_map[col] = auto

    return df.rename(columns=rename_map)


def _validate_unit_retail(df: pd.DataFrame) -> None:
    """
    Confirm that unit_retail (model alias for unit_price_actual) matches the
    source column within $0.01 mean absolute error.

    unit_retail = actual_revenue / units_sold must equal unit_price_actual.
    If mean divergence exceeds $0.01, a warning is printed so data issues are
    caught before model training or simulation produces incorrect predictions.
    """
    required = {"unit_retail", "unit_price_actual", "actual_revenue", "units_sold"}
    if not required.issubset(df.columns):
        warnings.warn(
            "[DATA VALIDATION SKIPPED] One or more columns required for "
            "unit_retail validation are missing: "
            f"{required - set(df.columns)}"
        )
        return

    computed = df["actual_revenue"] / df["units_sold"].replace(0, np.nan)
    diff = (computed - df["unit_price_actual"]).abs()
    mean_diff = diff.mean()
    max_diff = diff.max()
    pct_within_cent = (diff <= 0.01).mean() * 100

    if mean_diff > 0.01:
        warnings.warn(
            f"[DATA VALIDATION WARNING] unit_retail diverges from unit_price_actual.\n"
            f"  Mean absolute difference : ${mean_diff:.4f}\n"
            f"  Max  absolute difference : ${max_diff:.4f}\n"
            f"  Rows within $0.01        : {pct_within_cent:.1f}%\n"
            f"  Root cause: 'Actual Revenue' / 'Units Sold' does not reproduce\n"
            f"  'Unit Price Actual'. Check source data for rounding or data errors.\n"
            f"  IMPORTANT: If this persists, the elasticity model will train on\n"
            f"  incorrect per-unit prices and all simulations will be unreliable."
        )
    else:
        print(
            f"[DATA VALIDATION OK] unit_retail == unit_price_actual "
            f"(mean diff ${mean_diff:.4f}, max ${max_diff:.4f}, "
            f"{pct_within_cent:.1f}% of rows within $0.01)."
        )


def _auto_snake(s: str) -> str:
    """Convert a display column name to snake_case. Used as a fallback only."""
    s = s.strip().lower()
    s = re.sub(r"[\s/]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    return s
