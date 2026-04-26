"""
generate_synthetic_data.py
==========================
Generates a realistic synthetic dataset that mirrors the schema of the
original pricing analytics training data. No real company names, no real
pricing, no identifiable information.

Output: synthetic_data.xlsx  (same sheet/column structure the app expects)
        synthetic_data.csv   (flat version for inspection)

Run:
    python generate_synthetic_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

# ── Configuration ─────────────────────────────────────────────────────────────
N_CUSTOMERS   = 80
N_PARTS       = 120
N_ROWS        = 5000
DATE_START    = "2021-01-01"
DATE_END      = "2023-12-31"


# ── 1. Customer & Part pools ──────────────────────────────────────────────────

def _make_customers(n: int) -> pd.DataFrame:
    """Generate customer profiles."""
    ids = [f"CUST-{str(i).zfill(4)}" for i in range(1, n + 1)]
    is_national = RNG.choice([0, 1], size=n, p=[0.7, 0.3])
    # Annual spend ~ LogNormal: mean ~$180k, wide range
    annual_spend = RNG.lognormal(mean=12.0, sigma=1.2, size=n).clip(5_000, 2_000_000)
    return pd.DataFrame({
        "customer_id":      ids,
        "is_national_account": is_national,
        "base_annual_spend": annual_spend,
    })


def _make_parts(n: int) -> pd.DataFrame:
    """Generate part catalogue with price tiers and OEM flag."""
    ids = [f"PART-{str(i).zfill(5)}" for i in range(1, n + 1)]
    is_oem = RNG.choice([0, 1], size=n, p=[0.55, 0.45])

    # Base catalogue price: OEM parts tend to be pricier
    base_price = np.where(
        is_oem,
        RNG.lognormal(mean=4.5, sigma=0.8, size=n).clip(20, 2500),
        RNG.lognormal(mean=3.8, sigma=0.9, size=n).clip(8, 800),
    )
    # Cost: 40–65% of catalogue price
    cost_pct = RNG.uniform(0.40, 0.65, size=n)

    return pd.DataFrame({
        "part_id":   ids,
        "is_oem_part": is_oem,
        "catalogue_price": base_price,
        "cost_pct":  cost_pct,
    })


# ── 2. Macro time-series ──────────────────────────────────────────────────────

def _make_macro_calendar(start: str, end: str) -> pd.DataFrame:
    """
    Monthly macro index values. Column names are generic descriptors —
    no FRED codes, no traceable series names.
    """
    months = pd.date_range(start=start, end=end, freq="MS")
    n = len(months)

    # Simulate realistic index behaviour: smooth trend + seasonal + noise
    def _index(base, trend_per_yr, seasonal_amp, noise_std):
        t = np.arange(n)
        trend   = base + trend_per_yr * t / 12
        season  = seasonal_amp * np.sin(2 * np.pi * t / 12)
        noise   = RNG.normal(0, noise_std, n)
        return (trend + season + noise).clip(base * 0.6, base * 1.6)

    return pd.DataFrame({
        "invoice_month_ts":          months,
        "macro_freight_index":       _index(100, 3.0, 5.0, 2.5),
        "macro_truck_tonnage_index": _index(100, 1.5, 4.0, 2.0),
        "macro_vehicle_miles_index": _index(100, 2.0, 6.0, 1.8),
        "macro_heavy_truck_sales":   _index(100, 2.5, 8.0, 3.5),
        "macro_diesel_price_index":  _index(100, 4.0, 7.0, 4.0),
        "market_seasonality_index":  _index(100, 0.5, 10.0, 1.5),
    })


# ── 3. Transaction generator ──────────────────────────────────────────────────

def _generate_transactions(
    customers: pd.DataFrame,
    parts: pd.DataFrame,
    macro: pd.DataFrame,
    n_rows: int,
) -> pd.DataFrame:

    # Each transaction = random customer × random part
    cust_idx = RNG.integers(0, len(customers), size=n_rows)
    part_idx = RNG.integers(0, len(parts),     size=n_rows)

    cust = customers.iloc[cust_idx].reset_index(drop=True)
    part = parts.iloc[part_idx].reset_index(drop=True)

    # Random invoice dates (weighted toward recent)
    all_dates = pd.date_range(DATE_START, DATE_END, freq="D")
    weights = np.linspace(0.5, 1.5, len(all_dates))
    weights /= weights.sum()
    chosen_dates = RNG.choice(all_dates, size=n_rows, p=weights)
    invoice_dates = pd.to_datetime(chosen_dates)

    invoice_month = invoice_dates.month
    invoice_year  = invoice_dates.year

    # ── Pricing ──────────────────────────────────────────────────────────────
    # National accounts get 5–15% discount; regular get 0–10%
    discount_rate = np.where(
        cust["is_national_account"].values == 1,
        RNG.uniform(0.05, 0.15, n_rows),
        RNG.uniform(0.00, 0.10, n_rows),
    )
    unit_price_catalogue = part["catalogue_price"].values
    unit_price_actual    = unit_price_catalogue * (1 - discount_rate)
    unit_cost            = unit_price_catalogue * part["cost_pct"].values

    # ── Quantities ────────────────────────────────────────────────────────────
    # Demand driven by price-to-cost ratio and OEM flag with seasonal noise
    season_factor = 1 + 0.15 * np.sin(2 * np.pi * invoice_month / 12)
    base_qty = np.where(
        part["is_oem_part"].values == 1,
        RNG.lognormal(1.5, 0.8, n_rows),
        RNG.lognormal(2.0, 0.9, n_rows),
    )
    quantity = np.clip(base_qty * season_factor, 1, None).round().astype(int)

    # ── Revenue & Cost ────────────────────────────────────────────────────────
    actual_revenue   = unit_price_actual * quantity
    catalogue_revenue = unit_price_catalogue * quantity
    cost_of_goods    = unit_cost * quantity
    gross_profit     = actual_revenue - cost_of_goods

    # ── Competitor pricing ────────────────────────────────────────────────────
    comp_present = RNG.choice([0, 1], size=n_rows, p=[0.35, 0.65])
    comp_price   = np.where(
        comp_present == 1,
        unit_price_actual * RNG.uniform(0.85, 1.20, n_rows),
        0.0,
    )
    price_gap_vs_competitor = np.where(
        comp_present == 1,
        unit_price_actual - comp_price,
        np.nan,
    )

    # ── Customer behavior features ────────────────────────────────────────────
    # Approximated per-row — real values come from aggregation in model pipeline
    customer_avg_monthly_spend = cust["base_annual_spend"].values / 12 * RNG.uniform(0.7, 1.3, n_rows)
    customer_spend_zscore = (
        (cust["base_annual_spend"].values - cust["base_annual_spend"].values.mean())
        / (cust["base_annual_spend"].values.std() + 1e-6)
    )
    pair_transaction_count = RNG.integers(1, 30, n_rows)
    pair_avg_price_paid    = unit_price_actual * RNG.uniform(0.9, 1.1, n_rows)
    pair_avg_order_qty     = quantity * RNG.uniform(0.8, 1.2, n_rows)
    cust_share_of_part_vol = RNG.uniform(0.01, 0.40, n_rows)

    # ── Price position features ───────────────────────────────────────────────
    part_price_range_lo = unit_price_actual * 0.7
    part_price_range_hi = unit_price_actual * 1.3
    price_percentile    = (
        (unit_price_actual - part_price_range_lo)
        / (part_price_range_hi - part_price_range_lo + 1e-6)
    )
    price_percentile = np.clip(price_percentile, 0, 1)
    price_zscore_in_part   = RNG.normal(0, 1, n_rows)
    price_vs_customer_habit = unit_price_actual - pair_avg_price_paid
    price_to_cost_ratio    = unit_price_actual / (unit_cost + 1e-6)
    price_change_vs_prior  = RNG.normal(0, 2.5, n_rows)

    # ── Basket / co-purchase features ─────────────────────────────────────────
    is_multi_item = RNG.choice([0, 1], n_rows, p=[0.3, 0.7])
    is_bulk_order = (quantity >= 10).astype(int)
    basket_item_count  = np.where(is_multi_item, RNG.integers(2, 8, n_rows), 1)
    basket_total_value = actual_revenue * basket_item_count * RNG.uniform(0.8, 1.2, n_rows)
    basket_share_pct   = np.clip(actual_revenue / (basket_total_value + 1e-6), 0, 1)
    monthly_txn_seq    = RNG.integers(1, 5, n_rows)

    # ── Margin ────────────────────────────────────────────────────────────────
    gross_margin_pct = np.clip(gross_profit / (actual_revenue + 1e-6), 0, 1)

    # ── Annual spend (customer-level) ─────────────────────────────────────────
    customer_annual_spend = cust["base_annual_spend"].values

    # ── Merge macro by month/year ─────────────────────────────────────────────
    tx_df = pd.DataFrame({
        "invoice_month": invoice_month,
        "invoice_year":  invoice_year,
    })
    macro_lookup = macro.copy()
    macro_lookup["invoice_month"] = macro_lookup["invoice_month_ts"].dt.month
    macro_lookup["invoice_year"]  = macro_lookup["invoice_month_ts"].dt.year
    macro_lookup = macro_lookup.drop(columns=["invoice_month_ts"])
    tx_df = tx_df.merge(macro_lookup, on=["invoice_month", "invoice_year"], how="left")

    # ── Assemble final DataFrame ──────────────────────────────────────────────
    df = pd.DataFrame({
        # Identifiers
        "Customer ID":                  cust["customer_id"].values,
        "Part ID":                      part["part_id"].values,
        "Invoice Date":                 invoice_dates,
        "Invoice Month":                invoice_month,
        "Invoice Year":                 invoice_year,
        # Customer flags
        "Is National Account":          cust["is_national_account"].values,
        "Is OEM part":                  part["is_oem_part"].values,
        # Transaction
        "Units Sold":                   quantity,
        "Retail Sales":                 catalogue_revenue.round(2),
        "Cost of Goods":                cost_of_goods.round(2),
        "Actual Revenue":               actual_revenue.round(2),
        "Gross Profit":                 gross_profit.round(2),
        "Competitor Price":             np.where(comp_present, comp_price.round(2), 0.0),
        "Annual Spend":                 customer_annual_spend.round(2),
        # Per-unit pricing
        "Unit Price Actual":            unit_price_actual.round(4),
        "Unit Retail Price":            unit_price_catalogue.round(4),
        "Unit Cost":                    unit_cost.round(4),
        # Competitor signals
        "Competitor Present":           comp_present,
        "Price Gap vs Competitor":      np.where(comp_present, price_gap_vs_competitor.round(4), np.nan),
        # Price position
        "Discount Rate":                discount_rate.round(4),
        "Price Position in Part Range": price_percentile.round(4),
        "Prize Zscore within Part":     price_zscore_in_part.round(4),   # column name preserved (typo intentional for schema match)
        "Price Deviation from Habit":   price_vs_customer_habit.round(4),
        "Price to Cost Ratio":          price_to_cost_ratio.round(4),
        "Price Change vs Prior Month":  price_change_vs_prior.round(4),
        # Customer behavior
        "Customer Avg Monthly Spend":   customer_avg_monthly_spend.round(2),
        "Customer Spend Zscore":        customer_spend_zscore.round(4),
        "Pair Transaction Count":       pair_transaction_count,
        "Pair Avg Price Paid":          pair_avg_price_paid.round(4),
        "Pair Avg Order Quantity":      pair_avg_order_qty.round(2),
        "Customer Share of Part Volume": cust_share_of_part_vol.round(4),
        # Basket / co-purchase
        "Basket Total Value":           basket_total_value.round(2),
        "Basket Share pct":             basket_share_pct.round(4),
        "Basket Item Count":            basket_item_count,
        "Is Multi Item Order":          is_multi_item,
        "Is Bulk Order":                is_bulk_order,
        "Montly Transaction Sequence":  monthly_txn_seq,   # typo preserved for schema match
        # Margin
        "Gross Margin Pct":             gross_margin_pct.round(4),
        # Macro
        "Market Seasonality Index":     tx_df["market_seasonality_index"].round(4).values,
        "Freight Shipment Index":       tx_df["macro_freight_index"].round(4).values,
        "Truck Tonnage Index":          tx_df["macro_truck_tonnage_index"].round(4).values,
        "Vehicle Miles Traveled":       tx_df["macro_vehicle_miles_index"].round(4).values,
        "Heavy Truck Sales SAAR":       tx_df["macro_heavy_truck_sales"].round(4).values,
        "Diesel Price Index":           tx_df["macro_diesel_price_index"].round(4).values,
    })

    return df


# ── 4. Write Excel (two-row header matching original schema) ──────────────────

def _write_excel(df: pd.DataFrame, path: Path) -> None:
    """
    Write df to Excel with a group-label header row above the column names,
    matching the two-row header structure of the original dataset.
    """
    # Group label row — same groupings as original
    group_labels = {
        "Customer ID":                   "Identifiers",
        "Part ID":                       "Identifiers",
        "Invoice Date":                  "Identifiers",
        "Invoice Month":                 "Identifiers",
        "Invoice Year":                  "Identifiers",
        "Is National Account":           "Customer Flags",
        "Is OEM part":                   "Customer Flags",
        "Units Sold":                    "Raw Transaction",
        "Retail Sales":                  "Raw Transaction",
        "Cost of Goods":                 "Raw Transaction",
        "Actual Revenue":                "Raw Transaction",
        "Gross Profit":                  "Raw Transaction",
        "Competitor Price":              "Raw Transaction",
        "Annual Spend":                  "Raw Transaction",
        "Unit Price Actual":             "Per-Unit Pricing",
        "Unit Retail Price":             "Per-Unit Pricing",
        "Unit Cost":                     "Per-Unit Pricing",
        "Competitor Present":            "Competitor Signals",
        "Price Gap vs Competitor":       "Competitor Signals",
        "Discount Rate":                 "Price Position",
        "Price Position in Part Range":  "Price Position",
        "Prize Zscore within Part":      "Price Position",
        "Price Deviation from Habit":    "Price Position",
        "Price to Cost Ratio":           "Price Position",
        "Price Change vs Prior Month":   "Price Position",
        "Customer Avg Monthly Spend":    "Customer Behavior",
        "Customer Spend Zscore":         "Customer Behavior",
        "Pair Transaction Count":        "Customer Behavior",
        "Pair Avg Price Paid":           "Customer Behavior",
        "Pair Avg Order Quantity":       "Customer Behavior",
        "Customer Share of Part Volume": "Customer Behavior",
        "Basket Total Value":            "Basket / Co-Purchase",
        "Basket Share pct":              "Basket / Co-Purchase",
        "Basket Item Count":             "Basket / Co-Purchase",
        "Is Multi Item Order":           "Basket / Co-Purchase",
        "Is Bulk Order":                 "Basket / Co-Purchase",
        "Montly Transaction Sequence":   "Basket / Co-Purchase",
        "Gross Margin Pct":              "Margin",
        "Market Seasonality Index":      "Macro Economics",
        "Freight Shipment Index":        "Macro Economics",
        "Truck Tonnage Index":           "Macro Economics",
        "Vehicle Miles Traveled":        "Macro Economics",
        "Heavy Truck Sales SAAR":        "Macro Economics",
        "Diesel Price Index":            "Macro Economics",
    }

    group_row = [group_labels.get(c, "") for c in df.columns]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Row 0: group labels, Row 1: column display names, Rows 2+: data
        group_df  = pd.DataFrame([group_row], columns=df.columns)
        header_df = pd.DataFrame([df.columns.tolist()], columns=df.columns)
        combined  = pd.concat([group_df, header_df, df], ignore_index=True)
        combined.to_excel(writer, sheet_name="Training Data", index=False, header=False)

    print(f"[OK] Excel written → {path}  ({len(df):,} rows, {len(df.columns)} columns)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating synthetic pricing dataset...")

    customers = _make_customers(N_CUSTOMERS)
    parts     = _make_parts(N_PARTS)
    macro     = _make_macro_calendar(DATE_START, DATE_END)
    df        = _generate_transactions(customers, parts, macro, N_ROWS)

    out_dir = Path(__file__).parent
    excel_path = out_dir / "synthetic_data.xlsx"
    csv_path   = out_dir / "synthetic_data.csv"

    _write_excel(df, excel_path)
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV written   → {csv_path}  ({len(df):,} rows)")
    print(f"\nColumn count : {len(df.columns)}")
    print(f"Customers    : {df['Customer ID'].nunique()}")
    print(f"Parts        : {df['Part ID'].nunique()}")
    print(f"Date range   : {df['Invoice Date'].min().date()} → {df['Invoice Date'].max().date()}")
    print(f"Price range  : ${df['Unit Price Actual'].min():.2f} – ${df['Unit Price Actual'].max():.2f}")
    print(f"Qty range    : {df['Units Sold'].min()} – {df['Units Sold'].max()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
