"""
Pricing Elasticity Model
XGBoost demand model for B2B pricing elasticity and demand forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Optional, Tuple
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MACRO_FEATURES = [
    "macro_freight_idx", "macro_truck_tonnage_idx", "macro_vehicle_miles_idx", "macro_heavy_truck_sales_idx", "macro_diesel_price_idx"
]

NUMERIC_FEATURES = (
    ["unit_retail", "price_vs_comp", "catalogue_gap_rate", "annual_spend"]
    + MACRO_FEATURES
    + ["month", "year"]
    + [
        "cust_month_avg_spend", "month_qty_index", "annual_spend_zscore",
        "cust_part_tx_count", "cust_part_avg_price", "cust_part_avg_qty",
        "price_dev_from_habit", "comp_present", "comp_price_ratio",
        "price_pct_of_part_range", "price_zscore_within_part",
        "price_to_cost_ratio", "cust_share_of_part_vol",
        "part_price_mom_chg", "gross_margin_pct",
        "price_x_oem", "margin_pressure",
    ]
)

OHE_COLS = ["oem_flag"]


# ---------------------------------------------------------------------------
# Lookup table builder
# ---------------------------------------------------------------------------

def build_lookup_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute lookup DataFrames from training data."""

    cust_month_avg_spend = (
        df.groupby(["CustID", "month"])["actual_sales"]
        .mean()
        .reset_index()
        .rename(columns={"actual_sales": "cust_month_avg_spend"})
    )

    global_mean_qty = df["quantity"].mean()
    month_qty_index = (
        df.groupby("month")["quantity"]
        .mean()
        .div(global_mean_qty)
        .reset_index()
        .rename(columns={"quantity": "month_qty_index"})
    )

    annual_spend_stats = (
        df.groupby("CustID")["annual_spend"]
        .agg(cust_spend_mean="mean", cust_spend_std="std")
        .reset_index()
    )

    cust_part_agg = (
        df.groupby(["CustID", "PartID"])
        .agg(
            cust_part_tx_count=("quantity", "count"),
            cust_part_avg_price=("unit_retail", "mean"),
            cust_part_avg_qty=("quantity", "mean"),
        )
        .reset_index()
    )

    part_price_stats = (
        df.groupby("PartID")["unit_retail"]
        .agg(
            part_min_price="min",
            part_max_price="max",
            part_avg_price="mean",
            part_price_std="std",
        )
        .reset_index()
    )

    part_total_qty = (
        df.groupby("PartID")["quantity"].sum().rename("part_total_qty").reset_index()
    )
    cust_part_qty = df.groupby(["CustID", "PartID"])["quantity"].sum().reset_index()
    cust_part_qty = cust_part_qty.merge(part_total_qty, on="PartID")
    cust_part_qty["cust_share_of_part_vol"] = (
        cust_part_qty["quantity"] / cust_part_qty["part_total_qty"]
    )
    cust_share_of_part_vol = cust_part_qty[["CustID", "PartID", "cust_share_of_part_vol"]]

    monthly_avg = (
        df.groupby(["PartID", "year", "month"])["unit_retail"]
        .mean()
        .reset_index()
        .sort_values(["PartID", "year", "month"])
    )
    monthly_avg["part_price_mom_chg"] = monthly_avg.groupby("PartID")["unit_retail"].diff()
    part_price_mom_chg = monthly_avg[["PartID", "year", "month", "part_price_mom_chg"]]

    return {
        "cust_month_avg_spend": cust_month_avg_spend,
        "month_qty_index": month_qty_index,
        "annual_spend_stats": annual_spend_stats,
        "cust_part_agg": cust_part_agg,
        "part_price_stats": part_price_stats,
        "cust_share_of_part_vol": cust_share_of_part_vol,
        "part_price_mom_chg": part_price_mom_chg,
    }


# ---------------------------------------------------------------------------
# Internal: merge lookups and compute derived features
# ---------------------------------------------------------------------------

def _merge_lookups_and_derive(
    df: pd.DataFrame, lookups: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    df = df.merge(lookups["cust_month_avg_spend"], on=["CustID", "month"], how="left")
    df = df.merge(lookups["month_qty_index"], on="month", how="left")
    df = df.merge(lookups["annual_spend_stats"], on="CustID", how="left")
    df = df.merge(lookups["cust_part_agg"], on=["CustID", "PartID"], how="left")
    df = df.merge(lookups["part_price_stats"], on="PartID", how="left")
    df = df.merge(lookups["cust_share_of_part_vol"], on=["CustID", "PartID"], how="left")
    df = df.merge(lookups["part_price_mom_chg"], on=["PartID", "year", "month"], how="left")

    df["annual_spend_zscore"] = (df["annual_spend"] - df["cust_spend_mean"]) / (
        df["cust_spend_std"].fillna(0) + 1e-6
    )
    df["price_dev_from_habit"] = df["unit_retail"] - df["cust_part_avg_price"]
    df["comp_present"] = (df["comp_price"] > 0).astype(int)
    df["comp_price_ratio"] = np.where(
        df["comp_price"] > 0, df["unit_retail"] / df["comp_price"], np.nan
    )
    df["price_pct_of_part_range"] = (df["unit_retail"] - df["part_min_price"]) / (
        df["part_max_price"] - df["part_min_price"] + 1e-6
    )
    df["price_zscore_within_part"] = (df["unit_retail"] - df["part_avg_price"]) / (
        df["part_price_std"].fillna(1.0) + 1e-6
    )
    df["price_to_cost_ratio"] = df["unit_retail"] / (df["unit_cogs"] + 1e-6)
    if "actual_sales" in df.columns:
        df["gross_margin_pct"] = df["posgp"] / (df["actual_sales"] + 1e-6)
    else:
        df["gross_margin_pct"] = 0.0

    # Interaction features
    df["price_x_oem"] = df["unit_retail"] * df["oem_flag"]
    df["margin_pressure"] = (
        (df["unit_retail"] - df["unit_cogs"]) / (df["unit_retail"] + 1e-6)
    ).clip(0, 1)

    # Drop intermediates used only for zscore
    df = df.drop(columns=["cust_spend_mean", "cust_spend_std"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def load_and_preprocess(
    path,
    is_train: bool = True,
    lookups: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    # Peek at first row to detect two-header-row format
    _peek = pd.read_csv(path, nrows=0)
    _first_col = str(list(_peek.columns)[0]).lstrip("\ufeff")
    _two_row_header = _first_col.startswith("Original Columns")

    if _two_row_header:
        df = pd.read_csv(path, header=1, thousands=",")
    else:
        df = pd.read_csv(path, thousands=",")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Normalise column names from UPPERCASE WITH SPACES → snake_case
    col_map = {
        "INVOICE DATA":  "invoice_date",   # note: "DATA" is a typo for "DATE" in the CSV
        "INVOICE DATE":  "invoice_date",
        "NATL FLAG":     "natl_flag",
        "OEM FLAG":      "oem_flag",
        "ANNUAL SPEND":  "annual_spend",
        "QUANTITY":      "quantity",
        "RETAIL SALES":  "retail_sales",
        "COGS":          "cogs",
        "ACTUAL SALES":  "actual_sales",
        "POSGP":         "posgp",
        "COMP PRICE":    "comp_price",
        # Legacy: old CSV had unnamed date column
        "(No column name)": "invoice_date",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    df["invoice_date"] = pd.to_datetime(df["invoice_date"], format="mixed", dayfirst=False)
    df["month"] = df["invoice_date"].dt.month
    df["year"] = df["invoice_date"].dt.year

    df = df[(df["quantity"] > 0) & (df["retail_sales"] > 0)].copy()

    df["unit_retail"] = df["retail_sales"] / df["quantity"]
    df["unit_cogs"] = df["cogs"] / df["quantity"]
    df["price_vs_comp"] = np.where(
        df["comp_price"] == 0, np.nan, df["unit_retail"] - df["comp_price"]
    )

    # Reconstruct actual_sales for test set
    if not is_train:
        df["actual_sales"] = df["posgp"] + df["cogs"]

    df["catalogue_gap_rate"] = 1.0 - (df["actual_sales"] / df["retail_sales"])

    if is_train:
        lookups = build_lookup_tables(df)

    df = _merge_lookups_and_derive(df, lookups)

    # Drop raw columns — keep unit_cogs, comp_price, part_*_price stats for price_sweep base_row.
    # invoice_date is retained for directional accuracy evaluation.
    drop_cols = ["natl_flag", "retail_sales", "cogs", "posgp", "actual_sales"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df, lookups


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    cat_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode oem_flag and concatenate with numeric features.
    If cat_columns is provided (from training), align to that exact column set.
    """
    ohe_df = df[OHE_COLS].copy()
    # Ensure oem_flag is int so column names are oem_flag_0/oem_flag_1 (not _0.0)
    ohe_df["oem_flag"] = ohe_df["oem_flag"].round().astype(int)
    dummies = pd.get_dummies(ohe_df, columns=OHE_COLS, dtype=int)

    available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    num_df = df[available_num].reset_index(drop=True)
    dummies = dummies.reset_index(drop=True)
    X = pd.concat([num_df, dummies], axis=1)

    if cat_columns is not None:
        for col in cat_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[cat_columns]

    return X, list(X.columns)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    train_df: pd.DataFrame,
) -> Tuple[XGBRegressor, XGBRegressor, List[str], int]:
    """
    Returns (model_mean, model_median, feature_cols, n_excluded).

    Bulk outlier exclusion: rows where quantity > 90th-percentile within each
    PartID are dropped from training (kept in evaluation for honest test metrics).

    Two models are trained on the filtered set:
      model_mean   — standard squared-error regression (log1p target)
      model_median — quantile regression at alpha=0.5 (median, log1p target)

    Use model_median for the simulator / price sweep; model_mean for the
    evaluation tab to compare apples-to-apples with prior runs.
    """
    # --- bulk outlier exclusion ---
    p90 = train_df.groupby("PartID")["quantity"].transform(lambda x: x.quantile(0.90))
    mask_keep = train_df["quantity"] <= p90
    n_excluded = int((~mask_keep).sum())
    train_filtered = train_df[mask_keep].copy()

    X, feature_cols = build_features(train_filtered)
    y = np.log1p(train_filtered["quantity"].values)

    _shared = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=8,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model_mean = XGBRegressor(**_shared)
    model_mean.fit(X, y, verbose=False)

    model_median = XGBRegressor(
        **_shared,
        objective="reg:quantileerror",
        quantile_alpha=0.5,
    )
    model_median.fit(X, y, verbose=False)

    return model_mean, model_median, feature_cols, n_excluded


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _compute_directional_accuracy(df: pd.DataFrame, y_pred: np.ndarray) -> float:
    """
    For each CustID-PartID pair, sort transactions chronologically.
    For each consecutive pair where price changed and actual quantity changed,
    check whether the model predicted the same direction of quantity change.
    Returns fraction correct (NaN if no valid pairs exist).
    """
    work = df[["CustID", "PartID", "invoice_date", "unit_retail", "quantity"]].copy()
    work["_y_pred"] = y_pred

    correct = 0
    total = 0

    for _, grp in work.groupby(["CustID", "PartID"]):
        grp = grp.sort_values("invoice_date").reset_index(drop=True)
        if len(grp) < 2:
            continue

        delta_price  = grp["unit_retail"].diff().iloc[1:].values
        delta_actual = grp["quantity"].diff().iloc[1:].values
        delta_pred   = grp["_y_pred"].diff().iloc[1:].values

        # Only consider pairs where both price and actual qty moved
        mask = (np.abs(delta_price) > 1e-6) & (np.abs(delta_actual) > 1e-6)
        if mask.sum() == 0:
            continue

        correct += int((np.sign(delta_pred[mask]) == np.sign(delta_actual[mask])).sum())
        total   += int(mask.sum())

    if total == 0:
        return float("nan")
    return float(correct / total)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: XGBRegressor,
    feature_cols: List[str],
    df: pd.DataFrame,
) -> dict:
    """
    Evaluate on *full* df (including bulk outliers), so test metrics are honest.

    Metrics returned:
      r2                   — coefficient of determination
      mae                  — mean absolute error (original-scale units)
      rmse                 — root mean squared error (original-scale units)
      spearman_r           — Spearman rank correlation
      directional_accuracy — fraction of consecutive price moves where model
                             correctly predicts direction of quantity change
      y_true / y_pred      — arrays for plotting
    """
    X, _ = build_features(df, cat_columns=feature_cols)
    y_true = df["quantity"].values
    y_pred = np.expm1(np.maximum(model.predict(X), 0.0))
    y_pred = np.maximum(y_pred, 0.0)

    rho, _ = spearmanr(y_true, y_pred)
    dir_acc = _compute_directional_accuracy(df, y_pred)

    return {
        "r2":                   r2_score(y_true, y_pred),
        "mae":                  mean_absolute_error(y_true, y_pred),
        "rmse":                 float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman_r":           float(rho),
        "directional_accuracy": dir_acc,
        "y_true":               y_true,
        "y_pred":               y_pred,
    }


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def predict_demand(
    model: XGBRegressor,
    feature_cols: List[str],
    row: pd.Series,
) -> float:
    X, _ = build_features(pd.DataFrame([row]), cat_columns=feature_cols)
    return float(max(np.expm1(model.predict(X)[0]), 0.0))


def price_sweep(
    model: XGBRegressor,
    feature_cols: List[str],
    base_row: pd.Series,
    price_range: Tuple[float, float] = (0.80, 1.20),
    n_points: int = 120,
) -> pd.DataFrame:
    """
    Hold all features constant except unit_retail and price-derived features.
    Co-varies: price_vs_comp, comp_price_ratio, price_dev_from_habit,
               price_pct_of_part_range, price_zscore_within_part,
               price_to_cost_ratio, price_x_oem, margin_pressure.
    Returns DataFrame with unit_retail, predicted_quantity, elasticity,
    elasticity_zone, revenue, gross_margin, gross_margin_pct_sweep.
    """
    base_price          = float(base_row["unit_retail"])
    comp_price          = float(base_row.get("comp_price", 0) or 0)
    cust_part_avg_price = float(base_row.get("cust_part_avg_price", base_price) or base_price)
    part_min            = float(base_row.get("part_min_price", base_price * 0.5) or base_price * 0.5)
    part_max            = float(base_row.get("part_max_price", base_price * 2.0) or base_price * 2.0)
    part_avg            = float(base_row.get("part_avg_price", base_price) or base_price)
    part_std            = float(base_row.get("part_price_std", 1.0) or 1.0)
    unit_cogs           = float(base_row.get("unit_cogs", 0) or 0)
    oem_flag            = float(base_row.get("oem_flag", 0) or 0)

    prices = np.linspace(base_price * price_range[0], base_price * price_range[1], n_points)

    records = []
    for p in prices:
        r = base_row.copy()
        r["unit_retail"] = p

        if comp_price > 0:
            r["price_vs_comp"]  = p - comp_price
        r["comp_price_ratio"] = p / (comp_price + 1e-6)

        r["price_dev_from_habit"]     = p - cust_part_avg_price
        r["price_pct_of_part_range"]  = (p - part_min) / (part_max - part_min + 1e-6)
        r["price_zscore_within_part"] = (p - part_avg) / (part_std + 1e-6)
        r["price_to_cost_ratio"]      = p / (unit_cogs + 1e-6)
        r["price_x_oem"]              = p * oem_flag
        r["margin_pressure"]          = float(np.clip((p - unit_cogs) / (p + 1e-6), 0, 1))

        X, _ = build_features(pd.DataFrame([r]), cat_columns=feature_cols)
        q = float(max(np.expm1(model.predict(X)[0]), 0.0))
        records.append({"unit_retail": p, "predicted_quantity": q})

    result = pd.DataFrame(records)

    # Point elasticity via finite differences: ε = (dQ/dP) * (P/Q)
    dQ_dP = np.gradient(result["predicted_quantity"].values, result["unit_retail"].values)
    Q = result["predicted_quantity"].values
    P = result["unit_retail"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(Q > 0.001, dQ_dP * P / Q, 0.0)

    result["elasticity"] = e
    result["elasticity_zone"] = result["elasticity"].apply(
        lambda v: "Elastic" if v < -1 else ("Inelastic" if v > -1 else "Unitary Elastic")
    )
    result["revenue"]              = result["unit_retail"] * result["predicted_quantity"]
    result["gross_margin"]         = (result["unit_retail"] - unit_cogs) * result["predicted_quantity"]
    result["gross_margin_pct_sweep"] = np.where(
        result["unit_retail"] > 1e-6,
        (result["unit_retail"] - unit_cogs) / result["unit_retail"],
        0.0,
    )

    return result


def comp_price_sweep(
    model: XGBRegressor,
    feature_cols: List[str],
    base_row: pd.Series,
    comp_range: Tuple[float, float] = (0.5, 2.0),
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Hold unit_retail constant. Vary comp_price to sweep price_vs_comp.
    Returns DataFrame with comp_price, price_vs_comp, predicted_quantity.
    """
    base_unit  = float(base_row["unit_retail"])
    base_comp  = float(base_row.get("comp_price", base_unit) or base_unit)
    if base_comp == 0:
        base_comp = base_unit

    comp_prices = np.linspace(base_comp * comp_range[0], base_comp * comp_range[1], n_points)

    records = []
    for cp in comp_prices:
        r = base_row.copy()
        r["comp_price"]       = cp
        r["price_vs_comp"]    = base_unit - cp
        r["comp_price_ratio"] = base_unit / (cp + 1e-6)
        r["comp_present"]     = 1
        X, _ = build_features(pd.DataFrame([r]), cat_columns=feature_cols)
        q = float(max(np.expm1(model.predict(X)[0]), 0.0))
        records.append({
            "comp_price":         cp,
            "price_vs_comp":      base_unit - cp,
            "predicted_quantity": q,
        })

    return pd.DataFrame(records)
