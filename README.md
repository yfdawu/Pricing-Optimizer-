# B2B Pricing Analytics & Demand Forecasting Platform

A production-style pricing intelligence tool built for a large B2B industrial parts distributor. Combines an XGBoost elasticity model with a 5-tab Streamlit interface to give pricing analysts a live, interactive simulator for deal-level pricing decisions.

---

## What I Built

This project came out of a graduate consulting engagement. The business problem: a distributor with thousands of SKU/customer pairs had no systematic way to answer "what happens to demand if I change this price?" Reps were setting negotiated prices on instinct, with no visibility into elasticity, competitor positioning, or revenue impact.

I built a full analytics platform to solve that — from raw transaction data through to an interactive web app that any analyst can use without touching code.

**The core components:**

- **XGBoost elasticity model** trained on 5,000+ SKU/customer transaction pairs with 28 engineered features covering price position, customer behavior, basket context, and macroeconomic indices
- **Price sweep algorithm** that holds all features constant while varying price, producing predicted demand, elasticity zone, revenue, and gross margin curves across any price range
- **Streamlit app** with 5 analytical tabs, a global sidebar price slider, and live simulation state that updates every chart simultaneously

---

## Features

### Tab 1 — Deal Simulator
The primary tool. Select a customer and part, move the price slider, and every metric updates in real time:
- Revenue curve with cost floor and danger zone annotations
- Dual-axis demand overlay with ±15% uncertainty band
- Scenario comparison table: Current / Revenue-Optimal / Margin-Optimal / Simulated
- Recommendation sentence generated from the model's output
- Scenario status badge: Viable / Tight Margin / Uncontested / Below Cost Floor

### Tab 2 — Competitive Viability
Competitor price sweep — holds your price constant and varies the competitor's price to show how demand responds across the competitive gap. Includes competitor presence rate and historical price tracking.

### Tab 3 — Pricing Bands per SKU
Portfolio view of all parts for a selected customer. Each part gets a computed pricing band:
- Cost floor (absolute minimum)
- Min viable (15% margin floor)
- Revenue-optimal price
- Margin-optimal price
Revenue upside surfaced for each part relative to current negotiated price.

### Tab 4 — Basket Intelligence
Co-purchase analysis. Basket size, basket share, multi-item order rate, and monthly transaction sequence for the selected pair. Helps identify whether pricing a part aggressively risks losing a larger basket.

### Tab 5 — Customer Behavior Profile
Full transaction history for the selected customer-part pair: price paid over time, quantity trends, spend trajectory, and spend Z-score relative to the customer base.

---

## Model Details

| Component | Detail |
|---|---|
| Algorithm | XGBoost (gradient boosting, histogram method) |
| Objective | Quantile regression at α=0.5 (median) — robust to bulk order outliers |
| Target | log1p(quantity) — inverse-transformed at prediction time |
| Training rows | ~4,500 after 90th-percentile bulk outlier exclusion per part |
| Features | 28 total — see feature list below |
| Hyperparameters | n_estimators=300, max_depth=3, lr=0.05, subsample=0.8, colsample=0.6 |

**Feature groups:**
- **Price features**: unit price, price vs. competitor, price deviation from historical habit, price percentile within part range, price Z-score within part, price-to-cost ratio, OEM interaction
- **Customer features**: annual spend, spend Z-score, monthly spend index, customer share of part volume, pair transaction count, pair average price and quantity
- **Macro indices**: freight volume, truck tonnage, vehicle miles traveled, heavy truck sales, diesel price
- **Time**: month, year
- **Basket**: catalogue gap rate, gross margin

**Elasticity calculation**: Point elasticity via finite differences on the price sweep — ε = (dQ/dP) × (P/Q). Zones: Elastic (ε < −1), Unitary (ε = −1), Inelastic (ε > −1).

---

## Tech Stack

| Layer | Tool |
|---|---|
| Model | XGBoost, scikit-learn, SciPy |
| Data | pandas, NumPy, openpyxl |
| App | Streamlit |
| Charts | Plotly |
| Language | Python 3.10+ |

---

## Project Structure

```
├── pricing_app.py            # Streamlit app (5 tabs + global sidebar)
├── elasticity_model.py       # XGBoost model: features, training, price sweep
├── combined_data_loader.py   # Data pipeline: load, rename, validate, alias
├── generate_synthetic_data.py # Synthetic dataset generator (mirrors real schema)
├── synthetic_data.xlsx       # Demo dataset (80 customers, 120 parts, 5,000 rows)
└── README.md
```

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/pricing-analytics-platform.git
cd pricing-analytics-platform
```

**2. Install dependencies**
```bash
pip install streamlit xgboost scikit-learn pandas numpy plotly openpyxl scipy
```

**3. Generate the demo dataset** *(or skip — `synthetic_data.xlsx` is already included)*
```bash
python generate_synthetic_data.py
```

**4. Launch the app**
```bash
streamlit run pricing_app.py
```

The app loads data and trains the XGBoost model on startup (cached after first run — takes ~15 seconds on first load).

---

## Screenshots

> *(Add screenshots here after first run)*

| Tab | Description |
|---|---|
| ![Deal Simulator](screenshots/tab1_deal_simulator.png) | Revenue curve with demand overlay and scenario table |
| ![Pricing Bands](screenshots/tab3_pricing_bands.png) | Portfolio pricing band view per SKU |
| ![Customer Profile](screenshots/tab5_customer_profile.png) | Historical behavior and spend trajectory |

---

## Notes on the Dataset

The synthetic dataset in this repo was generated by `generate_synthetic_data.py` to mirror the schema of the original engagement data without exposing any proprietary pricing, customer, or transaction information. It produces statistically realistic distributions for price, quantity, competitor presence, basket context, and macroeconomic indices — sufficient to demonstrate the full model and app functionality.

---

## Author

**Daniel Wu** · M.S. Management Science, UT Dallas  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [Portfolio](https://YOUR_SITE.com)
