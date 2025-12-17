# TMHNA Unified Financial Intelligence Dashboard (PoC)

This is a **single-file Streamlit prototype** that simulates the end-state capabilities described in your spec:
- Consolidated **P&L**, **Balance Sheet**, **Cash/Liquidity**
- **Intercompany reconciliation** (mock)
- **Currency normalization** (CAD→USD, mock)
- Operational finance:
  - **Dealer profitability** (with geospatial view)
  - **VIN-level fleet lifecycle profitability** (with telematics usage)
  - **Spend analytics** (vendor de-duplication / normalization)
  - **Inventory valuation** across plants
- Predictive / AI-style features (simulated):
  - **Smart cash forecasting**
  - **GL anomaly detection**
  - **Predictive maintenance revenue** from telematics events
- Actionable workflows (write-back simulated via local SQLite):
  - **Journal entry simulation** (what-if overlay on P&L)
  - **Stewardship queue** (approve/reject AI suggestions)
  - **Commentary & annotation** write-back
- Governance:
  - **SSO via Entra ID** (simulated identity selector)
  - **Row-level security (RLS)** + **masking**
  - **Trace-to-source**
  - **Immutable audit log** (simulated)
  - **Controlled export to Excel** (watermark + audit)
  - **Alert thresholds** (simulated “Teams push” via audit entries)
- Bonus:
  - **Natural Language Query** (lightweight, rule-based)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- The database file `tmhna_poc.db` will be created in the same folder to persist:
  - audit logs
  - alerts
  - annotations
  - stewardship approvals/rejections
  - journal simulations
  - Link to Webpage: https://tmhna-unified-financial-intelligence-haskoaeypn6c3kvkawt3ro.streamlit.app/#action

Everything is mock data and safe to demo.
