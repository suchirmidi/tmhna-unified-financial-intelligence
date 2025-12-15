"""
TMHNA Unified Financial Intelligence Dashboard (PoC / Demo)
- Single-file Streamlit app with hard-coded + generated mock data.
- Simulates: consolidated reporting, operational KPIs, AI-like features, write-back workflows,
  governance/audit, RLS, masking, and "trace to source".
Author: ChatGPT (GPT-5.2 Thinking)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import random
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def set_page(page: str, **kwargs):
    """
    Updates the current page and any optional deep-link state (e.g., active section).
    """
    st.session_state["page"] = page
    for k, v in kwargs.items():
        st.session_state[k] = v

def format_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes dataframe for display:
    - Capitalizes headers (underscore -> space)
    - Formats floats as strings with separators (money/%)
    """
    out = df.copy()
    
    # 1. Column renaming
    out.columns = [c.replace("_", " ").title().replace("Usd", "USD").replace("Id", "ID") for c in out.columns]
    
    # 2. Value formatting (heuristic)
    for col in out.columns:
        # Check if float
        if pd.api.types.is_float_dtype(out[col]):
            # If column name suggests money
            if "USD" in col or "Amount" in col or "Profit" in col or "Cost" in col or "Price" in col:
                out[col] = out[col].apply(lambda x: f"${x:,.2f}")
            # If column name suggests percentage
            elif "%" in col or "Margin" in col or "Rate" in col or "Ratio" in col:
                out[col] = out[col].apply(lambda x: f"{x:.1%}")
            # General number
            else:
                 out[col] = out[col].apply(lambda x: f"{x:,.2f}")
                 
    return out

# -----------------------------
# Auth & Security (Simple Gate)
# -----------------------------
def check_password(user, password):
    """
    Checks credentials against st.secrets (if available) or defaults.
    """
    # 1. Try secrets
    if "auth" in st.secrets and "users" in st.secrets["auth"]:
        users = st.secrets["auth"]["users"]
        if user in users and users[user] == password:
            return True
    # 2. Fallback to demo/demo (with warning in log/console)
    #    In real production, this path should be disabled.
    else:
        if user == "demo" and password == "demo":
            return True
    return False


def login_screen():
    st.title(APP_TITLE)
    st.markdown("### Sign In")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if check_password(username, password):
                st.session_state["authenticated"] = True
                st.session_state["login_user"] = username
                st.session_state["actor"] = username.title() # Default actor name from username
                # Audit the login
                audit("LOGIN_SUCCESS", "AUTH", details={"username": username})
                st.rerun()
            else:
                audit("LOGIN_FAIL", "AUTH", details={"username": username})
                st.error("Invalid username or password.")
                
    if "auth" not in st.secrets:
        st.warning("⚠️ No secrets found. Using default `demo` / `demo`.")



# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="TMHNA Unified Financial Intelligence", layout="wide")

APP_TITLE = "TMHNA Unified Financial Intelligence"
APP_SUBTITLE = "PoC dashboard: unified P&L + Balance Sheet + Cash + Operational KPIs + Governance + AI-style insights (mock data)"
TZ = "America/Indiana/Indianapolis"


# -----------------------------
# Persistence (local SQLite inside working directory)
# -----------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "tmhna_poc.db")


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def init_db() -> None:
    con = db()
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT NOT NULL,
            role TEXT NOT NULL,
            action TEXT NOT NULL,
            object_type TEXT NOT NULL,
            object_id TEXT,
            details TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT NOT NULL,
            role TEXT NOT NULL,
            scope TEXT NOT NULL,          -- e.g., "P&L" / "Dealer Profitability"
            key TEXT NOT NULL,            -- e.g., "Raymond|Greene|Travel|2025-09"
            comment TEXT NOT NULL
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS stewardship_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT,
            decision_ts TEXT,
            status TEXT NOT NULL,         -- "PENDING" / "APPROVED" / "REJECTED"
            entity_type TEXT NOT NULL,    -- "VENDOR_MATCH" / "GL_CLASS"
            proposed_key TEXT NOT NULL,   -- e.g., "IBM~Intl Business Machines"
            proposed_value TEXT NOT NULL, -- e.g., "IBM"
            confidence REAL NOT NULL,
            rationale TEXT NOT NULL
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT NOT NULL,
            role TEXT NOT NULL,
            metric TEXT NOT NULL,         -- "Gross Margin %" etc
            threshold REAL NOT NULL,
            comparator TEXT NOT NULL,     -- "<" or ">"
            scope TEXT NOT NULL,          -- "Enterprise" / "Brand" etc
            scope_key TEXT NOT NULL       -- e.g., "TMHNA" / "TMH"
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS simulated_journals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT NOT NULL,
            role TEXT NOT NULL,
            brand TEXT NOT NULL,
            region TEXT NOT NULL,
            plant TEXT NOT NULL,
            account TEXT NOT NULL,
            amount REAL NOT NULL,
            memo TEXT,
            posted INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    con.commit()
    con.close()


def audit(action: str, object_type: str, object_id: Optional[str] = None, details: Optional[dict] = None) -> None:
    con = db()
    con.execute(
        "INSERT INTO audit_log(ts, actor, role, action, object_type, object_id, details) VALUES(?,?,?,?,?,?,?)",
        (
            dt.datetime.now().isoformat(timespec="seconds"),
            st.session_state.get("actor", "Unknown"),
            st.session_state.get("role", "Unknown"),
            action,
            object_type,
            object_id,
            json.dumps(details or {}, ensure_ascii=False),
        ),
    )
    con.commit()
    con.close()


def seed_stewardship_if_empty() -> None:
    con = db()
    cur = con.execute("SELECT COUNT(*) FROM stewardship_queue")
    n = cur.fetchone()[0]
    if n == 0:
        seeds = [
            ("VENDOR_MATCH", "IBM~Intl Business Machines", "IBM", 0.92, "High string similarity + shared tax ID pattern (mock)"),
            ("VENDOR_MATCH", "United Parcel Service~UPS", "UPS", 0.88, "Common alias detected (mock)"),
            ("VENDOR_MATCH", "3M Company~Minnesota Mining and Manufacturing", "3M", 0.84, "Corporate legal name normalization (mock)"),
            ("GL_CLASS", "Memo: 'Team dinner with dealer partners'", "Travel & Entertainment", 0.76, "Text classification suggests T&E (mock)"),
            ("GL_CLASS", "Memo: 'Forklift battery warranty reserve'", "Warranty Expense", 0.81, "Contains 'warranty' keyword (mock)"),
        ]
        for et, pk, pv, conf, rat in seeds:
            con.execute(
                """INSERT INTO stewardship_queue(ts,status,entity_type,proposed_key,proposed_value,confidence,rationale)
                   VALUES(?,?,?,?,?,?,?)""",
                (dt.datetime.now().isoformat(timespec="seconds"), "PENDING", et, pk, pv, float(conf), rat),
            )
        con.commit()
    con.close()


# -----------------------------
# Mock data generation
# -----------------------------
BRANDS = ["TMH", "Raymond"]
REGIONS = ["West", "Midwest", "South", "Northeast", "Canada"]
PLANTS = [
    ("West", "Phoenix"),
    ("Midwest", "Greene"),
    ("South", "Houston"),
    ("Northeast", "Buffalo"),
    ("Canada", "Konstant"),
]

GL_ACCOUNTS = [
    ("4000", "Revenue"),
    ("5000", "COGS"),
    ("6100", "Travel"),
    ("6200", "Marketing"),
    ("6300", "IT & Software"),
    ("6400", "Warranty Expense"),
    ("6500", "Freight"),
    ("6600", "SG&A Other"),
]

VENDOR_ALIASES = {
    "IBM": ["IBM", "Intl Business Machines", "I.B.M."],
    "UPS": ["UPS", "United Parcel Service", "U.P.S."],
    "Microsoft": ["Microsoft", "MSFT", "Microsoft Corp."],
    "Caterpillar": ["CAT", "Caterpillar", "Caterpillar Inc"],
    "3M": ["3M", "3M Company", "Minnesota Mining and Manufacturing"],
    "DHL": ["DHL", "DHL Express", "DHL Intl"],
}

DEALERS = [
    ("D001", "Sunset Lift", 34.0522, -118.2437),
    ("D002", "Great Lakes Material", 42.3314, -83.0458),
    ("D003", "Lone Star Handling", 29.7604, -95.3698),
    ("D004", "Empire Forklift", 40.7128, -74.0060),
    ("D005", "Prairie Industrial", 41.8781, -87.6298),
    ("D006", "Vancouver Lift", 49.2827, -123.1207),
]

random.seed(7)
np.random.seed(7)


@dataclass
class UserContext:
    actor: str
    role: str
    brand: str
    region: str
    plant: str
    off_network: bool
    mfa_ok: bool


def hash_id(*parts: str) -> str:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return h[:12]


@st.cache_data(show_spinner=False)
def make_mock_lakehouse() -> Dict[str, pd.DataFrame]:
    """
    Returns a dict of DataFrames representing: raw/cleansed/curated ("Bronze/Silver/Gold")
    plus source-level SAP/JDE-like transaction tables and telematics events.
    """
    today = dt.date.today()
    months = pd.date_range(today - pd.DateOffset(months=18), periods=19, freq="MS").date

    # Hierarchy mapping
    org_rows = []
    for b in BRANDS:
        for r in REGIONS:
            for pr, pl in PLANTS:
                if pr != r:
                    continue
                org_rows.append({"brand": b, "region": r, "plant": pl})
    org = pd.DataFrame(org_rows)

    # Transactions (GL-level)
    tx_rows = []
    for m in months:
        for _, o in org.iterrows():
            base_rev = np.random.normal(8_000_000, 1_200_000)
            season = 1.0 + 0.12 * np.sin((m.month / 12) * 2 * np.pi)
            rev = max(2_000_000, base_rev * season * (1.05 if o["brand"] == "TMH" else 0.95) * (1.03 if o["region"] != "Canada" else 0.92))
            cogs = -rev * np.random.uniform(0.62, 0.73)
            # opex mix
            travel = -abs(np.random.normal(120_000, 35_000))
            marketing = -abs(np.random.normal(160_000, 55_000))
            it = -abs(np.random.normal(240_000, 85_000))
            warranty = -abs(np.random.normal(220_000, 75_000))
            freight = -abs(np.random.normal(140_000, 45_000))
            sga = -abs(np.random.normal(180_000, 60_000))

            vals = {
                "4000": rev,
                "5000": cogs,
                "6100": travel,
                "6200": marketing,
                "6300": it,
                "6400": warranty,
                "6500": freight,
                "6600": sga,
            }

            for acct, amt in vals.items():
                acct_name = dict(GL_ACCOUNTS)[acct]
                src_system = "SAP_ECC" if o["brand"] == "TMH" else "JDE"
                currency = "USD" if o["region"] != "Canada" else "CAD"
                tx_id = hash_id(src_system, o["brand"], o["region"], o["plant"], acct, str(m))
                vendor_canon = random.choice(list(VENDOR_ALIASES.keys()))
                vendor_raw = random.choice(VENDOR_ALIASES[vendor_canon])

                memo_pool = [
                    "Monthly close adjustment",
                    "Service contract true-up",
                    "Dealer partner travel",
                    "Warranty reserve accrual",
                    "Cloud subscription renewal",
                    "Outbound freight reclass",
                    "Marketing campaign spend",
                    "Parts procurement",
                ]
                memo = random.choice(memo_pool)

                tx_rows.append(
                    {
                        "tx_id": tx_id,
                        "source_system": src_system,
                        "doc_no": f"{src_system[:3]}-{m.year%100:02d}{m.month:02d}-{random.randint(1000,9999)}",
                        "posting_month": m,
                        "brand": o["brand"],
                        "region": o["region"],
                        "plant": o["plant"],
                        "gl_account": acct,
                        "gl_name": acct_name,
                        "amount": float(round(amt, 2)),
                        "currency": currency,
                        "vendor_raw": vendor_raw,
                        "vendor_canonical": vendor_canon,
                        "memo": memo,
                        "cost_center": f"CC-{o['plant']}-{random.randint(10,99)}",
                    }
                )

    gl_tx = pd.DataFrame(tx_rows)

    # Intercompany (simplified)
    ic_rows = []
    for m in months:
        # between TMH and Raymond for parts/service
        for region, plant in PLANTS:
            amt = abs(np.random.normal(250_000, 90_000))
            ic_rows.append(
                {
                    "ic_id": hash_id("IC", str(m), plant),
                    "posting_month": m,
                    "seller_brand": "TMH",
                    "buyer_brand": "Raymond",
                    "region": region,
                    "plant": plant,
                    "amount_usd": float(round(amt, 2)),
                    "status": random.choice(["OPEN", "MATCHED", "OPEN", "OPEN"]),  # skew toward OPEN
                }
            )
    intercompany = pd.DataFrame(ic_rows)

    # Bank feeds + liquidity (mock)
    bank_rows = []
    for d in pd.date_range(today - pd.Timedelta(days=120), periods=121, freq="D"):
        cash = 120_000_000 + np.random.normal(0, 2_500_000)
        inflow = abs(np.random.normal(3_200_000, 900_000))
        outflow = -abs(np.random.normal(3_050_000, 850_000))
        bank_rows.append(
            {
                "date": d.date(),
                "bank": "JPM Chase",
                "cash_ending_usd": float(round(cash, 2)),
                "cash_inflow_usd": float(round(inflow, 2)),
                "cash_outflow_usd": float(round(outflow, 2)),
            }
        )
    bank = pd.DataFrame(bank_rows)

    # AP/AR aging
    aging_rows = []
    buckets = ["0-30", "31-60", "61-90", "90+"]
    for b in BRANDS:
        for r in REGIONS:
            for buck in buckets:
                aging_rows.append(
                    {
                        "brand": b,
                        "region": r,
                        "bucket": buck,
                        "ap_usd": float(round(abs(np.random.normal(8_000_000, 2_000_000)) * (1.0 if buck == "0-30" else 0.5 if buck == "31-60" else 0.3 if buck == "61-90" else 0.2), 2)),
                        "ar_usd": float(round(abs(np.random.normal(9_000_000, 2_300_000)) * (1.0 if buck == "0-30" else 0.55 if buck == "31-60" else 0.33 if buck == "61-90" else 0.22), 2)),
                    }
                )
    aging = pd.DataFrame(aging_rows)

    # Dealers profitability (mock)
    dealer_rows = []
    for code, name, lat, lon in DEALERS:
        for b in BRANDS:
            rev = abs(np.random.normal(18_000_000, 6_000_000))
            svc = abs(np.random.normal(5_800_000, 2_000_000))
            warranty = abs(np.random.normal(1_200_000, 450_000))
            margin = (rev + svc - warranty) * np.random.uniform(0.11, 0.22)
            dealer_rows.append(
                {
                    "dealer_code": code,
                    "dealer_name": name,
                    "brand": b,
                    "lat": lat,
                    "lon": lon,
                    "sales_revenue_usd": float(round(rev, 2)),
                    "service_margin_usd": float(round(svc, 2)),
                    "warranty_cost_usd": float(round(warranty, 2)),
                    "profit_usd": float(round(margin, 2)),
                }
            )
    dealers = pd.DataFrame(dealer_rows)

    # Fleet VIN-level profitability + telematics usage
    vins = [f"VIN{random.randint(100000,999999)}" for _ in range(120)]
    fleet_rows = []
    tele_rows = []
    for vin in vins:
        brand = random.choice(BRANDS)
        region = random.choice(REGIONS)
        plant = dict(PLANTS).get(region, random.choice([p for _, p in PLANTS]))
        sale_margin = float(round(abs(np.random.normal(12_500, 3_000)), 2))
        lifetime_service = float(round(abs(np.random.normal(9_500, 4_200)), 2))
        warranty = float(round(abs(np.random.normal(2_900, 1_600)), 2))
        hours = float(round(abs(np.random.normal(1400, 520)), 1))
        batt_cycles = int(abs(np.random.normal(580, 160)))
        utilization = min(1.0, max(0.1, hours / 2500.0))

        fleet_rows.append(
            {
                "vin": vin,
                "brand": brand,
                "region": region,
                "plant": plant,
                "sale_margin_usd": sale_margin,
                "lifetime_service_rev_usd": lifetime_service,
                "warranty_cost_usd": warranty,
                "vin_pnl_usd": float(round(sale_margin + lifetime_service - warranty, 2)),
                "usage_hours": hours,
                "battery_cycles": batt_cycles,
                "utilization": utilization,
            }
        )

        # "Telematics events" (high-velocity stream simulated by many rows)
        for _ in range(random.randint(30, 80)):
            day = today - dt.timedelta(days=random.randint(0, 180))
            tele_rows.append(
                {
                    "event_id": hash_id(vin, str(day), str(random.randint(1, 99999))),
                    "event_ts": dt.datetime.combine(day, dt.time(hour=random.randint(0, 23), minute=random.randint(0, 59))),
                    "vin": vin,
                    "metric": random.choice(["vibration", "motor_temp", "battery_health", "shock_event"]),
                    "value": float(round(abs(np.random.normal(1.0, 0.4)), 3)),
                }
            )

    fleet = pd.DataFrame(fleet_rows)
    telematics = pd.DataFrame(tele_rows)

    # Spend analytics (vendor family + duplicates)
    spend_rows = []
    for vcanon, aliases in VENDOR_ALIASES.items():
        for _ in range(75):
            raw = random.choice(aliases)
            amt = abs(np.random.normal(220_000, 180_000))
            family = random.choice(["IT Services", "Logistics", "Manufacturing Supplies", "Marketing", "Facilities"])
            spend_rows.append(
                {
                    "spend_id": hash_id("SPEND", raw, str(random.randint(1, 999999))),
                    "vendor_raw": raw,
                    "vendor_canonical": vcanon,
                    "vendor_family": family,
                    "amount_usd": float(round(amt, 2)),
                    "po_number": f"PO-{random.randint(100000,999999)}",
                }
            )
    spend = pd.DataFrame(spend_rows)

    # Inventory valuation across plants (mock)
    inv_rows = []
    for region, plant in PLANTS:
        for b in BRANDS:
            inv_val = abs(np.random.normal(45_000_000, 12_000_000)) * (1.05 if plant in ["Greene", "Houston"] else 0.9)
            turns = max(1.5, np.random.normal(6.2, 1.2))
            inv_rows.append(
                {
                    "brand": b,
                    "region": region,
                    "plant": plant,
                    "inventory_value_usd": float(round(inv_val, 2)),
                    "inventory_turns": float(round(turns, 2)),
                    "last_refresh": dt.datetime.now().isoformat(timespec="seconds"),
                }
            )
    inventory = pd.DataFrame(inv_rows)

    # Curated (Gold) aggregates for fast charts
    gl_tx["posting_month"] = pd.to_datetime(gl_tx["posting_month"])
    gold_pl = (
        gl_tx.groupby(["posting_month", "brand", "region", "plant", "gl_name"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "amount_local"})
    )
    # Currency normalization: CAD->USD conversion (mock fixed FX)
    FX_CADUSD = 0.74
    # For simplicity: tag CAD amounts by region Canada
    # We'll approximate by applying FX to rows with region==Canada
    gold_pl["amount_usd"] = gold_pl["amount_local"]
    gold_pl.loc[gold_pl["region"] == "Canada", "amount_usd"] = gold_pl.loc[gold_pl["region"] == "Canada", "amount_local"] * FX_CADUSD

    # Balance sheet snapshot (mock) per month
    bs_rows = []
    for m in months:
        for b in BRANDS:
            assets = abs(np.random.normal(1_350_000_000, 120_000_000)) * (1.08 if b == "TMH" else 0.95)
            liab = assets * np.random.uniform(0.52, 0.66)
            eq = assets - liab
            cash = abs(np.random.normal(160_000_000, 25_000_000))
            ar = abs(np.random.normal(230_000_000, 40_000_000))
            inv = abs(np.random.normal(280_000_000, 70_000_000))
            ap = abs(np.random.normal(170_000_000, 35_000_000))
            debt = abs(np.random.normal(420_000_000, 90_000_000))
            bs_rows.append(
                {
                    "asof_month": pd.to_datetime(m),
                    "brand": b,
                    "assets_usd": float(round(assets, 2)),
                    "liabilities_usd": float(round(liab, 2)),
                    "equity_usd": float(round(eq, 2)),
                    "cash_usd": float(round(cash, 2)),
                    "ar_usd": float(round(ar, 2)),
                    "inventory_usd": float(round(inv, 2)),
                    "ap_usd": float(round(ap, 2)),
                    "debt_usd": float(round(debt, 2)),
                }
            )
    balance_sheet = pd.DataFrame(bs_rows)

    return {
        # "Sources"
        "source_gl_tx": gl_tx,
        "source_intercompany": intercompany,
        "source_bank": bank,
        "source_aging": aging,
        "source_dealers": dealers,
        "source_fleet": fleet,
        "source_telematics": telematics,
        "source_spend": spend,
        "source_inventory": inventory,
        # "Gold / curated"
        "gold_pl": gold_pl,
        "gold_balance_sheet": balance_sheet,
    }


# -----------------------------
# Security simulation: Entra ID + RLS + masking
# -----------------------------
USERS = [
    # actor, role, default brand, default region, default plant
    ("Sarah Martinez", "Controller", "TMH", "Midwest", "Greene"),
    ("Avery Chen", "Executive", "TMH", "Northeast", "Buffalo"),
    ("Jordan Patel", "Plant Manager", "Raymond", "South", "Houston"),
    ("Taylor Brooks", "Data Steward", "TMH", "West", "Phoenix"),
    ("Riley Nguyen", "Auditor", "TMH", "Northeast", "Buffalo"),
    ("Casey Johnson", "Raymond Regional Manager", "Raymond", "Northeast", "Buffalo"),
]

ROLE_PERMS = {
    "Executive": {"can_export": True, "can_annotate": True, "can_steward": False, "can_journal": False, "can_audit": True},
    "Controller": {"can_export": True, "can_annotate": True, "can_steward": False, "can_journal": True, "can_audit": True},
    "Plant Manager": {"can_export": False, "can_annotate": True, "can_steward": False, "can_journal": False, "can_audit": False},
    "Data Steward": {"can_export": False, "can_annotate": False, "can_steward": True, "can_journal": False, "can_audit": True},
    "Auditor": {"can_export": True, "can_annotate": False, "can_steward": False, "can_journal": False, "can_audit": True},
    "Raymond Regional Manager": {"can_export": True, "can_annotate": True, "can_steward": False, "can_journal": False, "can_audit": True},
}



def rls_filter(df: pd.DataFrame, ctx: UserContext) -> pd.DataFrame:
    """
    Applies row-level security based on UserContext.
    Simple mock implementation: filter by brand / region / plant if columns exist.
    """
    out = df.copy()
    if ctx.brand == "TMHNA (Consolidated)":
        pass  # Show all brands
    elif "brand" in out.columns:
        out = out[out["brand"] == ctx.brand]

    if "region" in out.columns:
        if ctx.region in REGIONS:
             out = out[out["region"] == ctx.region]
    
    if "plant" in out.columns:
         # Filter by plant if selected and valid
        if ctx.plant in [p[1] for p in PLANTS]: 
             out = out[out["plant"] == ctx.plant]
        
    return out



def mask_sensitive(df: pd.DataFrame, ctx: UserContext) -> pd.DataFrame:
    """
    PII/sensitive masking simulation.
    - For non-Executive roles, watermark or mask "customer_contract_price" (not present) and some fields.
    - For Auditor: show doc numbers but mask cost center details.
    - For Plant Manager: mask vendor names.
    """
    out = df.copy()

    if "cost_center" in out.columns:
        if ctx.role in ["Auditor"]:
            out["cost_center"] = out["cost_center"].str.replace(r"CC-([A-Za-z]+)-\d+", r"CC-\1-XX", regex=True)

    if "vendor_raw" in out.columns:
        if ctx.role in ["Plant Manager"]:
            out["vendor_raw"] = out["vendor_raw"].apply(lambda _: "MASKED_VENDOR")
        if ctx.role in ["Raymond Regional Manager"]:
            out["vendor_raw"] = out["vendor_raw"].apply(lambda v: v if len(str(v)) <= 4 else (str(v)[:2] + "***"))

    if "memo" in out.columns and ctx.role in ["Plant Manager"]:
        out["memo"] = out["memo"].apply(lambda _: "MASKED_MEMO")

    return out


def require_mfa(ctx: UserContext) -> bool:
    """
    Conditional access simulation:
    - If off-network, MFA must be checked.
    """
    if ctx.off_network and not ctx.mfa_ok:
        st.error("Conditional Access: Off-network access requires MFA (simulate by checking the MFA box).")
        return False
    return True


# -----------------------------
# Utility formatting
# -----------------------------
def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(float(x))
    if x >= 1_000_000_000:
        return f"{sign}${x/1_000_000_000:.2f}B"
    if x >= 1_000_000:
        return f"{sign}${x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"{sign}${x/1_000:.2f}K"
    return f"{sign}${x:.0f}"


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


def kpi_card(label: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None):
    with st.container(border=True):
        st.caption(label)
        st.subheader(value)
        if delta is not None:
            st.write(delta)
        if help_text:
            st.caption(help_text)


def parse_nl_query(q: str) -> Dict[str, str]:
    """
    Very lightweight NLP parsing (rule-based):
    Examples:
      "Show me Q3 travel spend for Raymond Greene plant"
      "cash position last 30 days"
      "dealer profitability for TMH"
    """
    ql = q.lower().strip()
    out = {}

    # Brand
    if "raymond" in ql:
        out["brand"] = "Raymond"
    elif "tmh" in ql or "tmhna" in ql:
        out["brand"] = "TMH"

    # Plant (simple contains)
    for _, plant in PLANTS:
        if plant.lower() in ql:
            out["plant"] = plant

    # Region
    for r in REGIONS:
        if r.lower() in ql:
            out["region"] = r

    # Metric hints
    if "travel" in ql:
        out["gl_name"] = "Travel"
    if "marketing" in ql:
        out["gl_name"] = "Marketing"
    if "warranty" in ql:
        out["gl_name"] = "Warranty Expense"
    if "it" in ql and "software" in ql:
        out["gl_name"] = "IT & Software"

    # Quarter
    m = re.search(r"\bq([1-4])\b", ql)
    if m:
        out["quarter"] = f"Q{m.group(1)}"

    # Time window
    if "last 30" in ql:
        out["window_days"] = "30"
    if "last 90" in ql:
        out["window_days"] = "90"

    return out


def add_excel_watermark(df: pd.DataFrame, actor: str, role: str, note: str) -> bytes:
    """
    Creates an Excel file in-memory with a watermark cover sheet + exported data.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        cover = pd.DataFrame(
            {
                "Export Notice": [
                    "CONFIDENTIAL (PoC) — Export is logged to the immutable audit trail (simulated).",
                    f"Exported by: {actor} ({role})",
                    f"Timestamp: {dt.datetime.now().isoformat(timespec='seconds')}",
                    f"Note: {note}",
                ]
            }
        )
        cover.to_excel(writer, sheet_name="WATERMARK", index=False)
        df.to_excel(writer, sheet_name="DATA", index=False)

    return output.getvalue()


# -----------------------------
# Business logic: financial statements (P&L, BS, Cash)
# -----------------------------
def compute_pl_view(gold_pl: pd.DataFrame, ctx: UserContext, level: str) -> pd.DataFrame:
    df = gold_pl.copy()
    # apply RLS then additional "level" selections
    df = rls_filter(df, ctx)

    # level determines aggregation granularity
    group_cols = ["posting_month", "gl_name"]
    if level == "Enterprise":
        pass
    elif level == "Brand":
        group_cols = ["posting_month", "brand", "gl_name"]
    elif level == "Region":
        group_cols = ["posting_month", "brand", "region", "gl_name"]
    elif level == "Plant":
        group_cols = ["posting_month", "brand", "region", "plant", "gl_name"]

    out = df.groupby(group_cols, as_index=False)["amount_usd"].sum()
    return out


def compute_kpis(pl_df: pd.DataFrame) -> Dict[str, float]:
    # expects gl_name, amount_usd summed for latest month
    latest = pl_df["posting_month"].max()
    df = pl_df[pl_df["posting_month"] == latest]
    rev = df.loc[df["gl_name"] == "Revenue", "amount_usd"].sum()
    cogs = df.loc[df["gl_name"] == "COGS", "amount_usd"].sum()
    gp = rev + cogs  # cogs negative
    opex = df.loc[~df["gl_name"].isin(["Revenue", "COGS"]), "amount_usd"].sum()
    ebitda = gp + opex
    gm = gp / rev if rev != 0 else 0.0
    return {"Revenue": rev, "Gross Profit": gp, "Gross Margin %": gm, "EBITDA (mock)": ebitda}


def add_simulated_journal_to_pl(pl_df: pd.DataFrame, ctx: UserContext) -> pd.DataFrame:
    """
    Pull any simulated journals (posted=0) and overlay for simulation.
    """
    con = db()
    j = pd.read_sql_query(
        "SELECT brand, region, plant, account, amount, memo FROM simulated_journals WHERE posted=0",
        con,
    )
    con.close()
    if j.empty:
        return pl_df

    # Map account to gl_name
    acct_map = {a: n for a, n in GL_ACCOUNTS}
    j["gl_name"] = j["account"].map(acct_map).fillna("SG&A Other")
    j["posting_month"] = pl_df["posting_month"].max()
    # overlay only if journal is within user's visible scope
    j = rls_filter(j.rename(columns={"account": "gl_account"}), ctx)
    if j.empty:
        return pl_df

    overlay = j.groupby(["posting_month", "gl_name"], as_index=False)["amount"].sum().rename(columns={"amount": "amount_usd"})
    # merge by adding amounts
    out = pl_df.copy()
    out = out.merge(overlay, on=["posting_month", "gl_name"], how="left", suffixes=("", "_sim"))
    out["amount_usd"] = out["amount_usd"] + out["amount_usd_sim"].fillna(0.0)
    out = out.drop(columns=["amount_usd_sim"])
    return out


# -----------------------------
# AI-style features (mock)
# -----------------------------
def anomaly_detection(gl_tx: pd.DataFrame, ctx: UserContext, z_thresh: float = 2.6) -> pd.DataFrame:
    """
    Flag GL entries that deviate from historical norms for a cost center + gl_name.
    Uses z-score on amounts by (cost_center, gl_name).
    """
    df = rls_filter(gl_tx, ctx)
    df = mask_sensitive(df, ctx)
    df["posting_month"] = pd.to_datetime(df["posting_month"])
    # For anomaly scoring, use absolute magnitude
    grp = df.groupby(["cost_center", "gl_name"])["amount"].agg(["mean", "std"]).reset_index()
    df = df.merge(grp, on=["cost_center", "gl_name"], how="left")
    df["z"] = (df["amount"] - df["mean"]) / (df["std"].replace(0, np.nan))
    df["z_abs"] = df["z"].abs()
    out = df[df["z_abs"] >= z_thresh].copy()
    out = out.sort_values("z_abs", ascending=False).head(30)
    return out[["posting_month", "brand", "region", "plant", "gl_name", "amount", "cost_center", "vendor_raw", "doc_no", "memo", "z_abs"]]


def smart_cash_forecast(bank: pd.DataFrame, days_forward: int = 30) -> pd.DataFrame:
    """
    Simple forecasting proxy: rolling average of net cash movement + seasonality noise.
    """
    df = bank.sort_values("date").copy()
    df["net_flow"] = df["cash_inflow_usd"] + df["cash_outflow_usd"]
    roll = df["net_flow"].rolling(14, min_periods=7).mean().iloc[-1]
    last_cash = df["cash_ending_usd"].iloc[-1]
    start = df["date"].max()
    rows = []
    cash = last_cash
    for i in range(1, days_forward + 1):
        d = start + dt.timedelta(days=i)
        noise = np.random.normal(0, 450_000)
        cash += float(roll + noise)
        rows.append({"date": d, "cash_forecast_usd": float(round(cash, 2))})
    return pd.DataFrame(rows)


def predictive_maintenance_revenue(telematics: pd.DataFrame, fleet: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast service revenue based on telematics intensity (mock):
    - Higher vibration / shock events correlate with upcoming service needs.
    """
    t = telematics.copy()
    t["event_ts"] = pd.to_datetime(t["event_ts"])
    recent = t[t["event_ts"] >= (t["event_ts"].max() - pd.Timedelta(days=45))]
    scores = recent.groupby("vin")["value"].mean().reset_index().rename(columns={"value": "risk_score"})
    out = fleet[["vin", "brand", "region", "plant", "lifetime_service_rev_usd", "usage_hours", "battery_cycles"]].merge(scores, on="vin", how="left")
    out["risk_score"] = out["risk_score"].fillna(out["risk_score"].median())
    # Forecast: base + uplift by risk
    out["next_60d_service_rev_forecast_usd"] = (out["lifetime_service_rev_usd"] * 0.08) * (1.0 + (out["risk_score"] - out["risk_score"].mean()))
    out["next_60d_service_rev_forecast_usd"] = out["next_60d_service_rev_forecast_usd"].clip(lower=0.0)
    return out.sort_values("next_60d_service_rev_forecast_usd", ascending=False).head(25)


# -----------------------------
# UI sections
# -----------------------------

def sidebar_identity() -> UserContext:
    st.sidebar.markdown("### Identity (SSO via Entra ID — simulated)")
    
    # 1. Logout
    if st.sidebar.button("Sign Out"):
        st.session_state["authenticated"] = False
        st.session_state["login_user"] = None
        audit("LOGOUT", "AUTH", details={"username": st.session_state.get("actor")})
        st.rerun()

    actor = st.sidebar.selectbox("Signed in as", [u[0] for u in USERS], index=0)
    # resolve defaults
    u = next(x for x in USERS if x[0] == actor)
    role = st.sidebar.selectbox("Role", list(ROLE_PERMS.keys()), index=list(ROLE_PERMS.keys()).index(u[1]))
    
    # Brand Selector Logic
    # Available brands: Standards + potentially "TMHNA (Consolidated)"
    available_brands = list(BRANDS)
    
    # TMHNA option for Exec/Auditor
    if role in ["Executive", "Auditor"]:
        available_brands.append("TMHNA (Consolidated)")
        
    # Determine index. If user default is in list, use it. 
    # If they are Exec switching to TMHNA, we want to allow it.
    # We essentially let the user pick from 'available_brands'.
    
    # Default selection logic
    default_brand_idx = 0
    if u[2] in available_brands:
        default_brand_idx = available_brands.index(u[2])
        
    brand = st.sidebar.selectbox("Brand", available_brands, index=default_brand_idx)
    
    # Validation/Revert if role changes but brand remains stuck on restricted option (edge case in Streamlit state)
    if brand == "TMHNA (Consolidated)" and role not in ["Executive", "Auditor"]:
        st.warning(f"Role '{role}' cannot view Consolidated scope. Reverting to {u[2]}.")
        brand = u[2]

    region = st.sidebar.selectbox("Region", REGIONS, index=REGIONS.index(u[3]) if u[3] in REGIONS else 0)
    plants_in_region = [p for r, p in PLANTS if r == region]
    # For safety/simplicity, if list is empty (e.g. Canada?), handle gracefully or assume logic holds.
    # The constants have plants for all regions in mock data.
    
    # If TMHNA is selected, maybe we want 'All Plants'? 
    # The prompt says: "Update RLS logic so Executive/Auditor with TMHNA selection sees both brands, but other RLS dimensions (region/plant) still apply if the role requires them."
    # We will stick to the standard selectors for Region/Plant for now, 
    # effectively showing "TMHNA (all brands) in Region X".
    
    plant_idx = 0
    if u[4] in plants_in_region:
        plant_idx = plants_in_region.index(u[4])
    elif plants_in_region:
        plant_idx = 0
        
    plant = st.sidebar.selectbox("Plant", plants_in_region, index=plant_idx)

    st.sidebar.markdown("---")
    off_network = st.sidebar.toggle("Off-network access (simulate)", value=False)
    mfa_ok = st.sidebar.checkbox("MFA verified (simulate)", value=not off_network)
    st.sidebar.caption("Conditional Access: off-network requires MFA.")

    # Audit the scope change if it changes? (Optional, but good for tracking)
    # For now, just return context.
    
    ctx = UserContext(actor=actor, role=role, brand=brand, region=region, plant=plant, off_network=off_network, mfa_ok=mfa_ok)
    st.session_state["actor"] = actor
    st.session_state["role"] = role
    return ctx



# -----------------------------
# Landing Page (Revamped)
# -----------------------------
def persona_landing(ctx: UserContext):
    # Header
    st.title("Unified Financial Intelligence")
    st.markdown(f"#### Welcome, **{ctx.actor}**") 
    st.caption(f"Role: {ctx.role} | Brand Scope: {ctx.brand}")

    st.markdown("---")
    
    st.subheader("Quick Actions")
    
    # Row 1
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        with st.container(border=True):
            st.markdown("**Executive Overview**")
            st.caption("Consolidated P&L & Key Metrics")
            if st.button("Go to Financials", key="ql_fin", use_container_width=True):
                set_page("Financials", active_section_financials="P&L")
    
    with c2:
        with st.container(border=True):
            st.markdown("**Cash & Liquidity**")
            st.caption("Daily Cash Position & Sankey")
            if st.button("View Cash", key="ql_cash", use_container_width=True):
                set_page("Financials", active_section_financials="Cash & Liquidity")

    with c3:
        with st.container(border=True):
            st.markdown("**Dealer Profitability**")
            st.caption("Margin, Service, Warranty Heatmap")
            if st.button("View Dealers", key="ql_dealers", use_container_width=True):
                set_page("Operations", active_section_operations="Dealer Profitability")

    with c4:
        with st.container(border=True):
            st.markdown("**Audit Log**")
            st.caption("Governance Trail & History")
            if st.button("View Audit Log", key="ql_audit", use_container_width=True):
                set_page("Governance", active_section_governance="Audit Trail")

    st.markdown("") # Spacer

    # Row 2
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        with st.container(border=True):
            st.markdown("**Inventory Valuation**")
            st.caption("Plant-level Stock Value")
            if st.button("View Inventory", key="ql_inv", use_container_width=True):
                set_page("Operations", active_section_operations="Inventory Valuation")
    
    with c6:
        with st.container(border=True):
            st.markdown("**Intercompany**")
            st.caption("Reconciliation status & breaks")
            if st.button("View Intercompany", key="ql_ic", use_container_width=True):
                set_page("Financials", active_section_financials="Intercompany")
                
    with c7:
        with st.container(border=True):
            st.markdown("**Ask Finance (NLP)**")
            st.caption("Natural Language Query")
            if st.button("Ask Question", key="ql_nlp", use_container_width=True):
                set_page("NLP Query")

    with c8:
        with st.container(border=True):
            st.markdown("**Simulate Journal**")
            st.caption("Write-back Workflow")
            if st.button("Create Journal", key="ql_journal", use_container_width=True):
                set_page("Workflows", active_section_workflows="Journal Entry Simulation")

    st.markdown("---")



def module_financials(lake: Dict[str, pd.DataFrame], ctx: UserContext):
    st.header("Core Financial Reporting — Single Source of Truth")
    st.caption("Consolidated P&L, Balance Sheet, Cash & Liquidity, Intercompany, Currency normalization (CAD→USD).")

    if not require_mfa(ctx):
        return

    gold_pl = lake["gold_pl"]
    bs = lake["gold_balance_sheet"]
    bank = lake["source_bank"]
    aging = lake["source_aging"]
    intercompany = lake["source_intercompany"]
    gl_tx = lake["source_gl_tx"]

    # Global Controls
    top = st.columns([1.2, 1.0, 0.8])
    with top[0]:
        level = st.selectbox("View level", ["Enterprise", "Brand", "Region", "Plant"], index=0)
    with top[1]:
        show_sim = st.toggle("Include simulated journals (what-if)", value=True, help="Controller-only feature in real life; shown for demo.")
    with top[2]:
        fx = st.text_input("CAD→USD FX rate (mock)", value="0.74")

    # Deep-link support
    sections = ["P&L", "Balance Sheet", "Cash & Liquidity", "Intercompany", "Trace to Source"]
    default_idx = 0
    if "active_section_financials" in st.session_state:
        if st.session_state["active_section_financials"] in sections:
            default_idx = sections.index(st.session_state["active_section_financials"])
    
    section = st.radio("View", sections, index=default_idx, horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    # 1. P&L
    if section == "P&L":
        # compute view
        pl_view = compute_pl_view(gold_pl, ctx, level=level)

        # overlay simulated journals
        if show_sim:
            pl_view_sim = add_simulated_journal_to_pl(pl_view.rename(columns={"amount_usd": "amount_usd"}), ctx)
        else:
            pl_view_sim = pl_view

        # KPIs
        kpis = compute_kpis(pl_view_sim)
        k1, k2, k3, k4 = st.columns(4)
        kpi_card("Revenue (latest month)", money(kpis["Revenue"]), help_text="Definition: sum(Revenue) after CAD→USD normalization (mock).")
        kpi_card("Gross Profit", money(kpis["Gross Profit"]), help_text="Definition: Revenue + COGS.")
        kpi_card("Gross Margin %", pct(kpis["Gross Margin %"]), help_text="Definition: Gross Profit / Revenue.")
        kpi_card("EBITDA (mock)", money(kpis["EBITDA (mock)"]), help_text="Definition: Gross Profit + all Opex lines (simplified).")

        st.markdown("#### Consolidated P&L trend (rolling 12 months)")
        pl = pl_view_sim.copy()
        pl["posting_month"] = pd.to_datetime(pl["posting_month"])
        last12 = pl[pl["posting_month"] >= (pl["posting_month"].max() - pd.DateOffset(months=11))]
        # pivot to compute totals
        pivot = last12.groupby(["posting_month", "gl_name"], as_index=False)["amount_usd"].sum()
        fig = px.line(pivot, x="posting_month", y="amount_usd", color="gl_name", markers=False)
        fig.update_layout(height=380, legend_title_text="P&L Line")
        st.plotly_chart(fig, use_container_width=True)
        audit("VIEW", "FINANCIALS_PNL", details={"level": level, "include_simulated": bool(show_sim)})

    # 2. Balance Sheet
    elif section == "Balance Sheet":
        st.markdown("#### Balance Sheet snapshot + ratios")
        bs_view = bs.copy()
        bs_view = rls_filter(bs_view.rename(columns={"asof_month": "posting_month"}), ctx)
        bs_view = bs_view.rename(columns={"posting_month": "asof_month"})
        latest = bs_view["asof_month"].max()
        bs_latest = bs_view[bs_view["asof_month"] == latest].groupby(["brand"], as_index=False)[
            ["assets_usd", "liabilities_usd", "equity_usd", "cash_usd", "ar_usd", "inventory_usd", "ap_usd", "debt_usd"]
        ].sum()

        # Ratios
        bs_latest["debt_to_equity"] = bs_latest["debt_usd"] / bs_latest["equity_usd"].replace(0, np.nan)
        bs_latest["current_ratio"] = (bs_latest["cash_usd"] + bs_latest["ar_usd"] + bs_latest["inventory_usd"]) / (bs_latest["ap_usd"] + 1e-6)

        c1, c2 = st.columns([1.3, 1.0])
        with c1:
            st.dataframe(format_df(bs_latest), use_container_width=True, hide_index=True)
        with c2:
            # Simple waterfall-like bar
            fig2 = go.Figure()
            for _, row in bs_latest.iterrows():
                fig2.add_trace(go.Bar(name=row["brand"], x=["Assets", "Liabilities", "Equity"], y=[row["assets_usd"], row["liabilities_usd"], row["equity_usd"]]))
            fig2.update_layout(height=300, barmode="group", legend_title_text="Brand")
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Ratios (mock): Debt/Equity and Current Ratio computed from snapshot fields.")
        
        audit("VIEW", "FINANCIALS_BALANCE_SHEET", details={"asof_month": str(latest.date()) if hasattr(latest, "date") else str(latest)})

    # 3. Cash & Liquidity
    elif section == "Cash & Liquidity":
        st.markdown("#### Cash Flow & Liquidity (JPM Chase feed — mock)")
        bank_df = bank.sort_values("date")
        window = st.slider("Show last N days", min_value=14, max_value=120, value=60, step=1)
        bank_df = bank_df[bank_df["date"] >= (bank_df["date"].max() - dt.timedelta(days=int(window)))]

        left, right = st.columns([1.2, 1.0])
        with left:
            fig3 = px.line(bank_df, x="date", y="cash_ending_usd")
            fig3.update_layout(height=320, yaxis_title="USD")
            st.plotly_chart(fig3, use_container_width=True)

        with right:
            # Sankey: inflow -> cash -> outflow (simplified)
            inflow = float(bank_df["cash_inflow_usd"].sum())
            outflow = float((-bank_df["cash_outflow_usd"]).sum())
            sankey = go.Figure(
                data=[
                    go.Sankey(
                        node=dict(label=["Inflow", "Cash Position", "Outflow"]),
                        link=dict(source=[0, 1], target=[1, 2], value=[inflow, outflow]),
                    )
                ]
            )
            sankey.update_layout(height=320)
            st.plotly_chart(sankey, use_container_width=True)
            st.caption("Sankey visualizes Source → Use over the selected window (mock).")

        audit("VIEW", "FINANCIALS_CASH", details={"window_days": int(window)})

        # AP/AR aging
        st.markdown("#### AP/AR Aging (summary)")
        aging_view = aging.copy()
        # apply RLS-like filters by brand/region (if available)
        if ctx.role in ["Executive", "Auditor"]:
            pass
        else:
            aging_view = aging_view[(aging_view["brand"] == ctx.brand) & (aging_view["region"] == ctx.region)]

        pivot_aging = aging_view.groupby(["bucket"], as_index=False)[["ap_usd", "ar_usd"]].sum()
        fig4 = px.bar(pivot_aging, x="bucket", y=["ap_usd", "ar_usd"], barmode="group")
        fig4.update_layout(height=300, yaxis_title="USD")
        st.plotly_chart(fig4, use_container_width=True)

    # 4. Intercompany
    elif section == "Intercompany":
        st.markdown("#### Intercompany Reconciliation (automated matching — mock)")
        ic = intercompany.copy()
        if ctx.role not in ["Executive", "Auditor"]:
            # Restrict to their region/brand visibility
            if ctx.role == "Raymond Regional Manager":
                ic = ic[ic["buyer_brand"] == "Raymond"]
                ic = ic[ic["region"] == ctx.region]
            else:
                ic = ic[ic["region"] == ctx.region]
        matched = (ic["status"] == "MATCHED").mean() if len(ic) else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Items", f"{len(ic):,}")
        c2.metric("Matched rate", pct(matched))
        if "amount_usd" in ic.columns:
            open_amt = float(ic.loc[ic["status"] == "OPEN", "amount_usd"].sum())
        elif "amount" in ic.columns:
            open_amt = float(ic.loc[ic["status"] == "OPEN", "amount"].sum())
        else:
            open_amt = 0.0

        c3.metric("Open $ (USD)", money(open_amt))

        st.dataframe(format_df(ic.head(25)), use_container_width=True, hide_index=True)

        audit("VIEW", "FINANCIALS_INTERCOMPANY", details={"rows": int(len(ic))})

    # 5. Trace to Source
    elif section == "Trace to Source":
        # Trace to source: select KPI -> see underlying transactions
        st.markdown("#### Trace to Source (auditor-friendly)")
        line = st.selectbox("Pick a P&L line to trace", sorted(gold_pl["gl_name"].unique()), index=0)
        month = st.selectbox("Month", sorted(pd.to_datetime(gold_pl["posting_month"]).dt.date.unique()), index=len(sorted(pd.to_datetime(gold_pl["posting_month"]).dt.date.unique())) - 1)
        month_ts = pd.to_datetime(month)

        tx = gl_tx.copy()
        tx["posting_month"] = pd.to_datetime(tx["posting_month"])
        tx_f = tx[(tx["posting_month"] == month_ts) & (tx["gl_name"] == line)].copy()
        tx_f = rls_filter(tx_f, ctx)
        tx_f = mask_sensitive(tx_f, ctx)

        st.caption("Click a document number below to simulate lineage (doc → source system → transaction).")
        st.dataframe(format_df(tx_f[["source_system", "doc_no", "brand", "region", "plant", "gl_name", "amount", "currency", "vendor_raw", "cost_center", "memo"]].head(50)),
                     use_container_width=True, hide_index=True)
        audit("TRACE_TO_SOURCE", "GL_TRANSACTION", details={"gl_name": line, "month": str(month)})


def module_operations(lake: Dict[str, pd.DataFrame], ctx: UserContext):
    st.header("Operational Finance & KPI Tracking")
    st.caption("Dealer profitability, VIN-level fleet lifecycle profitability, spend analytics, inventory valuation.")

    if not require_mfa(ctx):
        return

    dealers = lake["source_dealers"]
    fleet = lake["source_fleet"]
    spend = lake["source_spend"]
    inventory = lake["source_inventory"]

    # Deep-link support: Programmatic Navigation
    sections = ["Dealer Profitability", "Fleet VIN P&L", "Spend Analytics", "Inventory Valuation"]
    default_idx = 0
    if "active_section_operations" in st.session_state:
        if st.session_state["active_section_operations"] in sections:
            default_idx = sections.index(st.session_state["active_section_operations"])
    
    # Use Radio for sub-nav instead of Tabs to support setting state (tabs don't support programmatic 'index' well yet)
    section = st.radio("View", sections, index=default_idx, horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    if section == "Dealer Profitability":
        df = dealers.copy()
        # RLS by brand for non-exec roles; exec sees both
        if ctx.role not in ["Executive", "Auditor"]:
            df = df[df["brand"] == ctx.brand]
        # map view
        st.markdown("##### Heatmap (geospatial) — North America")
        fig = px.scatter_geo(
            df,
            lat="lat",
            lon="lon",
            size="profit_usd",
            hover_name="dealer_name",
            hover_data={"dealer_code": True, "sales_revenue_usd": True, "service_margin_usd": True, "warranty_cost_usd": True, "lat": False, "lon": False},
            projection="natural earth",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Dealer Network Performance")
        st.dataframe(format_df(df.sort_values("profit_usd", ascending=False)), use_container_width=True, hide_index=True)
        audit("VIEW", "OPERATIONS_DEALERS")

    elif section == "Fleet VIN P&L":
        df = fleet.copy()
        df = rls_filter(df, ctx)
        st.markdown("##### VIN-level profitability (sale margin + lifetime service – warranty)")
        c1, c2 = st.columns([1.0, 1.2])
        with c1:
            st.dataframe(format_df(df.sort_values("vin_pnl_usd", ascending=False).head(25)), use_container_width=True, hide_index=True)
        with c2:
            fig = px.scatter(df, x="usage_hours", y="vin_pnl_usd", size="battery_cycles", hover_name="vin")
            fig.update_layout(height=380, xaxis_title="Usage hours (telematics)", yaxis_title="VIN P&L (USD)")
            st.plotly_chart(fig, use_container_width=True)
        audit("VIEW", "OPERATIONS_FLEET")

    elif section == "Spend Analytics":
        st.markdown("##### Duplicate Vendor Families (AI-normalized)")
        df = spend.copy()
        # For stewards allow all; otherwise restrict to brand
        if ctx.role not in ["Executive", "Auditor", "Data Steward"] and "brand" in df.columns:
             # spend mock might not have brand, but if it did...
             pass

        # Simulate "AI aggregation" by canonical
        fam = df.groupby(["vendor_family", "vendor_canonical"], as_index=False)["amount_usd"].sum()
        # Fix: Show textinfo so it's not empty boxes
        fig = px.treemap(fam, path=["vendor_family", "vendor_canonical"], values="amount_usd")
        fig.update_traces(textinfo="label+value+percent parent")
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Raw Vendor List (Potential Duplicates)")
        st.dataframe(format_df(df.sample(30, random_state=7)), use_container_width=True, hide_index=True)
        audit("VIEW", "OPERATIONS_SPEND")

    elif section == "Inventory Valuation":
        st.markdown("##### Inventory valuation across plants (real-time — mock)")
        df = inventory.copy()
        df = rls_filter(df, ctx)
        fig = px.bar(df, x="plant", y="inventory_value_usd", color="brand", barmode="group")
        fig.update_layout(height=360, yaxis_title="USD")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(format_df(df.sort_values("inventory_value_usd", ascending=False)), use_container_width=True, hide_index=True)
        audit("VIEW", "OPERATIONS_INVENTORY")


def module_ai(lake: Dict[str, pd.DataFrame], ctx: UserContext):
    st.header("Predictive & AI Capabilities (simulated)")
    st.caption("Smart cash forecasting, anomaly detection, predictive maintenance revenue from telematics.")

    if not require_mfa(ctx):
        return

    gl_tx = lake["source_gl_tx"]
    bank = lake["source_bank"]
    tele = lake["source_telematics"]
    fleet = lake["source_fleet"]

    # Deep-link support
    sections = ["Smart Forecasting", "Anomaly Detection", "Predictive Maintenance Revenue"]
    default_idx = 0
    if "active_section_ai" in st.session_state:
        if st.session_state["active_section_ai"] in sections:
            default_idx = sections.index(st.session_state["active_section_ai"])
    
    section = st.radio("View", sections, index=default_idx, horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    if section == "Smart Forecasting":
        st.markdown("##### Cash forecast (next 30 days) — ML proxy")
        days = st.slider("Days forward", 14, 90, 30, 1)
        fc = smart_cash_forecast(bank, days_forward=int(days))
        fig = px.line(fc, x="date", y="cash_forecast_usd")
        fig.update_layout(height=360, yaxis_title="USD")
        st.plotly_chart(fig, use_container_width=True)
        audit("VIEW", "AI_CASH_FORECAST", details={"days_forward": int(days)})

    elif section == "Anomaly Detection":
        st.markdown("##### GL anomaly alerts (z-score)")
        z = st.slider("Sensitivity (z-threshold)", 1.5, 4.0, 2.6, 0.1)
        out = anomaly_detection(gl_tx, ctx, z_thresh=float(z))
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.caption("Example alert: 'Unusual expense category for this cost center' (mock).")
        audit("VIEW", "AI_ANOMALY_DETECTION", details={"z_threshold": float(z), "rows": int(len(out))})

    elif section == "Predictive Maintenance Revenue":
        st.markdown("##### Service revenue forecast driven by iWAREHOUSE/MyInsights-like telematics (mock)")
        pred = predictive_maintenance_revenue(tele, fleet)
        pred = rls_filter(pred, ctx)
        st.dataframe(pred, use_container_width=True, hide_index=True)
        audit("VIEW", "AI_PRED_MAINT_REVENUE", details={"rows": int(len(pred))})


def module_workflows(lake: Dict[str, pd.DataFrame], ctx: UserContext):
    st.header("Actionable Workflows (CRUD / Write-Back — simulated)")
    st.caption("Journal entry simulation, stewardship queue (human-in-the-loop), commentary & annotation.")

    if not require_mfa(ctx):
        return

    perms = ROLE_PERMS.get(ctx.role, {})

    # Deep-link support
    sections = ["Journal Entry Simulation", "Stewardship Queue", "Commentary & Annotation"]
    default_idx = 0
    if "active_section_workflows" in st.session_state:
        if st.session_state["active_section_workflows"] in sections:
            default_idx = sections.index(st.session_state["active_section_workflows"])
    
    section = st.radio("View", sections, index=default_idx, horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    if section == "Journal Entry Simulation":
        st.markdown("##### Simulate a journal entry impact before posting")
        if not perms.get("can_journal", False):
            st.warning("Your role does not have journal simulation privileges in this PoC.")
        else:
            con = db()
            unposted = pd.read_sql_query("SELECT * FROM simulated_journals WHERE posted=0 ORDER BY ts DESC", con)
            con.close()

            c1, c2 = st.columns([1.0, 1.0])
            with c1:
                account = st.selectbox("GL Account", [f"{a} — {n}" for a, n in GL_ACCOUNTS], index=2)
                acct = account.split("—")[0].strip()
                amount = st.number_input("Amount (USD; negative for expense)", value=-50000.0, step=5000.0, format="%.2f")
                memo = st.text_input("Memo", value="Close adjustment (simulation)")
            with c2:
                st.caption("Scope (writes back to your org context)")
                st.write(f"Brand: **{ctx.brand}**  |  Region: **{ctx.region}**  |  Plant: **{ctx.plant}**")
                posted = st.toggle("Post now (simulated)", value=False)

            if st.button("Save journal simulation", type="primary"):
                con = db()
                con.execute(
                    """INSERT INTO simulated_journals(ts,actor,role,brand,region,plant,account,amount,memo,posted)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        dt.datetime.now().isoformat(timespec="seconds"),
                        ctx.actor,
                        ctx.role,
                        ctx.brand,
                        ctx.region,
                        ctx.plant,
                        acct,
                        float(amount),
                        memo,
                        1 if posted else 0,
                    ),
                )
                con.commit()
                con.close()
                audit("WRITE", "SIMULATED_JOURNAL", details={"account": acct, "amount": float(amount), "posted": bool(posted)})
                st.success("Saved. Toggle 'Include simulated journals' in Financials → P&L to see impact.")
                st.rerun()

            st.markdown("###### Current journal simulations (unposted)")
            st.dataframe(unposted, use_container_width=True, hide_index=True)

    elif section == "Stewardship Queue":
        st.markdown("##### Stewardship Queue (Tinder-like approve / reject)")
        if not perms.get("can_steward", False):
            st.warning("Your role does not have stewardship privileges in this PoC.")
        else:
            seed_stewardship_if_empty()
            con = db()
            q = pd.read_sql_query("SELECT * FROM stewardship_queue WHERE status='PENDING' ORDER BY confidence DESC, ts ASC", con)
            con.close()

            if q.empty:
                st.success("No pending items.")
            else:
                item = q.iloc[0].to_dict()
                with st.container(border=True):
                    st.subheader(f"{item['entity_type']} — confidence {item['confidence']:.2f}")
                    st.write(f"**Proposed:** `{item['proposed_key']}` → **{item['proposed_value']}**")
                    st.caption(item["rationale"])

                    b1, b2, b3 = st.columns([1, 1, 2])
                    with b1:
                        if st.button("✅ Approve", use_container_width=True):
                            con = db()
                            con.execute(
                                "UPDATE stewardship_queue SET status='APPROVED', actor=?, decision_ts=? WHERE id=?",
                                (ctx.actor, dt.datetime.now().isoformat(timespec="seconds"), int(item["id"])),
                            )
                            con.commit()
                            con.close()
                            audit("APPROVE", "STEWARDSHIP_ITEM", object_id=str(item["id"]), details={"proposed_key": item["proposed_key"]})
                            st.rerun()
                    with b2:
                        if st.button("❌ Reject", use_container_width=True):
                            con = db()
                            con.execute(
                                "UPDATE stewardship_queue SET status='REJECTED', actor=?, decision_ts=? WHERE id=?",
                                (ctx.actor, dt.datetime.now().isoformat(timespec="seconds"), int(item["id"])),
                            )
                            con.commit()
                            con.close()
                            audit("REJECT", "STEWARDSHIP_ITEM", object_id=str(item["id"]), details={"proposed_key": item["proposed_key"]})
                            st.rerun()
                    with b3:
                        st.caption("This simulates Human-in-the-Loop governance for AI vendor matching / GL classification.")

                st.markdown("###### Pending queue")
                st.dataframe(q, use_container_width=True, hide_index=True)

    elif section == "Commentary & Annotation":
        st.markdown("##### Commentary & annotation (write-back to data layer — simulated)")
        if not perms.get("can_annotate", False):
            st.warning("Your role does not have annotation privileges in this PoC.")
        else:
            scope = st.selectbox("Scope", ["P&L", "Dealer Profitability", "Inventory", "Cash", "Fleet"], index=0)
            key = st.text_input("Line key", value=f"{ctx.brand}|{ctx.region}|{ctx.plant}|Travel|{dt.date.today().isoformat()[:7]}")
            comment = st.text_area("Comment", value="Variance explained: elevated travel due to dealer visits + plant training (mock).", height=110)
            if st.button("Save annotation", type="primary"):
                con = db()
                con.execute(
                    "INSERT INTO annotations(ts,actor,role,scope,key,comment) VALUES(?,?,?,?,?,?)",
                    (dt.datetime.now().isoformat(timespec="seconds"), ctx.actor, ctx.role, scope, key, comment),
                )
                con.commit()
                con.close()
                audit("WRITE", "ANNOTATION", details={"scope": scope, "key": key})
                st.success("Saved annotation.")
                st.rerun()

            con = db()
            ann = pd.read_sql_query("SELECT * FROM annotations ORDER BY ts DESC LIMIT 200", con)
            con.close()
            # simple filter for context
            if ctx.role not in ["Executive", "Auditor"]:
                ann = ann[ann["actor"] == ctx.actor]
            st.markdown("###### Recent annotations")
            st.dataframe(ann, use_container_width=True, hide_index=True)


def module_governance(lake: Dict[str, pd.DataFrame], ctx: UserContext):
    st.header("Security, Governance, Compliance & Audit")
    st.caption("RLS, masking, traceability, immutable logs, controlled export, alerts, KPI definitions.")

    if not require_mfa(ctx):
        return

    perms = ROLE_PERMS.get(ctx.role, {})

    # Deep-link support
    sections = ["Access Controls", "Audit Trail", "Controlled Export", "Alerts + KPI Dictionary"]
    default_idx = 0
    if "active_section_governance" in st.session_state:
        if st.session_state["active_section_governance"] in sections:
            default_idx = sections.index(st.session_state["active_section_governance"])
    
    section = st.radio("View", sections, index=default_idx, horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    if section == "Access Controls":
        st.markdown("##### Simulated Row-Level Security (RLS) & Masking")
        st.write(
            f"""
            **Current User Scope:** Brand={ctx.brand}, Region={ctx.region}, Plant={ctx.plant}  
            **Access Policy:** Active
            """
        )
        st.caption("Try switching roles in the sidebar to observe data access changes (e.g., vendor masking, cost center redaction).")

        # show sample
        tx = lake["source_gl_tx"].copy()
        tx = tx.sample(25, random_state=7)
        a = st.columns(2)
        with a[0]:
            st.markdown("**Original Data (Pre-Policy)**")
            st.dataframe(format_df(tx[["source_system", "doc_no", "brand", "region", "plant", "gl_name", "amount", "vendor_raw", "cost_center", "memo"]]), use_container_width=True, hide_index=True)
        with a[1]:
            st.markdown("**Governed Data (Post-Policy)**")
            tx2 = mask_sensitive(rls_filter(tx, ctx), ctx)
            st.dataframe(format_df(tx2[["source_system", "doc_no", "brand", "region", "plant", "gl_name", "amount", "vendor_raw", "cost_center", "memo"]]), use_container_width=True, hide_index=True)
        audit("VIEW", "GOV_ACCESS_CONTROLS")

    elif section == "Audit Trail":
        st.markdown("##### Immutable audit log (simulated)")
        if not perms.get("can_audit", False):
            st.warning("Your role does not have audit viewing privileges in this PoC.")
        else:
            con = db()
            log = pd.read_sql_query("SELECT * FROM audit_log ORDER BY ts DESC LIMIT 500", con)
            con.close()
            st.dataframe(format_df(log), use_container_width=True, hide_index=True)
            st.caption("Immutable record of all business-critical actions.")
            audit("VIEW", "GOV_AUDIT_LOG")

    elif section == "Controlled Export":
        st.markdown("##### Export to Excel (controlled) — with watermark + audit logging")
        if not perms.get("can_export", False):
            st.warning("Your role does not have export privileges in this PoC.")
        else:
            st.write("Exports are watermarked and logged. This discourages shadow IT spreadsheets.")
            sample = lake["gold_pl"].copy()
            # Limit by RLS
            sample = rls_filter(sample.rename(columns={"amount_usd": "amount_usd"}), ctx)
            note = st.text_input("Export note", value="Controller analysis export (PoC).")
            data_bytes = add_excel_watermark(sample.head(500), ctx.actor, ctx.role, note)
            st.download_button(
                "Download Excel export",
                data=data_bytes,
                file_name="tmhna_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                on_click=lambda: audit("EXPORT", "EXCEL", details={"rows": int(min(500, len(sample))), "note": note}),
                type="primary",
            )

    elif section == "Alerts + KPI Dictionary":
        st.markdown("##### Alerting (thresholds push to Teams — simulated)")
        metric = st.selectbox("Metric", ["Gross Margin %", "EBITDA (mock)", "Cash Position"], index=0)
        comparator = st.selectbox("Comparator", ["<", ">"], index=0)
        threshold = st.number_input("Threshold", value=0.15 if metric == "Gross Margin %" else 100_000_000.0, step=0.01 if metric == "Gross Margin %" else 5_000_000.0)
        scope = st.selectbox("Scope", ["Enterprise", "Brand", "Region", "Plant"], index=1)
        scope_key = st.text_input("Scope key", value=ctx.brand if scope == "Brand" else ctx.region if scope == "Region" else ctx.plant if scope == "Plant" else "TMHNA")

        if st.button("Save alert", type="primary"):
            con = db()
            con.execute(
                "INSERT INTO alerts(ts,actor,role,metric,threshold,comparator,scope,scope_key) VALUES(?,?,?,?,?,?,?,?)",
                (
                    dt.datetime.now().isoformat(timespec="seconds"),
                    ctx.actor,
                    ctx.role,
                    metric,
                    float(threshold),
                    comparator,
                    scope,
                    scope_key,
                ),
            )
            con.commit()
            con.close()
            audit("WRITE", "ALERT", details={"metric": metric, "comparator": comparator, "threshold": float(threshold), "scope": scope, "scope_key": scope_key})
            st.success("Saved alert (simulation).")
            st.rerun()

        con = db()
        alerts = pd.read_sql_query("SELECT * FROM alerts ORDER BY ts DESC LIMIT 200", con)
        con.close()
        if ctx.role not in ["Executive", "Auditor"]:
            alerts = alerts[alerts["actor"] == ctx.actor]
        st.dataframe(alerts, use_container_width=True, hide_index=True)

        st.markdown("##### KPI dictionary (tooltips in a real deployment)")
        kpi_def = pd.DataFrame(
            [
                {"KPI": "Revenue", "Definition": "Sum of Revenue postings; CAD converted to USD for Canada (mock)."},
                {"KPI": "Gross Profit", "Definition": "Revenue + COGS (COGS stored as negative)."},
                {"KPI": "Gross Margin %", "Definition": "Gross Profit / Revenue."},
                {"KPI": "EBITDA (mock)", "Definition": "Gross Profit + Opex lines (simplified; excludes D&A and interest)."},
                {"KPI": "Cash Position", "Definition": "Ending cash balance from JPM Chase feed (mock)."},
            ]
        )
        st.dataframe(kpi_def, use_container_width=True, hide_index=True)

        st.caption("Teams notifications are represented by audit log entries in this PoC.")


def module_nlp(lake: Dict[str, pd.DataFrame], ctx: UserContext):
    st.header("Natural Language Query")
    st.caption("Ask questions about your data in plain English.")

    if not require_mfa(ctx):
        return

    q = st.text_input("Ask a question", value="Show me Q3 travel spend for Raymond Greene plant")
    if st.button("Run query", type="primary"):
        parsed = parse_nl_query(q)
        audit("QUERY", "NLP", details={"query": q, "parsed": parsed})
        # HIDDEN: st.write("**Parsed intent (rule-based):**")
        # HIDDEN: st.json(parsed)

        # Execute query over gold_pl
        pl = lake["gold_pl"].copy()
        pl["posting_month"] = pd.to_datetime(pl["posting_month"])
        # Apply RLS first
        pl = rls_filter(pl, ctx)

        if "brand" in parsed:
            pl = pl[pl["brand"] == parsed["brand"]]
        if "region" in parsed:
            pl = pl[pl["region"] == parsed["region"]]
        if "plant" in parsed:
            pl = pl[pl["plant"] == parsed["plant"]]
        if "gl_name" in parsed:
            pl = pl[pl["gl_name"] == parsed["gl_name"]]

        if "quarter" in parsed:
            qn = int(parsed["quarter"][1])
            pl = pl[pl["posting_month"].dt.quarter == qn]

        out = pl.groupby(["posting_month", "gl_name"], as_index=False)["amount_usd"].sum()
        if out.empty:
            st.warning("No data matched (likely due to RLS constraints). Try switching role or broadening scope.")
        else:
            fig = px.line(out, x="posting_month", y="amount_usd", color="gl_name")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(format_df(out), use_container_width=True, hide_index=True)



# -----------------------------
# Theming
# -----------------------------
def inject_theme(brand: str):
    """
    Injects CSS to override Streamlit styling based on brand selection.
    TMH: Black/Grey/Orange
    Raymond: Red/Black/White
    TMHNA (Consolidated): Neutral Dark Blue/Grey
    """
    # Default (TMH)
    primary_color = "#FF4D00"   # Orange
    bg_color = "#F0F2F6"        # Light Grey default
    sidebar_bg = "#FFFFFF"
    
    if brand == "Raymond":
        primary_color = "#E11A2B" # Raymond Red
        # Maybe a starker white/black contrast
    elif brand == "TMHNA (Consolidated)":
        primary_color = "#2C3E50" # Corporate Blue/Grey
    
    # CSS Injection
    # Note: Streamlit theming is usually config.toml, but we can override some elements via CSS.
    # We will target specific interactive elements if possible, or just the top bar color line if exposed.
    # Since we can't easily change the full theme dynamically without rerunning with different config,
    # we'll inject variable overrides for standard elements where CSS vars are used, 
    # or just colored headers/metrics to simulate the feel.
    
    st.markdown(
        f"""
        <style>
        /* Accent color for buttons using primary class */
        div.stButton > button:first-child {{
            background-color: {primary_color};
            color: white;
            border: none;
        }}
        div.stButton > button:first-child:hover {{
            background-color: {primary_color}DD; /* Slight transparency on hover */
            color: white;
            border: none;
        }}
        
        /* Metric Labels/Values */
        [data-testid="stMetricValue"] {{
            color: {primary_color};
        }}
        
        /* Sidebar styling override (simulated) */
        [data-testid="stSidebar"] {{
            border-right: 3px solid {primary_color};
        }}
        
        /* Headers */
        h1, h2, h3 {{
            border-bottom: 2px solid {primary_color}33; /* Faint underline */
            padding-bottom: 0.3rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Main
# -----------------------------
def main_app():
    init_db()
    # Ensure seed calling if needed or just let modules do it. 
    # Original main() just called init_db().
    lake = make_mock_lakehouse()

    # Sidebar identity
    ctx = sidebar_identity()
    
    # Inject Theme
    inject_theme(ctx.brand)

    # -----------------------------
    # Navigation State & Logic
    # -----------------------------
    PAGES = ["Landing", "Financials", "Operations", "AI Insights", "Workflows", "Governance", "NLP Query"]
    
    if "page" not in st.session_state:
        st.session_state["page"] = "Landing"

    # Top Navigation Bar (Redundant & Syncs with Sidebar)
    # Visual container for top nav
    with st.container():
        # Use columns: Logo/Title area | Nav Items | User Profile/Logout
        # We'll approximate an enterprise header layout
        tn1, tn2, tn3 = st.columns([2, 5, 1])
        
        with tn1:
            st.markdown("### TMHNA Intelligence")
        
        with tn2:
            # Horizontal buttons for nav
            # We use a trick: multiple columns for buttons to make them look like a bar
            nav_cols = st.columns(len(PAGES))
            for i, page_name in enumerate(PAGES):
                # Highlight active page button style if possible, or just standard
                # Streamlit buttons don't support "active" state styling easily without custom CSS,
                # but we can rely on the Sidebar/Page content to show where we are.
                if nav_cols[i].button(page_name, key=f"top_nav_{page_name}", use_container_width=True):
                    set_page(page_name)

        with tn3:
            # User profile / Logout
            # We already have logout in sidebar, adding a small icon/button here
            if st.button("👤 User", help=f"Signed in as {ctx.actor}"):
                pass # Just informational
            # Logout is redundant with sidebar, skipping to avoid clutter/state issues or adding simple one
            # if st.button("Logout", key="top_logout"):
            #    st.session_state["authenticated"] = False
            #    st.rerun()

    st.markdown("---")

    # Sidebar Sync: Use 'page' key to bind directly to session state
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    
    # We use accessibility helper (label_visibility) if needed, but standard is fine
    # Key="page" ensures bidirectional sync: if top nav updates state['page'], this updates.
    # If this updates, state['page'] updates.
    st.sidebar.radio(
        "Go to",
        PAGES,
        key="page",
        label_visibility="collapsed"
    )

    # -----------------------------
    # Routing
    # -----------------------------
    module = st.session_state["page"]

    if module == "Landing":
        persona_landing(ctx)
    elif module == "Financials":
        module_financials(lake, ctx)
    elif module == "Operations":
        module_operations(lake, ctx)
    elif module == "AI Insights":
        module_ai(lake, ctx)
    elif module == "Workflows":
        module_workflows(lake, ctx)
    elif module == "Governance":
        module_governance(lake, ctx)
    elif module == "NLP Query":
        module_nlp(lake, ctx)

    st.sidebar.markdown("---")
    st.sidebar.caption("PoC note: All data is mock. This app demonstrates the *shape* of the solution, not production integrations.")


if __name__ == "__main__":
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        init_db() # Ensure DB exists for audit logging of login attempts
        login_screen()
    else:
        main_app()
