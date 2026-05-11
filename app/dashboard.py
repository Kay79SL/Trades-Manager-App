"""
Trade Manager — Dashboard tab.

Renders four reactive KPIs (Total Revenue inc VAT, Materials Spend, Labour Cost,
Total Net Profit) with period-over-period directional arrows, plus three Plotly
charts. A county drop-down filter in the sidebar reactively re-computes every
aggregation. Chart hover labels expose invoice-level provenance.

Verified schema (RAGapp / trades_quotes):
    customers       : customer_id, first_name, last_name, email, phone, county,
                      eircode, preferred_trade, first_contact_date, address_*
    invoices        : invoice_id, customer_id, invoice_date, job_name, job_type_id,
                      trade, status, subtotal_ex_vat, materials_cost_ex_vat,
                      labour_cost_ex_vat, vat_23pct, total_inc_vat
    invoice_items   : invoice_id, line_no, item_id, item_name, quantity, unit,
                      unit_price_ex_vat, line_total_ex_vat
                      (no line_type field — items aren't split materials/labour,
                       use invoice.materials_cost_ex_vat instead)

Net profit definition for Trade Manager:
    net_profit = subtotal_ex_vat - materials_cost_ex_vat
    (since subtotal = materials + labour, this equals labour earned — the
     sole-trader's take-home before VAT)
"""
import os
from datetime import datetime, timedelta, date

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

# Load .env so MONGO_URI is available when run standalone
load_dotenv()


# =============================================================================
# CONFIG — verified against actual RAGapp schema
# =============================================================================
CONFIG = {
    "db_name": os.environ.get("MONGO_DB", "trades_quotes"),

    # Customer fields
    "customer_id_field": "customer_id",
    "customer_county_field": "county",

    # Invoice fields (verified)
    "invoice_id_field": "invoice_id",
    "invoice_customer_field": "customer_id",
    "invoice_date_field": "invoice_date",
    "invoice_job_name_field": "job_name",
    "invoice_trade_field": "trade",
    "invoice_status_field": "status",
    "invoice_subtotal_field": "subtotal_ex_vat",
    "invoice_materials_field": "materials_cost_ex_vat",
    "invoice_labour_field": "labour_cost_ex_vat",
    "invoice_vat_field": "vat_23pct",
    "invoice_total_inc_vat_field": "total_inc_vat",

    # Invoice-item fields (verified — used only for top-items chart)
    "item_invoice_field": "invoice_id",
    "item_name_field": "item_name",
    "item_line_total_field": "line_total_ex_vat",
}


# =============================================================================
# DB connection (cached so we reuse it across reruns)
# =============================================================================
@st.cache_resource
def get_db():
    uri = os.environ.get("MONGO_URI")
    if not uri:
        st.error("MONGO_URI not set. Add it to .env or .streamlit/secrets.toml")
        st.stop()
    client = MongoClient(uri)
    return client[CONFIG["db_name"]]


# =============================================================================
# Cached lookups
# =============================================================================
@st.cache_data(ttl=300)
def get_county_list():
    """Return sorted list of counties from the customers collection."""
    db = get_db()
    counties = sorted(c for c in db.customers.distinct(CONFIG["customer_county_field"]) if c)
    return ["All counties"] + counties


@st.cache_data(ttl=60)
def get_customer_ids_for_county(county: str) -> list:
    """Return list of customer IDs in the chosen county.
    Empty list if 'All counties' — caller should skip the filter then."""
    if county == "All counties":
        return []
    db = get_db()
    cursor = db.customers.find(
        {CONFIG["customer_county_field"]: county},
        {CONFIG["customer_id_field"]: 1}
    )
    return [doc[CONFIG["customer_id_field"]] for doc in cursor]


# =============================================================================
# KPI computation
# =============================================================================
def build_invoice_filter(start: date, end: date, county: str) -> dict:
    """Build the MongoDB $match filter for invoices in date range and county."""
    f = {
        CONFIG["invoice_date_field"]: {
            "$gte": datetime.combine(start, datetime.min.time()),
            "$lt":  datetime.combine(end + timedelta(days=1), datetime.min.time()),
        }
    }
    customer_ids = get_customer_ids_for_county(county)
    if customer_ids:
        f[CONFIG["invoice_customer_field"]] = {"$in": customer_ids}
    return f


def compute_kpis(start: date, end: date, county: str) -> dict:
    """Compute the four KPI values for the given window.

    Returns dict with keys:
        total_revenue_inc_vat, materials, labour, net_profit, invoice_count

    Uses pre-computed materials_cost_ex_vat and labour_cost_ex_vat fields on
    the invoice document — no $lookup into invoice_items needed.
    """
    db = get_db()
    invoice_filter = build_invoice_filter(start, end, county)

    pipeline = [
        {"$match": invoice_filter},
        {"$group": {
            "_id": None,
            "total_revenue_inc_vat": {"$sum": f"${CONFIG['invoice_total_inc_vat_field']}"},
            "total_subtotal_ex_vat": {"$sum": f"${CONFIG['invoice_subtotal_field']}"},
            "materials": {"$sum": f"${CONFIG['invoice_materials_field']}"},
            "labour":    {"$sum": f"${CONFIG['invoice_labour_field']}"},
            "invoice_count": {"$sum": 1},
        }},
    ]

    result = list(db.invoices.aggregate(pipeline))
    if not result:
        return {
            "total_revenue_inc_vat": 0,
            "materials": 0,
            "labour": 0,
            "net_profit": 0,
            "invoice_count": 0,
        }

    r = result[0]
    # Net profit for sole-trader = subtotal − materials (labour is income, not expense)
    net_profit = r["total_subtotal_ex_vat"] - r["materials"]
    return {
        "total_revenue_inc_vat": r["total_revenue_inc_vat"],
        "materials": r["materials"],
        "labour":    r["labour"],
        "net_profit": net_profit,
        "invoice_count": r["invoice_count"],
    }


def pct_change(current: float, previous: float):
    """Return percentage change, or None if previous is zero."""
    if previous == 0:
        return None
    return (current - previous) / previous * 100


def format_delta(pct):
    """Format the % change for st.metric's delta arg."""
    if pct is None:
        return ""
    return f"{pct:+.1f}% vs previous period"


# =============================================================================
# Chart helpers
# =============================================================================
@st.cache_data(ttl=60)
def get_invoices_df(start: date, end: date, county: str) -> pd.DataFrame:
    """Return invoice records joined with customer info for hover labels."""
    db = get_db()
    invoice_filter = build_invoice_filter(start, end, county)

    pipeline = [
        {"$match": invoice_filter},
        {"$lookup": {
            "from": "customers",
            "localField": CONFIG["invoice_customer_field"],
            "foreignField": CONFIG["customer_id_field"],
            "as": "customer",
        }},
        {"$unwind": {"path": "$customer", "preserveNullAndEmptyArrays": True}},
        {"$project": {
            "_id": 0,
            "invoice_id":      f"${CONFIG['invoice_id_field']}",
            "invoice_date":    f"${CONFIG['invoice_date_field']}",
            "total_inc_vat":   f"${CONFIG['invoice_total_inc_vat_field']}",
            "materials":       f"${CONFIG['invoice_materials_field']}",
            "labour":          f"${CONFIG['invoice_labour_field']}",
            "job_name":        f"${CONFIG['invoice_job_name_field']}",
            "trade":           f"${CONFIG['invoice_trade_field']}",
            "status":          f"${CONFIG['invoice_status_field']}",
            "customer_first":  "$customer.first_name",
            "customer_last":   "$customer.last_name",
            "county":          f"$customer.{CONFIG['customer_county_field']}",
        }},
    ]

    rows = list(db.invoices.aggregate(pipeline))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["customer_name"] = (
            df["customer_first"].fillna("") + " " + df["customer_last"].fillna("")
        ).str.strip()
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def render_revenue_trend(df: pd.DataFrame):
    """Stacked bar of monthly revenue by trade with hover provenance."""
    if df.empty:
        st.info("No invoice data for the selected filters.")
        return
    df = df.copy()
    df["month"] = df["invoice_date"].dt.to_period("M").astype(str)
    agg = (
        df.groupby(["month", "trade"], as_index=False)
          .agg(total_inc_vat=("total_inc_vat", "sum"),
               invoice_count=("invoice_id", "count"))
    )
    fig = px.bar(
        agg, x="month", y="total_inc_vat", color="trade",
        title="Revenue by month, broken down by trade",
        labels={"month": "Month", "total_inc_vat": "Revenue inc VAT (€)", "trade": "Trade"},
        hover_data={"invoice_count": True, "trade": True, "month": False},
    )
    fig.update_layout(barmode="stack", height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_invoice_scatter(df: pd.DataFrame):
    """Per-invoice scatter with hover provenance."""
    if df.empty:
        return
    fig = px.scatter(
        df, x="invoice_date", y="total_inc_vat", color="trade",
        title="Individual invoices — hover for details",
        labels={"invoice_date": "Date", "total_inc_vat": "Total inc VAT (€)", "trade": "Trade"},
        hover_data={
            "invoice_id": True,
            "customer_name": True,
            "county": True,
            "job_name": True,
            "status": True,
            "invoice_date": "|%Y-%m-%d",
        },
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_top_items(start: date, end: date, county: str):
    """Top 10 catalogue items by total spend in selected window."""
    db = get_db()
    invoice_filter = build_invoice_filter(start, end, county)

    matching_invoice_ids = [
        doc[CONFIG["invoice_id_field"]]
        for doc in db.invoices.find(invoice_filter, {CONFIG["invoice_id_field"]: 1})
    ]
    if not matching_invoice_ids:
        return

    pipeline = [
        {"$match": {CONFIG["item_invoice_field"]: {"$in": matching_invoice_ids}}},
        {"$group": {
            "_id": f"${CONFIG['item_name_field']}",
            "total_spend": {"$sum": f"${CONFIG['item_line_total_field']}"},
            "times_used": {"$sum": 1},
        }},
        {"$sort": {"total_spend": -1}},
        {"$limit": 10},
    ]
    rows = list(db.invoice_items.aggregate(pipeline))
    if not rows:
        return

    df = pd.DataFrame([
        {"item": r["_id"], "total_spend": r["total_spend"], "times_used": r["times_used"]}
        for r in rows
    ])
    fig = px.bar(
        df, x="total_spend", y="item", orientation="h",
        title="Top 10 catalogue items by total spend (selected period)",
        labels={"total_spend": "Total spend ex VAT (€)", "item": "Item"},
        hover_data={"times_used": True},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_county_distribution():
    """All-time customer count by county."""
    db = get_db()
    pipeline = [
        {"$group": {"_id": f"${CONFIG['customer_county_field']}", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    rows = list(db.customers.aggregate(pipeline))
    if not rows:
        return
    df = pd.DataFrame([{"county": r["_id"], "customers": r["count"]} for r in rows])
    fig = px.bar(
        df, x="county", y="customers",
        title="Customer distribution by county (all-time)",
        labels={"county": "County", "customers": "Number of customers"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MAIN — call this from your Streamlit app for the Dashboard tab
# =============================================================================
def render_dashboard():
    st.title("Trade Manager — Dashboard")

    # ----- Sidebar filters --------------------------------------------------
    st.sidebar.header("Dashboard filters")

    counties = get_county_list()
    selected_county = st.sidebar.selectbox("County", counties, index=0)

    today = date.today()
    default_start = today - timedelta(days=365)
    date_range = st.sidebar.date_input(
        "Date range",
        value=(default_start, today),
        max_value=today,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, today

    # ----- Compute current + previous period KPIs ---------------------------
    period_length = end_date - start_date
    prev_start = start_date - period_length - timedelta(days=1)
    prev_end = start_date - timedelta(days=1)

    current = compute_kpis(start_date, end_date, selected_county)
    previous = compute_kpis(prev_start, prev_end, selected_county)

    # ----- Header summary ---------------------------------------------------
    st.markdown(
        f"**{current['invoice_count']} invoices** in "
        f"**{selected_county.lower()}** from "
        f"**{start_date}** to **{end_date}**"
    )

    # ----- KPI cards (2 x 2 grid) -------------------------------------------
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.metric(
            label="Total revenue (inc VAT)",
            value=f"€{current['total_revenue_inc_vat']:,.2f}",
            delta=format_delta(pct_change(
                current["total_revenue_inc_vat"],
                previous["total_revenue_inc_vat"])),
        )
    with row1_col2:
        st.metric(
            label="Materials spend",
            value=f"€{current['materials']:,.2f}",
            delta=format_delta(pct_change(current["materials"], previous["materials"])),
            delta_color="inverse",  # rising materials is bad for the owner
        )

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.metric(
            label="Labour cost",
            value=f"€{current['labour']:,.2f}",
            delta=format_delta(pct_change(current["labour"], previous["labour"])),
        )
    with row2_col2:
        st.metric(
            label="Total net profit",
            value=f"€{current['net_profit']:,.2f}",
            delta=format_delta(pct_change(current["net_profit"], previous["net_profit"])),
            help="Gross margin: subtotal ex VAT minus materials cost (labour charged to customer is the owner's income)",
        )

    st.divider()

    # ----- Charts ----------------------------------------------------------
    invoices_df = get_invoices_df(start_date, end_date, selected_county)

    render_revenue_trend(invoices_df)
    render_invoice_scatter(invoices_df)
    render_top_items(start_date, end_date, selected_county)

    with st.expander("Customer distribution (all-time, no date filter)"):
        render_county_distribution()


# =============================================================================
# Standalone test:  streamlit run app/dashboard.py
# =============================================================================
if __name__ == "__main__":
    render_dashboard()
