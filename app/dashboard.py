"""
Business Intelligence Dashboard
==========================================

Trade-level KPIs and charts sourced live from the trades_quotes database.

Data-modelling note 
-------------------------------------------------------
Trade is treated as living on `job_types` only - the source of truth.
Although `invoices` carries a denormalised `trade` field, this dashboard
joins `invoices.job_type_id` -> `job_types.job_type_id` and reads the
trade from there. A trade reclassification on a job_type would be
reflected immediately across all historical invoices, with no risk of
stale denormalised values.

Schema (as exported from MongoDB Atlas, 2026-05-04):
    customers     (121 docs) - customer_id, first/last name, county, eircode...
    invoices      (150 docs) - invoice_id, customer_id, job_type_id,
                                invoice_date, total_inc_vat, ...
    invoice_items (750 docs) - invoice_id, item_id, item_name, quantity,
                                unit_price_ex_vat, line_total_ex_vat
    items          (88 docs) - item_id, item_name, category, unit_price_ex_vat
    job_types      (60 docs) - job_type_id, job_name, trade, ...
    pos            (15 docs) - po_number, po_date (string), job_type (string),
                                materials_subtotal, total_inc_vat, line_items[]

Foreign keys are STRING fields. All $lookup joins use string equality.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo import MongoClient

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

COLLECTIONS = {
    "customers": "customers",
    "invoices": "invoices",
    "invoice_items": "invoice_items",
    "purchase_orders": "pos",
    "items": "items",
    "job_types": "job_types",
}

# Autumn pastel palette 
CARD_BG = "#FDFAF6"
TERRA = "#B85A5A"
SAGE = "#587858"
SAGE_PALE = "#E8EEE3"
BORDER = "#E5DDD2"
TEXT = "#2D2520"
MUTED = "#786558"

# Distinct accent colour per trade for the multi-trade revenue chart.
DEFAULT_TRADE_PALETTE = ["#A8B5C9", "#C9A57B", "#D4896B", "#A89484", "#8FA08F"]

# Custom CSS for KPI cards and overall styling. This is injected into the Streamlit app using st.markdown with unsafe_allow_html=True.
# it includes styles for the KPI cards, titles, values, subtitles, dashboard title, section headers, and horizontal rules. 
# The colors are defined using the constants above to maintain a consistent theme across the dashboard.
THEME_CSS = f"""
<style>
.kpi-card {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    background: {CARD_BG};
    height: 100%;
}}
.kpi-title {{
    font-size: 0.82rem;
    color: {MUTED};
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}}
.kpi-value {{
    font-size: 1.6rem;
    font-weight: 800;
    color: {TEXT};
    margin: 6px 0 4px 0;
}}
.kpi-sub {{
    font-size: 0.78rem;
    color: {MUTED};
    margin: 0;
}}
.dashboard-title {{
    color: {TERRA};
    font-weight: 800;
    margin-bottom: 1rem;
}}
.section-header {{
    color: {TEXT};
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0.6rem;
}}
hr.dashboard-rule {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 1.2rem 0;
}}
</style>
"""

# -----------------------------------------------------------------------------
# Connection
# -----------------------------------------------------------------------------

# The get_mongo_db function establishes a connection to the MongoDB database using the URI provided in the environment variables or Streamlit secrets.
@st.cache_resource
def get_mongo_db():
    uri = os.getenv("MONGO_URI") or st.secrets.get("MONGO_URI", None)
    if not uri: # If the URI is not found in either the environment variables or Streamlit secrets, an error message is displayed to the user, 
        # and the execution of the app is stopped using st.stop(). This ensures that the app does not attempt to run without a valid database connection, which would lead to further errors down the line.
        st.error(
            "MONGO_URI not configured. Set it in `.env` or "
            ".streamlit/secrets.toml."
        )
        st.stop()
    db_name = os.getenv("MONGO_DB", "trades_quotes") # The database name is also read from the environment variables, with a default value of "trades_quotes" if not specified.
    client = MongoClient(uri)
    return client[db_name]


# -----------------------------------------------------------------------------
# Pipeline helpers
# -----------------------------------------------------------------------------

#  The _date_match function is a helper function that creates a MongoDB aggregation pipeline stage for filtering documents based on a date range.
def _date_match(start: Optional[date], end: Optional[date], field: str) -> Optional[dict]:
    if start is None or end is None:
        return None
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())
    return {"$match": {field: {"$gte": start_dt, "$lte": end_dt}}} # The function takes a start date, an end date, and the name of the date field to filter on. 
    # It returns a MongoDB aggregation stage that matches documents where the specified date field is between the start and end dates (inclusive). 
    # If either the start or end date is not provided, it returns None, indicating that no date filtering should be applied.

# The _trade_match function is another helper function that creates a MongoDB aggregation pipeline stage for filtering documents based on a trade value.
def _trade_match(trade: str, field: str) -> Optional[dict]:
    if trade == "All":
        return None
    return {"$match": {field: trade}}

#  The _invoice_job_lookup function creates a MongoDB aggregation pipeline stage for joining invoice documents with their corresponding job types.
def _invoice_job_lookup(local_prefix: str = "") -> list:
    
    # The function takes an optional local_prefix argument, which allows it to be used in different contexts where the invoice documents may be nested under a different field name.
    local_field = f"{local_prefix}.job_type_id" if local_prefix else "job_type_id"
    return [
        {
            "$lookup": {
                "from": COLLECTIONS["job_types"], # The lookup stage joins the current collection (which would be "invoices" or a nested field containing invoice documents) with the "job_types" collection based on the job_type_id field.
                "localField": local_field, # The localField is constructed using the local_prefix if provided, allowing for flexibility in how the function can be used in different aggregation pipelines.
                "foreignField": "job_type_id", # The foreignField is the job_type_id in the job_types collection, which is the field that will be matched against the localField in the current collection.
                "as": "job", # The results of the lookup are stored in a new field called "job", which will be an array containing the matching job_type document(s) for each invoice.
            }
        },
        {"$unwind": "$job"}, # The unwind stage is used to deconstruct the "job" array created by the lookup stage, so that each invoice document is paired with a single job_type document.
    ]

#   The _po_job_lookup function is similar to the _invoice_job_lookup function but is designed to handle the purchase orders collection, 
# where the job_type field may contain either the job_type_id or the job_name due to inconsistencies in PDF extraction.
def _po_job_lookup() -> list:
    # This function creates a MongoDB aggregation pipeline stage for joining purchase order documents with their corresponding job types,
    # but it accounts for the fact that the job_type field in the purchase orders collection may contain either the job_type_id or the job_name due to inconsistencies in how the data was extracted from PDFs.
    return [
        {
            "$lookup": {
                "from": COLLECTIONS["job_types"],
                "let": {"j": "$job_type"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$or": [
                                    {"$eq": ["$job_type_id", "$$j"]}, # The match condition uses the $or operator to check if the job_type field in the purchase order matches either the job_type_id or the job_name in the job_types collection.
                                    {"$eq": ["$job_name", "$$j"]}, # This allows the lookup to succeed even if the job_type field in the purchase order contains the job name instead of the ID, which is a common issue when data is extracted from PDFs and may not be perfectly structured.
                                ]
                            }
                        }
                    }
                ],
                "as": "job",
            }
        },
        {"$unwind": {"path": "$job", "preserveNullAndEmptyArrays": False}}, # The unwind stage is used to deconstruct the "job" array created by the lookup stage, similar to the _invoice_job_lookup function. 
        # However, in this case, preserveNullAndEmptyArrays is set to False to ensure that only purchase orders with a matching job type are included in the results, since the job_type field in the purchase
    ]


# -----------------------------------------------------------------------------
# Cached metadata queries
# -----------------------------------------------------------------------------

# The get_available_trades function retrieves the distinct trade values from the job_types collection in MongoDB.
@st.cache_data(ttl=600, show_spinner=False)
def get_available_trades() -> list[str]:
    db = get_mongo_db()
    try:
        trades = db[COLLECTIONS["job_types"]].distinct("trade")
        return sorted([t for t in trades if t])
    except Exception:
        return []

# The get_invoice_date_range function retrieves the minimum and maximum invoice dates from the invoices collection in MongoDB to determine the range of available data for filtering.
@st.cache_data(ttl=600, show_spinner=False)
def get_invoice_date_range() -> tuple[Optional[date], Optional[date]]: # This function queries the invoices collection to find the earliest and latest invoice dates, which can be used to set the default date range for filtering the dashboard.
    db = get_mongo_db() # It uses the find_one method with sorting to get the first and last documents based on the invoice_date field. 
                        # The projection is used to only retrieve the invoice_date field for efficiency. If successful, it returns a tuple of the minimum and maximum dates. If there is an error during the query, it returns (None, None).
    try:
        first = db[COLLECTIONS["invoices"]].find_one(
            {}, sort=[("invoice_date", 1)], projection={"invoice_date": 1} # The first query sorts the invoices in ascending order by invoice_date to get the earliest date, while the second query sorts in descending order to get the latest date.
        )
        last = db[COLLECTIONS["invoices"]].find_one(
            {}, sort=[("invoice_date", -1)], projection={"invoice_date": 1}
        )
        if first and last:
            return first["invoice_date"].date(), last["invoice_date"].date()
    except Exception:
        pass
    return None, None


# -----------------------------------------------------------------------------
# KPI queries
# -----------------------------------------------------------------------------

#   The query_kpi_customer_count function calculates the total number of unique customers served based on the invoices and their associated job types, filtered by trade and date range.

@st.cache_data(ttl=300, show_spinner=False)

# The function constructs a MongoDB aggregation pipeline that first applies date filtering if start and end dates are provided, 
# then performs a lookup to join invoices with job types, applies trade filtering if a specific trade is selected, and finally groups the results by customer_id to count the total number of unique customers.

def query_kpi_customer_count(trade: str, start: Optional[date], end: Optional[date]) -> int: # This function calculates the total number of unique customers served based on the invoices and their associated job types, filtered by trade and date range.
    db = get_mongo_db()
    pipeline: list = []
    date_stage = _date_match(start, end, "invoice_date") # The pipeline is constructed step by step, starting with an optional date match stage that filters invoices based on the invoice_date field.
    if date_stage: # If the date_stage is not None (i.e., if both start and end dates are provided), it is appended to the pipeline.
        pipeline.append(date_stage)
    pipeline += _invoice_job_lookup() # The pipeline then includes the stages from the _invoice_job_lookup function, which performs a lookup to join the invoices with their corresponding job types based on the job_type_id field.
    trade_stage = _trade_match(trade, "job.trade") # Next, an optional trade match stage is added to the pipeline if a specific trade is selected (i.e., if trade is not "All"). This stage filters the joined documents based on the trade field in the job document.
    if trade_stage: 
        pipeline.append(trade_stage) #  Finally, the pipeline includes stages to group the results by customer_id and count the total number of unique customers. 
        # The $group stage groups the documents by customer_id, and the $count stage counts the number of unique customer_id groups, resulting in the total number of customers served.
    pipeline += [
        {"$group": {"_id": "$customer_id"}},
        {"$count": "total"},
    ]
    result = list(db[COLLECTIONS["invoices"]].aggregate(pipeline))
    return result[0]["total"] if result else 0


@st.cache_data(ttl=300, show_spinner=False)
def query_kpi_jobs_value(trade: str, start: Optional[date], end: Optional[date]) -> float: # This function calculates the total value of jobs served based on the invoices and their associated job types, filtered by trade and date range.
    db = get_mongo_db()
    pipeline: list = []
    date_stage = _date_match(start, end, "invoice_date")
    if date_stage:
        pipeline.append(date_stage)
    pipeline += _invoice_job_lookup()
    trade_stage = _trade_match(trade, "job.trade")
    if trade_stage:
        pipeline.append(trade_stage)
    pipeline += [{"$group": {"_id": None, "total": {"$sum": "$total_inc_vat"}}}]
    result = list(db[COLLECTIONS["invoices"]].aggregate(pipeline))
    return float(result[0]["total"]) if result else 0.0


@st.cache_data(ttl=300, show_spinner=False)
def query_kpi_materials_spend(trade: str, start: Optional[date], end: Optional[date]) -> float: # This function calculates the total spend on materials based on the purchase orders and their associated job types, filtered by trade and date range.
    db = get_mongo_db()
    pipeline: list = list(_po_job_lookup())
    trade_stage = _trade_match(trade, "job.trade")
    if trade_stage:
        pipeline.append(trade_stage)
    pipeline += [{"$group": {"_id": None, "total": {"$sum": "$materials_subtotal"}}}]
    result = list(db[COLLECTIONS["purchase_orders"]].aggregate(pipeline))
    return float(result[0]["total"]) if result else 0.0


# The query_kpi_net_profit function calculates net profit per period.
# Definition: subtotal_ex_vat − materials_cost_ex_vat (gross margin for a sole trader).
# Labour charged to the customer is income for the owner, not a cost, so it is not subtracted.
# VAT is excluded because it is collected for Revenue, not the owner's money.
@st.cache_data(ttl=300, show_spinner=False)
def query_kpi_net_profit(trade: str, start: Optional[date], end: Optional[date]) -> float:
    """Net profit for the selected trade and date window.

    Computed from invoices as subtotal_ex_vat − materials_cost_ex_vat. This is the
    gross margin the owner keeps after paying suppliers, treating labour charged
    to the customer as income (the sole-trader case).
    """
    db = get_mongo_db()
    pipeline: list = []
    date_stage = _date_match(start, end, "invoice_date")
    if date_stage:
        pipeline.append(date_stage)
    pipeline += _invoice_job_lookup()
    trade_stage = _trade_match(trade, "job.trade")
    if trade_stage:
        pipeline.append(trade_stage)
    pipeline += [{
        "$group": {
            "_id": None,
            "profit": {
                "$sum": {
                    "$subtract": ["$subtotal_ex_vat", "$materials_cost_ex_vat"]
                }
            }
        }
    }]
    result = list(db[COLLECTIONS["invoices"]].aggregate(pipeline))
    return float(result[0]["profit"]) if result else 0.0


# -----------------------------------------------------------------------------
# Chart queries
# -----------------------------------------------------------------------------

# The query_revenue_trend function retrieves the revenue trend over time, grouped by month and trade, based on the invoices and their associated job types, filtered by trade and date range.
@st.cache_data(ttl=300, show_spinner=False)
# This function constructs a MongoDB aggregation pipeline that filters invoices by date and trade, joins them with job types, 
# and then groups the results by month and trade to calculate the total revenue and invoice count for each group.
def query_revenue_trend(trade: str, start: Optional[date], end: Optional[date]) -> pd.DataFrame: 
    db = get_mongo_db()
    pipeline: list = []
    date_stage = _date_match(start, end, "invoice_date")
    if date_stage:
        pipeline.append(date_stage)
    pipeline += _invoice_job_lookup()
    trade_stage = _trade_match(trade, "job.trade")
    if trade_stage:
        pipeline.append(trade_stage)
    pipeline += [
        {
            "$group": { # The $group stage groups the documents by a composite key consisting of the month (extracted from the invoice_date) and the trade (from the joined job document).
                "_id": {
                    "month": {
                        "$dateToString": {"format": "%Y-%m", "date": "$invoice_date"}
                    },
                    "trade": "$job.trade",
                },
                "revenue": {"$sum": "$total_inc_vat"},
                "invoice_count": {"$sum": 1},
            }
        },
        {"$sort": {"_id.month": 1}},
    ]
    # The function then executes the aggregation pipeline on the invoices collection and processes the results into a pandas DataFrame with columns for month, 
    # trade, revenue, and invoice count. If there are no results, it returns an empty DataFrame with the appropriate columns.
    rows = list(db[COLLECTIONS["invoices"]].aggregate(pipeline))
    if not rows:
        return pd.DataFrame(columns=["month", "trade", "revenue", "invoice_count"])
    return pd.DataFrame(
        [
            {
                "month": r["_id"]["month"],
                "trade": r["_id"]["trade"],
                "revenue": float(r["revenue"]),
                "invoice_count": int(r["invoice_count"]),
            }
            for r in rows
        ]
    )

# The query_top_items function retrieves the top items by invoice value based on the invoice items, their associated invoices, and job types, filtered by trade and date range.
@st.cache_data(ttl=300, show_spinner=False)
#   This function constructs a MongoDB aggregation pipeline that joins invoice items with their corresponding invoices and job types, applies date and trade filtering, 
# and then groups the results by item name to calculate the total value and quantity for each item. The results are sorted by total value in descending order and limited to the specified number of top items.
def query_top_items(
    trade: str, start: Optional[date], end: Optional[date], limit: int = 10
) -> pd.DataFrame:
    db = get_mongo_db()
    pipeline: list = [
        {
            "$lookup": {
                "from": COLLECTIONS["invoices"],
                "localField": "invoice_id",
                "foreignField": "invoice_id",
                "as": "inv",
            }
        },
        {"$unwind": "$inv"},
    ]
    # The pipeline starts with a lookup to join the invoice items with their corresponding invoices based on the invoice_id field.
    date_stage = _date_match(start, end, "inv.invoice_date")
    if date_stage:
        pipeline.append(date_stage)
    pipeline += [
        {
            "$lookup": {
                "from": COLLECTIONS["job_types"],
                "localField": "inv.job_type_id",
                "foreignField": "job_type_id",
                "as": "job",
            }
        },
        {"$unwind": "$job"},
    ]
    #   Next, another lookup is performed to join the invoices with their corresponding job types based on the job_type_id field.
    trade_stage = _trade_match(trade, "job.trade")
    if trade_stage:
        pipeline.append(trade_stage)
    pipeline += [
        {
            "$group": {
                "_id": "$item_name",
                "total_value": {"$sum": "$line_total_ex_vat"},
                "total_qty": {"$sum": "$quantity"},
            }
        },
        {"$sort": {"total_value": -1}},
        {"$limit": limit},
    ]
    #   Then, an optional trade match stage is added to filter the results based on the selected trade. Finally, the pipeline groups the results by item_name to calculate the total value and quantity for each item, 
    # sorts them by total value in descending order, and limits the results to the specified number of top items.
    rows = list(db[COLLECTIONS["invoice_items"]].aggregate(pipeline))
    if not rows:
        return pd.DataFrame(columns=["item", "total_value", "total_qty"])
    return pd.DataFrame(
        [
            {
                "item": r["_id"],
                "total_value": float(r["total_value"]),
                "total_qty": int(r["total_qty"]),
            }
            for r in rows
        ]
    )

#   The query_county_activity function retrieves the invoice count and total revenue by customer county based on the invoices, their associated customers, and job types, filtered by trade and date range.
@st.cache_data(ttl=300, show_spinner=False)
# This function constructs a MongoDB aggregation pipeline that joins invoices with their corresponding job types and customers, 
# applies date and trade filtering, and then groups the results by customer county to calculate the total invoice count and revenue for each county. The results are sorted by revenue in descending order.
def query_county_activity(trade: str, start: Optional[date], end: Optional[date]) -> pd.DataFrame:
    db = get_mongo_db()
    pipeline: list = []
    date_stage = _date_match(start, end, "invoice_date")
    if date_stage:
        pipeline.append(date_stage)
    pipeline += _invoice_job_lookup()
    trade_stage = _trade_match(trade, "job.trade")
    if trade_stage:
        pipeline.append(trade_stage)
    pipeline += [
        {
            "$lookup": {
                "from": COLLECTIONS["customers"],
                "localField": "customer_id",
                "foreignField": "customer_id",
                "as": "customer",
            }
        },
        {"$unwind": "$customer"},
        {
            "$group": {
                "_id": "$customer.county",
                "invoice_count": {"$sum": 1},
                "revenue": {"$sum": "$total_inc_vat"},
            }
        },
        {"$sort": {"revenue": -1}},
    ]
    #   The pipeline starts with optional date and trade filtering stages, followed by lookups to join the invoices with their corresponding job types and customers.
    rows = list(db[COLLECTIONS["invoices"]].aggregate(pipeline))
    if not rows:
        return pd.DataFrame(columns=["county", "invoice_count", "revenue"])
    return pd.DataFrame(
        [
            {
                "county": r["_id"] or "Unknown",
                "invoice_count": int(r["invoice_count"]),
                "revenue": float(r["revenue"]),
            }
            for r in rows
        ]
    )


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------

# The _kpi_card function generates HTML for a KPI card that displays a title, a value, an optional subtitle, and an optional period-over-period delta arrow.
# The card is styled using the custom CSS defined in THEME_CSS. The delta arrow is colour-coded: green for growth and red for drop on standard KPIs,
# reversed (red for growth, green for drop) when inverse=True for "lower is better" metrics like materials spend.
def _kpi_card(title: str, value: str, sub: Optional[str] = None,
              delta_pct: Optional[float] = None, inverse: bool = False) -> str:
    sub_html = f'<p class="kpi-sub">{sub}</p>' if sub else ""
    arrow_html = ""
    if delta_pct is not None:
        rising = delta_pct >= 0
        # Green for good direction, red for bad. Inverse flips it (materials spend rising is bad).
        if inverse:
            color = TERRA if rising else SAGE
        else:
            color = SAGE if rising else TERRA
        arrow = "▲" if rising else "▼"
        arrow_html = (
            f'<p style="color:{color}; margin:2px 0 4px 0; '
            f'font-size:0.85rem; font-weight:600;">'
            f'{arrow} {abs(delta_pct):.1f}% vs previous period</p>'
        )
    return (
        f'<div class="kpi-card">'
        f'<p class="kpi-title">{title}</p>'
        f'<p class="kpi-value">{value}</p>'
        f'{arrow_html}'
        f"{sub_html}"
        f"</div>"
    )


# Helpers for the period-over-period KPI arrows.
# _previous_period returns the equivalent prior window, _pct_change computes the percentage change.
def _previous_period(start: Optional[date], end: Optional[date]) -> tuple[Optional[date], Optional[date]]:
    """Return the equivalent previous period for the given window.

    For a 90-day window ending today, returns the 90 days before that. When the
    user has selected 'All time' (start and end both None) returns (None, None)
    so the caller can omit the delta arrow.
    """
    if start is None or end is None:
        return None, None
    period_length = end - start
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - period_length
    return prev_start, prev_end


def _pct_change(current: float, previous: float) -> Optional[float]:
    """Return percentage change current vs previous, or None if comparison invalid."""
    if previous is None or previous == 0 or current is None:
        return None
    return (current - previous) / previous * 100

# The _fmt_money function formats a float value as a string representing a monetary amount in euros, with a euro symbol and comma as a thousands separator.
def _fmt_money(x: float) -> str:
    return f"€{x:,.0f}"

# The _styled_plotly_layout function applies a consistent styling to a Plotly figure, setting the background colors, font, margins, and other layout properties to match the overall theme of the dashboard.
def _styled_plotly_layout(fig):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color=TEXT, size=12),
        margin=dict(t=10, b=40, l=20, r=20),
    )
    return fig

#   The _trade_palette function generates a color palette for the trades based on the DEFAULT_TRADE_PALETTE. It creates a mapping of trade names to colors, 
# ensuring that each trade is assigned a distinct color from the palette, and if there are more trades than colors, it cycles through the palette again.
def _trade_palette(trades: list[str]) -> dict:
    return {
        t: DEFAULT_TRADE_PALETTE[i % len(DEFAULT_TRADE_PALETTE)]
        for i, t in enumerate(sorted(trades))
    }


# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------

# The _render_sidebar_filters function renders the sidebar filters for the dashboard, allowing the user to select a trade, a date range, 
# and whether to show all time data. It returns the selected trade and date range for use in the main dashboard queries.
def _render_sidebar_filters(available_trades: list[str]) -> tuple:
    with st.sidebar:
        st.markdown("### Dashboard filters")

        trade = st.radio(
            "Trade",
            ["All"] + available_trades,
            index=0,
            key="bi_trade",
        )

        all_time = st.checkbox(
            "All time",
            value=True,
            key="bi_all_time",
            help="Show every invoice, no date filter.",
        )

        if all_time:
            start, end = None, None
            st.caption("Showing all dates")
        else: # If the "All time" checkbox is not selected, the function retrieves the date range of available invoice data using the get_invoice_date_range function. It then sets the default start and end dates for the date input widget based on this range, 
            # defaulting to the last 365 days if no data is available. The user can select a custom date range using the st.date_input widget, and the selected start and end dates are returned for use in filtering the dashboard data.
            data_start, data_end = get_invoice_date_range()
            default_start = data_start or date.today() - timedelta(days=365)
            default_end = data_end or date.today()
            date_range = st.date_input(
                "Date range",
                value=(default_start, default_end),
                key="bi_date_range",
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start, end = date_range
            else:
                start, end = None, None

        if st.button("Refresh data", use_container_width=True, key="bi_refresh"):
            st.cache_data.clear()
            st.rerun()

        st.divider()

    return trade, start, end


# -----------------------------------------------------------------------------
# Main render function
# -----------------------------------------------------------------------------

#   The render_dashboard function is the main function that renders the entire dashboard. It sets up the page layout, applies the custom CSS, and orchestrates the rendering of the KPIs and charts based on the selected filters.
def render_dashboard() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    st.markdown(
        '<h1 class="dashboard-title">Business Intelligence</h1>',
        unsafe_allow_html=True,
    )
    # The function first injects the custom CSS defined in THEME_CSS to style the dashboard, then it renders the main title of the dashboard. 
    # It retrieves the available trades from the database and renders the sidebar filters using the _render_sidebar_filters function, which returns the selected trade and date range.
    available_trades = get_available_trades()
    if not available_trades:
        st.warning(
            "Could not read trade values from `job_types.trade`. "
            "Check your MongoDB connection in `.env`."
        )
        return
    # If the available trades cannot be retrieved (e.g., due to a database connection issue), a warning message is displayed to the user, and the function returns early to prevent further errors. 
    # If the trades are successfully retrieved, the dashboard proceeds to render the KPIs and charts based on the selected filters.
    trade, start, end = _render_sidebar_filters(available_trades)
    palette = _trade_palette(available_trades)

    # KPI row — four KPIs with period-over-period arrows
    # Customers served, Total jobs value, Materials spend, and Net profit.
    # Each card shows a green ▲ when the metric rises versus the equivalent prior window
    # and a red ▼ when it falls. Materials spend uses inverse colouring (rising is bad).
    try:
        cust_count = query_kpi_customer_count(trade, start, end)
        jobs_value = query_kpi_jobs_value(trade, start, end)
        po_spend   = query_kpi_materials_spend(trade, start, end)
        net_profit = query_kpi_net_profit(trade, start, end)

        # Previous period values for delta arrows. When "All time" is selected
        # prev_start is None and we skip the comparison so no arrow renders.
        prev_start, prev_end = _previous_period(start, end)
        if prev_start is not None:
            d_cust   = _pct_change(cust_count,  query_kpi_customer_count(trade, prev_start, prev_end))
            d_jobs   = _pct_change(jobs_value,  query_kpi_jobs_value(trade, prev_start, prev_end))
            d_spend  = _pct_change(po_spend,    query_kpi_materials_spend(trade, prev_start, prev_end))
            d_profit = _pct_change(net_profit,  query_kpi_net_profit(trade, prev_start, prev_end))
        else:
            d_cust = d_jobs = d_spend = d_profit = None
    except Exception as e:
        st.error(f"KPI query failed: {e}")
        return
    # The function retrieves the KPI values using the corresponding query functions. If any of the queries fail (e.g., due to a database error), an error message is displayed to the user, and the function returns early.
    period_label = (
        f"{start.strftime('%b %Y')} – {end.strftime('%b %Y')}"
        if start and end
        else "all time"
    )
    sub_label = "all trades" if trade == "All" else trade.lower()
    # The period_label is constructed based on the selected start and end dates, while the sub_label is determined by the selected trade. These labels are used in the KPI cards to provide context for the displayed values.
    kc1, kc2, kc3, kc4 = st.columns(4, gap="medium")
    with kc1:
        st.markdown(
            _kpi_card("Customers served", f"{cust_count}",
                      f"{sub_label} · {period_label}",
                      delta_pct=d_cust),
            unsafe_allow_html=True,
        )
    with kc2:
        st.markdown(
            _kpi_card("Total jobs value", _fmt_money(jobs_value),
                      f"{sub_label} · inc VAT",
                      delta_pct=d_jobs),
            unsafe_allow_html=True,
        )
    with kc3:
        st.markdown(
            _kpi_card("Materials spend (POs)", _fmt_money(po_spend),
                      f"{sub_label} · all POs (15 total)",
                      delta_pct=d_spend, inverse=True),
            unsafe_allow_html=True,
        )
    with kc4:
        st.markdown(
            _kpi_card("Net profit", _fmt_money(net_profit),
                      f"{sub_label} · ex VAT, ex materials",
                      delta_pct=d_profit),
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="dashboard-rule">', unsafe_allow_html=True)

    # Chart 1: Revenue trend
    # The first chart displays the revenue trend over time, grouped by month and trade. It uses the query_revenue_trend function to retrieve the data, 
    # and if there are no invoices in the selected period, an informational message is displayed. Otherwise, a line chart is rendered using Plotly Express, with custom styling applied to match the dashboard theme.
    st.markdown('<h3 class="section-header">Revenue trend</h3>', unsafe_allow_html=True)
    df_rev = query_revenue_trend(trade, start, end)
    if df_rev.empty:
        st.info("No invoices in this period for the selected trade.")
    else:
        fig = px.line(
            df_rev,
            x="month",
            y="revenue",
            color="trade",
            color_discrete_map=palette,
            markers=True,
            labels={"month": "Month", "revenue": "Revenue (€)", "trade": "Trade"},
        )
        _styled_plotly_layout(fig)
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#eee", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    # Chart 2: Top items
    # The second chart displays the top 10 catalogue items by invoice value based on the invoice items, their associated invoices, and job types, filtered by trade and date range. 
    # It uses the query_top_items function to retrieve the data, and if there are no line items in the selected period, an informational message is displayed. Otherwise, a horizontal bar chart is rendered using Plot
    st.markdown(
        '<h3 class="section-header">Top 10 catalogue items by invoice value</h3>',
        unsafe_allow_html=True,
    )
    # The chart shows the total value of each item (excluding VAT) across all invoices in the selected period, allowing the user to quickly identify which items are generating the most revenue.
    df_items = query_top_items(trade, start, end, limit=10)
    if df_items.empty:
        st.info("No line items in this period.")
    else:
        df_items = df_items.sort_values("total_value", ascending=True)
        fig = px.bar(
            df_items,
            x="total_value",
            y="item",
            orientation="h",
            color_discrete_sequence=[TERRA],
            labels={"total_value": "Total value (€)", "item": ""},
        )
        _styled_plotly_layout(fig)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(showgrid=True, gridcolor="#eee", tickformat=",.0f")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    # Chart 3: County activity (sage gradient)
    #   The third chart displays the invoice count and total revenue by customer county based on the invoices, their associated customers, and job types, filtered by trade and date range.
    st.markdown(
        '<h3 class="section-header">Activity by county</h3>',
        unsafe_allow_html=True,
    )
    #  The chart uses the query_county_activity function to retrieve the data, and if there are no customer activities in the selected period, an informational message is displayed. 
    # Otherwise, a horizontal bar chart is rendered using Plotly Express, with a color gradient based on the invoice count to visually differentiate counties with higher activity.
    df_county = query_county_activity(trade, start, end)
    if df_county.empty:
        st.info("No customer activity in this period.")
    else:
        df_county = df_county.head(15).sort_values("revenue", ascending=True)
        fig = px.bar(
            df_county,
            x="revenue",
            y="county",
            orientation="h",
            color="invoice_count",
            color_continuous_scale=[SAGE_PALE, SAGE],
            labels={"revenue": "Revenue (€)", "county": "", "invoice_count": "Invoices"},
        )
        _styled_plotly_layout(fig)
        fig.update_xaxes(showgrid=True, gridcolor="#eee", tickformat=",.0f")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)


# Standalone test entry point
if __name__ == "__main__":
    st.set_page_config(page_title="Trade Manager - Dashboard", layout="wide")
    render_dashboard()
