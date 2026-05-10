"""
streamlit_app.py
=================

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gridfs
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from retrieve.orchestrator import answer_query
from dashboard import render_dashboard
from dashboard import get_mongo_db


# ─────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quotes Manager",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #FBF8F2; }
    .confidence-high   { color: #587858; font-weight: 600; }
    .confidence-medium { color: #C2942F; font-weight: 600; }
    .confidence-low    { color: #B85A5A; font-weight: 600; }
    .chip-row .stButton > button {
        background: #FDFAF6; border: 1px solid #E5DDD2; color: #2D2520;
        font-size: 0.85rem; font-weight: 500; padding: 6px 12px;
        border-radius: 18px; transition: all 0.15s;
    }
    .chip-row .stButton > button:hover {
        background: #F5EFE5; border-color: #B85A5A; color: #B85A5A;
    }
    .chip-label {
        color: #786558; font-size: 0.85rem; margin: 0.5rem 0 0.4rem 0;
        text-transform: uppercase; letter-spacing: 0.04em;
    }
    .page-footer {
        margin-top: 3rem; padding-top: 0.6rem;
        border-top: 1px solid #E5DDD2; color: #A89484;
        font-size: 0.7rem; line-height: 1.4;
    }
    .footer-status { display: flex; flex-wrap: wrap; gap: 18px; align-items: center; }
    .footer-status .label { font-weight: 600; color: #786558; }
    .footer-clear .stButton > button {
        background: transparent; border: 1px solid #E5DDD2; color: #786558;
        font-size: 0.7rem; padding: 2px 10px; border-radius: 6px;
        height: auto; min-height: 0; line-height: 1.2;
    }
    .footer-clear .stButton > button:hover { border-color: #B85A5A; color: #B85A5A; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def preload_models():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") # Preload the sentence transformer model for embedding generation. This is a one-time operation that can take around 10 seconds, 
                            # so we use Streamlit's caching mechanism to ensure that the model is loaded only once and reused across interactions, improving performance for subsequent queries that require embeddings.

with st.spinner("Loading embedding model (one-time, ~10 sec)..."): # Show a spinner while the model is loading to provide feedback to the user, since loading the model can take some time. 
    # This enhances the user experience by indicating that the system is working and prevents confusion during the initial load.
    _ = preload_models()


@st.cache_resource
def get_gridfs_buckets(): # Get GridFS bucket instances for the three types of files we manage: CSVs, PDFs, and emails. This function connects to MongoDB and returns a dictionary of GridFS bucket instances that can be used 
    # to read and write files in those buckets. By caching this resource, we ensure that we only establish the MongoDB connection and create the GridFS instances once, improving performance for file operations throughout the app.
    """
    Returns all three named GridFS buckets:
      csv_files   — seed CSVs
      po_files    — supplier PO PDFs
      email_files — raw .eml files
    """
    db = get_mongo_db()
    return {
        "csv":   gridfs.GridFS(db, collection="csv_files"),
        "pdf":   gridfs.GridFS(db, collection="po_files"),
        "email": gridfs.GridFS(db, collection="email_files"),
    }


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if "messages"        not in st.session_state: st.session_state.messages        = [] # Initialise session state variables for managing the chat conversation, pending queries, and ingest logs. 
                                        # The "messages" list will hold the history of user and assistant messages in the chat interface, allowing us to render the conversation history and maintain context across interactions.
if "last_prediction" not in st.session_state: st.session_state.last_prediction = None
if "pending_query"   not in st.session_state: st.session_state.pending_query   = None
if "ingest_log"      not in st.session_state: st.session_state.ingest_log      = []


# ─────────────────────────────────────────────────────────────
# SIDEBAR 
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Trade Manager")
    st.caption("AI assistant for Irish trades")


# ─────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────
st.title("Quotes and Customer Manager")
st.caption("AI assistant for Irish trades")


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_dashboard, tab_chat, tab_upload = st.tabs(["Dashboard", "Chatbot", "Data Upload"])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — BI DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab_dashboard:
    render_dashboard()


# ══════════════════════════════════════════════════════════════
#  TAB 2 — CHATBOT
# ══════════════════════════════════════════════════════════════
with tab_chat:

    st.markdown('<p class="chip-label">Try one of these</p>', unsafe_allow_html=True) # Render a set of sample query buttons for quick testing. These buttons allow users to easily try out common queries 
    # without having to type them manually. When a button is clicked, it sets the pending_query in the session state and triggers a rerun of the app to process that query and display the results.

    sample_queries = [
        "How much for a boiler installation?",
        "Show me PO-2026-P0042",
        "What's Gerard Walsh's phone number?",
        "I need a quote for floor sanding",
        "Has cust_0001 had work done before?",
        "Show me all POs",
        "Find all customers for Carpenter trade",
    ]

    st.markdown('<div class="chip-row">', unsafe_allow_html=True) # Render the sample query buttons in a row with custom styling. We use Streamlit's columns to layout the buttons in a responsive way, 
                        # and apply CSS classes for consistent styling of the buttons as "chips". Each button corresponds to a sample query that users can click to quickly see how the chatbot responds to common questions about quotes, customers, and purchase orders.
    chip_cols = st.columns(3)
    for i, q in enumerate(sample_queries):
        with chip_cols[i % 3]:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider() # Add a visual divider between the sample query buttons and the main chat interface for better separation of UI elements and improved user experience.

    col_chat, col_quote = st.columns([2, 1]) #  Set up a two-column layout for the chat interface and the predicted quote display. The left column (col_chat) will be used to render the conversation history and user input,
    # while the right column (col_quote) will display the predicted quote details based on the user's queries. This layout allows users to easily view both the chat conversation and the relevant quote information side by side.

    with col_quote: #    Render the predicted quote details in the right column. This section displays the job type, confidence level, materials, labour, totals, and benchmark information based on the last prediction made by the chatbot.
        st.subheader("Predicted Quote")
        pred = st.session_state.last_prediction

        if not pred: #  If there is no prediction available (i.e., the user has not asked a question that triggers a quote prediction), we show an informational message prompting the user to ask a pricing question. 
            # This provides guidance on how to use the chatbot to see predictions and encourages users to interact with the system to get quote estimates.
            st.info(
                "Ask a pricing question to see predictions here.\n\n"
                "Example: *How much for a boiler installation?*"
            )
        else: # If a prediction is available, we display the details of the predicted quote. This includes the job type, confidence level (with color-coded styling), breakdown of materials and labour costs, totals including VAT, 
            # and benchmark information comparing the predicted quote to similar past purchase orders.
            st.markdown(f"**Job:** {pred.get('job_type', '?')}")
            confidence = pred.get("confidence", "unknown")
            st.markdown( # Display the confidence level with color-coded styling based on whether it is high, medium, or low. This provides a visual indication of how confident the model is in its prediction, which can help users gauge the reliability of the quote estimate.
                f"**Confidence:** "
                f"<span class='confidence-{confidence}'>{confidence.upper()}</span>",
                unsafe_allow_html=True,
            )
            st.divider() #  Add a divider between the confidence level and the detailed breakdown of materials, labour, and totals for better visual separation of information. 
                        # This helps users easily distinguish between the overall prediction summary and the specific cost components of the quote.
            mat = pred.get("materials", {})
            if mat.get("subtotal", 0) > 0: # If there are materials costs included in the prediction, we display the subtotal for materials along with a breakdown of individual items.
                st.markdown(f"**Materials**  €{mat['subtotal']:,.2f}") # The materials section shows the total estimated cost for materials needed for the job, and we also list out the individual items with their quantities and unit prices. 
                                                        # If there are more than 5 items, we show a caption indicating that there are additional items not displayed to keep the interface clean and focused on the most relevant information.
                st.caption(f"From {mat.get('n_recipe_items', 0)} graph recipe items")
                for item in mat.get("items", [])[:5]: # Iterate over the first 5 items in the materials list and display them with their name, quantity, and unit price. 
                    #   This gives users a detailed view of the key materials contributing to the quote estimate, while keeping the display concise if there are many items.
                    st.markdown( # Display each material item with its name, quantity, and unit price in a formatted way. We use a small font size for the item details to differentiate it from the main subtotal and to fit more information in a compact space.
                        f"<small>• {item['item_name']}  "
                        f"({item['quantity']} × €{item['unit_price']:.2f})</small>",
                        unsafe_allow_html=True,
                    )
                if len(mat.get("items", [])) > 5: # If there are more than 5 material items, we show a caption indicating that there are additional items not displayed. This helps manage the amount of information shown in the interface while still 
                    # acknowledging that there are more materials contributing to the quote.
                    st.caption(f"...and {len(mat['items']) - 5} more")

            lab = pred.get("labour", {}) # If there are labour costs included in the prediction, we display the median labour cost along with a summary of past invoices that contribute to that estimate. 
                                            # This provides users with insight into how the labour cost was derived based on historical data.
            if lab.get("median_eur", 0) > 0: # The labour section shows the median estimated cost for labour based on past invoices for similar jobs. We also provide a caption that indicates the number 
                                            # of past invoices that were used to calculate the median, as well as the range of costs (minimum and maximum) observed in those invoices.
                st.divider()
                st.markdown(f"**Labour**  €{lab['median_eur']:,.2f}")
                st.caption(
                    f"Median of {lab.get('n_invoices', 0)} past invoices  "
                    f"(€{lab.get('min_eur', 0):.0f}–€{lab.get('max_eur', 0):.0f})"
                )

            totals = pred.get("totals", {}) # If there are total costs included in the prediction, we display the subtotal excluding VAT, the VAT amount, and the total including VAT. 
            # This gives users a clear breakdown of the overall cost estimate for the job, including tax considerations.
            if totals.get("total_inc_vat", 0) > 0: # The totals section shows the overall cost estimate for the job, including a breakdown of the subtotal excluding VAT, the VAT amount (calculated at 23%), and the total cost including VAT.
                st.divider()
                st.markdown(f"**Subtotal ex VAT**  €{totals.get('subtotal_ex_vat', 0):,.2f}")
                st.markdown(f"**VAT 23%**  €{totals.get('vat_23pct', 0):,.2f}")
                st.markdown(f"**Total inc VAT**  **€{totals.get('total_inc_vat', 0):,.2f}**")

            bench = pred.get("benchmark", {}) # If there is benchmark information available, we display the average total cost for similar past purchase orders along with the number of similar POs that were used to calculate that average.
            if bench.get("n_pos", 0) > 0: # The benchmark section provides a comparison of the predicted quote to historical data by showing the average total cost for similar past purchase orders. 
                # This helps users understand how the predicted quote aligns with what has been observed in the past for similar jobs, giving them additional context for evaluating the estimate.
                st.divider()
                st.caption(
                    f"**Benchmark:** €{bench.get('avg_total', 0):,.2f} "
                    f"(avg of {bench['n_pos']} similar PO(s))"
                )

            ev = pred.get("evidence", {}) # If there is evidence information available, we display the sources that were used to generate the prediction. This can include references to specific purchase orders, invoices, or other documents that contributed to the model's estimate.
            if ev.get("stores_used"):
                st.divider()
                st.caption("**Stores used:** " + ", ".join(ev["stores_used"]))

    with col_chat: # Render the chat interface. We display the conversation history by iterating over st.session_state.messages, which contains all user and assistant messages. For each message, we use st.chat_message to render it in the appropriate style (user or assistant).
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                #if msg.get("metadata"): # Optionally show metadata for each message in an expander. This can include the sources retrieved, latency, and routing information for each assistant response, which can be useful for debugging and evaluation purposes.
                    #meta = msg["metadata"]
                    #with st.expander(
                        #f"Sources ({len(meta.get('sources', []))}) · "
                        #f"{meta.get('latency_ms', 0)} ms · "
                        #f"{meta.get('routing', {}).get('intent', '?')}"
                    #):
                        #if meta.get("sources"):
                            #st.markdown("**Sources:**")
                            #for src in meta["sources"]:
                                #st.markdown(f"- {src}")
                        #if meta.get("routing"):
                            #st.markdown("---")
                            #st.markdown("**Routing:**")
                            #st.json(meta["routing"])

        typed_input = st.chat_input("Ask about a customer, quote, or job...")

        user_input = None # Determine the user input to process. We check if there is a new typed input from the chat input box, and if not, we check if there is a pending query set by one of the sample query buttons.
        if typed_input:
            user_input = typed_input
        elif st.session_state.pending_query:
            user_input = st.session_state.pending_query
            st.session_state.pending_query = None

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input}) # Add the user's input to the session state messages so that it will be rendered in the chat history. 
            # This allows us to maintain a complete conversation history that includes both user inputs and assistant responses, providing context for the ongoing interaction.
            with st.chat_message("user"):
                st.markdown(user_input)

            result  = None
            latency = 0
            with st.chat_message("assistant"): # Show a spinner while we process the user's query and generate a response. This provides feedback to the user that the system is working on their request, especially since querying the databases and generating a response can take some time.
                with st.spinner("Searching MongoDB, Neo4j, and Atlas Vector Search..."): # We call the answer_query function to process the user's input and generate a response. This function interacts with MongoDB, Neo4j, 
                                                                                                # and Atlas Vector Search to retrieve relevant information and generate an answer based on the user's query.
                    try:
                        t_start = time.time()
                        result  = answer_query(user_input, verbose=True)
                        latency = round((time.time() - t_start) * 1000)
                    except Exception as e: # If there is an error during the query processing, we catch the exception and display an appropriate error message to the user. We check the error message for specific 
                        # keywords to determine if it is related to a paused Neo4j AuraDB instance or an authentication error, and provide tailored messages for those cases. For any other errors, we display a generic error message with the exception details.
                        err_str = str(e).lower()
                        if "defunct" in err_str or "service unavailable" in err_str:
                            st.warning(
                                "**Neo4j AuraDB is paused.** Wake it at "
                                "[console.neo4j.io](https://console.neo4j.io), then retry."
                            )
                        elif "auth" in err_str: # Check for authentication errors and prompt the user to check their Neo4j credentials in the .env file. 
                            # This helps users troubleshoot common configuration issues that may prevent the chatbot from accessing the necessary data in Neo4j.
                            st.error("Authentication error. Check Neo4j credentials in .env.")
                        else:
                            st.error(f"Error: {e}")

                if result is not None: # If we successfully get a result from the answer_query function, we display the answer in the chat interface. We also check if there are raw results 
                    # included in the response (such as a "predict" field) and if so, we update the last_prediction in the session state with that information.
                    st.markdown(result["answer"])
                    raw_results = result.get("raw_results", {})
                    if "predict" in raw_results:
                        st.session_state.last_prediction = raw_results["predict"]



            if result is not None: # After processing the query and generating a response, we add the assistant's answer to the session state messages along with metadata about the sources used, latency, and routing information. 
                # This allows us to maintain a complete conversation history that includes the assistant's responses and the relevant metadata for each response, which can be useful for users to understand how the answer was generated and for debugging purposes.
                st.session_state.messages.append({
                    "role":     "assistant",
                    "content":  result["answer"],
                    "metadata": {
                        "sources":    result.get("sources", []),
                        "latency_ms": result.get("latency_ms", latency),
                        "routing":    result.get("routing", {}),
                    },
                })
                st.rerun()


# ══════════════════════════════════════════════════════════════
#  TAB 3 — DATA UPLOAD & INGEST ORCHESTRATOR
# ══════════════════════════════════════════════════════════════
with tab_upload: # Render the data upload and ingest orchestrator interface. This section allows users to upload CSV and PDF files directly into the appropriate GridFS buckets, 
    # view the files currently stored in GridFS, and run the various ingestion steps in order without needing to use the command line.

    st.markdown("## Data Upload & Ingest Orchestrator")
    st.caption(
        "Upload CSVs or PDFs into the correct GridFS bucket, "
        "then run each ingestion step without touching the command line."
    )

    # ── Import ingest runner ──────────────────────────────────
    try:
        from ingest.ingest_runner import ( # Import the ingest runner functions for each step of the ingestion pipeline. These functions correspond to the different stages of processing the uploaded data, such as loading 
                                          # CSVs into MongoDB, extracting information from PDFs, and embedding documents for vector search.
            run_load_mongo,
            run_extract_pos,
            run_extract_entities,
            run_load_neo4j,
            run_load_pos_neo4j,
            run_embed_documents,
        )
        runner_available = True
    except ImportError:
        runner_available = False

    buckets = get_gridfs_buckets()

    # ── SECTION 1: Upload ─────────────────────────────────────
    st.markdown("### Upload files") # Provide an interface for uploading CSV and PDF files directly into the appropriate GridFS buckets. We use Streamlit's file_uploader to allow users to select multiple files at once, and we show a progress bar during the upload process.

    up_col1, up_col2 = st.columns([3, 1])
    with up_col1: # Render the file uploader in the left column, allowing users to select CSV and PDF files for upload. We specify the accepted file types and allow multiple files to be uploaded at once. We also provide a help tooltip to guide users on which files go into which buckets.
        uploaded_files = st.file_uploader( # Render the file uploader component for uploading CSV and PDF files. We specify the accepted file types (CSV and PDF) and allow users to select multiple files at once. 
                                          # The uploaded files will be stored in the "uploaded_files" variable, which we can then process when the user clicks the "Upload to GridFS" button.
            "Choose CSVs or PDFs",
            type=["csv", "pdf"],
            accept_multiple_files=True,
            key="gridfs_uploader",
            help="CSVs → csv_files bucket · PDFs → po_files bucket",
        )
    with up_col2: # Provide a caption in the right column that explains which types of files should be uploaded to which GridFS buckets. This helps users understand how to organize their uploads correctly, 
        # ensuring that CSV files go into the "csv_files" bucket and PDF files go into the "po_files" bucket for proper processing in the ingestion pipeline.
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("**CSVs** → `csv_files` bucket  \n**PDFs** → `po_files` bucket")

    if uploaded_files: # If there are files uploaded, we show an "Upload to GridFS" button. When the user clicks this button, we process each uploaded file, determine the appropriate GridFS bucket based on the file type, 
        # and upload the file to MongoDB. We also handle duplicate files by checking if a file with the same name already exists in the bucket and skip it if so. During the upload process, we show a progress bar to provide feedback to the user on the upload status.
        if st.button("Upload to GridFS", type="primary", key="btn_upload"):
            upload_results = []
            progress = st.progress(0)

            for idx, uf in enumerate(uploaded_files): # Iterate over each uploaded file and process it for upload to GridFS. We determine the file type based on the extension, select the appropriate bucket, 
                # and attempt to upload the file. We also handle duplicates and errors, recording the results of each upload attempt for later display.
                filename    = uf.name
                is_pdf      = filename.lower().endswith(".pdf")
                bucket      = buckets["pdf"] if is_pdf else buckets["csv"]
                ctype       = "application/pdf" if is_pdf else "text/csv"
                bucket_name = "po_files" if is_pdf else "csv_files"

                if bucket.find_one({"filename": filename}):
                    upload_results.append(("skip", filename, bucket_name))
                else:
                    try:
                        file_id = bucket.put( # Upload the file to GridFS by reading its content and storing it with the original filename and content type. We use the put method of the GridFS bucket to save the file, 
                                             # which returns a unique file ID that we can use for reference.
                            uf.getvalue(),
                            filename=filename,
                            content_type=ctype,
                        )
                        upload_results.append(("ok", filename, bucket_name, str(file_id))) # If the upload is successful, we record the result as "ok" along with the filename, bucket name, and the generated file ID. 
                        # This information will be displayed to the user after the upload process is complete.
                    except Exception as exc:
                        upload_results.append(("err", filename, bucket_name, str(exc)))

                progress.progress((idx + 1) / len(uploaded_files))

            progress.empty() # After processing all uploaded files, we clear the progress bar and display the results of the upload attempts. We show a success message for successful uploads, an info message for skipped duplicates, and an error message for any failures.
            for row in upload_results:
                if row[0] == "ok":
                    st.success(f"✓ **{row[1]}** → `{row[2]}` (id: `{row[3]}`)")
                elif row[0] == "skip":
                    st.info(f"↷ **{row[1]}** already in `{row[2]}` — skipped")
                else:
                    st.error(f"✗ **{row[1]}** failed: {row[3]}")

    st.divider()

    # ── SECTION 2: Files in GridFS ────────────────────────────
    st.markdown("### Files in GridFS")

    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        st.button("Refresh", key="btn_refresh_gridfs") # Provide a "Refresh" button to allow users to reload the list of files in GridFS after uploading new files or making changes. When the user clicks this button, 
        # it triggers a rerun of the app, which will re-query GridFS and update the displayed list of files accordingly.

    try: # Display the files currently stored in GridFS across the three buckets (CSV files, PO PDFs, and email files). We query each bucket for its contents and compile a list of files with their details such as filename, content type, size, and upload date.
        bucket_map = {
            "csv_files":   buckets["csv"],
            "po_files":    buckets["pdf"],
            "email_files": buckets["email"],
        }

        rows = []
        for bucket_name, fs in bucket_map.items(): # Iterate over each GridFS bucket and query for the files stored in it. For each file found, we extract relevant details such as the filename, content type, size in KB, and upload date, 
            # and compile this information into a list of rows that we can display in a table format.
            for d in fs.find():
                rows.append({
                    "Bucket":    bucket_name,
                    "Filename":  d.filename,
                    "Type":      d.content_type or "—",
                    "Size (KB)": round(d.length / 1024, 1),
                    "Uploaded":  d.upload_date.strftime("%Y-%m-%d %H:%M") if d.upload_date else "—",
                })

        if not rows:
            st.info("No files found in any GridFS bucket.") #   If there are no files found in any of the GridFS buckets, we display an informational message to the user indicating that there are currently no files stored. 
            # This provides feedback on the state of the file storage and encourages users to upload files if they haven't already.
        else:
            df_fs = pd.DataFrame(rows).sort_values(["Bucket", "Filename"])
            st.dataframe(df_fs, use_container_width=True, hide_index=True)
            st.caption(
                f"{len(rows)} file(s) total · "
                f"{sum(r['Size (KB)'] for r in rows):.1f} KB · "
                f"across {len(bucket_map)} buckets"
            )

    except Exception as e:
        st.error(f"Could not read GridFS: {e}")

    st.divider()

    # ── SECTION 3: Full ingest pipeline ──────────────────────
    st.markdown("### Ingest pipeline") # instructions to run the full ingest pipeline after uploading new data. We provide a clear caption that explains the order in which the steps should be run based on the type of data uploaded (new PO PDFs, new CSV data, or new emails).
    st.caption(
        "Run steps in order after uploading. "
        "Results show which collections were updated."
    )
    st.info("""
    **Which steps to run after uploading:**

     **New PO PDF** → run in this order: **② then ⑤ then ⑥**
    - ② extracts fields from the PDF into the `pos` collection
    - ⑤ creates the PO node in Neo4j and links to customer, job and materials
    - ⑥ embeds the PO description so the chatbot can find it via vector search

     **New CSV data** (customers, invoices, items) → run in this order: **① then ④ then ⑥**
    - ① loads CSV records into MongoDB collections
    - ④ projects the updated data into the Neo4j graph
    - ⑥ regenerates embeddings to reflect new records

     **New emails** → run in this order: **③ then ④ then ⑥**
    - ③ extracts customer and job entities from emails
    - ④ adds email nodes to the Neo4j graph
    - ⑥ embeds email bodies for vector search

     **Full rebuild** → click **▶ Run all steps**
    """)

    if not runner_available:
        st.warning(
            "`ingest/ingest_runner.py` not found — pipeline buttons disabled. "
            "Add it to the `ingest/` folder and redeploy."
        )

    # All 6 pipeline steps
    pipeline_steps = [
        {
            "key":   "step_load_mongo",
            "label": "① Load CSVs → MongoDB",
            "desc":  "Reads CSVs from `csv_files` bucket · upserts into `customers`, "
                     "`invoices`, `job_types`, `items`, `job_items`, `invoice_items`.",
            "fn":    "run_load_mongo",
        },
        {
            "key":   "step_extract_pos",
            "label": "② Extract PO PDFs → `pos`",
            "desc":  "Reads PDFs from `po_files` bucket · Claude Haiku extracts fields · "
                     "upserts structured records into `pos` collection.",
            "fn":    "run_extract_pos",
        },
        {
            "key":   "step_extract_entities",
            "label": "③ Extract entities from emails",
            "desc":  "Reads emails from `email_files` bucket · Claude Haiku extracts "
                     "customer and job entities · writes into `emails` collection.",
            "fn":    "run_extract_entities",
        },
        {
            "key":   "step_load_neo4j",
            "label": "④ Load MongoDB → Neo4j graph",
            "desc":  "Projects customers, invoices, job types and items from MongoDB "
                     "into Neo4j as nodes and relationships.",
            "fn":    "run_load_neo4j",
        },
        {
            "key":   "step_load_pos_neo4j",
            "label": "⑤ Load POs → Neo4j graph",
            "desc":  "Creates PO nodes in Neo4j · connects to Customer, JobType and "
                     "Item nodes via FOR_CUSTOMER, FOR_JOB, CONTAINS_ITEM relationships.",
            "fn":    "run_load_pos_neo4j",
        },
        {
            "key":   "step_embed",
            "label": "⑥ Generate embeddings → Vector index",
            "desc":  "Embeds email bodies, PO descriptions and customer notes using "
                     "all-MiniLM-L6-v2 · writes 384-dim vectors into `embeddings`.",
            "fn":    "run_embed_documents",
        },
    ]

    # Initialise step states in session state if not already set. We track the state of each pipeline step (idle, running, done, error) in the session state to manage the UI and provide feedback to the user on the status of each step. 
    # This allows us to disable buttons while a step is running and show appropriate labels based on the current state
    for step in pipeline_steps:
        if step["key"] not in st.session_state:
            st.session_state[step["key"]] = "idle"

    # Build function map
    fn_map = {} #   Build a mapping of function names to the actual functions imported from the ingest runner. This allows us to dynamically call the appropriate function for each pipeline step when the user clicks the corresponding button.
    if runner_available:
        fn_map = {
            "run_load_mongo":        run_load_mongo,
            "run_extract_pos":       run_extract_pos,
            "run_extract_entities":  run_extract_entities,
            "run_load_neo4j":        run_load_neo4j,
            "run_load_pos_neo4j":    run_load_pos_neo4j,
            "run_embed_documents":   run_embed_documents,
        }

    for step in pipeline_steps: # Iterate over each pipeline step and render a container with the step label, description, and a button to run the step. The button's label and disabled state are determined based on the current state of the step 
        # (idle, running, done, error). When the button is clicked, we update the state to "running", call the corresponding function from the
        with st.container():
            c_label, c_btn = st.columns([4, 1])
            with c_label:
                st.markdown(f"**{step['label']}**")
                st.caption(step["desc"])
            with c_btn:
                state     = st.session_state[step["key"]]
                btn_label = {
                    "idle":    "Run",
                    "running": "Running…",
                    "done":    "✓ Done",
                    "error":   "✗ Error",
                }.get(state, "Run")

                disabled = (not runner_available) or (state == "running") # Disable the button if the ingest runner is not available or if the step is currently running to prevent multiple simultaneous executions of the same step, which could lead to conflicts or inconsistent states in the database.
                if st.button(
                    btn_label,
                    key=f"btn_{step['key']}",
                    disabled=disabled,
                    use_container_width=True,
                ):
                    st.session_state[step["key"]] = "running"
                    st.session_state.ingest_log.append(
                        f"[{time.strftime('%H:%M:%S')}] Starting: {step['label']}"
                    )
                    try:
                        result_msg = fn_map[step["fn"]]()
                        st.session_state[step["key"]] = "done"
                        st.session_state.ingest_log.append(
                            f"[{time.strftime('%H:%M:%S')}] ✓ {step['label']}: {result_msg}"
                        )
                    except Exception as exc:
                        st.session_state[step["key"]] = "error"
                        st.session_state.ingest_log.append(
                            f"[{time.strftime('%H:%M:%S')}] ✗ {step['label']} FAILED: {exc}"
                        )
                    st.rerun()

        st.markdown( # Add a horizontal rule between each pipeline step for better visual separation. This helps users distinguish between the different steps in the pipeline and improves the overall readability of the interface.
            "<hr style='border:none;border-top:1px solid #E5DDD2;margin:8px 0'>",
            unsafe_allow_html=True,
        )

    col_reset, col_runall = st.columns([1, 1]) #  At the bottom of the pipeline steps, we provide two buttons: "Reset pipeline" to reset the state of all steps back to idle and clear the ingest log, and " Run all steps" 
    # to execute all pipeline steps in order without needing to click each button individually.
    with col_reset:
        if st.button("Reset pipeline", key="btn_reset_pipeline", use_container_width=True):
            for step in pipeline_steps:
                st.session_state[step["key"]] = "idle"
            st.session_state.ingest_log = []
            st.rerun()
    with col_runall:
        if st.button(
            "▶ Run all steps",
            key="btn_run_all",
            type="primary",
            disabled=not runner_available,
            use_container_width=True,
        ):
            for step in pipeline_steps: #   When the "Run all steps" button is clicked, we iterate over each pipeline step and execute them in order. We update the state of each step to "running", call the corresponding function, and handle the results and any exceptions that may occur.
                st.session_state[step["key"]] = "running"
                st.session_state.ingest_log.append(
                    f"[{time.strftime('%H:%M:%S')}] Starting: {step['label']}"
                )
                try:
                    result_msg = fn_map[step["fn"]]() # Call the function corresponding to the current pipeline step and capture any result message it returns. This allows us to provide feedback on the outcome of each step in the ingest log, which can be useful for monitoring the progress and results of the pipeline execution.
                    st.session_state[step["key"]] = "done"
                    st.session_state.ingest_log.append(
                        f"[{time.strftime('%H:%M:%S')}] ✓ {step['label']}: {result_msg}"
                    )
                except Exception as exc: # If there is an exception during the execution of any pipeline step, we catch the exception, update the state of the step to "error", and log the failure in the ingest log with the exception details. This helps users identify which step failed and what the error was, allowing for easier troubleshooting and resolution.
                    st.session_state[step["key"]] = "error"
                    st.session_state.ingest_log.append(
                        f"[{time.strftime('%H:%M:%S')}] ✗ {step['label']} FAILED: {exc}"
                    )
            st.rerun()




# ─────────────────────────────────────────────────────────────
# PAGE-WIDE FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="page-footer">', unsafe_allow_html=True)

footer_col_status, footer_col_clear = st.columns([5, 1])

with footer_col_status:
    st.markdown(
        """
<div class="footer-status">
  <span><span class="label">MongoDB</span> 1,468 records</span>
  <span><span class="label">Neo4j</span> 537 nodes</span>
  <span><span class="label">Vector index</span> 263 chunks</span>
  <span><span class="label">Pricing engine</span> live</span>
</div>
""",
        unsafe_allow_html=True,
    )

with footer_col_clear: # Provide a "Clear chat" button in the footer that allows users to reset the conversation history and any stored predictions. When the user clicks this button, we clear the session state messages and last prediction, 
    # and then rerun the app to reflect the cleared state in the interface.
    st.markdown('<div class="footer-clear">', unsafe_allow_html=True)
    if st.button("Clear chat", key="footer_clear_btn", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.last_prediction = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
