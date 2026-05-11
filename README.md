# Trade Manager

A multi-database RAG system for Irish trades businesses (plumbers, electricians, carpenters). Combines MongoDB Atlas, Neo4j AuraDB, and Atlas Vector Search with Claude for natural-language quoting, customer history queries, and procurement insights.

Applied Data Science final project, DkIT.

## Live demo

(https://trades-manager-app-yjskqkjbsz6luhl9drzhhu.streamlit.app/)

## Tech stack

- **MongoDB Atlas** — records (customers, invoices, items, POs)
- **Neo4j AuraDB** — relationships and materials recipes
- **Atlas Vector Search** — semantic search over emails and PO descriptions
- **Claude (Anthropic)** — query routing, PO extraction, response generation
- **Streamlit** — web UI
- **Python 3.11**

