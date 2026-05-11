"""
generate_trades_dataset.py
===========================
Synthetic dataset generator for the Irish trades quote-request RAG project.

Domain: customers emailing plumbers, carpenters, or electricians asking for
job quotes. Each trade has past customers, past invoices, job types, and items.

Outputs (written to ./data/):
    customers.csv        - 120 past customers (40 per trade), Irish names + Eircodes
    job_types.csv        -  60 job types (20 per trade), with labour cost + skill level
    items.csv            - master items catalogue (materials across all trades)
    job_items.csv        - 300 rows: 5 typical items needed per job type
    invoices.csv         - 150 past invoices linking customers -> job_types
    invoice_items.csv    - ~600 rows: line items per invoice with quantity + price
    emails.csv           - 100 incoming quote-request emails (full text + headers)
    emails/*.eml         - same 100 emails as individual MIME .eml files

Usage:
    python generate_trades_dataset.py

No external dependencies - Python 3.8+ standard library only.
"""

from __future__ import annotations

import csv
import json
import random
import re
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import format_datetime, make_msgid
from pathlib import Path

# Reproducibility - same seed = same dataset every time
random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT_DIR = Path("data")
EML_DIR = OUT_DIR / "emails"

TRADES = ["plumber", "carpenter", "electrician"]
VAT_RATE = 0.23
CUSTOMERS_PER_TRADE = 40
NUM_INVOICES = 150
NUM_EMAILS = 100

TODAY = datetime(2026, 4, 22)

# ---------------------------------------------------------------------------
# Irish flavour data
# ---------------------------------------------------------------------------

IRISH_FIRST_NAMES = [
    "Sean", "Aoife", "Conor", "Niamh", "Liam", "Siobhan", "Declan", "Orla",
    "Ruairi", "Sinead", "Cillian", "Grainne", "Oisin", "Caoimhe", "Padraig",
    "Roisin", "Fergal", "Eimear", "Brendan", "Maire", "Kieran", "Fiona",
    "Diarmuid", "Aisling", "Tadhg", "Clodagh", "Eamon", "Saoirse", "Donal",
    "Ciara", "Ronan", "Blathnaid", "Aidan", "Nuala", "Eoin", "Dervla",
    "Feargal", "Mairead", "Paddy", "Una", "Gerard", "Bridget"
]

IRISH_SURNAMES = [
    "Murphy", "Kelly", "O'Brien", "Byrne", "O'Sullivan", "Ryan", "O'Connor",
    "Walsh", "Doyle", "McCarthy", "Gallagher", "O'Neill", "Reilly", "Doherty",
    "Kennedy", "Lynch", "Quinn", "Brennan", "Burke", "Collins", "Fitzgerald",
    "Moore", "Daly", "Hughes", "Farrell", "Martin", "Nolan", "Flynn",
    "O'Donnell", "Power", "McGrath", "Buckley", "O'Reilly", "Boyle", "Healy",
    "Sheehan", "Dunne", "Barry", "Sweeney", "Mahony"
]

# County -> list of routing key prefixes used in Eircodes
COUNTIES = {
    "Dublin": ["D01", "D02", "D03", "D04", "D06", "D07", "D08", "D09",
               "D12", "D14", "D15", "D16", "D18", "D22", "D24"],
    "Cork": ["T12", "T23", "P25", "P31"],
    "Galway": ["H91", "H54", "H53"],
    "Limerick": ["V94", "V35", "V42"],
    "Waterford": ["X91", "X42"],
    "Meath": ["A82", "A83", "C15"],
    "Kildare": ["W12", "W34", "R51"],
    "Wicklow": ["A63", "A67"],
    "Kilkenny": ["R95", "R93"],
    "Louth": ["A91", "A92"],
}

STREETS = [
    "Main St", "High St", "Church Rd", "Oak Drive", "The Crescent",
    "Willow Park", "Parnell St", "Pearse St", "O'Connell Ave",
    "St Patrick's Rd", "The Green", "Maple Grove", "Meadow Lane",
    "Riverside", "Castle View", "Priory Rd", "Abbey Ct", "Grove Ave",
    "Bishop St", "Dean's Close", "Newtown Ave", "Hillside Dr",
    "Beech Park", "Sycamore Rd", "Elm Grove", "Ash Lane"
]

EMAIL_DOMAINS = ["gmail.com", "hotmail.com", "yahoo.ie", "eircom.net",
                 "outlook.com", "icloud.com", "live.ie"]

# ---------------------------------------------------------------------------
# Job types (20 per trade)
# ---------------------------------------------------------------------------
# Tuple schema: (job_type_id, job_name, typical_hours, base_labour_cost_eur,
#                skill_level_1to5, description)

JOB_TYPES_DATA: dict[str, list[tuple]] = {
    "plumber": [
        ("pl_01", "Boiler installation",         8.0,  450, 4, "Install new gas or oil boiler including commissioning and safety test"),
        ("pl_02", "Boiler service",              1.5,   95, 2, "Annual boiler maintenance, clean, and safety check"),
        ("pl_03", "Leaking pipe repair",         2.0,  120, 2, "Locate and repair leaking pipework under floors or behind walls"),
        ("pl_04", "Bathroom suite installation", 16.0, 800, 4, "Remove old suite and install new bath, WC, basin with plumbing"),
        ("pl_05", "Radiator replacement",        2.0,  150, 2, "Drain system, remove old radiator, fit and balance new"),
        ("pl_06", "Toilet installation",         3.0,  180, 2, "Fit new toilet pan, cistern, and connect to soil stack"),
        ("pl_07", "Shower installation",         4.0,  220, 3, "Fit electric or thermostatic mixer shower including wall prep"),
        ("pl_08", "Tap replacement",             1.0,   75, 1, "Replace kitchen or bathroom taps"),
        ("pl_09", "Hot water cylinder install",  6.0,  350, 3, "Install unvented or vented hot water cylinder"),
        ("pl_10", "Drain unblocking",            1.5,  110, 2, "Clear blocked drains using rods or high-pressure jet"),
        ("pl_11", "Pipe insulation",             3.0,  140, 1, "Insulate pipework against frost in attics or outbuildings"),
        ("pl_12", "Gas safety certificate",      1.5,   85, 3, "Landlord annual gas safety inspection and certificate"),
        ("pl_13", "Underfloor heating install",  20.0, 1200, 5, "Install wet underfloor heating system with manifold"),
        ("pl_14", "Waste disposal unit",         2.0,  180, 2, "Fit sink waste disposal unit with wiring"),
        ("pl_15", "Washing machine plumbing",    1.0,   95, 1, "Connect washing machine to supply and waste"),
        ("pl_16", "Dishwasher plumbing",         1.0,   90, 1, "Connect dishwasher to cold supply and waste"),
        ("pl_17", "Immersion heater replace",    2.0,  160, 2, "Replace electric immersion heater element and thermostat"),
        ("pl_18", "Outside tap installation",    2.5,  145, 2, "Fit frost-protected outside tap with isolation valve"),
        ("pl_19", "Burst pipe emergency",        2.0,  180, 3, "Emergency burst pipe repair, including make-safe"),
        ("pl_20", "Pressure booster install",    4.0,  280, 3, "Install water pressure booster pump and expansion vessel"),
    ],
    "carpenter": [
        ("cp_01", "Kitchen installation",        24.0, 1400, 5, "Full kitchen unit installation including worktop and appliance cut-outs"),
        ("cp_02", "Fitted wardrobe building",    16.0,  900, 4, "Build custom fitted wardrobes to alcove or wall"),
        ("cp_03", "Door hanging",                 2.0,  130, 2, "Hang internal or external door including hinges and latch"),
        ("cp_04", "Wooden floor laying",         14.0,  750, 3, "Lay solid or engineered wood flooring with underlay"),
        ("cp_05", "Skirting board fitting",       6.0,  260, 2, "Fit MDF or timber skirting with mitred corners"),
        ("cp_06", "Staircase repair",             5.0,  340, 4, "Repair damaged stair treads, risers, or balustrade"),
        ("cp_07", "Decking construction",        20.0, 1100, 3, "Build garden deck with treated timber frame and boards"),
        ("cp_08", "Shelving installation",        2.0,  110, 1, "Fit bespoke or flat-pack shelving"),
        ("cp_09", "Window frame repair",          4.0,  220, 3, "Repair rotted timber window frames or cills"),
        ("cp_10", "Attic flooring",               8.0,  420, 2, "Lay chipboard or tongue-and-groove flooring in attic"),
        ("cp_11", "Built-in bookcase",           10.0,  560, 3, "Build floor-to-ceiling bookcase in study or living room"),
        ("cp_12", "Loft hatch installation",      3.0,  180, 2, "Cut opening and fit insulated loft hatch and ladder"),
        ("cp_13", "Kitchen cabinet repair",       3.0,  150, 2, "Repair hinges, doors, or drawer runners"),
        ("cp_14", "Garden shed build",           16.0,  820, 3, "Build 6x4 or 8x6 timber garden shed on prepared base"),
        ("cp_15", "Fence panel replacement",      4.0,  190, 1, "Replace damaged fence panels with feather-edge or lap"),
        ("cp_16", "Garden gate installation",     5.0,  280, 2, "Fit timber garden gate with hinges, latch, and hardware"),
        ("cp_17", "Architrave fitting",           3.0,  140, 2, "Fit architrave around doors with mitred corners"),
        ("cp_18", "Worktop replacement",          6.0,  380, 3, "Replace kitchen worktop including sink cut-out"),
        ("cp_19", "Floor sanding",                8.0,  380, 3, "Sand and refinish wooden floors with lacquer or oil"),
        ("cp_20", "Staircase balustrade",        12.0,  680, 4, "Install new balustrade, newel posts, and handrail"),
    ],
    "electrician": [
        ("el_01", "Full house rewiring",         40.0, 2400, 5, "Complete electrical rewire of 3-bed home including certification"),
        ("el_02", "Extra socket installation",    1.5,   90, 1, "Fit additional double socket including chase and make-good"),
        ("el_03", "Light fitting installation",   1.0,   80, 1, "Install pendant, ceiling, or wall light"),
        ("el_04", "Consumer unit upgrade",        6.0,  480, 4, "Replace fuse board with modern RCD-protected consumer unit"),
        ("el_05", "EV charger installation",      6.0,  550, 4, "Install type-2 home EV charger including SEAI grant paperwork"),
        ("el_06", "Smoke alarm installation",     2.0,  140, 2, "Fit interconnected mains-powered smoke and heat alarms"),
        ("el_07", "Outdoor lighting",             3.0,  220, 2, "Install garden or driveway lighting with weatherproof cable"),
        ("el_08", "Doorbell installation",        1.0,   70, 1, "Fit wired or wireless doorbell with chime unit"),
        ("el_09", "Electrical safety cert",       2.5,  180, 3, "Periodic inspection report and RECI certificate"),
        ("el_10", "CCTV installation",            5.0,  380, 3, "Install 4-camera CCTV system with NVR and cabling"),
        ("el_11", "Bathroom extractor fan",       2.0,  130, 2, "Fit humidity-sensing IP-rated extractor fan"),
        ("el_12", "Immersion heater wiring",      2.0,  120, 2, "Wire immersion heater to timer switch and isolator"),
        ("el_13", "Cooker circuit installation",  3.0,  180, 3, "Dedicated 32A circuit for electric cooker or hob"),
        ("el_14", "Garden socket installation",   2.5,  150, 2, "Install weatherproof IP66 garden socket with RCD"),
        ("el_15", "Bathroom lights upgrade",      2.0,  140, 3, "Upgrade to IP-rated LED bathroom lighting zones 1 and 2"),
        ("el_16", "Security light install",       1.5,  100, 2, "Fit PIR-activated LED security light"),
        ("el_17", "Intercom system",              4.0,  280, 3, "Install audio or video door intercom system"),
        ("el_18", "Shower pull cord install",     1.0,   85, 2, "Fit bathroom shower isolation pull-cord switch"),
        ("el_19", "Fuse box test",                1.5,   95, 2, "Test and certify existing consumer unit"),
        ("el_20", "Landlord electrical report",   3.0,  200, 3, "RECI periodic inspection report for rental property"),
    ],
}

# ---------------------------------------------------------------------------
# Items catalogue
# ---------------------------------------------------------------------------
# Schema: (item_id, name, unit, unit_price_ex_vat, category)

ITEMS_DATA = [
    # Plumbing items
    ("it_pl_001", "15mm copper pipe",          "metre",  4.50,  "plumbing"),
    ("it_pl_002", "22mm copper pipe",          "metre",  6.80,  "plumbing"),
    ("it_pl_003", "PTFE tape",                 "roll",   1.20,  "plumbing"),
    ("it_pl_004", "15mm compression fitting",  "each",   2.50,  "plumbing"),
    ("it_pl_005", "22mm compression fitting",  "each",   3.80,  "plumbing"),
    ("it_pl_006", "Lead-free solder 250g",     "pack",   8.50,  "plumbing"),
    ("it_pl_007", "Pipe clip pack of 10",      "pack",   3.20,  "plumbing"),
    ("it_pl_008", "Radiator valve pair",       "pair",  18.00,  "plumbing"),
    ("it_pl_009", "Gas boiler 24kW",           "unit", 1100.00,  "plumbing"),
    ("it_pl_010", "Toilet pan and cistern",    "unit",  180.00,  "plumbing"),
    ("it_pl_011", "Bathroom suite white",      "unit",  420.00,  "plumbing"),
    ("it_pl_012", "Electric shower 8.5kW",     "unit",  160.00,  "plumbing"),
    ("it_pl_013", "Mixer shower thermostatic", "unit",  210.00,  "plumbing"),
    ("it_pl_014", "Kitchen tap chrome",        "unit",   75.00,  "plumbing"),
    ("it_pl_015", "Bathroom mixer tap",        "unit",   85.00,  "plumbing"),
    ("it_pl_016", "Hot water cylinder 210L",   "unit",  340.00,  "plumbing"),
    ("it_pl_017", "Immersion heater element",  "each",   42.00,  "plumbing"),
    ("it_pl_018", "Pipe insulation 15mm 2m",   "length", 4.80,  "plumbing"),
    ("it_pl_019", "Drain rod set",             "set",    35.00,  "plumbing"),
    ("it_pl_020", "Waste disposal unit",       "unit",  150.00,  "plumbing"),
    ("it_pl_021", "Radiator type-22 600x1000", "unit",   95.00,  "plumbing"),
    ("it_pl_022", "Underfloor heating mat",    "sqm",    28.00,  "plumbing"),
    ("it_pl_023", "Manifold 6-port",           "unit",  185.00,  "plumbing"),
    ("it_pl_024", "Pressure booster pump",     "unit",  220.00,  "plumbing"),
    ("it_pl_025", "Outside tap kit",           "kit",    32.00,  "plumbing"),
    ("it_pl_026", "Boiler flue kit",           "kit",    85.00,  "plumbing"),
    ("it_pl_027", "Washing machine hose pair", "pair",   12.00,  "plumbing"),

    # Carpentry items
    ("it_cp_001", "MDF sheet 18mm 8x4",        "sheet",  38.00,  "carpentry"),
    ("it_cp_002", "Timber C16 2x4 4.8m",       "length", 18.00,  "carpentry"),
    ("it_cp_003", "Timber C24 2x6 4.8m",       "length", 28.00,  "carpentry"),
    ("it_cp_004", "Wood screws 4.0x40 box",    "box",    14.00,  "carpentry"),
    ("it_cp_005", "Wood screws 5.0x90 box",    "box",    22.00,  "carpentry"),
    ("it_cp_006", "Butt hinges pair brass",    "pair",    6.50,  "carpentry"),
    ("it_cp_007", "Internal door oak",         "unit",  120.00,  "carpentry"),
    ("it_cp_008", "External door hardwood",    "unit",  340.00,  "carpentry"),
    ("it_cp_009", "Engineered oak flooring",   "sqm",    42.00,  "carpentry"),
    ("it_cp_010", "Solid oak flooring",        "sqm",    68.00,  "carpentry"),
    ("it_cp_011", "Floor underlay 15sqm",      "roll",   28.00,  "carpentry"),
    ("it_cp_012", "MDF skirting 18x145mm 4m",  "length", 16.00,  "carpentry"),
    ("it_cp_013", "Architrave MDF 4m",         "length", 12.00,  "carpentry"),
    ("it_cp_014", "Decking board 4.8m",        "board",  24.00,  "carpentry"),
    ("it_cp_015", "Fence panel 6x6ft",         "panel",  38.00,  "carpentry"),
    ("it_cp_016", "Fence post treated",        "each",   18.00,  "carpentry"),
    ("it_cp_017", "Kitchen cabinet 600mm",     "unit",   95.00,  "carpentry"),
    ("it_cp_018", "Worktop laminate 3m",       "length", 140.00,  "carpentry"),
    ("it_cp_019", "Worktop solid oak 3m",      "length", 320.00,  "carpentry"),
    ("it_cp_020", "Shelf bracket pair",        "pair",    6.00,  "carpentry"),
    ("it_cp_021", "Handle knob chrome",        "each",    4.50,  "carpentry"),
    ("it_cp_022", "Door latch set",            "set",     8.50,  "carpentry"),
    ("it_cp_023", "Garden shed kit 8x6",       "kit",   680.00,  "carpentry"),
    ("it_cp_024", "Loft hatch insulated",      "unit",   95.00,  "carpentry"),
    ("it_cp_025", "Loft ladder 3-section",     "unit",  145.00,  "carpentry"),
    ("it_cp_026", "Wood filler tub",           "tub",     8.00,  "carpentry"),
    ("it_cp_027", "Sanding discs pack",        "pack",   14.00,  "carpentry"),
    ("it_cp_028", "Floor lacquer 5L",          "tin",    58.00,  "carpentry"),
    ("it_cp_029", "Balustrade oak spindle",    "each",    9.50,  "carpentry"),
    ("it_cp_030", "Newel post oak",            "each",   48.00,  "carpentry"),
    ("it_cp_031", "Handrail oak 3m",           "length", 62.00,  "carpentry"),

    # Electrical items
    ("it_el_001", "Twin & earth 2.5mm 100m",    "reel",  120.00,  "electrical"),
    ("it_el_002", "Twin & earth 6mm 50m",       "reel",  165.00,  "electrical"),
    ("it_el_003", "Twin & earth 1.5mm 100m",    "reel",   85.00,  "electrical"),
    ("it_el_004", "Double socket white",        "each",    3.80,  "electrical"),
    ("it_el_005", "Single socket white",        "each",    2.80,  "electrical"),
    ("it_el_006", "Light switch 1-gang",        "each",    3.20,  "electrical"),
    ("it_el_007", "Light switch 2-gang",        "each",    4.50,  "electrical"),
    ("it_el_008", "Junction box 30A",           "each",    5.50,  "electrical"),
    ("it_el_009", "MCB 16A type-B",             "each",    6.80,  "electrical"),
    ("it_el_010", "MCB 32A type-B",             "each",    7.50,  "electrical"),
    ("it_el_011", "RCD 40A 30mA",               "each",   28.00,  "electrical"),
    ("it_el_012", "Consumer unit 10-way",       "unit",  145.00,  "electrical"),
    ("it_el_013", "Smoke alarm mains optical",  "unit",   38.00,  "electrical"),
    ("it_el_014", "Heat alarm mains",           "unit",   32.00,  "electrical"),
    ("it_el_015", "LED pendant fitting",        "unit",   24.00,  "electrical"),
    ("it_el_016", "LED downlight IP65",         "each",    9.50,  "electrical"),
    ("it_el_017", "Outdoor PIR light LED",      "unit",   42.00,  "electrical"),
    ("it_el_018", "EV charger 7.4kW type-2",    "unit",  420.00,  "electrical"),
    ("it_el_019", "CCTV camera 4MP",            "each",   85.00,  "electrical"),
    ("it_el_020", "CCTV NVR 8-channel",         "unit",  220.00,  "electrical"),
    ("it_el_021", "CAT6 cable 100m",            "reel",   45.00,  "electrical"),
    ("it_el_022", "Extractor fan humidity",     "unit",   48.00,  "electrical"),
    ("it_el_023", "Doorbell wired",             "kit",    22.00,  "electrical"),
    ("it_el_024", "Intercom video kit",         "kit",  180.00,  "electrical"),
    ("it_el_025", "Garden socket IP66",         "each",   18.00,  "electrical"),
    ("it_el_026", "Bathroom pull cord switch",  "each",    6.50,  "electrical"),
    ("it_el_027", "Cable clip pack 100",        "pack",    4.50,  "electrical"),
    ("it_el_028", "Back box metal 35mm",        "each",    1.80,  "electrical"),
    ("it_el_029", "Earth bond cable 10mm",      "metre",   2.20,  "electrical"),
    ("it_el_030", "Cooker switch 45A",          "each",   14.00,  "electrical"),
]

# ---------------------------------------------------------------------------
# Mapping of items to job types (5 typical items per job)
# ---------------------------------------------------------------------------

JOB_ITEMS_MAP: dict[str, list[str]] = {
    # Plumber
    "pl_01": ["it_pl_009", "it_pl_026", "it_pl_001", "it_pl_004", "it_pl_003"],
    "pl_02": ["it_pl_003", "it_pl_007", "it_pl_018", "it_pl_004", "it_pl_006"],
    "pl_03": ["it_pl_001", "it_pl_004", "it_pl_003", "it_pl_006", "it_pl_007"],
    "pl_04": ["it_pl_011", "it_pl_010", "it_pl_015", "it_pl_001", "it_pl_004"],
    "pl_05": ["it_pl_021", "it_pl_008", "it_pl_003", "it_pl_004", "it_pl_001"],
    "pl_06": ["it_pl_010", "it_pl_003", "it_pl_004", "it_pl_007", "it_pl_001"],
    "pl_07": ["it_pl_012", "it_pl_001", "it_pl_004", "it_pl_003", "it_pl_007"],
    "pl_08": ["it_pl_014", "it_pl_003", "it_pl_004", "it_pl_001", "it_pl_007"],
    "pl_09": ["it_pl_016", "it_pl_017", "it_pl_002", "it_pl_005", "it_pl_006"],
    "pl_10": ["it_pl_019", "it_pl_003", "it_pl_007", "it_pl_004", "it_pl_001"],
    "pl_11": ["it_pl_018", "it_pl_007", "it_pl_003", "it_pl_001", "it_pl_004"],
    "pl_12": ["it_pl_003", "it_pl_007", "it_pl_001", "it_pl_004", "it_pl_018"],
    "pl_13": ["it_pl_022", "it_pl_023", "it_pl_001", "it_pl_004", "it_pl_008"],
    "pl_14": ["it_pl_020", "it_pl_001", "it_pl_004", "it_pl_003", "it_pl_007"],
    "pl_15": ["it_pl_027", "it_pl_001", "it_pl_004", "it_pl_003", "it_pl_007"],
    "pl_16": ["it_pl_027", "it_pl_001", "it_pl_004", "it_pl_003", "it_pl_007"],
    "pl_17": ["it_pl_017", "it_pl_003", "it_pl_006", "it_pl_007", "it_pl_001"],
    "pl_18": ["it_pl_025", "it_pl_001", "it_pl_004", "it_pl_003", "it_pl_007"],
    "pl_19": ["it_pl_001", "it_pl_004", "it_pl_005", "it_pl_006", "it_pl_003"],
    "pl_20": ["it_pl_024", "it_pl_002", "it_pl_005", "it_pl_003", "it_pl_006"],

    # Carpenter
    "cp_01": ["it_cp_017", "it_cp_018", "it_cp_006", "it_cp_004", "it_cp_021"],
    "cp_02": ["it_cp_001", "it_cp_002", "it_cp_004", "it_cp_006", "it_cp_021"],
    "cp_03": ["it_cp_007", "it_cp_006", "it_cp_022", "it_cp_004", "it_cp_013"],
    "cp_04": ["it_cp_009", "it_cp_011", "it_cp_004", "it_cp_012", "it_cp_026"],
    "cp_05": ["it_cp_012", "it_cp_004", "it_cp_026", "it_cp_013", "it_cp_002"],
    "cp_06": ["it_cp_002", "it_cp_004", "it_cp_029", "it_cp_031", "it_cp_026"],
    "cp_07": ["it_cp_014", "it_cp_003", "it_cp_005", "it_cp_004", "it_cp_016"],
    "cp_08": ["it_cp_020", "it_cp_001", "it_cp_004", "it_cp_002", "it_cp_026"],
    "cp_09": ["it_cp_002", "it_cp_026", "it_cp_004", "it_cp_006", "it_cp_013"],
    "cp_10": ["it_cp_001", "it_cp_002", "it_cp_004", "it_cp_005", "it_cp_027"],
    "cp_11": ["it_cp_001", "it_cp_002", "it_cp_004", "it_cp_020", "it_cp_028"],
    "cp_12": ["it_cp_024", "it_cp_025", "it_cp_002", "it_cp_004", "it_cp_005"],
    "cp_13": ["it_cp_006", "it_cp_021", "it_cp_022", "it_cp_004", "it_cp_026"],
    "cp_14": ["it_cp_023", "it_cp_003", "it_cp_005", "it_cp_004", "it_cp_002"],
    "cp_15": ["it_cp_015", "it_cp_016", "it_cp_005", "it_cp_004", "it_cp_003"],
    "cp_16": ["it_cp_002", "it_cp_005", "it_cp_006", "it_cp_022", "it_cp_003"],
    "cp_17": ["it_cp_013", "it_cp_004", "it_cp_026", "it_cp_002", "it_cp_027"],
    "cp_18": ["it_cp_018", "it_cp_004", "it_cp_026", "it_cp_002", "it_cp_027"],
    "cp_19": ["it_cp_027", "it_cp_028", "it_cp_026", "it_cp_004", "it_cp_002"],
    "cp_20": ["it_cp_029", "it_cp_030", "it_cp_031", "it_cp_005", "it_cp_004"],

    # Electrician
    "el_01": ["it_el_001", "it_el_003", "it_el_011", "it_el_012", "it_el_004"],
    "el_02": ["it_el_004", "it_el_001", "it_el_028", "it_el_027", "it_el_009"],
    "el_03": ["it_el_015", "it_el_003", "it_el_006", "it_el_027", "it_el_029"],
    "el_04": ["it_el_012", "it_el_011", "it_el_009", "it_el_010", "it_el_029"],
    "el_05": ["it_el_018", "it_el_002", "it_el_010", "it_el_011", "it_el_029"],
    "el_06": ["it_el_013", "it_el_014", "it_el_003", "it_el_027", "it_el_028"],
    "el_07": ["it_el_017", "it_el_001", "it_el_027", "it_el_029", "it_el_016"],
    "el_08": ["it_el_023", "it_el_003", "it_el_027", "it_el_028", "it_el_006"],
    "el_09": ["it_el_011", "it_el_009", "it_el_029", "it_el_027", "it_el_028"],
    "el_10": ["it_el_019", "it_el_020", "it_el_021", "it_el_027", "it_el_028"],
    "el_11": ["it_el_022", "it_el_003", "it_el_027", "it_el_028", "it_el_006"],
    "el_12": ["it_el_003", "it_el_006", "it_el_027", "it_el_028", "it_el_029"],
    "el_13": ["it_el_002", "it_el_030", "it_el_010", "it_el_011", "it_el_027"],
    "el_14": ["it_el_025", "it_el_001", "it_el_011", "it_el_027", "it_el_029"],
    "el_15": ["it_el_016", "it_el_003", "it_el_006", "it_el_027", "it_el_029"],
    "el_16": ["it_el_017", "it_el_003", "it_el_027", "it_el_028", "it_el_006"],
    "el_17": ["it_el_024", "it_el_021", "it_el_003", "it_el_027", "it_el_028"],
    "el_18": ["it_el_026", "it_el_003", "it_el_027", "it_el_028", "it_el_029"],
    "el_19": ["it_el_011", "it_el_009", "it_el_010", "it_el_029", "it_el_027"],
    "el_20": ["it_el_011", "it_el_009", "it_el_029", "it_el_027", "it_el_028"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def eircode(county: str) -> str:
    prefix = random.choice(COUNTIES[county])
    routing = "".join(random.choices("ABCDEFGHKLMNPRSTVWXY", k=1)) \
              + "".join(random.choices("0123456789", k=1)) \
              + "".join(random.choices("ABCDEFGHKLMNPRSTVWXY", k=2))
    return f"{prefix} {routing}"


def irish_name() -> tuple[str, str]:
    return random.choice(IRISH_FIRST_NAMES), random.choice(IRISH_SURNAMES)


def irish_email(first: str, last: str) -> str:
    first_clean = re.sub(r"[^a-z]", "", first.lower())
    last_clean = re.sub(r"[^a-z]", "", last.lower())
    patterns = [
        f"{first_clean}.{last_clean}",
        f"{first_clean}{last_clean}",
        f"{first_clean[0]}{last_clean}",
        f"{first_clean}.{last_clean}{random.randint(1, 99)}",
    ]
    return f"{random.choice(patterns)}@{random.choice(EMAIL_DOMAINS)}"


def irish_phone() -> str:
    prefix = random.choice(["083", "085", "086", "087", "089"])
    return f"{prefix} {random.randint(1000000, 9999999)}"


def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=seconds)


def round_eur(amount: float) -> float:
    return round(amount + 0.005, 2)


def item_by_id(items_list: list, item_id: str) -> tuple:
    for it in items_list:
        if it[0] == item_id:
            return it
    raise KeyError(item_id)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_customers() -> list[dict]:
    """Create 40 past customers per trade, 120 total."""
    customers = []
    cust_id = 1
    for trade in TRADES:
        for _ in range(CUSTOMERS_PER_TRADE):
            first, last = irish_name()
            county = random.choice(list(COUNTIES.keys()))
            street_no = random.randint(1, 180)
            first_contact = random_date(TODAY - timedelta(days=1095), TODAY - timedelta(days=30))
            customers.append({
                "customer_id": f"cust_{cust_id:04d}",
                "first_name": first,
                "last_name": last,
                "email": irish_email(first, last),
                "phone": irish_phone(),
                "address_line_1": f"{street_no} {random.choice(STREETS)}",
                "address_line_2": "",
                "county": county,
                "eircode": eircode(county),
                "preferred_trade": trade,
                "first_contact_date": first_contact.strftime("%Y-%m-%d"),
            })
            cust_id += 1
    return customers


def generate_job_types() -> list[dict]:
    rows = []
    for trade, jobs in JOB_TYPES_DATA.items():
        for job_id, name, hours, labour, skill, desc in jobs:
            rows.append({
                "job_type_id": job_id,
                "trade": trade,
                "job_name": name,
                "typical_hours": hours,
                "base_labour_cost_eur": labour,
                "skill_level": skill,
                "description": desc,
            })
    return rows


def generate_items() -> list[dict]:
    return [
        {
            "item_id": it[0],
            "item_name": it[1],
            "unit": it[2],
            "unit_price_ex_vat": it[3],
            "category": it[4],
        }
        for it in ITEMS_DATA
    ]


def generate_job_items_mapping() -> list[dict]:
    rows = []
    for job_id, item_ids in JOB_ITEMS_MAP.items():
        for idx, item_id in enumerate(item_ids, start=1):
            rows.append({
                "job_type_id": job_id,
                "item_id": item_id,
                "typical_quantity": random.choice([1, 1, 2, 3, 5, 1]),
                "position": idx,
            })
    return rows


def generate_invoices(customers: list[dict], job_types: list[dict]) -> tuple[list[dict], list[dict]]:
    """Generate invoices and per-invoice line items."""
    invoices, invoice_items = [], []
    job_types_by_id = {j["job_type_id"]: j for j in job_types}
    items_by_id = {i[0]: i for i in ITEMS_DATA}

    for i in range(1, NUM_INVOICES + 1):
        # pick a customer then a job for their preferred trade
        cust = random.choice(customers)
        trade_jobs = [j for j in job_types if j["trade"] == cust["preferred_trade"]]
        job = random.choice(trade_jobs)

        invoice_date = random_date(TODAY - timedelta(days=730), TODAY - timedelta(days=1))
        invoice_id = f"INV-{invoice_date.year}-{i:04d}"

        # Labour
        labour_cost = float(job["base_labour_cost_eur"])
        # Random minor adjustment +/- 15%
        labour_cost = round_eur(labour_cost * random.uniform(0.88, 1.15))

        # Items
        item_ids = JOB_ITEMS_MAP.get(job["job_type_id"], [])
        line_items = []
        materials_total = 0.0
        for pos, item_id in enumerate(item_ids, start=1):
            it = items_by_id[item_id]
            qty = random.choice([1, 1, 2, 2, 3, 4])
            unit_price = it[3]
            line_total = round_eur(qty * unit_price)
            materials_total += line_total
            line_items.append({
                "invoice_id": invoice_id,
                "line_no": pos,
                "item_id": item_id,
                "item_name": it[1],
                "quantity": qty,
                "unit": it[2],
                "unit_price_ex_vat": unit_price,
                "line_total_ex_vat": line_total,
            })

        subtotal = round_eur(labour_cost + materials_total)
        vat = round_eur(subtotal * VAT_RATE)
        total = round_eur(subtotal + vat)

        invoices.append({
            "invoice_id": invoice_id,
            "customer_id": cust["customer_id"],
            "trade": cust["preferred_trade"],
            "job_type_id": job["job_type_id"],
            "job_name": job["job_name"],
            "invoice_date": invoice_date.strftime("%Y-%m-%d"),
            "labour_cost_ex_vat": labour_cost,
            "materials_cost_ex_vat": round_eur(materials_total),
            "subtotal_ex_vat": subtotal,
            "vat_23pct": vat,
            "total_inc_vat": total,
            "status": random.choice(["paid", "paid", "paid", "paid", "overdue"]),
        })
        invoice_items.extend(line_items)

    return invoices, invoice_items


# ---------------------------------------------------------------------------
# Email generation
# ---------------------------------------------------------------------------

EMAIL_TEMPLATES = {
    "formal_specific": [
        """Dear {trade_title},

I hope this email finds you well. I am writing to request a quotation for {job_phrase} at my property in {county}.

{detail_line}

Would it be possible to arrange a site visit at your earliest convenience? My availability is generally evenings after 6pm and Saturdays.

Kind regards,
{first} {last}
{phone}""",
        """Hello,

I would like to obtain a quote for the following work: {job_phrase}.

{detail_line}

Please let me know what additional information you need from me and when you could come out to have a look.

Best regards,
{first} {last}""",
    ],
    "casual_specific": [
        """Hi there,

Looking to get a quote for {job_phrase}. {detail_line} Based in {county}.

Let me know when you could pop out.

Cheers,
{first}""",
        """Hey,

Hoping you can give me a rough price on {job_phrase}? {detail_line}

Thanks,
{first} {last}""",
    ],
    "vague_symptom": [
        """Hi,

{symptom} Not sure what's involved but was hoping to get someone out to take a look.

I'm based in {county}. Let me know your availability and a rough idea of cost.

Thanks,
{first} {last}
{phone}""",
        """Hello,

{symptom} Can you come out and give me a quote?

Address is {address_line_1}, {county}.

{first}""",
    ],
    "urgent": [
        """URGENT

{urgent_issue}

Please call me back ASAP on {phone} or reply here. {county} area.

{first} {last}""",
        """Hi, emergency call out please.

{urgent_issue} Happened about an hour ago, need someone today if at all possible.

{first} {last} - {phone}""",
    ],
    "multi_trade": [
        """Hi,

We're doing a kitchen renovation and need quotes for a few bits. {multi_detail}

The property is in {county}. Could you let me know what you can cover and send a quote for your part?

Thanks,
{first} {last}""",
        """Hello,

Planning a bathroom refit and I'm trying to line up the trades. {multi_detail}

Can you come out and give me a quote for the {trade} side? Property in {county}.

{first} {last}
{phone}""",
    ],
    "returning_customer": [
        """Hi again,

{first} {last} here - you did {previous_job} for me {months_ago} months back.

{new_need}

Would you be able to come out and have a look?

Cheers,
{first}""",
    ],
}

SYMPTOMS = {
    "plumber": [
        "There's a damp patch appearing on the ceiling downstairs, I think something is leaking upstairs.",
        "The water pressure in the shower has gone really weak over the last week.",
        "My boiler keeps losing pressure and I have to top it up every few days.",
        "There's a strange gurgling sound coming from the kitchen sink when it drains.",
        "The toilet keeps running on and on after I flush it.",
        "Radiator in the living room is cold at the top and warm at the bottom.",
        "Hot water stopped working this morning, cold is fine.",
    ],
    "carpenter": [
        "One of the stairs is creaking really badly and I'm worried it's getting worse.",
        "The back door doesn't close properly anymore, it sticks at the top.",
        "Kitchen cabinet door has come off its hinges.",
        "There's a soft patch in the floor upstairs, feels spongy when you walk on it.",
        "Window frame is rotting and the window is hard to open.",
        "Skirting board has pulled away from the wall in a few places.",
    ],
    "electrician": [
        "The lights in the kitchen keep flickering.",
        "A socket in the living room isn't working - tripped a few times then stopped.",
        "Smoke alarm is beeping every minute or so, won't stop even with a new battery.",
        "Bathroom extractor fan is making a really loud noise.",
        "One of the sockets sparked when I plugged in the hoover.",
        "My fuse board is really old and I'm not sure it's safe anymore.",
    ],
}

URGENT_ISSUES = {
    "plumber": [
        "Burst pipe under the kitchen sink, water everywhere, I've turned off the mains.",
        "No hot water and no heating, boiler showing an error code F22.",
        "Toilet completely blocked and overflowing, can't use the bathroom.",
        "Massive leak coming from somewhere in the attic, water coming through the ceiling.",
    ],
    "carpenter": [
        "Someone tried to break in last night and damaged the back door, I need it secured today.",
        "Tree fell on the garden fence during the storm, panels are down and I have a dog.",
    ],
    "electrician": [
        "Whole house has no power, trip switch won't stay up when I reset it.",
        "Burning smell coming from a socket and it's hot to touch, I've turned off that circuit at the board.",
        "Power went off in half the house and won't come back on.",
    ],
}

JOB_DETAIL_LINES = {
    "plumber": [
        "It's a semi-detached house, roughly 10 years old.",
        "The existing installation is fairly old, probably 20+ years.",
        "We have a combi boiler already in place.",
        "Access should be straightforward, the area is tiled.",
        "We'd like it done before the end of the month if possible.",
    ],
    "carpenter": [
        "The property is a 3-bed semi, relatively modern construction.",
        "We have the materials picked out already but happy to take recommendations.",
        "It's a weekend when we're both off, but evenings work too.",
        "The area is about 20 square metres.",
    ],
    "electrician": [
        "The house is from the mid-90s, original fuse box is still in place.",
        "We've recently extended so there's some new wiring in one section.",
        "Would need certification for insurance purposes.",
        "Ideally during school hours when we're out of the house.",
    ],
}

MULTI_TRADE_DETAILS = [
    "We need the old units ripped out, new ones fitted, plumbing rerouted for the sink and dishwasher, and the lighting upgraded with under-cabinet LEDs.",
    "Existing bath to come out and be replaced with a walk-in shower, plus new tiling, new extractor fan, and the lights moved.",
    "Converting the attic - need flooring, some lighting and sockets added, and a radiator brought up there.",
]

TRADE_TITLES = {
    "plumber": ["plumber", "plumbing contractor"],
    "carpenter": ["carpenter", "carpentry contractor"],
    "electrician": ["electrician", "electrical contractor"],
}

# Map job names to natural phrasings a customer would use
JOB_PHRASES = {
    "Boiler installation":       "replacing our old boiler with a new one",
    "Boiler service":            "a boiler service",
    "Leaking pipe repair":       "repairing a leaking pipe behind the kitchen wall",
    "Bathroom suite installation": "a full bathroom refit",
    "Radiator replacement":      "replacing two old radiators",
    "Toilet installation":       "installing a new toilet",
    "Shower installation":       "installing a new shower",
    "Tap replacement":           "replacing the kitchen and bathroom taps",
    "Hot water cylinder install": "fitting a new hot water cylinder",
    "Drain unblocking":          "unblocking a kitchen drain",
    "Pipe insulation":           "insulating the pipes in the attic",
    "Gas safety certificate":    "a landlord gas safety certificate",
    "Underfloor heating install": "installing wet underfloor heating in the kitchen",
    "Waste disposal unit":       "fitting a waste disposal unit",
    "Washing machine plumbing":  "plumbing in a new washing machine",
    "Dishwasher plumbing":       "plumbing in a new dishwasher",
    "Immersion heater replace":  "replacing a broken immersion heater",
    "Outside tap installation":  "fitting an outside tap",
    "Burst pipe emergency":      "repairing a burst pipe",
    "Pressure booster install":  "installing a pressure booster pump",

    "Kitchen installation":      "fitting a new kitchen",
    "Fitted wardrobe building":  "building fitted wardrobes in two bedrooms",
    "Door hanging":              "hanging three internal doors",
    "Wooden floor laying":       "laying engineered oak flooring in the living room",
    "Skirting board fitting":    "fitting new skirting boards throughout",
    "Staircase repair":          "repairing the staircase",
    "Decking construction":      "building a garden deck",
    "Shelving installation":     "fitting built-in shelving in the alcove",
    "Window frame repair":       "repairing two rotten window frames",
    "Attic flooring":            "putting flooring down in the attic for storage",
    "Built-in bookcase":         "building a floor-to-ceiling bookcase",
    "Loft hatch installation":   "fitting a loft hatch and ladder",
    "Kitchen cabinet repair":    "repairing some kitchen cabinet doors",
    "Garden shed build":         "building a garden shed",
    "Fence panel replacement":   "replacing a few fence panels",
    "Garden gate installation":  "fitting a new garden gate",
    "Architrave fitting":        "fitting new architrave around the doors",
    "Worktop replacement":       "replacing the kitchen worktop",
    "Floor sanding":             "sanding and refinishing the hall floor",
    "Staircase balustrade":      "fitting a new balustrade on the stairs",

    "Full house rewiring":       "rewiring the full house",
    "Extra socket installation": "adding a couple of extra sockets",
    "Light fitting installation": "hanging a new pendant light",
    "Consumer unit upgrade":     "upgrading the fuse board",
    "EV charger installation":   "installing a home EV charger",
    "Smoke alarm installation":  "fitting mains-connected smoke alarms",
    "Outdoor lighting":          "fitting garden lighting",
    "Doorbell installation":     "fitting a new doorbell",
    "Electrical safety cert":    "an electrical safety certificate",
    "CCTV installation":         "installing a CCTV system",
    "Bathroom extractor fan":    "replacing the bathroom extractor fan",
    "Immersion heater wiring":   "wiring in a new immersion heater timer",
    "Cooker circuit installation": "wiring a new cooker circuit",
    "Garden socket installation": "fitting an outdoor socket",
    "Bathroom lights upgrade":   "upgrading the bathroom lights",
    "Security light install":    "fitting a security light at the back",
    "Intercom system":           "installing a video door intercom",
    "Shower pull cord install":  "fitting a shower isolation pull cord",
    "Fuse box test":             "testing the fuse box",
    "Landlord electrical report": "an electrical report for a rental property",
}

def generate_emails(customers: list[dict], job_types: list[dict], invoices: list[dict]) -> list[dict]:
    """Generate 100 varied quote-request emails."""
    emails = []
    job_types_by_id = {j["job_type_id"]: j for j in job_types}

    # ~60% from NEW prospects, ~40% from returning customers
    for i in range(1, NUM_EMAILS + 1):
        is_returning = random.random() < 0.40 and len(invoices) > 0
        trade = random.choice(TRADES)
        trade_jobs = [j for j in job_types if j["trade"] == trade]
        job = random.choice(trade_jobs)

        if is_returning:
            cust_invoice = random.choice([inv for inv in invoices if inv["trade"] == trade])
            cust = next(c for c in customers if c["customer_id"] == cust_invoice["customer_id"])
            months_ago = random.randint(3, 22)
            prev_job = cust_invoice["job_name"].lower()
            template_kind = "returning_customer"
        else:
            first, last = irish_name()
            county = random.choice(list(COUNTIES.keys()))
            cust = {
                "customer_id": None,   # new prospect
                "first_name": first,
                "last_name": last,
                "email": irish_email(first, last),
                "phone": irish_phone(),
                "address_line_1": f"{random.randint(1, 180)} {random.choice(STREETS)}",
                "county": county,
                "eircode": eircode(county),
                "preferred_trade": trade,
            }
            template_kind = random.choices(
                ["formal_specific", "casual_specific", "vague_symptom", "urgent", "multi_trade"],
                weights=[25, 30, 20, 10, 15],
            )[0]

        template = random.choice(EMAIL_TEMPLATES[template_kind])

        # Fill template variables
        fill = {
            "first":            cust["first_name"],
            "last":             cust["last_name"],
            "phone":            cust["phone"],
            "address_line_1":   cust["address_line_1"],
            "county":           cust["county"],
            "trade":            trade,
            "trade_title":      random.choice(TRADE_TITLES[trade]),
            "job_phrase":       JOB_PHRASES.get(job["job_name"], job["job_name"].lower()),
            "detail_line":      random.choice(JOB_DETAIL_LINES[trade]),
            "symptom":          random.choice(SYMPTOMS[trade]),
            "urgent_issue":     random.choice(URGENT_ISSUES[trade]),
            "multi_detail":     random.choice(MULTI_TRADE_DETAILS),
            "previous_job":     prev_job if is_returning else "",
            "months_ago":       months_ago if is_returning else "",
            "new_need":         f"Now we're hoping to get a quote for {JOB_PHRASES.get(job['job_name'], job['job_name'].lower())}." if is_returning else "",
        }
        body = template.format(**fill)

        subject_options = {
            "formal_specific":    f"Quote request: {job['job_name']}",
            "casual_specific":    f"Looking for a quote - {job['job_name'].lower()}",
            "vague_symptom":      "Issue at home, need someone to take a look",
            "urgent":             f"URGENT - {trade} needed",
            "multi_trade":        f"Multiple trades needed - {trade}",
            "returning_customer": f"Follow up - {job['job_name'].lower()}",
        }
        subject = subject_options[template_kind]

        sent_at = random_date(TODAY - timedelta(days=45), TODAY)
        trade_contact = f"info@{trade}s-ireland.ie"

        emails.append({
            "email_id": f"msg_{i:04d}",
            "from_name": f"{cust['first_name']} {cust['last_name']}",
            "from_email": cust["email"],
            "to_email": trade_contact,
            "trade_contacted": trade,
            "subject": subject,
            "sent_at": sent_at.strftime("%Y-%m-%d %H:%M:%S"),
            "template_kind": template_kind,
            "is_returning_customer": is_returning,
            "customer_id_if_known": cust.get("customer_id") or "",
            "target_job_type_id": job["job_type_id"],
            "target_job_name": job["job_name"],
            "body": body.strip(),
        })
    return emails


def write_eml_files(emails: list[dict]) -> None:
    """Write each email as a MIME .eml file in data/emails/."""
    EML_DIR.mkdir(parents=True, exist_ok=True)
    for e in emails:
        msg = EmailMessage()
        msg["From"] = f"{e['from_name']} <{e['from_email']}>"
        msg["To"] = e["to_email"]
        msg["Subject"] = e["subject"]
        sent_dt = datetime.strptime(e["sent_at"], "%Y-%m-%d %H:%M:%S")
        msg["Date"] = format_datetime(sent_dt)
        msg["Message-ID"] = make_msgid(domain="quotes-ireland.ie")
        msg.set_content(e["body"])

        out_path = EML_DIR / f"{e['email_id']}.eml"
        with out_path.open("wb") as f:
            f.write(bytes(msg))


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic dataset ...")
    customers = generate_customers()
    job_types = generate_job_types()
    items = generate_items()
    job_items = generate_job_items_mapping()
    invoices, invoice_items = generate_invoices(customers, job_types)
    emails = generate_emails(customers, job_types, invoices)

    write_csv(OUT_DIR / "customers.csv",       customers)
    write_csv(OUT_DIR / "job_types.csv",       job_types)
    write_csv(OUT_DIR / "items.csv",           items)
    write_csv(OUT_DIR / "job_items.csv",       job_items)
    write_csv(OUT_DIR / "invoices.csv",        invoices)
    write_csv(OUT_DIR / "invoice_items.csv",   invoice_items)
    write_csv(OUT_DIR / "emails.csv",          emails)

    write_eml_files(emails)

    print("\nSummary")
    print("-------")
    print(f"  customers.csv      {len(customers):>5} rows")
    print(f"  job_types.csv      {len(job_types):>5} rows")
    print(f"  items.csv          {len(items):>5} rows")
    print(f"  job_items.csv      {len(job_items):>5} rows")
    print(f"  invoices.csv       {len(invoices):>5} rows")
    print(f"  invoice_items.csv  {len(invoice_items):>5} rows")
    print(f"  emails.csv         {len(emails):>5} rows")
    print(f"  emails/*.eml       {len(emails):>5} files")

    # Quality check: ensure referential integrity
    customer_ids = {c["customer_id"] for c in customers}
    job_ids = {j["job_type_id"] for j in job_types}
    item_ids = {i["item_id"] for i in items}

    bad = 0
    for inv in invoices:
        if inv["customer_id"] not in customer_ids: bad += 1
        if inv["job_type_id"] not in job_ids: bad += 1
    for li in invoice_items:
        if li["item_id"] not in item_ids: bad += 1

    print(f"\nReferential integrity: {'OK' if bad == 0 else f'{bad} broken references'}")
    print(f"\nAll files written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
