"""
News Perception Study
=====================

Streamlit app for the two-phase news-feed experiment.

    Phase 1 (feed 1): every participant sees the source colour and the
                      original text.
    Phase 2 (feed 2): the between-subject treatment (2 x 2):
                      source visible / hidden  x  original / LLM-rewritten.

Each phase is ONE scrolling feed with all its articles (topics x outlets,
e.g. 4 x 4 = 16), in an order randomised independently for each phase and
each participant. So that the feed never seems to end, the same sequence is
repeated REPEAT_ROUNDS times (default 10) one after the other.

The state of an article is shared by all its copies: if the participant
opens it ("Read more" / "Read full article") or likes it on one copy, every
other copy of that article shows it opened / liked too.

Interaction per article:
    preview -> "... Read more" -> "... Read full article" (whole text),
    plus an optional Like toggle.

The "Continue" / "Finish" button sits in a bar fixed at the bottom of the
screen (always reachable, however far the participant scrolls). It unlocks
only after MIN_FEED_SECONDS (default 2 minutes), with a countdown on it.
Between the two feeds a short break screen shows a brief "loading the second
part" wait (BREAK_LOADING_SECONDS) before the participant can go on.

Every card shows the headline from the `Title` column (the same headline in
both text versions). The LLM-rewritten text comes from the `LLM_text` column.

When the source is visible, the whole card (background + border) is coloured
with the outlet's colour.

Run locally:     streamlit run app.py
Documentation:   app_documentation.txt
"""

import hashlib
import html as html_lib
import json
import os
import random
import re
import string
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="News Perception Study",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

APP_VERSION = "2026-10-07.3"

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTICLES_CSV = os.path.join(DATA_DIR, "articoli_selezionati - articoli scelti.csv")
LOCAL_BACKUP_DIR = os.path.join(DATA_DIR, "responses")

# CSV column names. For each field the app uses the first name it finds.
# Edit these lists if your CSV uses different names.
COLUMN_CANDIDATES = {
    "id": ["id", "article_id"],
    "topic": ["topic"],
    "outlet": ["domain", "outlet", "source"],
    "title": ["Title", "title", "headline"],
    "text": ["clean_text", "text"],
    "text_rewritten": ["LLM_text", "llm_text", "rewritten_text", "text_rewritten",
                       "clean_text_rewritten", "ai_text", "rewritten"],
    "reliability": ["reliability_class", "reliability_label"],
    "rating": ["rating"],
    "year": ["year"],
    "month": ["month"],
}
REQUIRED_FIELDS = ("topic", "outlet", "title", "text", "text_rewritten")

ARTICLES_PER_CELL = 2      # articles per outlet per topic (1 per phase)

# Text visible at each stage, in characters (spaces included). The cut is
# moved back to the nearest word boundary, so no word is ever split.
PREVIEW_CHARS = 144         # visible before "Read more"
READMORE_CHARS = 500       # visible after "Read more" (total)

# How many times the article sequence of a feed is repeated, one after the
# other, so that the feed never seems to end (16 articles x 10 = 160 cards).
REPEAT_ROUNDS = 10

# Minimum time on each feed before "Continue" / "Finish" can be clicked.
# Set to 0 to disable (e.g. while testing).
MIN_FEED_SECONDS = 120

# Break screen between the two feeds: how long the "loading the second part"
# wait lasts before the participant can start feed 2.
BREAK_LOADING_SECONDS = 3

# Card tint strength when the source is visible: 0 = white, 1 = full colour.
CARD_TINT = 0.16

# Okabe-Ito colours, chosen to stay distinguishable with colour-vision
# deficiencies. Shuffled and assigned to the outlets for each participant.
OUTLET_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

CONDITIONS = {
    1: {"label": "source visible / original text", "source_visible": True, "rewritten": False},
    2: {"label": "source hidden / original text", "source_visible": False, "rewritten": False},
    3: {"label": "source visible / rewritten text", "source_visible": True, "rewritten": True},
    4: {"label": "source hidden / rewritten text", "source_visible": False, "rewritten": True},
}

# Balanced assignment: a participant who started less than this many minutes
# ago and has not finished yet still holds a place in their condition.
PENDING_WINDOW_MIN = 60

WARN_ON_LEAVE = True       # browser asks for confirmation before leaving mid-study
DEV_MODE = False           # True: ?condition=1..4 in the URL forces a condition

ESTIMATED_MINUTES = "15–20"
PROLIFIC_RETURN_URL = "https://app.prolific.com/submissions/complete?cc={code}"

# New tab names (v3): the columns changed with the repeated feed, so the
# data go to fresh tabs instead of being mixed with the v2 data.
SHEET_ASSIGNMENTS = "assignments_v3"
SHEET_SESSIONS = "sessions_v3"
SHEET_RESPONSES = "responses_v3"

ASSIGNMENT_HEADERS = ["session_id", "prolific_pid", "condition", "assigned_at_utc"]

SESSION_HEADERS = [
    "session_id", "prolific_pid", "study_id", "prolific_session_id",
    "condition", "condition_label", "assignment_method",
    "color_map", "phase1_order", "phase2_order", "article_split",
    "opened_at_utc", "started_at_utc", "completed_at_utc",
    "duration_min", "n_articles", "n_readmore", "n_full", "n_likes",
    "phase1_max_round", "phase2_max_round",
    "user_agent", "app_version",
]

RESPONSE_HEADERS = [
    "session_id", "prolific_pid", "condition", "condition_label",
    "phase", "page_number", "topic", "position",
    "article_uid", "outlet", "reliability_class", "outlet_color",
    "source_visible", "text_version", "title_shown", "n_sentences", "n_chars",
    "readmore_available", "readmore_clicked", "readmore_ms", "readmore_ts", "readmore_round",
    "full_available", "full_clicked", "full_ms", "full_ts", "full_round",
    "liked_final", "like_toggles", "like_history",
    "rounds_shown", "max_round_reached",
    "page_enter_ts", "page_exit_ts", "page_duration_ms", "page_duration_server_s",
    "year", "month", "rating",
]

# =============================================================================
# 2. SMALL HELPERS
# =============================================================================

_rng = random.SystemRandom()


def utc_now():
    return datetime.now(timezone.utc)


def utc_iso(dt=None):
    return (dt or utc_now()).isoformat(timespec="seconds")


def make_session_id():
    suffix = "".join(_rng.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    return f"{utc_now().strftime('%Y%m%d_%H%M%S')}_{suffix}"


def to_py(value):
    """numpy/pandas scalars -> plain Python, NaN -> None."""
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and value != value:
        return None
    return value


def cell(value):
    """Value ready for a Google Sheets / CSV cell."""
    value = to_py(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def get_secret(section, key, default=None):
    try:
        return st.secrets[section][key]
    except Exception:
        return default


def number_word(n):
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
             "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
    return words[n] if 0 <= n < len(words) else str(n)


def duration_words(seconds):
    """120 -> 'two minutes', 60 -> 'one minute', 90 -> '90 seconds'."""
    if seconds % 60 == 0:
        m = seconds // 60
        return f"{number_word(m)} minute{'s' if m != 1 else ''}"
    return f"{seconds} seconds"


def tint_hex(hex_color, alpha):
    """Opaque mix of `hex_color` with white (same formula as in the feed)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: round(255 - (255 - c) * alpha)
    return f"rgb({mix(r)}, {mix(g)}, {mix(b)})"


def log(message):
    print(f"[news-study {utc_iso()}] {message}", flush=True)


# =============================================================================
# 3. SENTENCE SPLITTING
# =============================================================================

_ABBREVIATIONS = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "Mt.", "Gov.",
    "Sen.", "Rep.", "Gen.", "Lt.", "Col.", "Capt.", "Sgt.", "Rev.", "Hon.",
    "U.S.", "U.K.", "U.N.", "E.U.", "D.C.", "a.m.", "p.m.", "e.g.", "i.e.",
    "vs.", "etc.", "approx.", "Inc.", "Ltd.", "Co.", "Corp.", "No.", "Jan.",
    "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.",
    "Nov.", "Dec.",
]
_ABBR_RE = re.compile(
    r"(?<![\w.])(" + "|".join(re.escape(a) for a in sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")"
)
_INITIAL_RE = re.compile(r"\b([A-Z])\.(?=\s+[A-Z])")
_SPLIT_RE = re.compile(r'''([.!?…]+["')\]]*)\s+(?=["'(\[]?[A-Z0-9À-Ý])''')
_DOT = "․"   # placeholder for protected full stops
_SEP = "⁣"   # placeholder for sentence boundaries


def split_sentences(text):
    """Return [{"t": sentence, "p": paragraph_index}, ...].

    Any line break in the source text starts a new paragraph."""
    sentences = []
    paragraphs = [p for p in re.split(r"\s*\n\s*", str(text).strip()) if p.strip()]
    for p_idx, para in enumerate(paragraphs):
        para = re.sub(r"\s+", " ", para).strip()
        protected = _ABBR_RE.sub(lambda m: m.group(0).replace(".", _DOT), para)
        protected = _INITIAL_RE.sub(lambda m: m.group(1) + _DOT, protected)
        marked = _SPLIT_RE.sub(lambda m: m.group(1) + _SEP, protected)
        for chunk in marked.split(_SEP):
            chunk = chunk.replace(_DOT, ".").strip()
            if chunk:
                sentences.append({"t": chunk, "p": p_idx})
    return sentences


def n_chars(sentences):
    """Length of the text as displayed (sentences joined by one space or one
    paragraph break). Matches the character count used by the feed."""
    if not sentences:
        return 0
    return sum(len(s["t"]) for s in sentences) + len(sentences) - 1


# =============================================================================
# 4. ARTICLES: LOADING AND VALIDATION
# =============================================================================

def _pick_column(df, field):
    for name in COLUMN_CANDIDATES[field]:
        if name in df.columns:
            return name
    return None


def _is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _clean(value):
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return to_py(value)


@st.cache_data(show_spinner=False)
def load_study_data(csv_path, mtime):
    """Load and validate the article set. Returns (data, errors).
    `mtime` is part of the cache key, so editing the CSV reloads it."""
    if not os.path.exists(csv_path):
        return None, [f"The articles file was not found: {csv_path}"]

    df = pd.read_csv(csv_path).reset_index(drop=True)
    cols = {field: _pick_column(df, field) for field in COLUMN_CANDIDATES}

    errors = [
        f"No column for '{f}'. Expected one of: {', '.join(COLUMN_CANDIDATES[f])}."
        for f in REQUIRED_FIELDS if cols[f] is None
    ]
    if errors:
        return None, errors

    warnings = []
    articles = {}
    for i, row in df.iterrows():
        def val(field):
            return _clean(row[cols[field]]) if cols[field] else None

        uid = str(val("id")) if cols["id"] and val("id") is not None else f"a{i:02d}"
        text = str(val("text") or "").strip()
        text_rw = str(val("text_rewritten") or "").strip()
        title = str(val("title") or "").strip()

        if uid in articles:
            errors.append(f"Duplicate article id: {uid}.")
        if not title:
            errors.append(f"Article {uid} has no title ('{cols['title']}' column).")
        if not text:
            errors.append(f"Article {uid} has no original text.")
        if not text_rw:
            errors.append(f"Article {uid} has no rewritten text ('{cols['text_rewritten']}' column).")

        articles[uid] = {
            "uid": uid,
            "row": int(i),
            "topic": str(val("topic")).strip(),
            "outlet": str(val("outlet")).strip(),
            "title": title,
            "sentences_original": split_sentences(text),
            "sentences_rewritten": split_sentences(text_rw),
            "reliability": str(val("reliability")).strip() if val("reliability") is not None else None,
            "rating": val("rating"),
            "year": val("year"),
            "month": val("month"),
        }

    topics = sorted({a["topic"] for a in articles.values()})
    outlets = sorted({a["outlet"] for a in articles.values()})

    if len(outlets) > len(OUTLET_PALETTE):
        errors.append(f"Found {len(outlets)} outlets but the palette has only "
                      f"{len(OUTLET_PALETTE)} colours.")

    cells = {t: {o: [] for o in outlets} for t in topics}
    for a in articles.values():
        cells[a["topic"]][a["outlet"]].append(a["uid"])
    for t in topics:
        for o in outlets:
            n = len(cells[t][o])
            if n != ARTICLES_PER_CELL:
                errors.append(f"Topic '{t}' / outlet '{o}': found {n} articles, "
                              f"expected {ARTICLES_PER_CELL}.")

    # Reliability class, in order of preference:
    #   1. its own column (reliability_class / reliability_label);
    #   2. the 'rating' column when it holds labels (e.g. "Reliable"/"Unreliable");
    #   3. the 'rating' column when it holds numbers (higher outlet mean = reliable).
    if cols["reliability"] is None:
        rating_values = [a["rating"] for a in articles.values() if a["rating"] is not None]
        numeric = bool(rating_values) and all(_is_number(v) for v in rating_values)
        if rating_values and not numeric:
            for a in articles.values():
                a["reliability"] = str(a["rating"]).strip().lower() if a["rating"] is not None else "NA"
            warnings.append("Reliability class taken from the labels in the 'rating' column.")
        elif numeric:
            means = {}
            for o in outlets:
                ratings = [float(a["rating"]) for a in articles.values()
                           if a["outlet"] == o and a["rating"] is not None]
                means[o] = sum(ratings) / len(ratings) if ratings else float("-inf")
            ranked = sorted(outlets, key=lambda o: means[o], reverse=True)
            reliable = set(ranked[: len(ranked) // 2])
            for a in articles.values():
                a["reliability"] = "reliable" if a["outlet"] in reliable else "unreliable"
            warnings.append("Reliability class derived from the numeric 'rating' column.")
        else:
            for a in articles.values():
                a["reliability"] = "NA"
            warnings.append("No reliability information found.")

    for o in outlets:
        classes = {a["reliability"] for a in articles.values() if a["outlet"] == o}
        if len(classes) > 1:
            warnings.append(f"Outlet '{o}' has more than one reliability class: {sorted(classes)}.")

    for w in warnings:
        log(f"WARNING: {w}")

    data = {"articles": articles, "topics": topics, "outlets": outlets,
            "cells": cells, "columns": cols}
    return (None, errors) if errors else (data, [])


# =============================================================================
# 5. STORAGE (Google Sheets, with a local CSV backup)
# =============================================================================

class SheetStore:
    def __init__(self, spreadsheet):
        self.sh = spreadsheet
        self.lock = threading.Lock()
        self._worksheets = {}

    def _worksheet(self, name, headers):
        import gspread
        if name not in self._worksheets:
            try:
                ws = self.sh.worksheet(name)
            except gspread.WorksheetNotFound:
                ws = self.sh.add_worksheet(title=name, rows=1000, cols=len(headers) + 2)
            if not ws.row_values(1):
                ws.append_row(headers, value_input_option="RAW")
            self._worksheets[name] = ws
        return self._worksheets[name]

    def read(self, name, headers):
        with self.lock:
            values = self._worksheet(name, headers).get_all_values()
        if len(values) < 2:
            return []
        head = values[0]
        return [dict(zip(head, row)) for row in values[1:]]

    def append(self, name, headers, rows):
        if not rows:
            return
        with self.lock:
            self._worksheet(name, headers).append_rows(rows, value_input_option="RAW")


@st.cache_resource(show_spinner=False)
def _connect_store():
    key = get_secret("sheets", "spreadsheet_key")
    try:
        creds = dict(st.secrets["gcp_service_account"])
    except Exception:
        creds = None
    if not key or not creds:
        raise RuntimeError("Google Sheets is not configured (see secrets.toml).")
    import gspread
    client = gspread.service_account_from_dict(creds)
    return SheetStore(client.open_by_key(key))


def get_store():
    try:
        return _connect_store()
    except Exception as exc:
        log(f"Storage unavailable: {exc}")
        return None


def _parse_time(value):
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _as_int(value):
    try:
        return int(value)
    except Exception:
        return None


def assign_condition(session_id, prolific_pid):
    """Balanced assignment counting completed participants plus recent,
    still-running ones. Returns (condition, method)."""
    if DEV_MODE:
        forced = st.query_params.get("condition")
        if forced in {str(c) for c in CONDITIONS}:
            return int(forced), "forced_dev"

    store = get_store()
    if store is None:
        return _rng.choice(list(CONDITIONS)), "random_no_storage"

    try:
        completed = store.read(SHEET_SESSIONS, SESSION_HEADERS)
        reserved = store.read(SHEET_ASSIGNMENTS, ASSIGNMENT_HEADERS)
        counts = {c: 0 for c in CONDITIONS}
        finished_ids = set()
        for r in completed:
            c = _as_int(r.get("condition"))
            if c in counts:
                counts[c] += 1
                finished_ids.add(r.get("session_id"))
        cutoff = utc_now() - timedelta(minutes=PENDING_WINDOW_MIN)
        for r in reserved:
            if r.get("session_id") in finished_ids:
                continue
            c = _as_int(r.get("condition"))
            t = _parse_time(r.get("assigned_at_utc"))
            if c in counts and t is not None and t >= cutoff:
                counts[c] += 1
        lowest = min(counts.values())
        condition = _rng.choice([c for c, n in counts.items() if n == lowest])
        store.append(SHEET_ASSIGNMENTS, ASSIGNMENT_HEADERS,
                     [[session_id, prolific_pid, condition, utc_iso()]])
        return condition, "balanced"
    except Exception as exc:
        log(f"Balanced assignment failed, using random: {exc}")
        return _rng.choice(list(CONDITIONS)), "random_storage_error"


# =============================================================================
# 6. FEED COMPONENT (HTML/JS rendered inside Streamlit)
# =============================================================================

THUMB_SVG = (
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
    '<path d="M7 10v12"/>'
    '<path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 '
    '1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z"/></svg>'
)

FEED_COMPONENT_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,500&family=Schibsted+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --canvas: #ECEEF1;
    --surface: #FFFFFF;
    --ink: #191C20;
    --ink-2: #2E3238;
    --ink-3: #5C626A;
    --line: #D8DCE1;
    --line-strong: #B9BEC5;
    --sans: "Schibsted Grotesk", system-ui, -apple-system, "Segoe UI", sans-serif;
    --serif: "Literata", Georgia, "Times New Roman", serif;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: transparent; color: var(--ink); }
  body { font-family: var(--sans); -webkit-font-smoothing: antialiased; }
  #root { padding: 2px 2px 10px; }

  .progress { display: flex; align-items: center; gap: 14px; margin: 0 0 20px; }
  .progress-label { font-size: 14px; font-weight: 500; color: var(--ink-3);
                    white-space: nowrap; font-variant-numeric: tabular-nums; }
  .track { flex: 1; height: 4px; border-radius: 2px; background: var(--line); overflow: hidden; }
  .fill { height: 100%; border-radius: 2px; background: var(--ink);
          transition: width .6s cubic-bezier(.2, .7, .2, 1); }

  .feed { display: flex; flex-direction: column; gap: 14px; }
  .card { position: relative; background: var(--surface); border: 1px solid var(--line);
          border-radius: 12px; padding: 22px 26px 18px; }
  /* Source visible: the card is tinted with the outlet colour (set inline). */
  .card.sourced { border-width: 2px; padding: 21px 25px 17px; padding-top: 46px; }
  .ribbon { position: absolute; top: -1px; left: 25px; width: 18px; height: 28px;
            clip-path: polygon(0 0, 100% 0, 100% 100%, 50% 74%, 0 100%); }

  .headline { font-family: var(--sans); font-weight: 600; font-size: 21px; line-height: 1.28;
              letter-spacing: -0.012em; margin: 0 0 10px; color: var(--ink); text-wrap: pretty; }
  .body { font-family: var(--serif); font-optical-sizing: auto; font-size: 17px;
          line-height: 1.66; color: var(--ink-2); }
  .body p { margin: 0 0 .75em; }
  .body p:last-child { margin-bottom: 0; }
  .fresh { animation: fresh .6s ease-out both; }
  @keyframes fresh { from { opacity: 0; } to { opacity: 1; } }

  .more { appearance: none; border: 0; background: none; padding: 0; margin: 0;
          font: 600 15px/1 var(--sans); color: var(--ink); cursor: pointer; white-space: nowrap;
          text-decoration: underline; text-decoration-thickness: 1.5px;
          text-decoration-color: var(--line-strong); text-underline-offset: 4px; border-radius: 3px; }
  .more:hover { text-decoration-color: var(--ink); }

  .foot { margin-top: 16px; display: flex; }
  .like { appearance: none; display: inline-flex; align-items: center; gap: 7px; height: 36px;
          padding: 0 15px 0 12px; border-radius: 18px; border: 1px solid var(--line);
          background: var(--surface); color: var(--ink-3); font: 500 14px/1 var(--sans);
          cursor: pointer; transition: background-color .15s, border-color .15s, color .15s; }
  .like:hover { border-color: var(--line-strong); color: var(--ink); }
  .like svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 1.8;
              stroke-linecap: round; stroke-linejoin: round; }
  .like.on { background: var(--ink); border-color: var(--ink); color: #FFFFFF; }
  .like.on svg path:last-child { fill: currentColor; }
  .like.pop svg { animation: pop .28s ease-out; }
  @keyframes pop { 0% { transform: scale(1); } 40% { transform: scale(.78); } 100% { transform: scale(1); } }

  /* Fallback only: used when the fixed bar cannot be placed in the page. */
  .actions { display: flex; justify-content: flex-end; margin: 0 0 20px; }
  .next { appearance: none; border: 0; height: 48px; padding: 0 30px; border-radius: 24px;
          background: var(--ink); color: #FFFFFF; font: 600 16px/1 var(--sans); cursor: pointer;
          font-variant-numeric: tabular-nums; transition: opacity .15s; }
  .next:disabled { opacity: .55; cursor: default; }

  button:focus-visible { outline: 2px solid var(--ink); outline-offset: 3px; }

  @media (max-width: 560px) {
    .card { padding: 18px 18px 16px; }
    .card.sourced { padding: 17px 17px 15px; padding-top: 42px; }
    .ribbon { left: 17px; }
    .headline { font-size: 19px; }
    .body { font-size: 16.5px; }
    .next { width: 100%; }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
  }
</style>
</head>
<body>
<div id="root"></div>
<script>
(function () {
  "use strict";

  var THUMB = '__THUMB_SVG__';
  var BAR_ID = "nps-feedbar";
  var root = document.getElementById("root");
  var state = null;
  var currentPage = null;
  var lastHeight = -1;
  var countdown = null;
  var viewObserver = null;

  // ---- Streamlit component protocol -------------------------------------
  function send(type, data) {
    var msg = { isStreamlitMessage: true, type: type };
    if (data) { for (var k in data) { msg[k] = data[k]; } }
    window.parent.postMessage(msg, "*");
  }
  function updateHeight() {
    var h = Math.ceil(root.getBoundingClientRect().height) + 4;
    if (h !== lastHeight) { lastHeight = h; send("streamlit:setFrameHeight", { height: h }); }
  }
  if (window.ResizeObserver) { new ResizeObserver(updateHeight).observe(root); }
  window.addEventListener("resize", updateHeight);

  // ---- helpers -----------------------------------------------------------
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }
  function ms() { return Math.round(performance.now() - state.t0); }
  function iso() { return new Date().toISOString(); }
  function clock(s) { var m = Math.floor(s / 60), r = s % 60; return m + ":" + (r < 10 ? "0" : "") + r; }

  // Opaque mix of a hex colour with white (alpha 0 = white, 1 = full colour).
  function tint(hex, alpha) {
    var h = String(hex).replace("#", "");
    if (h.length === 3) h = h.charAt(0) + h.charAt(0) + h.charAt(1) + h.charAt(1) + h.charAt(2) + h.charAt(2);
    var n = parseInt(h, 16);
    var rgb = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map(function (c) {
      return Math.round(255 - (255 - c) * alpha);
    });
    return "rgb(" + rgb.join(", ") + ")";
  }

  // ---- parent page (same origin as the Streamlit app) --------------------
  function parentDoc() {
    try { var d = window.parent.document; return (d && d.body) ? d : null; }
    catch (err) { return null; }
  }

  // The element that scrolls the Streamlit page (the closest scrollable
  // ancestor of this iframe, or the document itself).
  function parentScroller() {
    var d = parentDoc();
    if (!d) return null;
    try {
      var n = window.frameElement ? window.frameElement.parentElement : null;
      while (n && n !== d.body && n !== d.documentElement) {
        var oy = window.parent.getComputedStyle(n).overflowY;
        if ((oy === "auto" || oy === "scroll") && n.scrollHeight > n.clientHeight) return n;
        n = n.parentElement;
      }
    } catch (err) { /* ignore */ }
    return d.scrollingElement || d.documentElement;
  }

  function scrollParentToTop() {
    try {
      var d = window.parent.document;
      var sel = ['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]', 'section.main', '.main'];
      for (var i = 0; i < sel.length; i++) { var n = d.querySelector(sel[i]); if (n) n.scrollTop = 0; }
      if (d.scrollingElement) d.scrollingElement.scrollTop = 0;
      window.parent.scrollTo(0, 0);
    } catch (err) { /* cross-origin: ignore */ }
  }
  function installLeaveGuard() {
    try {
      var w = window.parent;
      if (w.__npsLeaveGuard) return;
      w.__npsLeaveGuard = new w.Function("e", "e.preventDefault(); e.returnValue = ''; return '';");
      w.addEventListener("beforeunload", w.__npsLeaveGuard);
    } catch (err) { /* ignore */ }
  }
  function removeLeaveGuard() {
    try {
      var w = window.parent;
      if (w.__npsLeaveGuard) { w.removeEventListener("beforeunload", w.__npsLeaveGuard); w.__npsLeaveGuard = null; }
    } catch (err) { /* ignore */ }
  }

  // Continue / Finish button in a bar fixed at the bottom of the screen
  // (placed in the Streamlit page, styled by the app CSS: #nps-feedbar).
  function removeBar() {
    var d = parentDoc();
    if (!d) return;
    var old = d.getElementById(BAR_ID);
    if (old && old.parentNode) old.parentNode.removeChild(old);
  }
  function mountBar() {
    var d = parentDoc();
    if (!d) return null;
    try {
      removeBar();
      var bar = d.createElement("div");
      bar.id = BAR_ID;
      var inner = d.createElement("div");
      inner.className = "nps-feedbar-inner";
      var btn = d.createElement("button");
      btn.type = "button";
      btn.className = "nps-feedbar-btn";
      inner.appendChild(btn);
      bar.appendChild(inner);
      d.body.appendChild(bar);
      return btn;
    } catch (err) { return null; }
  }
  window.addEventListener("pagehide", removeBar);

  // ---- article text ------------------------------------------------------
  // The text is flattened to one string: sentences joined by a space,
  // paragraphs by "\n". Limits are counted in characters on this string.
  function flatten(sentences) {
    var out = "", para = null;
    for (var i = 0; i < sentences.length; i++) {
      var s = sentences[i];
      if (i > 0) out += (s.p !== para) ? "\n" : " ";
      out += s.t;
      para = s.p;
    }
    return out;
  }

  // Cut position for a limit of `limit` characters: moved back to the last
  // word boundary, then trailing spaces and weak punctuation are dropped.
  function cutAt(flat, limit) {
    if (flat.length <= limit) return flat.length;
    var i = limit;
    while (i > 0 && !/\s/.test(flat.charAt(i))) i--;
    if (i === 0) i = limit;
    while (i > 0 && /[\s,;:–—\-]/.test(flat.charAt(i - 1))) i--;
    return i;
  }

  // The reading stage belongs to the article, so all its copies share it.
  function visibleChars(a) {
    if (a.stage === 0) return cutAt(a.flat, state.previewChars);
    if (a.stage === 1) return cutAt(a.flat, state.readmoreChars);
    return a.flat.length;
  }

  // Renders the visible text of one copy, grouped in paragraphs, with the
  // inline "... Read more" / "... Read full article" link after it. Text past
  // `prevVisible` fades in.
  function renderBody(a, inst, prevVisible) {
    var body = inst.bodyEl;
    var vis = visibleChars(a);
    body.textContent = "";
    var paras = a.flat.split("\n");
    var offset = 0, p = null;
    for (var k = 0; k < paras.length; k++) {
      var start = offset, end = offset + paras[k].length;
      offset = end + 1;
      if (start >= vis) break;
      p = el("p");
      body.appendChild(p);
      var stop = Math.min(end, vis);
      var split = Math.max(start, Math.min(stop, prevVisible));
      if (split > start) p.appendChild(document.createTextNode(a.flat.slice(start, split)));
      if (stop > split) p.appendChild(el("span", "fresh", a.flat.slice(split, stop)));
    }
    var label = null, nextStage = null;
    if (a.stage === 0 && a.readmoreAvailable) { label = "Read more"; nextStage = 1; }
    else if (a.stage === 1 && a.fullAvailable) { label = "Read full article"; nextStage = 2; }
    if (!label) return null;
    if (!p) { p = el("p"); body.appendChild(p); }
    p.appendChild(document.createTextNode(" "));
    var link = el("button", "more", "… " + label);
    link.type = "button";
    link.addEventListener("click", function () { expand(a, inst, nextStage); });
    p.appendChild(link);
    return link;
  }

  // Opens the article on every copy. Copies above the clicked one grow too,
  // so the page is scrolled by the same amount to keep the clicked card still.
  function expand(a, inst, nextStage) {
    if (state.submitted || a.stage >= nextStage) return;
    var prevVisible = visibleChars(a);
    var t = ms(), ts = iso();
    if (nextStage === 1) { a.readmore_ms = t; a.readmore_ts = ts; a.readmore_round = inst.round; }
    else { a.full_ms = t; a.full_ts = ts; a.full_round = inst.round; }
    a.stage = nextStage;

    var topBefore = inst.card.getBoundingClientRect().top;
    var nextLink = null;
    for (var i = 0; i < a.instances.length; i++) {
      var other = a.instances[i];
      if (other === inst) nextLink = renderBody(a, other, prevVisible);
      else renderBody(a, other, Number.MAX_SAFE_INTEGER);
    }
    var shift = inst.card.getBoundingClientRect().top - topBefore;
    if (Math.abs(shift) >= 1) {
      var scroller = parentScroller();
      if (scroller) scroller.scrollTop += shift;
    }
    if (nextLink) { try { nextLink.focus({ preventScroll: true }); } catch (e) {} }
    updateHeight();
  }

  // Like is shared by all copies of the article.
  function paintLike(b, liked) {
    b.classList.toggle("on", liked);
    b.setAttribute("aria-pressed", liked ? "true" : "false");
  }
  function buildLike(a, inst) {
    var b = el("button", "like");
    b.type = "button";
    b.innerHTML = THUMB + "<span>Like</span>";
    paintLike(b, a.liked);
    b.addEventListener("click", function () {
      if (state.submitted) return;
      a.liked = !a.liked;
      a.history.push({ liked: a.liked, round: inst.round, ms: ms(), ts: iso() });
      for (var i = 0; i < a.likeButtons.length; i++) paintLike(a.likeButtons[i], a.liked);
      b.classList.remove("pop"); void b.offsetWidth; b.classList.add("pop");
    });
    a.likeButtons.push(b);
    return b;
  }

  function buildCard(a, round) {
    var card = el("article", "card");
    if (a.color) {
      card.classList.add("sourced");
      card.style.background = tint(a.color, state.tintAlpha);
      card.style.borderColor = a.color;
      var ribbon = el("div", "ribbon");
      ribbon.style.background = a.color;
      ribbon.setAttribute("aria-hidden", "true");
      card.appendChild(ribbon);
    }
    if (a.title) card.appendChild(el("h2", "headline", a.title));
    var inst = { round: round, card: card, bodyEl: el("div", "body") };
    card.appendChild(inst.bodyEl);
    var foot = el("div", "foot");
    foot.appendChild(buildLike(a, inst));
    card.appendChild(foot);
    a.instances.push(inst);
    renderBody(a, inst, Number.MAX_SAFE_INTEGER);
    card.setAttribute("data-round", String(round));
    if (viewObserver) viewObserver.observe(card);
    return card;
  }

  // ---- minimum time on the feed -------------------------------------------
  // The Continue / Finish button stays disabled, with a countdown, until
  // `minMs` have passed since the feed was shown. performance.now() keeps
  // counting correctly even if the browser throttles timers in a background tab.
  function startCountdown(button, label) {
    if (countdown) { clearInterval(countdown); countdown = null; }
    function tick() {
      var left = state.minMs - ms();
      if (left <= 0) {
        if (countdown) { clearInterval(countdown); countdown = null; }
        if (!state.submitted) { button.disabled = false; button.textContent = label; }
        return;
      }
      button.disabled = true;
      button.textContent = label + " in " + clock(Math.ceil(left / 1000));
    }
    tick();
    if (state.minMs > 0) countdown = setInterval(tick, 250);
  }

  // ---- page --------------------------------------------------------------
  function buildPage(args) {
    if (viewObserver) { viewObserver.disconnect(); viewObserver = null; }
    state = {
      t0: performance.now(), enterTs: iso(), submitted: false, isLast: !!args.is_last,
      previewChars: args.preview_chars || 50, readmoreChars: args.readmore_chars || 144,
      tintAlpha: (typeof args.card_tint === "number") ? args.card_tint : 0.16,
      minMs: Math.max(0, (args.min_seconds || 0) * 1000),
      rounds: Math.max(1, args.repeat_rounds || 1), maxRound: 0, articles: []
    };
    root.textContent = "";

    // Furthest round the participant scrolled to (a card at least half visible).
    if (window.IntersectionObserver) {
      viewObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (!en.isIntersecting) return;
          var r = Number(en.target.getAttribute("data-round")) || 0;
          if (r > state.maxRound) state.maxRound = r;
          viewObserver.unobserve(en.target);
        });
      }, { threshold: 0.5 });
    }

    var head = el("div", "progress");
    head.appendChild(el("span", "progress-label", "Feed " + args.page_number + " of " + args.total_pages));
    var track = el("div", "track");
    track.setAttribute("role", "progressbar");
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", String(args.total_pages));
    track.setAttribute("aria-valuenow", String(args.page_number));
    var fill = el("div", "fill");
    fill.style.width = (100 * (args.page_number - 1) / args.total_pages) + "%";
    track.appendChild(fill);
    head.appendChild(track);
    root.appendChild(head);

    var label = args.is_last ? "Finish" : "Continue";
    var next = mountBar();
    if (!next) {
      // Fallback: the page could not be reached, so the button goes on top.
      var actions = el("div", "actions");
      next = el("button", "next");
      next.type = "button";
      actions.appendChild(next);
      root.appendChild(actions);
    }
    next.textContent = label;
    next.addEventListener("click", function () { submit(next); });

    state.articles = (args.articles || []).map(function (art, i) {
      var sentences = art.sentences || [];
      var a = {
        uid: String(art.uid), position: i + 1, color: art.color || null, title: art.title || "",
        sentences: sentences, flat: flatten(sentences),
        stage: 0, readmore_ms: null, readmore_ts: null, readmore_round: null,
        full_ms: null, full_ts: null, full_round: null,
        liked: false, history: [], instances: [], likeButtons: []
      };
      a.readmoreAvailable = a.flat.length > state.previewChars;
      a.fullAvailable = a.flat.length > state.readmoreChars;
      return a;
    });

    // The same sequence, repeated `rounds` times.
    var feed = el("div", "feed");
    for (var r = 1; r <= state.rounds; r++) {
      for (var i = 0; i < state.articles.length; i++) feed.appendChild(buildCard(state.articles[i], r));
    }
    root.appendChild(feed);

    startCountdown(next, label);

    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        fill.style.width = (100 * args.page_number / args.total_pages) + "%";
      });
    });
    if (args.warn_on_leave) installLeaveGuard();
    scrollParentToTop();
    updateHeight();
  }

  function submit(button) {
    if (state.submitted || ms() < state.minMs) return;
    state.submitted = true;
    button.disabled = true;
    button.textContent = "One moment…";
    if (viewObserver) { viewObserver.disconnect(); viewObserver = null; }
    var payload = {
      page_id: currentPage,
      page_enter_ts: state.enterTs,
      page_exit_ts: iso(),
      page_duration_ms: ms(),
      rounds_shown: state.rounds,
      max_round_reached: state.maxRound,
      articles: state.articles.map(function (a) {
        return {
          uid: a.uid, position: a.position, n_sentences: a.sentences.length, n_chars: a.flat.length,
          readmore_available: a.readmoreAvailable, readmore_ms: a.readmore_ms,
          readmore_ts: a.readmore_ts, readmore_round: a.readmore_round,
          full_available: a.fullAvailable, full_ms: a.full_ms,
          full_ts: a.full_ts, full_round: a.full_round,
          liked: a.liked, like_history: a.history
        };
      })
    };
    if (state.isLast) removeLeaveGuard();
    send("streamlit:setComponentValue", { value: payload, dataType: "json" });
  }

  window.addEventListener("message", function (event) {
    var d = event.data;
    if (!d || d.type !== "streamlit:render") return;
    var args = d.args || {};
    if (args.page_id && args.page_id !== currentPage) {
      currentPage = args.page_id;
      buildPage(args);
    } else {
      updateHeight();
    }
  });
  send("streamlit:componentReady", { apiVersion: 1 });
})();
</script>
</body>
</html>
""".replace("__THUMB_SVG__", THUMB_SVG)

_COMPONENT_DIR = os.path.join(
    tempfile.gettempdir(),
    "nps_feed_" + hashlib.sha1(FEED_COMPONENT_HTML.encode("utf-8")).hexdigest()[:12],
)


def _ensure_component_files():
    path = os.path.join(_COMPONENT_DIR, "index.html")
    if not os.path.exists(path):
        os.makedirs(_COMPONENT_DIR, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{_rng.randint(0, 10**9)}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(FEED_COMPONENT_HTML)
        os.replace(tmp, path)


_ensure_component_files()
feed_component = components.declare_component("news_feed", path=_COMPONENT_DIR)

# =============================================================================
# 7. PAGE STYLE
# =============================================================================

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,500&family=Schibsted+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --canvas: #ECEEF1;
  --surface: #FFFFFF;
  --ink: #191C20;
  --ink-2: #2E3238;
  --ink-3: #5C626A;
  --line: #D8DCE1;
  --line-strong: #B9BEC5;
  --sans: "Schibsted Grotesk", system-ui, -apple-system, "Segoe UI", sans-serif;
  --serif: "Literata", Georgia, "Times New Roman", serif;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--canvas) !important;
  color: var(--ink);
}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], #MainMenu, footer { display: none !important; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 700px !important;
  padding: 3.2rem 1.25rem 4rem !important;
}
[data-stale="true"], .stale-element { opacity: 1 !important; transition: none !important; }
iframe { border: 0 !important; }
.stApp, .stApp p, .stApp li { font-family: var(--sans); }

/* ---------- intro ---------- */
.nps-title {
  font-family: var(--sans); font-weight: 700; font-size: 44px; line-height: 1.06;
  letter-spacing: -0.025em; color: var(--ink); margin: 0 0 20px; padding: 0;
}
.nps-lead {
  font-family: var(--serif); font-optical-sizing: auto; font-size: 20px; line-height: 1.58;
  color: var(--ink-2); margin: 0 0 14px; max-width: 34em;
}
.nps-sub { font-size: 15px; line-height: 1.5; color: var(--ink-3); margin: 0 0 46px; }
.nps-h2 {
  font-family: var(--sans); font-weight: 600; font-size: 20px; letter-spacing: -0.01em;
  color: var(--ink); margin: 0 0 18px; padding: 0;
}
ul.nps-rows { list-style: none !important; margin: 0 0 30px !important; padding: 0 !important;
              display: grid; gap: 20px; }
li.nps-row { display: grid; grid-template-columns: 150px 1fr; gap: 22px; align-items: center;
             margin: 0 !important; padding: 0 !important; }
li.nps-row::marker { content: none; }
.nps-row p { margin: 0; font-size: 16px; line-height: 1.55; color: var(--ink-2); }
.nps-row b { color: var(--ink); font-weight: 600; }
.nps-token { display: flex; align-items: center; justify-content: flex-start; }
.tok-swatches { gap: 8px; }
.tok-swatches span {
  display: block; width: 26px; height: 20px; border-radius: 5px; border: 2px solid;
}
.tok-more {
  font-weight: 600; font-size: 15px; color: var(--ink); text-decoration: underline;
  text-decoration-thickness: 1.5px; text-decoration-color: var(--line-strong);
  text-underline-offset: 4px; white-space: nowrap;
}
.tok-like {
  display: inline-flex; align-items: center; gap: 7px; height: 34px; padding: 0 14px 0 11px;
  border-radius: 17px; border: 1px solid var(--line); background: var(--surface);
  color: var(--ink-3); font-weight: 500; font-size: 14px;
}
.tok-thumb {
  display: inline-block; width: 17px; height: 17px; background: currentColor;
  -webkit-mask: url("__THUMB_MASK__") center / contain no-repeat;
  mask: url("__THUMB_MASK__") center / contain no-repeat;
}
.tok-next {
  display: inline-flex; align-items: center; height: 34px; padding: 0 18px; border-radius: 17px;
  background: var(--ink); color: #FFFFFF; font-weight: 600; font-size: 14px;
}
.nps-plain { font-size: 16px; line-height: 1.55; color: var(--ink-2); margin: 0 0 40px; }

/* ---------- Continue / Finish bar, fixed at the bottom during a feed ---------- */
#nps-feedbar {
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 999990;
  padding: 28px 16px calc(16px + env(safe-area-inset-bottom));
  background: linear-gradient(to bottom, rgba(236, 238, 241, 0), var(--canvas) 55%);
  pointer-events: none;
}
.nps-feedbar-inner {
  max-width: 700px; margin: 0 auto; padding: 0 1.25rem;
  display: flex; justify-content: flex-end;
}
.nps-feedbar-btn {
  pointer-events: auto; appearance: none; border: 0; height: 48px; padding: 0 30px;
  border-radius: 24px; background: var(--ink); color: #FFFFFF;
  font: 600 16px/1 var(--sans); cursor: pointer; font-variant-numeric: tabular-nums;
  box-shadow: 0 6px 18px rgba(25, 28, 32, .18); transition: background-color .15s;
}
.nps-feedbar-btn:disabled { background: #8B9097; cursor: default; box-shadow: none; }
.nps-feedbar-btn:focus-visible { outline: 2px solid var(--ink); outline-offset: 3px; }
body:has(#nps-feedbar) [data-testid="stMainBlockContainer"],
body:has(#nps-feedbar) .block-container { padding-bottom: 120px !important; }

/* ---------- break between the feeds ---------- */
.nps-kicker {
  font-size: 14px; font-weight: 600; color: var(--ink-3); letter-spacing: .02em;
  margin: 0 0 14px; font-variant-numeric: tabular-nums;
}
.nps-break .nps-sub { margin-bottom: 30px; }
.nps-loading {
  display: flex; align-items: center; gap: 12px; min-height: 48px;
  font-size: 15px; color: var(--ink-3);
}
.nps-spinner {
  flex: none; width: 18px; height: 18px; border-radius: 50%;
  border: 2px solid var(--line); border-top-color: var(--ink);
  animation: nps-spin .8s linear infinite;
}
@keyframes nps-spin { to { transform: rotate(360deg); } }

/* ---------- end ---------- */
.nps-code {
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 24px 26px; margin: 26px 0 18px;
}
.nps-code p { margin: 0; font-size: 15px; color: var(--ink-3); line-height: 1.55; }
.nps-code strong {
  display: block; font-family: var(--sans); font-weight: 700; font-size: 34px;
  letter-spacing: 0.04em; color: var(--ink); margin: 6px 0 12px; user-select: all;
}
.nps-alert {
  border-left: 3px solid var(--ink); background: var(--surface); padding: 14px 18px;
  font-size: 15px; line-height: 1.55; color: var(--ink-2); margin: 0 0 18px; border-radius: 0 8px 8px 0;
}

/* ---------- Streamlit widgets ---------- */
.stButton > button, .stLinkButton > a,
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"],
[data-testid="stBaseLinkButton-primary"] {
  background: var(--ink) !important; color: #FFFFFF !important; border: 0 !important;
  border-radius: 24px !important; min-height: 48px !important; padding: 0 30px !important;
  font-family: var(--sans) !important; font-weight: 600 !important; font-size: 16px !important;
  box-shadow: none !important;
}
.stButton > button p, .stLinkButton > a p { color: #FFFFFF !important; font-weight: 600 !important; font-size: 16px !important; }
.stButton > button:disabled { opacity: .3 !important; cursor: not-allowed !important; }
.stButton > button:focus-visible, .stLinkButton > a:focus-visible { outline: 2px solid var(--ink) !important; outline-offset: 3px !important; }

@media (max-width: 560px) {
  .nps-title { font-size: 34px; }
  .nps-lead { font-size: 18px; }
  li.nps-row { grid-template-columns: 1fr; gap: 10px; }
  .nps-code { padding: 20px 18px 16px; }
  .nps-feedbar-inner { padding: 0; }
  .nps-feedbar-btn { width: 100%; }
}
@media (prefers-reduced-motion: reduce) {
  .nps-spinner { animation-duration: 2.4s; }
}
""".replace(
    "__THUMB_MASK__",
    "data:image/svg+xml," + quote(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
        "stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'>"
        "<path d='M7 10v12'/><path d='M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 "
        "17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 "
        "3 3.88Z'/></svg>"
    ),
)

# =============================================================================
# 8. SESSION STATE
# =============================================================================

def init_session():
    qp = st.query_params
    try:
        user_agent = st.context.headers.get("User-Agent", "")
    except Exception:
        user_agent = ""
    st.session_state.update({
        "nps_ready": True,
        "session_id": make_session_id(),
        "prolific_pid": qp.get("PROLIFIC_PID", ""),
        "study_id": qp.get("STUDY_ID", ""),
        "prolific_session_id": qp.get("SESSION_ID", ""),
        "user_agent": user_agent,
        "stage": "intro",
        "opened_at": utc_iso(),
        "started_at": None,
        "started_ts": None,
        "completed_at": None,
        "condition": None,
        "assignment_method": None,
        "color_map": {},
        "article_order": {},
        "split": {},
        "pages": [],
        "page_idx": 0,
        "processed": [],
        "page_server_start": {},
        "break_ready": False,
        "responses": [],
        "save_status": None,
        "saved_parts": [],
    })


def start_study(data):
    """Randomise everything for this participant and build the two feeds."""
    ss = st.session_state
    topics, outlets = list(data["topics"]), list(data["outlets"])

    colors = OUTLET_PALETTE[: len(outlets)]
    _rng.shuffle(colors)
    color_map = dict(zip(outlets, colors))

    # For each topic x outlet, one article goes to phase 1 and one to phase 2.
    split = {"1": {t: [] for t in topics}, "2": {t: [] for t in topics}}
    for t in topics:
        for o in outlets:
            uids = list(data["cells"][t][o])
            _rng.shuffle(uids)
            split["1"][t].append(uids[0])
            split["2"][t].append(uids[1])

    # One feed per phase with all its articles; the order mixes topics and
    # outlets and is drawn independently for the two phases. The feed repeats
    # this same order REPEAT_ROUNDS times.
    order = {}
    for phase in ("1", "2"):
        uids = [u for t in topics for u in split[phase][t]]
        _rng.shuffle(uids)
        order[phase] = uids

    condition, method = assign_condition(ss.session_id, ss.prolific_pid)
    cond = CONDITIONS[condition]

    pages = []
    for number, phase in enumerate(("1", "2"), start=1):
        if phase == "1":
            visible, version = True, "original"
        else:
            visible = cond["source_visible"]
            version = "rewritten" if cond["rewritten"] else "original"
        pages.append({
            "page_number": number, "phase": int(phase),
            "source_visible": visible, "text_version": version,
            "articles": order[phase],
        })

    ss.update({
        "condition": condition, "assignment_method": method, "color_map": color_map,
        "split": split, "article_order": order, "pages": pages, "page_idx": 0,
        "stage": "feed", "started_at": utc_iso(), "started_ts": time.time(),
    })
    log(f"session {ss.session_id} started: condition {condition} ({method})")


def record_page(data, page, page_id, result, cards):
    ss = st.session_state
    shown = {c["uid"]: c for c in cards}
    server_s = round(time.time() - ss.page_server_start.get(page_id, time.time()), 2)
    for item in result.get("articles", []):
        uid = str(item.get("uid"))
        if uid not in shown:
            continue
        art = data["articles"][uid]
        history = item.get("like_history") or []
        ss.responses.append({
            "session_id": ss.session_id,
            "prolific_pid": ss.prolific_pid,
            "condition": ss.condition,
            "condition_label": CONDITIONS[ss.condition]["label"],
            "phase": page["phase"],
            "page_number": page["page_number"],
            "topic": art["topic"],
            "position": item.get("position"),
            "article_uid": uid,
            "outlet": art["outlet"],
            "reliability_class": art["reliability"],
            "outlet_color": ss.color_map.get(art["outlet"]),
            "source_visible": page["source_visible"],
            "text_version": page["text_version"],
            "title_shown": shown[uid]["title"],
            "n_sentences": len(shown[uid]["sentences"]),
            "n_chars": item.get("n_chars", n_chars(shown[uid]["sentences"])),
            "readmore_available": bool(item.get("readmore_available")),
            "readmore_clicked": item.get("readmore_ms") is not None,
            "readmore_ms": item.get("readmore_ms"),
            "readmore_ts": item.get("readmore_ts"),
            "readmore_round": item.get("readmore_round"),
            "full_available": bool(item.get("full_available")),
            "full_clicked": item.get("full_ms") is not None,
            "full_ms": item.get("full_ms"),
            "full_ts": item.get("full_ts"),
            "full_round": item.get("full_round"),
            "liked_final": bool(item.get("liked")),
            "like_toggles": len(history),
            "like_history": json.dumps(history),
            "rounds_shown": result.get("rounds_shown"),
            "max_round_reached": result.get("max_round_reached"),
            "page_enter_ts": result.get("page_enter_ts"),
            "page_exit_ts": result.get("page_exit_ts"),
            "page_duration_ms": result.get("page_duration_ms"),
            "page_duration_server_s": server_s,
            "year": art["year"],
            "month": art["month"],
            "rating": art["rating"],
        })


def build_session_row():
    ss = st.session_state
    resp = ss.responses
    duration = round((time.time() - ss.started_ts) / 60, 2) if ss.started_ts else None

    def max_round(phase):
        rows = [r for r in resp if r["phase"] == phase]
        return rows[0]["max_round_reached"] if rows else None

    return {
        "session_id": ss.session_id,
        "prolific_pid": ss.prolific_pid,
        "study_id": ss.study_id,
        "prolific_session_id": ss.prolific_session_id,
        "condition": ss.condition,
        "condition_label": CONDITIONS[ss.condition]["label"],
        "assignment_method": ss.assignment_method,
        "color_map": json.dumps(ss.color_map),
        "phase1_order": json.dumps(ss.article_order.get("1", [])),
        "phase2_order": json.dumps(ss.article_order.get("2", [])),
        "article_split": json.dumps(ss.split),
        "opened_at_utc": ss.opened_at,
        "started_at_utc": ss.started_at,
        "completed_at_utc": ss.completed_at,
        "duration_min": duration,
        "n_articles": len(resp),
        "n_readmore": sum(1 for r in resp if r["readmore_clicked"]),
        "n_full": sum(1 for r in resp if r["full_clicked"]),
        "n_likes": sum(1 for r in resp if r["liked_final"]),
        "phase1_max_round": max_round(1),
        "phase2_max_round": max_round(2),
        "user_agent": ss.user_agent,
        "app_version": APP_VERSION,
    }


def save_results():
    """Single write at the end: responses + session row to Google Sheets,
    plus a local CSV/JSON backup."""
    ss = st.session_state
    session_row = build_session_row()
    response_rows = [[cell(r.get(h)) for h in RESPONSE_HEADERS] for r in ss.responses]
    session_values = [cell(session_row.get(h)) for h in SESSION_HEADERS]

    try:
        os.makedirs(LOCAL_BACKUP_DIR, exist_ok=True)
        pd.DataFrame(ss.responses, columns=RESPONSE_HEADERS).to_csv(
            os.path.join(LOCAL_BACKUP_DIR, f"{ss.session_id}_responses.csv"), index=False)
        with open(os.path.join(LOCAL_BACKUP_DIR, f"{ss.session_id}_session.json"), "w",
                  encoding="utf-8") as f:
            json.dump(session_row, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        log(f"Local backup failed: {exc}")

    store = get_store()
    if store is None:
        ss.save_status = "local_only"
        return

    for attempt in range(3):
        try:
            if "responses" not in ss.saved_parts:
                store.append(SHEET_RESPONSES, RESPONSE_HEADERS, response_rows)
                ss.saved_parts.append("responses")
            if "session" not in ss.saved_parts:
                store.append(SHEET_SESSIONS, SESSION_HEADERS, [session_values])
                ss.saved_parts.append("session")
            ss.save_status = "sheets"
            log(f"session {ss.session_id} saved to Google Sheets")
            return
        except Exception as exc:
            log(f"Save attempt {attempt + 1} failed: {exc}")
            time.sleep(1.5 * (attempt + 1))
    ss.save_status = "sheets_failed"


# =============================================================================
# 9. SCREENS
# =============================================================================

def render_setup_error(errors):
    items = "".join(f"<li>{html_lib.escape(e)}</li>" for e in errors)
    st.html(f"""
<section>
  <h1 class="nps-title">The study isn't set up correctly</h1>
  <p class="nps-sub">Fix the following in the articles file or in the configuration at the top of app.py, then reload the page.</p>
  <ul class="nps-plain">{items}</ul>
</section>""")


def render_intro(data):
    n_feeds = 2
    swatches = "".join(
        f'<span style="background:{tint_hex(c, CARD_TINT)};border-color:{c}"></span>'
        for c in OUTLET_PALETTE[: len(data["outlets"])]
    )
    if MIN_FEED_SECONDS > 0:
        wait_note = (f" The button becomes active after {duration_words(MIN_FEED_SECONDS)} "
                     f"on each feed.")
    else:
        wait_note = ""
    st.html(f"""
<section class="nps-intro">
  <h1 class="nps-title">News Perception Study</h1>
  <p class="nps-lead">You'll browse {number_word(n_feeds)} news feeds. Read them the way you'd read news online.</p>
  <p class="nps-sub">It takes about {ESTIMATED_MINUTES} minutes, with a short break between the two feeds. Please finish in one sitting and don't refresh the page.</p>

  <h2 class="nps-h2">How it works</h2>
  <ul class="nps-rows">
    <li class="nps-row">
      <div class="nps-token"><span class="tok-more">…&nbsp;Read more</span></div>
      <p>Each article opens with a short preview. Click <b>Read more</b> to keep reading. After that, <b>Read full article</b> shows the whole text.</p>
    </li>
    <li class="nps-row">
      <div class="nps-token tok-swatches">{swatches}</div>
      <p>The colour of each article's box shows which news outlet published it. Each colour is a different outlet.</p>
    </li>
    <li class="nps-row">
      <div class="nps-token"><span class="tok-like"><span class="tok-thumb" aria-hidden="true"></span><span>Like</span></span></div>
      <p>Click <b>Like</b> on any article you like. Click it again to undo.</p>
    </li>
    <li class="nps-row">
      <div class="nps-token"><span class="tok-next">Continue</span></div>
      <p>When you want to move on, click <b>Continue</b> at the bottom of the screen.{wait_note} You can't go back to an earlier feed.</p>
    </li>
  </ul>
  <p class="nps-plain">There's no right or wrong way to do this.</p>
</section>""")

    if st.button("Start the study", type="primary", key="start_button"):
        start_study(data)
        st.rerun()


def render_feed(data):
    ss = st.session_state
    page = ss.pages[ss.page_idx]
    page_id = f"{ss.session_id}-p{page['page_number']}"
    ss.page_server_start.setdefault(page_id, time.time())

    rewritten = page["text_version"] == "rewritten"
    cards = []
    for uid in page["articles"]:
        art = data["articles"][uid]
        cards.append({
            "uid": uid,
            "color": ss.color_map[art["outlet"]] if page["source_visible"] else None,
            "title": art["title"],   # same headline (Title column) in both versions
            "sentences": art["sentences_rewritten"] if rewritten else art["sentences_original"],
        })

    result = feed_component(
        page_id=page_id,
        page_number=page["page_number"],
        total_pages=len(ss.pages),
        is_last=page["page_number"] == len(ss.pages),
        articles=cards,
        repeat_rounds=REPEAT_ROUNDS,
        preview_chars=PREVIEW_CHARS,
        readmore_chars=READMORE_CHARS,
        min_seconds=MIN_FEED_SECONDS,
        card_tint=CARD_TINT,
        warn_on_leave=WARN_ON_LEAVE,
        key=f"feed-{page_id}",
        default=None,
    )

    if isinstance(result, dict) and result.get("page_id") == page_id and page_id not in ss.processed:
        record_page(data, page, page_id, result, cards)
        ss.processed.append(page_id)
        ss.page_idx += 1
        if ss.page_idx >= len(ss.pages):
            ss.stage = "end"
            ss.completed_at = utc_iso()
        else:
            ss.stage = "break"
            ss.break_ready = False
        st.rerun()


def render_break():
    """Short pause between the two feeds: a brief 'loading the second part'
    wait, then the button to start feed 2. The break length can be computed
    from the data as page_enter_ts (feed 2) - page_exit_ts (feed 1)."""
    ss = st.session_state
    done = ss.page_idx
    total = len(ss.pages)
    st.html(f"""
<section class="nps-break">
  <p class="nps-kicker">Feed {done} of {total} complete</p>
  <h1 class="nps-title">Short break</h1>
  <p class="nps-lead">You've finished the first part. Take a moment, then start the second one.</p>
  <p class="nps-sub">It works the same way as the first. Please don't refresh or close this page.</p>
</section>""")

    slot = st.empty()
    if not ss.break_ready and BREAK_LOADING_SECONDS > 0:
        slot.html('<div class="nps-loading" role="status">'
                  '<span class="nps-spinner" aria-hidden="true"></span>'
                  'Loading the second part…</div>')
        time.sleep(BREAK_LOADING_SECONDS)
        slot.empty()
    ss.break_ready = True

    if st.button("Start the second part", type="primary", key="break_button"):
        ss.stage = "feed"
        st.rerun()


def render_end():
    ss = st.session_state
    if ss.save_status is None:
        with st.spinner("Saving your responses…"):
            save_results()

    code = str(get_secret("prolific", "completion_code", "") or "").strip()
    failed = ss.save_status == "sheets_failed"

    if failed:
        lead = "You've finished the study, but your responses couldn't be uploaded."
        alert = ('<div class="nps-alert">Click <b>Try again</b>. If it still fails, send a message '
                 'to the researcher on Prolific before submitting.</div>')
    else:
        lead = "You've finished the study and your responses have been saved."
        alert = ""

    if code:
        code_block = (f'<div class="nps-code"><p>Your completion code</p>'
                      f'<strong>{html_lib.escape(code)}</strong>'
                      f'<p>Use the button below to return to Prolific, or enter the code there yourself.</p></div>')
    else:
        code_block = '<p class="nps-sub">You can now close this window.</p>'

    st.html(f"""
<section class="nps-end">
  <h1 class="nps-title">Thank you</h1>
  <p class="nps-lead">{lead}</p>
  {alert}
  {code_block}
</section>""")

    if failed and st.button("Try again", key="retry_save"):
        ss.save_status = None
        st.rerun()
    if code:
        st.link_button("Return to Prolific", PROLIFIC_RETURN_URL.format(code=code), type="primary")


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    st.html(f"<style>{APP_CSS}</style>")
    if "nps_ready" not in st.session_state:
        init_session()

    stage = st.session_state.stage
    if stage != "feed":
        # The Continue bar belongs to the feed only.
        st.html("<style>#nps-feedbar { display: none !important; }</style>")

    mtime = os.path.getmtime(ARTICLES_CSV) if os.path.exists(ARTICLES_CSV) else 0.0
    data, errors = load_study_data(ARTICLES_CSV, mtime)
    if errors:
        render_setup_error(errors)
        return

    if stage == "intro":
        render_intro(data)
    elif stage == "feed":
        render_feed(data)
    elif stage == "break":
        render_break()
    else:
        render_end()


main()