"""
News Perception Study
=====================

Streamlit app for the two-phase news-feed experiment.

    Phase 1 (pages 1-4): every participant sees the source colour and the
                         original text.
    Phase 2 (pages 5-8): the between-subject treatment (2 x 2):
                         source visible / hidden  x  original / LLM-rewritten.

Each page is one topic and shows one article from each of the 4 outlets, in
random order. Interaction per article:
    headline + 2-sentence snippet -> "... Read more" (5 sentences)
    -> "... Read full article" (whole text), plus an optional Like toggle.

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

APP_VERSION = "2026-10-03"

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
    "title": ["title", "headline"],
    "title_rewritten": ["title_rewritten", "rewritten_title", "headline_rewritten"],
    "text": ["clean_text", "text"],
    "text_rewritten": ["rewritten_text", "text_rewritten", "clean_text_rewritten",
                       "llm_text", "ai_text", "rewritten"],
    "reliability": ["reliability_class", "reliability_label"],
    "rating": ["rating"],
    "year": ["year"],
    "month": ["month"],
}
REQUIRED_FIELDS = ("topic", "outlet", "text", "text_rewritten")

ARTICLES_PER_CELL = 2      # articles per outlet per topic (1 per phase)
SNIPPET_SENTENCES = 2      # sentences visible before "Read more"
READMORE_SENTENCES = 5     # sentences visible after "Read more" (total)

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
RESEARCHER_CONTACT = "[Researcher name, institution, e-mail]"
ETHICS_STATEMENT = "[Ethics approval reference]"
PROLIFIC_RETURN_URL = "https://app.prolific.com/submissions/complete?cc={code}"

SHEET_ASSIGNMENTS = "assignments"
SHEET_SESSIONS = "sessions"
SHEET_RESPONSES = "responses"

ASSIGNMENT_HEADERS = ["session_id", "prolific_pid", "condition", "assigned_at_utc"]

SESSION_HEADERS = [
    "session_id", "prolific_pid", "study_id", "prolific_session_id",
    "condition", "condition_label", "assignment_method",
    "color_map", "phase1_topic_order", "phase2_topic_order", "article_split",
    "opened_at_utc", "consent_at_utc", "started_at_utc", "completed_at_utc",
    "duration_min", "n_articles", "n_readmore", "n_full", "n_likes",
    "user_agent", "app_version",
]

RESPONSE_HEADERS = [
    "session_id", "prolific_pid", "condition", "condition_label",
    "phase", "page_number", "page_in_phase", "topic", "position",
    "article_uid", "outlet", "reliability_class", "outlet_color",
    "source_visible", "text_version", "title_shown", "n_sentences",
    "readmore_available", "readmore_clicked", "readmore_ms", "readmore_ts",
    "full_available", "full_clicked", "full_ms", "full_ts",
    "liked_final", "like_toggles", "like_history",
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
             "eight", "nine", "ten", "eleven", "twelve"]
    return words[n] if 0 <= n < len(words) else str(n)


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
_SPLIT_RE = re.compile(r"([.!?…]+[\"”’')\]]*)\s+(?=[\"“‘(\[]?[A-Z0-9À-Ý])")
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


# =============================================================================
# 4. ARTICLES: LOADING AND VALIDATION
# =============================================================================

def _pick_column(df, field):
    for name in COLUMN_CANDIDATES[field]:
        if name in df.columns:
            return name
    return None


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
    if cols["title"] is None:
        warnings.append("No headline column found: cards will show the text only.")

    articles = {}
    for i, row in df.iterrows():
        def val(field):
            return _clean(row[cols[field]]) if cols[field] else None

        uid = str(val("id")) if cols["id"] and val("id") is not None else f"a{i:02d}"
        text = str(val("text") or "").strip()
        text_rw = str(val("text_rewritten") or "").strip()
        title = str(val("title") or "").strip()
        title_rw = str(val("title_rewritten") or "").strip()

        if uid in articles:
            errors.append(f"Duplicate article id: {uid}.")
        if not text:
            errors.append(f"Article {uid} has no original text.")
        if not text_rw:
            errors.append(f"Article {uid} has no rewritten text.")

        articles[uid] = {
            "uid": uid,
            "row": int(i),
            "topic": str(val("topic")).strip(),
            "outlet": str(val("outlet")).strip(),
            "title": title,
            "title_rewritten": title_rw,
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

    # Reliability class: from its own column if present, otherwise derived from
    # the outlet rating (higher mean rating = reliable).
    if cols["reliability"] is None:
        if cols["rating"] is not None:
            means = {}
            for o in outlets:
                ratings = [float(a["rating"]) for a in articles.values()
                           if a["outlet"] == o and a["rating"] is not None]
                means[o] = sum(ratings) / len(ratings) if ratings else float("-inf")
            ranked = sorted(outlets, key=lambda o: means[o], reverse=True)
            reliable = set(ranked[: len(ranked) // 2])
            for a in articles.values():
                a["reliability"] = "reliable" if a["outlet"] in reliable else "unreliable"
            warnings.append("Reliability class derived from the 'rating' column.")
        else:
            for a in articles.values():
                a["reliability"] = "NA"
            warnings.append("No reliability information found.")

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
  .card.has-source { padding-top: 46px; }
  .ribbon { position: absolute; top: -1px; left: 25px; width: 18px; height: 28px;
            clip-path: polygon(0 0, 100% 0, 100% 100%, 50% 74%, 0 100%); }

  .headline { font-family: var(--sans); font-weight: 600; font-size: 21px; line-height: 1.28;
              letter-spacing: -0.012em; margin: 0 0 10px; color: var(--ink); text-wrap: pretty; }
  .body { font-family: var(--serif); font-optical-sizing: auto; font-size: 17px;
          line-height: 1.66; color: var(--ink-2); }
  .body p { margin: 0 0 .75em; }
  .body p:last-child { margin-bottom: 0; }
  .fresh { animation: fresh .6s ease-out both; }
  @keyframes fresh { from { opacity: 0; background: rgba(25, 28, 32, .07); }
                     to { opacity: 1; background: transparent; } }

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

  .actions { display: flex; justify-content: flex-end; margin-top: 26px; }
  .next { appearance: none; border: 0; height: 48px; padding: 0 30px; border-radius: 24px;
          background: var(--ink); color: #FFFFFF; font: 600 16px/1 var(--sans); cursor: pointer;
          transition: opacity .15s; }
  .next:disabled { opacity: .55; cursor: default; }

  button:focus-visible { outline: 2px solid var(--ink); outline-offset: 3px; }

  @media (max-width: 560px) {
    .card { padding: 18px 18px 16px; }
    .card.has-source { padding-top: 42px; }
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
  var root = document.getElementById("root");
  var state = null;
  var currentPage = null;
  var lastHeight = -1;

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

  // ---- article text ------------------------------------------------------
  function visibleCount(a) {
    var n = a.sentences.length;
    if (a.stage === 0) return Math.min(state.snippet, n);
    if (a.stage === 1) return Math.min(state.readmore, n);
    return n;
  }

  // Renders the visible sentences grouped in paragraphs, with the inline
  // "... Read more" / "... Read full article" link after the last one.
  function renderBody(a, prevVisible) {
    var body = a.bodyEl;
    var vis = visibleCount(a);
    body.textContent = "";
    var p = null, para = null;
    for (var i = 0; i < vis; i++) {
      var s = a.sentences[i];
      if (p === null || s.p !== para) { p = el("p"); body.appendChild(p); para = s.p; }
      else { p.appendChild(document.createTextNode(" ")); }
      p.appendChild(el("span", i >= prevVisible ? "fresh" : null, s.t));
    }
    var label = null, nextStage = null;
    if (a.stage === 0 && a.readmoreAvailable) { label = "Read more"; nextStage = 1; }
    else if (a.stage === 1 && a.fullAvailable) { label = "Read full article"; nextStage = 2; }
    if (!label) return null;
    if (!p) { p = el("p"); body.appendChild(p); }
    p.appendChild(document.createTextNode(" "));
    var link = el("button", "more", "… " + label);
    link.type = "button";
    link.addEventListener("click", function () { expand(a, nextStage, vis); });
    p.appendChild(link);
    return link;
  }

  function expand(a, nextStage, prevVisible) {
    if (state.submitted || a.stage >= nextStage) return;
    var t = ms(), ts = iso();
    if (nextStage === 1) { a.readmore_ms = t; a.readmore_ts = ts; }
    else { a.full_ms = t; a.full_ts = ts; }
    a.stage = nextStage;
    var nextLink = renderBody(a, prevVisible);
    if (nextLink) { try { nextLink.focus({ preventScroll: true }); } catch (e) {} }
    updateHeight();
  }

  function buildLike(a) {
    var b = el("button", "like");
    b.type = "button";
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = THUMB + "<span>Like</span>";
    b.addEventListener("click", function () {
      if (state.submitted) return;
      a.liked = !a.liked;
      a.history.push({ liked: a.liked, ms: ms(), ts: iso() });
      b.classList.toggle("on", a.liked);
      b.setAttribute("aria-pressed", a.liked ? "true" : "false");
      b.classList.remove("pop"); void b.offsetWidth; b.classList.add("pop");
    });
    return b;
  }

  // ---- page --------------------------------------------------------------
  function buildPage(args) {
    state = {
      t0: performance.now(), enterTs: iso(), submitted: false, isLast: !!args.is_last,
      snippet: args.snippet_sentences || 2, readmore: args.readmore_sentences || 5, items: []
    };
    root.textContent = "";

    var head = el("div", "progress");
    head.appendChild(el("span", "progress-label", "Page " + args.page_number + " of " + args.total_pages));
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

    var feed = el("div", "feed");
    (args.articles || []).forEach(function (art, i) {
      var a = {
        uid: String(art.uid), position: i + 1, sentences: art.sentences || [], stage: 0,
        readmore_ms: null, readmore_ts: null, full_ms: null, full_ts: null,
        liked: false, history: []
      };
      a.readmoreAvailable = a.sentences.length > state.snippet;
      a.fullAvailable = a.sentences.length > state.readmore;

      var card = el("article", "card" + (art.color ? " has-source" : ""));
      if (art.color) {
        var ribbon = el("div", "ribbon");
        ribbon.style.background = art.color;
        ribbon.setAttribute("aria-hidden", "true");
        card.appendChild(ribbon);
      }
      if (art.title) card.appendChild(el("h2", "headline", art.title));
      a.bodyEl = el("div", "body");
      card.appendChild(a.bodyEl);
      var foot = el("div", "foot");
      foot.appendChild(buildLike(a));
      card.appendChild(foot);
      renderBody(a, Number.MAX_SAFE_INTEGER);
      state.items.push(a);
      feed.appendChild(card);
    });
    root.appendChild(feed);

    var actions = el("div", "actions");
    var next = el("button", "next", args.is_last ? "Finish" : "Continue");
    next.type = "button";
    next.addEventListener("click", function () { submit(next); });
    actions.appendChild(next);
    root.appendChild(actions);

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
    if (state.submitted) return;
    state.submitted = true;
    button.disabled = true;
    button.textContent = "One moment…";
    var payload = {
      page_id: currentPage,
      page_enter_ts: state.enterTs,
      page_exit_ts: iso(),
      page_duration_ms: ms(),
      articles: state.items.map(function (a) {
        return {
          uid: a.uid, position: a.position, n_sentences: a.sentences.length,
          readmore_available: a.readmoreAvailable, readmore_ms: a.readmore_ms, readmore_ts: a.readmore_ts,
          full_available: a.fullAvailable, full_ms: a.full_ms, full_ts: a.full_ts,
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
.tok-ribbons { gap: 9px; }
.tok-ribbons span {
  display: block; width: 15px; height: 24px;
  clip-path: polygon(0 0, 100% 0, 100% 100%, 50% 74%, 0 100%);
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
.nps-consent {
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 24px 26px 20px; margin: 0 0 8px;
}
.nps-consent p { font-size: 15px; line-height: 1.6; color: var(--ink-2); margin: 0 0 12px; }
.nps-consent .nps-contact { color: var(--ink-3); font-size: 14px; margin: 16px 0 0; }

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
.stCheckbox p, .stCheckbox label { color: var(--ink) !important; font-size: 15px !important; }
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
  .nps-consent, .nps-code { padding: 20px 18px 16px; }
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
        "consent_at": None,
        "started_at": None,
        "started_ts": None,
        "completed_at": None,
        "condition": None,
        "assignment_method": None,
        "color_map": {},
        "topic_order": {},
        "split": {},
        "pages": [],
        "page_idx": 0,
        "processed": [],
        "page_server_start": {},
        "responses": [],
        "save_status": None,
        "saved_parts": [],
    })


def start_study(data):
    """Randomise everything for this participant and build the 8 pages."""
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

    order = {"1": topics[:], "2": topics[:]}
    _rng.shuffle(order["1"])
    _rng.shuffle(order["2"])

    condition, method = assign_condition(ss.session_id, ss.prolific_pid)
    cond = CONDITIONS[condition]

    pages, number = [], 0
    for phase in ("1", "2"):
        for k, topic in enumerate(order[phase], start=1):
            number += 1
            uids = list(split[phase][topic])
            _rng.shuffle(uids)
            if phase == "1":
                visible, version = True, "original"
            else:
                visible = cond["source_visible"]
                version = "rewritten" if cond["rewritten"] else "original"
            pages.append({
                "page_number": number, "phase": int(phase), "page_in_phase": k,
                "topic": topic, "source_visible": visible, "text_version": version,
                "articles": uids,
            })

    ss.update({
        "condition": condition, "assignment_method": method, "color_map": color_map,
        "split": split, "topic_order": order, "pages": pages, "page_idx": 0,
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
            "page_in_phase": page["page_in_phase"],
            "topic": page["topic"],
            "position": item.get("position"),
            "article_uid": uid,
            "outlet": art["outlet"],
            "reliability_class": art["reliability"],
            "outlet_color": ss.color_map.get(art["outlet"]),
            "source_visible": page["source_visible"],
            "text_version": page["text_version"],
            "title_shown": shown[uid]["title"],
            "n_sentences": len(shown[uid]["sentences"]),
            "readmore_available": bool(item.get("readmore_available")),
            "readmore_clicked": item.get("readmore_ms") is not None,
            "readmore_ms": item.get("readmore_ms"),
            "readmore_ts": item.get("readmore_ts"),
            "full_available": bool(item.get("full_available")),
            "full_clicked": item.get("full_ms") is not None,
            "full_ms": item.get("full_ms"),
            "full_ts": item.get("full_ts"),
            "liked_final": bool(item.get("liked")),
            "like_toggles": len(history),
            "like_history": json.dumps(history),
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
    return {
        "session_id": ss.session_id,
        "prolific_pid": ss.prolific_pid,
        "study_id": ss.study_id,
        "prolific_session_id": ss.prolific_session_id,
        "condition": ss.condition,
        "condition_label": CONDITIONS[ss.condition]["label"],
        "assignment_method": ss.assignment_method,
        "color_map": json.dumps(ss.color_map),
        "phase1_topic_order": json.dumps(ss.topic_order.get("1", [])),
        "phase2_topic_order": json.dumps(ss.topic_order.get("2", [])),
        "article_split": json.dumps(ss.split),
        "opened_at_utc": ss.opened_at,
        "consent_at_utc": ss.consent_at,
        "started_at_utc": ss.started_at,
        "completed_at_utc": ss.completed_at,
        "duration_min": duration,
        "n_articles": len(resp),
        "n_readmore": sum(1 for r in resp if r["readmore_clicked"]),
        "n_full": sum(1 for r in resp if r["full_clicked"]),
        "n_likes": sum(1 for r in resp if r["liked_final"]),
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
    n_pages = 2 * len(data["topics"])
    n_per_page = len(data["outlets"])
    ribbons = "".join(f'<span style="background:{c}"></span>' for c in OUTLET_PALETTE[:n_per_page])
    contact = html_lib.escape(RESEARCHER_CONTACT)
    ethics = html_lib.escape(ETHICS_STATEMENT)
    st.html(f"""
<section class="nps-intro">
  <h1 class="nps-title">News Perception Study</h1>
  <p class="nps-lead">You'll browse {number_word(n_pages)} short news feeds with {number_word(n_per_page)} articles each. Read them the way you'd read news online: open what interests you, like what you enjoy, and move on whenever you want.</p>
  <p class="nps-sub">It takes about {ESTIMATED_MINUTES} minutes. Please finish in one sitting and don't refresh the page.</p>

  <h2 class="nps-h2">How it works</h2>
  <ul class="nps-rows">
    <li class="nps-row">
      <div class="nps-token"><span class="tok-more">…&nbsp;Read more</span></div>
      <p>Each article opens with a headline and a short preview. Click <b>Read more</b> to keep reading. After that, <b>Read full article</b> shows the whole text.</p>
    </li>
    <li class="nps-row">
      <div class="nps-token tok-ribbons">{ribbons}</div>
      <p>The coloured marker in the top-left corner shows which news outlet published the article. Each colour is a different outlet.</p>
    </li>
    <li class="nps-row">
      <div class="nps-token"><span class="tok-like"><span class="tok-thumb" aria-hidden="true"></span><span>Like</span></span></div>
      <p>Click <b>Like</b> on any article you like. Click it again to undo.</p>
    </li>
    <li class="nps-row">
      <div class="nps-token"><span class="tok-next">Continue</span></div>
      <p>When you've finished a feed, click <b>Continue</b>. You can't go back to earlier feeds.</p>
    </li>
  </ul>
  <p class="nps-plain">There's no right or wrong way to do this. Reading, liking and skipping are all equally useful to us.</p>

  <div class="nps-consent">
    <h2 class="nps-h2">Your participation</h2>
    <p>Taking part is voluntary. You can stop at any time by closing this window; if you do, none of your responses will be stored.</p>
    <p>We record how you interact with the articles (which ones you open and like, and how long you spend on each feed) together with your Prolific ID. The data are stored securely and used only for research.</p>
    <p class="nps-contact">Contact: {contact}<br>{ethics}</p>
  </div>
</section>""")

    consent = st.checkbox("I have read this information and agree to take part.", key="consent_box")
    if st.button("Start the study", type="primary", disabled=not consent, key="start_button"):
        st.session_state.consent_at = utc_iso()
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
        title = art["title_rewritten"] if (rewritten and art["title_rewritten"]) else art["title"]
        cards.append({
            "uid": uid,
            "color": ss.color_map[art["outlet"]] if page["source_visible"] else None,
            "title": title or "",
            "sentences": art["sentences_rewritten"] if rewritten else art["sentences_original"],
        })

    result = feed_component(
        page_id=page_id,
        page_number=page["page_number"],
        total_pages=len(ss.pages),
        is_last=page["page_number"] == len(ss.pages),
        articles=cards,
        snippet_sentences=SNIPPET_SENTENCES,
        readmore_sentences=READMORE_SENTENCES,
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

    mtime = os.path.getmtime(ARTICLES_CSV) if os.path.exists(ARTICLES_CSV) else 0.0
    data, errors = load_study_data(ARTICLES_CSV, mtime)
    if errors:
        render_setup_error(errors)
        return

    stage = st.session_state.stage
    if stage == "intro":
        render_intro(data)
    elif stage == "feed":
        render_feed(data)
    else:
        render_end()


main()