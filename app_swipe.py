"""
News Perception Study
=====================

Streamlit app for the news-feed experiment: an outlet survey, then two feeds.

    Outlets, step 1:     the participant sees the SURVEY_OUTLETS (logo and
                         name, in random order) and says for each one whether
                         they know it.
    Outlets, step 2:     for each outlet they know, a Likert scale: how much
                         they would like to come across it in their news feed.
                         Skipped when they know none of them.
    Phase 1 (feed 1):    every participant sees the outlet logo and the
                         original text.
    Phase 2 (feed 2):    the between-subject treatment (2 x 2):
                         source visible / hidden  x  original / LLM-rewritten.

The article set has 3 topics x 4 outlets x 3 articles = 36 articles:
PHASE1_ARTICLES (18) in phase 1 and the remaining 18 in phase 2. Both phases
are balanced by outlet, topic, reliability class and topic x reliability
class: each gets half of every group, rounded down or up (e.g. 4 or 5
articles per outlet, 6 per topic, 9 reliable and 9 unreliable). For each
participant the split is drawn at random among all the balanced ones (see
balanced_counts and compute_split_options); within each topic x outlet cell,
which articles go to which phase is random too.

FEED_STYLE picks how a feed is shown: "scroll" (app.py) or "swipe"
(app_swipe.py: one card at a time, swiped left or right, or Next / arrow
keys, to move on; the swipe has no meaning beyond moving on, liking stays the
Like button; each time a card is on screen is saved as a view with its time
on screen). The two files differ only in that line.

Each phase is ONE scrolling feed with all its articles, in an order
randomised independently for each phase and each participant. So that the
feed never seems to end, the same sequence is repeated REPEAT_ROUNDS times
(default 10).

The state of an article is shared by all its copies: if the participant
opens it ("Read more" / "Read full article") or likes it on one copy, every
other copy of that article shows it opened / liked too.

Interaction per article:
    preview -> "... Read more" -> "... Read full article" (whole text),
    plus an optional Like toggle.

The "Continue" / "Finish" button sits in a bar fixed at the bottom of the
screen (always reachable, however far the participant scrolls). It unlocks
only after MIN_FEED_SECONDS (default 2 minutes), but the countdown appears
on it only during the last COUNTDOWN_VISIBLE_SECONDS, so the feed never
looks like a timed task. Before each feed a "loading the feed" wait runs
(FIRST_FEED_LOADING_SECONDS before feed 1, BREAK_LOADING_SECONDS before
feed 2).

Each card shows the headline that goes with the text version it displays:
the `Title` column with the original text, the `LLM_title` column with the
LLM-rewritten text (from the `LLM_text` column).

Outlet logos: image files in data/logos/, one per outlet, named after the
outlet: for the feed outlets as written in the CSV `domain` column, for the
survey outlets as written in SURVEY_OUTLETS (e.g. data/logos/reuters.com.png).
PNG, SVG, JPG and WEBP are accepted. If a logo is missing, the app shows the
exact file names it expects. LOGO_FILES can map an outlet to a different
file name.

Prolific integration:
    - PROLIFIC_PID, STUDY_ID and SESSION_ID are read from the URL, so the
      participant never types their ID. Without PROLIFIC_PID the study does
      not start (REQUIRE_PROLIFIC_PID; DEV_MODE skips the check).
    - A PROLIFIC_PID that already has a completed session in SHEET_SESSIONS
      is blocked (BLOCK_DUPLICATE_PID).
    - At the end the completion code is shown on screen, with a button that
      returns to Prolific with the code already filled in. The code is read
      from secrets ([prolific] completion_code), or PROLIFIC_COMPLETION_CODE.

Run locally:     streamlit run app.py
Documentation:   app_documentation.txt
"""

import hashlib
import html as html_lib
import itertools
import json
import math
import os
import random
import re
import string
import base64
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from fractions import Fraction
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

APP_VERSION = "2026-11-04.2"

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTICLES_CSV = os.path.join(DATA_DIR, "articoli_selezionati - articoli scelti.csv")
LOCAL_BACKUP_DIR = os.path.join(DATA_DIR, "responses")

# Outlet logos: one image per outlet in this folder, named after the outlet
# exactly as written in the CSV `domain` column, e.g. "nytimes.com.png".
# Matching ignores upper/lower case and a leading "www."; spaces and other
# symbols may also be written as "_" or "-". If a logo is missing, the app
# lists the file names it expects.
LOGO_DIR = os.path.join(DATA_DIR, "logos")
LOGO_EXTENSIONS = (".png", ".svg", ".webp", ".jpg", ".jpeg", ".gif", ".avif")
# Optional: outlet -> file name inside LOGO_DIR, when a file can't be named
# after the outlet. Example: {"www.nytimes.com": "nyt.png"}
LOGO_FILES = {}
# Logos larger than this trigger a warning in the log (they still work).
LOGO_MAX_KB = 400
# Logo height on the cards, in pixels (the width follows the image), and
# corner rounding, in pixels (0 = square corners; half the height = pill /
# circle for square logos). On phones the logo is shown at 80% of this size.
LOGO_HEIGHT_PX = 104
LOGO_RADIUS_PX = 18
# Logo height in the outlet survey (both steps) and in the intro, in pixels.
SURVEY_LOGO_HEIGHT_PX = 96
INTRO_LOGO_HEIGHT_PX = 64
# Intro ("How it works"): how many example logos to show. They are drawn at
# random, for each participant, among ALL the logo images in LOGO_DIR (the
# feed outlets, the survey outlets and any extra logo you put there), so the
# intro doesn't reveal which outlets the feeds contain.
INTRO_LOGO_COUNT = 4
# Names for extra logos that are in LOGO_DIR but in neither the feeds nor
# SURVEY_OUTLETS (shown under them in the intro). Without a name the domain
# is shown. Example: {"nytimes.com": "The New York Times"}
EXTRA_OUTLET_NAMES = {}

# ---- outlet survey (before the feeds) ----------------------------------------
# The outlets shown in the survey: domain (also the logo file name, as for
# the feed outlets), the name shown under the logo, and reference data that
# goes into the saved rows only (never shown to the participant).
# NewsGuard ratings from newsguard_22_02_24.csv (T = trustworthy,
# N = not trustworthy, score 0-100, threshold 60).
SURVEY_OUTLETS = [
    {"domain": "bbc.com",         "name": "BBC",          "group": "feed",         "ng_rating": "T", "ng_score": 95.0},
    {"domain": "cnn.com",         "name": "CNN",          "group": "feed",         "ng_rating": "T", "ng_score": 80.0},
    {"domain": "theblaze.com",    "name": "TheBlaze",     "group": "feed",         "ng_rating": "N", "ng_score": 49.5},
    {"domain": "rt.com",          "name": "RussiaToday",  "group": "feed",         "ng_rating": "N", "ng_score": 20.0},
    {"domain": "reuters.com",     "name": "Reuters",      "group": "famous",       "ng_rating": "T", "ng_score": 100.0},
    {"domain": "theguardian.com", "name": "The Guardian", "group": "famous",       "ng_rating": "T", "ng_score": 100.0},
    {"domain": "infowars.com",    "name": "Infowars",     "group": "famous",       "ng_rating": "N", "ng_score": 7.5},
    {"domain": "zerohedge.com",   "name": "ZeroHedge",    "group": "famous",       "ng_rating": "N", "ng_score": 15.0},
    {"domain": "capegazette.com", "name": "Cape Gazette", "group": "lesser_known", "ng_rating": "T", "ng_score": 100.0},
    {"domain": "mintpressnews.com", "name": "MintPress News", "group": "lesser_known", "ng_rating": "N", "ng_score": 22.5},
]
# True: the outlets appear in a random order, drawn independently for each
# participant and for each step.
SHUFFLE_SURVEY_OUTLETS = True

FAMILIARITY_TITLE = "Do you know these news outlets?"
FAMILIARITY_LEAD = ("For each outlet, tell us whether you know it. It's completely fine "
                    "not to know some of them: just answer honestly.")
FAMILIARITY_OPTIONS = [("yes", "I know it"), ("no", "I don't know it")]

PREFERENCE_TITLE = "Would you like to see them in your feed?"
PREFERENCE_LEAD = ("For each outlet you know, tell us how much you would like to come "
                   "across its articles in your news feed.")
# Likert scale: one value per point, with labels at the two ends and in the
# middle (an odd number of points puts the middle label on the centre point).
PREFERENCE_POINTS = 7
PREFERENCE_ANCHORS = ("Definitely not", "Neutral", "Definitely yes")

# CSV column names. For each field the app uses the first name it finds.
# Edit these lists if your CSV uses different names.
COLUMN_CANDIDATES = {
    "id": ["ID", "id", "article_id"],
    "topic": ["topic"],
    "outlet": ["domain", "outlet", "source"],
    "title": ["Title", "title", "headline"],
    "title_rewritten": ["LLM_title", "llm_title", "rewritten_title", "title_rewritten",
                        "ai_title", "Title_rewritten"],
    "text": ["clean_text", "text"],
    "text_rewritten": ["LLM_text", "llm_text", "rewritten_text", "text_rewritten",
                       "clean_text_rewritten", "ai_text", "rewritten"],
    "reliability": ["reliability_class", "reliability_label"],
    "rating": ["rating"],
    "year": ["year"],
    "month": ["month"],
}
REQUIRED_FIELDS = ("topic", "outlet", "title", "title_rewritten", "text", "text_rewritten")

ARTICLES_PER_CELL = 3      # articles per outlet per topic (3 topics x 4 outlets x 3 = 36)

# Phase lengths: phase 1 has PHASE1_ARTICLES and phase 2 all the others,
# 18 + 18 = 36. With 4 outlets that is 4 or 5 articles per outlet per phase.
PHASE1_ARTICLES = 18

# Safety limit on how many candidate splits are checked at start-up.
MAX_SPLIT_COMBOS = 2 ** 16

# Text visible at each stage, in characters (spaces included). The cut is
# moved back to the nearest word boundary, so no word is ever split.
PREVIEW_CHARS = 144         # visible before "Read more"
READMORE_CHARS = 500       # visible after "Read more" (total)

# How many times the article sequence of a feed is repeated, one after the
# other, so that the feed never seems to end (18 articles x 10 = 180 cards
# in each feed).
REPEAT_ROUNDS = 10

# Minimum time on each feed before "Continue" / "Finish" can be clicked.
# Set to 0 to disable (e.g. while testing).
MIN_FEED_SECONDS = 120

# How long before the unlock the countdown becomes visible on the button.
# Until then the button just reads "Continue" / "Finish" (greyed out), so
# the participant never sees a clock running against them.
# Set to 0 to never show the countdown.
COUNTDOWN_VISIBLE_SECONDS = 10

# "Loading the feed" screen before each feed: how long it lasts before the
# participant can go on. FIRST_FEED_LOADING_SECONDS comes after the outlet
# survey, BREAK_LOADING_SECONDS between the two feeds.
FIRST_FEED_LOADING_SECONDS = 3
BREAK_LOADING_SECONDS = 10

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
DEV_MODE = False           # True: ?condition=1..4 in the URL forces a condition,
                           # and the study starts even without PROLIFIC_PID

ESTIMATED_MINUTES = "15–20"
PROLIFIC_RETURN_URL = "https://app.prolific.com/submissions/complete?cc={code}"

# Prolific ID read from the URL (?PROLIFIC_PID=...&STUDY_ID=...&SESSION_ID=...).
# True: without PROLIFIC_PID the study does not start (DEV_MODE skips this check).
REQUIRE_PROLIFIC_PID = True

# True: a PROLIFIC_PID that already has a completed session in SHEET_SESSIONS
# cannot take the study again.
BLOCK_DUPLICATE_PID = True

# Fallback completion code, used only when [prolific] completion_code is
# missing from secrets.toml. Leave "" to rely on secrets only.
PROLIFIC_COMPLETION_CODE = ""

# Feed style: "scroll" = one scrolling feed with all the cards (app.py);
# "swipe" = one card at a time, swiped left or right (or Next / arrow keys)
# to see the next one, Tinder-style (app_swipe.py). The swipe only moves on:
# liking stays the Like button. Everything else is identical.
FEED_STYLE = "swipe"

# New tab names (v5): the practice phase is gone and the outlet survey is
# new, so the data go to fresh tabs instead of being mixed with v4 data.
# The swipe version writes to its own tabs (suffix "_swipe"), and balances
# the conditions on its own participants.
_SHEET_SUFFIX = "_swipe" if FEED_STYLE == "swipe" else ""
SHEET_ASSIGNMENTS = "assignments_v5" + _SHEET_SUFFIX
SHEET_SESSIONS = "sessions_v5" + _SHEET_SUFFIX
SHEET_RESPONSES = "responses_v5" + _SHEET_SUFFIX
SHEET_OUTLETS = "outlets_v5" + _SHEET_SUFFIX     # one row per participant x survey outlet

ASSIGNMENT_HEADERS = ["session_id", "prolific_pid", "condition", "assigned_at_utc"]

SESSION_HEADERS = [
    "session_id", "prolific_pid", "study_id", "prolific_session_id",
    "condition", "condition_label", "assignment_method",
    "logo_files", "intro_logos", "phase1_order", "phase2_order", "article_split",
    "familiarity_order", "preference_order", "n_outlets_known",
    "familiarity_duration_ms", "preference_duration_ms",
    "opened_at_utc", "started_at_utc", "completed_at_utc",
    "duration_min", "n_articles", "n_readmore", "n_full", "n_likes",
    "phase1_max_round", "phase2_max_round",
    "feed_style", "user_agent", "app_version",
]

RESPONSE_HEADERS = [
    "session_id", "prolific_pid", "condition", "condition_label",
    "phase", "page_number", "topic", "position",
    "article_uid", "outlet", "reliability_class", "logo_file",
    "source_visible", "text_version", "title_shown", "n_sentences", "n_chars",
    "readmore_available", "readmore_clicked", "readmore_ms", "readmore_ts", "readmore_round",
    "full_available", "full_clicked", "full_ms", "full_ts", "full_round",
    "liked_final", "like_toggles", "like_history",
    "rounds_shown", "max_round_reached",
    # swipe version only (empty in the scroll version): how many times the
    # card was on screen, total time on screen, swipes by direction, and
    # every view as JSON [{round, enter_ms, exit_ms, dir, method}].
    "n_views", "dwell_total_ms", "dwell_first_ms", "n_swipe_left", "n_swipe_right", "views",
    "page_enter_ts", "page_exit_ts", "page_duration_ms", "page_duration_server_s",
    "year", "month", "rating",
]

OUTLET_HEADERS = [
    "session_id", "prolific_pid", "condition", "condition_label",
    "outlet", "outlet_name", "outlet_group", "in_feed", "newsguard_rating", "newsguard_score",
    "logo_file",
    "familiarity_position", "familiar", "familiarity_first_ms", "familiarity_last_ms",
    "familiarity_changes",
    "preference_shown", "preference_position", "preference", "preference_label",
    "preference_first_ms", "preference_last_ms", "preference_changes",
    "familiarity_enter_ts", "familiarity_exit_ts",
    "preference_enter_ts", "preference_exit_ts",
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
# 4. ARTICLES: LOADING, VALIDATION AND PHASE SPLIT
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


def _near(x):
    """x rounded down and up (a single value when x is a whole number)."""
    return sorted({math.floor(x), math.ceil(x)})


def balanced_counts(keys, sizes, outlet_class, n):
    """All the ways to take `n` articles from the cells so that the selection
    is balanced.

    `keys` lists the cells as (topic, outlet) and `sizes` how many articles
    each cell still has. A selection says how many articles to take from each
    cell; it is admissible when it has exactly `n` articles and every outlet,
    topic, reliability class and topic x reliability class gets its
    proportional share (share = n / total), rounded down or up. Each cell
    also gets its own share, rounded down or up.

    `outlet_class` maps outlet -> reliability class, or is None when the
    reliability classes can't be balanced. Returns (options, error)."""
    total = sum(sizes)
    if not 0 < n <= total:
        return [], f"Cannot take {n} articles out of {total}."

    share = Fraction(n, total)
    choices = [[c for c in _near(size * share) if 0 <= c <= size] for size in sizes]

    n_combos = 1
    for c in choices:
        n_combos *= len(c)
    if n_combos > MAX_SPLIT_COMBOS:
        return [], (f"Too many candidate splits to check ({n_combos}); "
                    f"raise MAX_SPLIT_COMBOS or use fewer cells.")

    groups = {}
    for i, (t, o) in enumerate(keys):
        groups.setdefault(("outlet", o), []).append(i)
        groups.setdefault(("topic", t), []).append(i)
        if outlet_class:
            groups.setdefault(("reliability", outlet_class[o]), []).append(i)
            groups.setdefault(("topic x reliability", t, outlet_class[o]), []).append(i)
    limits = [(idx, set(_near(sum(sizes[i] for i in idx) * share)))
              for idx in groups.values()]

    options = []
    for combo in itertools.product(*choices):
        if sum(combo) != n:
            continue
        if all(sum(combo[i] for i in idx) in allowed for idx, allowed in limits):
            options.append(tuple(combo))
    if not options:
        return [], (f"No balanced way to take {n} of {total} articles while keeping "
                    f"outlets, topics and reliability classes balanced.")
    return options, None


def compute_split_options(articles, cells, topics, outlets, phase1_n):
    """All the admissible ways to split the articles between the two phases.

    Phase 1 gets `phase1_n` articles taken, balanced, from the whole set (see
    balanced_counts); phase 2 gets the rest, which is then balanced too.
    With 36 articles and phase1_n = 18: 4 or 5 articles per outlet in each
    phase, 6 per topic, 9 reliable and 9 unreliable, and 3 per topic x
    reliability class.

    Returns (cell_keys, options, error). `cell_keys` lists the cells as
    (topic, outlet); each option has one phase-1 count per cell. Phase 2 gets
    size - phase1 per cell."""
    total = len(articles)
    if not 0 < phase1_n < total:
        return [], [], (f"PHASE1_ARTICLES must be between 1 and {total - 1} "
                        f"(found {phase1_n}).")

    keys = [(t, o) for t in topics for o in outlets]
    sizes = [len(cells[t][o]) for t, o in keys]

    # Reliability is balanced only when each outlet has a single known class
    # and there are at least two classes.
    outlet_class = {}
    for o in outlets:
        classes = {a["reliability"] for a in articles.values() if a["outlet"] == o}
        outlet_class[o] = next(iter(classes)) if len(classes) == 1 else None
    use_reliability = (all(c not in (None, "NA") for c in outlet_class.values())
                       and len(set(outlet_class.values())) > 1)
    oc = outlet_class if use_reliability else None

    options, error = balanced_counts(keys, sizes, oc, phase1_n)
    if error:
        return keys, [], (f"Phase 1: {error} Change PHASE1_ARTICLES or the article set.")
    return keys, options, None


# ---- outlet logos -----------------------------------------------------------

_LOGO_MIME = {".png": "image/png", ".svg": "image/svg+xml", ".webp": "image/webp",
              ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
              ".avif": "image/avif"}


def _logo_key(name):
    """Normalised name used to match a logo file to an outlet: lower case,
    no leading "www." (also "www_", "www-" or a typo like "wwww."), and every run of
    symbols turned into "_"."""
    name = str(name).strip().lower()
    name = re.sub(r"^(https?://)?w{3,}[._\-\s]+", "", name)   # also "wwww." typos
    return re.sub(r"[^a-z0-9.]+", "_", name).strip("_")


def _logo_tokens(key):
    """The words of a normalised name: "infowars.com_logo" -> {infowars, com, logo}."""
    return {t for t in re.split(r"[._]+", key) if t}


def _logo_index():
    """Signature of the logo folder (names, sizes, mtimes), used as cache key."""
    if not os.path.isdir(LOGO_DIR):
        return ()
    out = []
    for f in sorted(os.listdir(LOGO_DIR)):
        path = os.path.join(LOGO_DIR, f)
        if os.path.isfile(path):
            st_ = os.stat(path)
            out.append((f, st_.st_size, st_.st_mtime))
    return tuple(out)


@st.cache_data(show_spinner=False)
def load_logos(outlets, index):
    """Find and read the logo of each outlet.

    Returns (logos, missing): logos maps outlet -> {"file", "src"}, where src
    is a data: URI; missing lists the outlets with no logo file.
    `index` (see _logo_index) is part of the cache key, so adding or
    replacing a logo reloads it."""
    files = {}
    for f, _, _ in index:
        stem, ext = os.path.splitext(f.strip())
        if ext.lower() in LOGO_EXTENSIONS:
            files.setdefault(_logo_key(stem), f)
    names = {f for f, _, _ in index}

    def loose_match(key):
        """Fallback for names like "www.infowars.com_logo.png" or
        "Infowars-logo.png": the one image whose name contains the outlet's
        name ("infowars") as a whole word. None if zero or several match."""
        base = key.split(".")[0]
        hits = [f for k, f in files.items() if base and base in _logo_tokens(k)]
        if len(hits) > 1:
            exact = [f for k, f in files.items() if k.startswith(key)]
            hits = exact if len(exact) == 1 else hits
        if len(hits) > 1:
            log(f"WARNING: several logo files could be '{key}': {hits}; use LOGO_FILES.")
        return hits[0] if len(hits) == 1 else None

    logos, missing = {}, []
    for o in outlets:
        f = LOGO_FILES.get(o)
        if f and f not in names:
            log(f"WARNING: LOGO_FILES['{o}'] = '{f}' not found in {LOGO_DIR}")
            f = None
        if not f:
            # "nytimes.com" matches nytimes.com.png and also nytimes.png
            key = _logo_key(o)
            f = files.get(key) or files.get(key.rsplit(".", 1)[0]) or loose_match(key)
        if not f:
            missing.append(o)
            continue
        path = os.path.join(LOGO_DIR, f)
        with open(path, "rb") as fh:
            raw = fh.read()
        if len(raw) > LOGO_MAX_KB * 1024:
            log(f"WARNING: logo {f} is {len(raw) // 1024} KB; a smaller file loads faster.")
        mime = _LOGO_MIME[os.path.splitext(f.strip())[1].lower()]
        logos[o] = {"file": f,
                    "src": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"}
    return logos, missing


@st.cache_data(show_spinner=False)
def load_logo_pool(index):
    """Every logo image in LOGO_DIR, one per outlet (duplicates of the same
    outlet, e.g. a .png and a .svg, count once): [{"key", "file", "src"}]."""
    pool, seen = [], set()
    for f, _, _ in index:
        stem, ext = os.path.splitext(f.strip())
        if ext.lower() not in LOGO_EXTENSIONS:
            continue
        key = _logo_key(stem)
        if key in seen:
            continue
        seen.add(key)
        with open(os.path.join(LOGO_DIR, f), "rb") as fh:
            raw = fh.read()
        mime = _LOGO_MIME[ext.lower()]
        pool.append({"key": key, "file": f,
                     "src": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"})
    return pool


def logo_hint(outlet):
    """Expected logo file name for an outlet, for error messages."""
    return f"{_logo_key(outlet)}.png"


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

        # IDs read as floats (because some are missing) become "12", not "12.0".
        raw_id = val("id") if cols["id"] else None
        if isinstance(raw_id, float) and raw_id.is_integer():
            raw_id = int(raw_id)
        uid = str(raw_id).strip() if raw_id is not None else f"a{i:02d}"
        if raw_id is None and cols["id"]:
            warnings.append(f"Row {i} has no id: using '{uid}'.")

        text = str(val("text") or "").strip()
        text_rw = str(val("text_rewritten") or "").strip()
        title = str(val("title") or "").strip()
        title_rw = str(val("title_rewritten") or "").strip()

        if uid in articles:
            errors.append(f"Duplicate article id: {uid}.")
        if not title:
            errors.append(f"Article {uid} has no title ('{cols['title']}' column).")
        if not title_rw:
            errors.append(f"Article {uid} has no rewritten title ('{cols['title_rewritten']}' column).")
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
            "title_rewritten": title_rw,
            "sentences_original": split_sentences(text),
            "sentences_rewritten": split_sentences(text_rw),
            "reliability": str(val("reliability")).strip() if val("reliability") is not None else None,
            "rating": val("rating"),
            "year": val("year"),
            "month": val("month"),
        }

    titles_seen = {}
    for a in articles.values():
        if a["title"] and a["title"] in titles_seen:
            warnings.append(f"Articles {titles_seen[a['title']]} and {a['uid']} have the same title.")
        titles_seen.setdefault(a["title"], a["uid"])

    topics = sorted({a["topic"] for a in articles.values()})
    outlets = sorted({a["outlet"] for a in articles.values()})

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

    # Admissible phase splits (checked once here, drawn per participant).
    split_cells, split_options = [], []
    if not errors:
        split_cells, split_options, split_error = compute_split_options(
            articles, cells, topics, outlets, PHASE1_ARTICLES)
        if split_error:
            errors.append(split_error)
        else:
            n2 = len(articles) - PHASE1_ARTICLES
            log(f"{len(split_options)} balanced phase splits available "
                f"({PHASE1_ARTICLES} + {n2} articles).")

    for w in warnings:
        log(f"WARNING: {w}")

    data = {"articles": articles, "topics": topics, "outlets": outlets,
            "cells": cells, "columns": cols,
            "split_cells": split_cells, "split_options": split_options}
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


def pid_already_completed(prolific_pid):
    """True if this PROLIFIC_PID already has a completed session, False if not,
    None when the check could not be done (no storage or read error).

    SHEET_SESSIONS gets a row only when a participant finishes the study, so
    a match there means the study was already completed with this ID."""
    pid = (prolific_pid or "").strip().lower()
    if not pid:
        return False
    store = get_store()
    if store is None:
        return None
    try:
        rows = store.read(SHEET_SESSIONS, SESSION_HEADERS)
    except Exception as exc:
        log(f"Duplicate PID check failed: {exc}")
        return None
    return any(str(r.get("prolific_pid", "")).strip().lower() == pid for r in rows)


def check_access():
    """Decide once per session whether this participant may start.
    Returns 'ok', 'ok_unchecked', 'missing_pid' or 'duplicate'."""
    ss = st.session_state
    if not ss.prolific_pid:
        return "ok" if (DEV_MODE or not REQUIRE_PROLIFIC_PID) else "missing_pid"
    if BLOCK_DUPLICATE_PID:
        done = pid_already_completed(ss.prolific_pid)
        if done:
            log(f"Blocked duplicate PROLIFIC_PID {ss.prolific_pid}")
            return "duplicate"
        if done is None:
            log(f"Duplicate check not possible for {ss.prolific_pid}: letting them in")
            return "ok_unchecked"
    return "ok"


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
  /* Source visible: the outlet logo and name sit at the top of the card.
     The name goes beside the logo, or below it when the logo is too wide. */
  .source { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px;
            margin: -4px 0 18px; padding: 0; }
  .source img { display: block; flex: none; height: var(--logo-h, 80px); width: auto;
                max-width: min(100%, 480px);
                object-fit: contain; object-position: left center;
                border-radius: var(--logo-r, 16px); overflow: hidden; }
  .source-name { font-family: var(--sans); font-weight: 600; font-size: 18px; line-height: 1.25;
                 color: var(--ink); }

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
  .actions { display: flex; flex-direction: column; align-items: center; gap: 9px; margin: 0 0 20px; }
  /* min-height keeps the button still when the hint fades in. */
  .hint { margin: 0; font-size: 13.5px; line-height: 1.4; min-height: 1.4em; color: var(--ink-3);
          text-align: center; opacity: 0; transition: opacity .35s ease; }
  .hint.on { opacity: 1; }
  .next { appearance: none; border: 0; height: 48px; padding: 0 30px; border-radius: 24px;
          background: var(--ink); color: #FFFFFF; font: 600 16px/1 var(--sans); cursor: pointer;
          min-width: 215px; text-align: center;
          font-variant-numeric: tabular-nums; transition: background-color .15s; }
  .next[aria-disabled="true"], .next:disabled { background: #8B9097; cursor: default; }

  button:focus-visible { outline: 2px solid var(--ink); outline-offset: 3px; }

  @media (max-width: 560px) {
    .card { padding: 18px 18px 16px; }
    .source img { height: calc(var(--logo-h, 80px) * 0.8); max-width: 100%;
                  border-radius: calc(var(--logo-r, 16px) * 0.8); }
    .source-name { font-size: 17px; }
    .headline { font-size: 19px; }
    .body { font-size: 16.5px; }
    .actions { align-items: stretch; }
    .next { width: 100%; }
  }
  /* ---------- swipe mode: one card at a time ---------- */
  .deck-hint { margin: 0 0 12px; font-size: 14px; line-height: 1.4; color: var(--ink-3); text-align: center; }
  /* Bottom margin: room for the card shadow inside this frame. */
  .deck { position: relative; height: var(--deck-h, 560px); margin: 0 auto 28px; max-width: 620px; }
  .deck .card { position: absolute; inset: 0; padding: 0; display: flex; flex-direction: column;
                overflow: hidden; touch-action: pan-y; user-select: none; -webkit-user-select: none;
                box-shadow: 0 12px 22px -12px rgba(25, 28, 32, .22); will-change: transform;
                cursor: grab; }
  .deck .card.dragging { cursor: grabbing; transition: none !important; }
  .deck .card.settle { transition: transform .25s cubic-bezier(.2, .7, .2, 1); }
  .deck .card.fly { transition: transform .3s ease-in, opacity .3s ease-in; opacity: 0; }
  .deck .card.peek { transform: translateY(12px) scale(.955); box-shadow: none; cursor: default;
                     transition: transform .3s cubic-bezier(.2, .7, .2, 1); }
  .deck .card.peek.up { transform: none; }
  .deck .card.peek > * { visibility: hidden; }
  /* Horizontal finger movement must reach the drag code: no element in the
     card may pan sideways (the scrolling text box included). */
  .deck .card, .deck .card * { touch-action: pan-y; }
  .card-scroll { flex: 1; min-height: 0; overflow-x: hidden; overflow-y: auto; overscroll-behavior: contain;
                 padding: 22px 26px 18px; -webkit-overflow-scrolling: touch; }
  .card-scroll .body { user-select: text; -webkit-user-select: text; }
  .card-foot { flex: none; display: flex; align-items: center; justify-content: space-between;
               gap: 12px; padding: 12px 26px; border-top: 1px solid var(--line); background: var(--surface); }
  .deck-next { appearance: none; display: inline-flex; align-items: center; gap: 8px; height: 36px;
               padding: 0 14px 0 16px; border-radius: 18px; border: 1px solid var(--line);
               background: var(--surface); color: var(--ink-2); font: 500 14px/1 var(--sans);
               cursor: pointer; }
  .deck-next:hover { border-color: var(--line-strong); color: var(--ink); }
  .deck-next svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 2;
                   stroke-linecap: round; stroke-linejoin: round; }
  .deck-end { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center;
              justify-content: center; text-align: center; padding: 24px; background: var(--surface);
              border: 1px solid var(--line); border-radius: 12px; }
  .deck-end p { margin: 0; font-size: 16px; line-height: 1.5; color: var(--ink-2); max-width: 26em; }
  @media (max-width: 560px) {
    .card-scroll { padding: 18px 18px 16px; }
    .card-foot { padding: 10px 18px; }
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

  var NEXT_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></svg>';
  var SWIPE_HINT = "Swipe the card left or right to see the next article.";
  var THUMB = '__THUMB_SVG__';
  var BAR_ID = "nps-feedbar";
  // Shown once the button unlocks, and when it is clicked too early.
  var UNLOCK_HINT = "Take your time \u2014 you can now proceed whenever you want.";
  var LOCKED_HINT = "You'll be able to continue in a moment.";
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

  // Logos arrive as data: URIs; each becomes one blob URL, shared by all
  // the cards of that outlet (no copy of the image per card).
  function logoUrls(map) {
    var out = {};
    for (var key in (map || {})) {
      var src = map[key].src, url = src;
      try {
        var comma = src.indexOf(",");
        var mime = src.slice(5, src.indexOf(";"));
        var bin = atob(src.slice(comma + 1));
        var bytes = new Uint8Array(bin.length);
        for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        url = URL.createObjectURL(new Blob([bytes], { type: mime }));
      } catch (err) { url = src; }
      out[key] = { url: url, alt: map[key].alt || "", name: map[key].name || "" };
    }
    return out;
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
      var hint = d.createElement("p");
      hint.className = "nps-feedbar-hint";
      var btn = d.createElement("button");
      btn.type = "button";
      btn.className = "nps-feedbar-btn";
      inner.appendChild(hint);
      inner.appendChild(btn);
      bar.appendChild(inner);
      d.body.appendChild(bar);
      return { btn: btn, hint: hint };
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
    if (!state.swipe && Math.abs(shift) >= 1) {
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
    // Swipe mode: the content scrolls inside the card and the Like / Next
    // buttons sit in a footer that is always visible.
    var box = card;
    if (state.swipe) {
      box = el("div", "card-scroll");
      card.appendChild(box);
      a.instances = [];       // only the card on screen is kept in sync
      a.likeButtons = [];
    }
    var logo = a.logo ? state.logos[a.logo] : null;
    if (logo) {
      card.classList.add("sourced");
      var src = el("div", "source");
      var img = el("img");
      img.src = logo.url;
      img.alt = logo.name ? "" : logo.alt;   // the name next to it already says it
      img.decoding = "async";
      img.draggable = false;
      src.appendChild(img);
      if (logo.name) src.appendChild(el("span", "source-name", logo.name));
      box.appendChild(src);
    }
    if (a.title) box.appendChild(el("h2", "headline", a.title));
    var inst = { round: round, card: card, bodyEl: el("div", "body") };
    box.appendChild(inst.bodyEl);
    var foot = el("div", state.swipe ? "card-foot" : "foot");
    foot.appendChild(buildLike(a, inst));
    if (state.swipe) {
      var nb = el("button", "deck-next");
      nb.type = "button";
      nb.innerHTML = "<span>Next</span>" + NEXT_SVG;
      nb.addEventListener("click", function () { flyOut(1, "button"); });
      foot.appendChild(nb);
    }
    card.appendChild(foot);
    a.instances.push(inst);
    renderBody(a, inst, Number.MAX_SAFE_INTEGER);
    card.setAttribute("data-round", String(round));
    if (viewObserver) viewObserver.observe(card);
    return card;
  }

  // ---- swipe mode ----------------------------------------------------------
  // One card at a time. The deck is the article sequence repeated `rounds`
  // times. Dragging the card left or right past SWIPE_PX (or flicking it),
  // the Next button and the arrow keys all move to the next card; the
  // direction is only recorded, it has no meaning. Every time a card is on
  // screen counts as one view of that article: {round, enter_ms, exit_ms,
  // dir (-1 left / 1 right / null), method (drag / button / key / continue)}.
  var SWIPE_PX = 90, FLICK_PX_PER_MS = 0.6;

  function deckHeight() {
    var ph = 0;
    try { ph = window.parent.innerHeight; } catch (e) { ph = 0; }
    if (!ph) ph = 800;
    // Room for the page padding, the progress bar, the hint and the fixed
    // Continue bar at the bottom.
    return Math.max(340, Math.min(760, ph - 300));
  }
  function sizeDeck() {
    if (state && state.swipe) root.style.setProperty("--deck-h", deckHeight() + "px");
  }
  try { window.parent.addEventListener("resize", function () { sizeDeck(); updateHeight(); }); } catch (e) {}

  function buildDeck() {
    state.deck = [];
    for (var r = 1; r <= state.rounds; r++) {
      for (var i = 0; i < state.articles.length; i++) state.deck.push({ a: state.articles[i], round: r });
    }
    state.pos = 0;
    state.current = null;
    installParentArrows();
    sizeDeck();
    root.appendChild(el("p", "deck-hint", SWIPE_HINT));
    state.deckEl = el("div", "deck");
    root.appendChild(state.deckEl);
    showCard();
  }

  function showCard() {
    var deck = state.deckEl;
    deck.textContent = "";
    state.current = null;
    if (state.pos >= state.deck.length) {
      var end = el("div", "deck-end");
      end.appendChild(el("p", null, "You've seen every article in this feed. Click Continue at the bottom of the screen to move on."));
      deck.appendChild(end);
      updateHeight();
      return;
    }
    // An empty card behind the current one, for depth.
    if (state.pos + 1 < state.deck.length) {
      var peek = el("div", "card peek");
      peek.setAttribute("aria-hidden", "true");
      deck.appendChild(peek);
      state.peekEl = peek;
    } else {
      state.peekEl = null;
    }
    var entry = state.deck[state.pos];
    var card = buildCard(entry.a, entry.round);
    deck.appendChild(card);
    if (entry.round > state.maxRound) state.maxRound = entry.round;
    state.current = { entry: entry, card: card, enter: ms(), leaving: false };
    attachDrag(card);
    updateHeight();
  }

  function closeView(dir, method) {
    var cur = state.current;
    if (!cur) return;
    cur.entry.a.views.push({ round: cur.entry.round, enter_ms: cur.enter, exit_ms: ms(),
                             dir: dir, method: method });
  }

  function flyOut(dir, method) {
    var cur = state.current;
    if (!cur || cur.leaving || state.submitted) return;
    cur.leaving = true;
    closeView(dir, method);
    var card = cur.card;
    var w = card.getBoundingClientRect().width || 400;
    card.classList.remove("dragging", "settle");
    card.classList.add("fly");
    card.style.transform = "translateX(" + (dir * w * 1.3) + "px) rotate(" + (dir * 14) + "deg)";
    if (state.peekEl) state.peekEl.classList.add("up");
    var done = false;
    function next() {
      if (done) return;
      done = true;
      state.pos += 1;
      showCard();
    }
    card.addEventListener("transitionend", next, { once: true });
    setTimeout(next, 380);   // in case transitionend never fires (reduced motion)
  }

  function attachDrag(card) {
    var startX = 0, startY = 0, dx = 0, t0 = 0, id = null, dragging = false;
    card.addEventListener("pointerdown", function (e) {
      if (state.submitted || e.button > 0) return;
      if (e.target.closest && e.target.closest("button, a")) return;
      id = e.pointerId; startX = e.clientX; startY = e.clientY; dx = 0; dragging = false;
      t0 = performance.now();
    });
    card.addEventListener("pointermove", function (e) {
      if (e.pointerId !== id) return;
      var mx = e.clientX - startX, my = e.clientY - startY;
      if (!dragging) {
        // Only a clearly horizontal movement becomes a drag; vertical
        // movement keeps scrolling the text.
        if (Math.abs(mx) < 10 || Math.abs(mx) < Math.abs(my) * 1.2) {
          if (Math.abs(my) > 10) id = null;
          return;
        }
        dragging = true;
        card.classList.remove("settle");
        card.classList.add("dragging");
        try { card.setPointerCapture(id); } catch (err) {}
        try { window.getSelection().removeAllRanges(); } catch (err) {}
      }
      dx = mx;
      card.style.transform = "translateX(" + dx + "px) rotate(" + (dx / 22) + "deg)";
      e.preventDefault();
    });
    function release(e) {
      if (e.pointerId !== id) return;
      id = null;
      if (!dragging) return;
      dragging = false;
      card.classList.remove("dragging");
      var speed = Math.abs(dx) / Math.max(1, performance.now() - t0);
      if (Math.abs(dx) > SWIPE_PX || (Math.abs(dx) > 30 && speed > FLICK_PX_PER_MS)) {
        flyOut(dx > 0 ? 1 : -1, "drag");
      } else {
        card.classList.add("settle");
        card.style.transform = "";
      }
    }
    card.addEventListener("pointerup", release);
    card.addEventListener("pointercancel", release);
  }

  // Arrow keys work wherever the focus is: in this frame or in the page
  // around it (e.g. right after clicking a Streamlit button). The page-level
  // listener is replaced by each new feed, so only the live one answers.
  function onArrow(e) {
    if (!state || !state.swipe || state.submitted) return;
    var t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName || ""))) return;
    if (e.key === "ArrowRight") { e.preventDefault(); flyOut(1, "key"); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); flyOut(-1, "key"); }
  }
  document.addEventListener("keydown", onArrow);
  function installParentArrows() {
    try {
      var d = window.parent.document;
      if (d === document) return;
      if (window.parent.__npsArrows) d.removeEventListener("keydown", window.parent.__npsArrows);
      window.parent.__npsArrows = onArrow;
      d.addEventListener("keydown", onArrow);
    } catch (err) { /* cross-origin: keys work inside the frame only */ }
  }
  function removeParentArrows() {
    try {
      if (window.parent.__npsArrows === onArrow) {
        window.parent.document.removeEventListener("keydown", onArrow);
        window.parent.__npsArrows = null;
      }
    } catch (err) { /* ignore */ }
  }
  window.addEventListener("pagehide", removeParentArrows);

  // ---- minimum time on the feed -------------------------------------------
  // The Continue / Finish button stays locked until `minMs` have passed since
  // the feed was shown. The countdown only appears during the last `revealMs`,
  // so the feed doesn't look like a timed task; before that the button simply
  // reads "Continue" / "Finish", greyed out. performance.now() keeps counting
  // correctly even if the browser throttles timers in a background tab.
  function setHint(hint, text) {
    if (!hint || hint.__text === text) return;
    hint.__text = text;
    hint.textContent = text || "";
    hint.classList.toggle("on", !!text);
  }

  function setLocked(button, locked) {
    button.setAttribute("aria-disabled", locked ? "true" : "false");
  }

  function startCountdown(button, hint, label) {
    if (countdown) { clearInterval(countdown); countdown = null; }
    function tick() {
      var left = state.minMs - ms();
      if (left <= 0) {
        if (countdown) { clearInterval(countdown); countdown = null; }
        if (!state.submitted) {
          setLocked(button, false);
          button.textContent = label;
          setHint(hint, UNLOCK_HINT);
        }
        return;
      }
      setLocked(button, true);
      if (state.revealMs > 0 && left <= state.revealMs) {
        button.textContent = "You can proceed in " + Math.ceil(left / 1000) + "s";
      } else {
        button.textContent = label;
      }
      setHint(hint, (performance.now() < state.nudgeUntil) ? LOCKED_HINT : "");
    }
    tick();
    if (state.minMs > 0) countdown = setInterval(tick, 250);
  }

  // ---- page --------------------------------------------------------------
  function buildPage(args) {
    if (viewObserver) { viewObserver.disconnect(); viewObserver = null; }
    if (state && state.logos) {
      for (var k in state.logos) { try { URL.revokeObjectURL(state.logos[k].url); } catch (e) {} }
    }
    state = {
      t0: performance.now(), enterTs: iso(), submitted: false, isLast: !!args.is_last,
      previewChars: args.preview_chars || 50, readmoreChars: args.readmore_chars || 144,
      logos: logoUrls(args.logos),
      minMs: Math.max(0, (args.min_seconds || 0) * 1000),
      revealMs: Math.max(0, (args.countdown_visible_seconds || 0) * 1000),
      nudgeUntil: 0, hintEl: null,
      rounds: Math.max(1, args.repeat_rounds || 1), maxRound: 0, articles: [],
      swipe: args.feed_style === "swipe"
    };
    root.textContent = "";
    if (args.logo_height) root.style.setProperty("--logo-h", args.logo_height + "px");
    if (typeof args.logo_radius === "number") root.style.setProperty("--logo-r", args.logo_radius + "px");

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
    head.appendChild(el("span", "progress-label", args.progress_label || ""));
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

    var label = "Continue";
    var mounted = mountBar();
    var next, hint;
    if (mounted) {
      next = mounted.btn;
      hint = mounted.hint;
    } else {
      // Fallback: the page could not be reached, so the button goes on top.
      var actions = el("div", "actions");
      hint = el("p", "hint");
      next = el("button", "next");
      next.type = "button";
      actions.appendChild(hint);
      actions.appendChild(next);
      root.appendChild(actions);
    }
    state.hintEl = hint;
    next.textContent = label;
    setLocked(next, state.minMs > 0);
    next.addEventListener("click", function () { submit(next); });

    state.articles = (args.articles || []).map(function (art, i) {
      var sentences = art.sentences || [];
      var a = {
        uid: String(art.uid), position: i + 1, logo: art.logo || null, title: art.title || "",
        sentences: sentences, flat: flatten(sentences),
        stage: 0, readmore_ms: null, readmore_ts: null, readmore_round: null,
        full_ms: null, full_ts: null, full_round: null,
        liked: false, history: [], instances: [], likeButtons: [], views: []
      };
      a.readmoreAvailable = a.flat.length > state.previewChars;
      a.fullAvailable = a.flat.length > state.readmoreChars;
      return a;
    });

    if (state.swipe) {
      buildDeck();
    } else {
      // The same sequence, repeated `rounds` times.
      var feed = el("div", "feed");
      for (var r = 1; r <= state.rounds; r++) {
        for (var i = 0; i < state.articles.length; i++) feed.appendChild(buildCard(state.articles[i], r));
      }
      root.appendChild(feed);
    }

    startCountdown(next, hint, label);

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
    if (ms() < state.minMs) {
      // Clicked too early: say so instead of looking broken.
      state.nudgeUntil = performance.now() + 2600;
      setHint(state.hintEl, LOCKED_HINT);
      return;
    }
    state.submitted = true;
    setLocked(button, true);
    button.textContent = "One moment…";
    setHint(state.hintEl, "");
    if (viewObserver) { viewObserver.disconnect(); viewObserver = null; }
    if (state.swipe && state.current && !state.current.leaving) closeView(null, "continue");
    if (state.swipe) removeParentArrows();
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
          liked: a.liked, like_history: a.history, views: a.views
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


# ---- outlet survey component -------------------------------------------------
# One page with a card per outlet (logo + name). Two modes:
#   "familiarity": two choices per outlet (FAMILIARITY_OPTIONS);
#   "preference":  a Likert scale per outlet (PREFERENCE_POINTS points).
# "Continue" unlocks once every outlet has an answer; the component then
# returns the answers, every change with its time, and the page timing.

SURVEY_COMPONENT_HTML = r"""<!doctype html>
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
    --warn: #A3361F;
    --sans: "Schibsted Grotesk", system-ui, -apple-system, "Segoe UI", sans-serif;
    --serif: "Literata", Georgia, "Times New Roman", serif;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: transparent; color: var(--ink); }
  body { font-family: var(--sans); -webkit-font-smoothing: antialiased; }
  #root { padding: 2px 2px 10px; }

  .step { font-size: 14px; font-weight: 500; color: var(--ink-3); margin: 0 0 14px;
          font-variant-numeric: tabular-nums; }
  h1 { font-weight: 700; font-size: 34px; line-height: 1.1; letter-spacing: -0.022em;
       margin: 0 0 14px; text-wrap: balance; }
  .lead { font-family: var(--serif); font-optical-sizing: auto; font-size: 18px;
          line-height: 1.58; color: var(--ink-2); margin: 0 0 28px; max-width: 34em; }

  .list { display: flex; flex-direction: column; gap: 12px; }
  .item { background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
          padding: 20px 24px; transition: border-color .2s, box-shadow .2s; }
  .item.missing { border-color: var(--warn); box-shadow: 0 0 0 1px var(--warn); }

  .ident { display: flex; flex-direction: column; align-items: flex-start; gap: 10px; min-width: 0; }
  /* Every logo gets the same height (small files are scaled up too) and
     keeps its shape; only a logo too wide for the card is scaled down. */
  .logo { height: var(--logo-h, 96px); width: 100%; display: flex; align-items: center; }
  .logo img { display: block; height: 100%; width: auto; max-width: 100%;
              object-fit: contain; object-position: left center;
              border-radius: var(--logo-r, 18px); }
  .name { font-weight: 600; font-size: 18px; line-height: 1.25; color: var(--ink); }

  /* familiarity: identity on the left, the two choices on the right */
  .item.binary { display: grid; grid-template-columns: minmax(0, 1fr) auto;
                 align-items: center; gap: 18px 24px; }
  .choices { display: flex; gap: 8px; }
  .opt { appearance: none; height: 44px; padding: 0 18px; border-radius: 22px;
         border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2);
         font: 500 15px/1 var(--sans); cursor: pointer; white-space: nowrap;
         transition: background-color .15s, border-color .15s, color .15s; }
  .opt:hover { border-color: var(--ink); color: var(--ink); }
  .opt[aria-checked="true"] { background: var(--ink); border-color: var(--ink); color: #FFFFFF; }

  /* preference: identity on top, the scale below at full width */
  .scale { display: grid; grid-template-columns: repeat(var(--points, 7), minmax(0, 1fr));
           gap: 6px; margin-top: 18px; }
  .pt { appearance: none; height: 46px; padding: 0; border-radius: 10px;
        border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2);
        font: 600 16px/1 var(--sans); cursor: pointer; font-variant-numeric: tabular-nums;
        transition: background-color .15s, border-color .15s, color .15s; }
  .pt:hover { border-color: var(--ink); color: var(--ink); }
  .pt[aria-checked="true"] { background: var(--ink); border-color: var(--ink); color: #FFFFFF; }
  .anchors { display: grid; grid-template-columns: repeat(var(--points, 7), minmax(0, 1fr));
             gap: 6px; margin-top: 8px; font-size: 13px; line-height: 1.3; color: var(--ink-3); }
  .anchors .a-left { text-align: left; }
  .anchors .a-mid { text-align: center; }
  .anchors .a-right { text-align: right; }

  .actions { display: flex; flex-direction: column; align-items: center; gap: 10px;
             margin: 30px 0 8px; }
  .count { margin: 0; font-size: 14px; color: var(--ink-3); font-variant-numeric: tabular-nums; }
  .count.warn { color: var(--warn); font-weight: 500; }
  .next { appearance: none; border: 0; height: 48px; padding: 0 30px; border-radius: 24px;
          background: var(--ink); color: #FFFFFF; font: 600 16px/1 var(--sans); cursor: pointer;
          min-width: 215px; transition: background-color .15s; }
  .next[aria-disabled="true"] { background: #8B9097; cursor: default; }

  button:focus-visible { outline: 2px solid var(--ink); outline-offset: 3px; }

  @media (max-width: 560px) {
    h1 { font-size: 28px; }
    .lead { font-size: 17px; }
    .item { padding: 18px 18px; }
    .logo { height: calc(var(--logo-h, 96px) * 0.8); }
    .logo img { border-radius: calc(var(--logo-r, 18px) * 0.8); }
    .item.binary { grid-template-columns: 1fr; }
    .choices .opt { flex: 1; padding: 0 10px; }
    .pt { height: 44px; font-size: 15px; border-radius: 8px; }
    .scale, .anchors { gap: 4px; }
    .anchors { font-size: 12px; }
    .actions { align-items: stretch; }
    .next { width: 100%; }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { transition: none !important; }
  }
</style>
</head>
<body>
<div id="root"></div>
<script>
(function () {
  "use strict";

  var root = document.getElementById("root");
  var state = null;
  var currentPage = null;
  var lastHeight = -1;

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
    } catch (err) { /* ignore */ }
  }
  function installLeaveGuard() {
    try {
      var w = window.parent;
      if (w.__npsLeaveGuard) return;
      w.__npsLeaveGuard = new w.Function("e", "e.preventDefault(); e.returnValue = ''; return '';");
      w.addEventListener("beforeunload", w.__npsLeaveGuard);
    } catch (err) { /* ignore */ }
  }

  // A logo too wide for its box at full height is drawn smaller, at its
  // own proportions (no empty bands, the rounded corners stay on the image).
  function fitLogo(img) {
    var box = img.parentNode;
    if (!box || !img.naturalWidth || !img.naturalHeight) return;
    var fullW = img.naturalWidth / img.naturalHeight * box.clientHeight;
    var tooWide = fullW > box.clientWidth + 0.5;
    img.style.height = tooWide ? "auto" : "";
    img.style.width = tooWide ? "100%" : "";
  }
  window.addEventListener("resize", function () {
    if (state && state.logoImgs) state.logoImgs.forEach(fitLogo);
  });

  function answered() {
    var n = 0;
    for (var i = 0; i < state.items.length; i++) {
      if (state.answers[state.items[i].key] !== undefined) n++;
    }
    return n;
  }

  function refresh() {
    var n = answered(), total = state.items.length, done = n === total;
    state.next.setAttribute("aria-disabled", done ? "false" : "true");
    if (done) {
      state.count.textContent = "All " + total + " answered";
      state.count.classList.remove("warn");
    } else if (!state.count.classList.contains("warn")) {
      state.count.textContent = n + " of " + total + " answered";
    }
  }

  // A radio group of buttons: one choice per outlet, changeable at any time.
  function choose(item, value, buttons) {
    if (state.submitted) return;
    state.answers[item.key] = value;
    state.history.push({ key: item.key, value: value, ms: ms() });
    for (var i = 0; i < buttons.length; i++) {
      var on = buttons[i].__value === value;
      buttons[i].setAttribute("aria-checked", on ? "true" : "false");
      buttons[i].tabIndex = on ? 0 : -1;
    }
    item.card.classList.remove("missing");
    if (state.count.classList.contains("warn")) {
      state.count.classList.remove("warn");
    }
    refresh();
  }

  function buildGroup(item, container, options, cls, ariaLabel) {
    container.setAttribute("role", "radiogroup");
    container.setAttribute("aria-label", ariaLabel);
    var buttons = [];
    options.forEach(function (opt, i) {
      var b = el("button", cls, opt.label);
      b.type = "button";
      b.__value = opt.value;
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", "false");
      if (opt.aria) b.setAttribute("aria-label", opt.aria);
      b.tabIndex = i === 0 ? 0 : -1;
      b.addEventListener("click", function () { choose(item, opt.value, buttons); });
      b.addEventListener("keydown", function (e) {
        var step = (e.key === "ArrowRight" || e.key === "ArrowDown") ? 1
                 : (e.key === "ArrowLeft" || e.key === "ArrowUp") ? -1 : 0;
        if (!step) return;
        e.preventDefault();
        var j = (i + step + buttons.length) % buttons.length;
        buttons[j].focus();
        choose(item, buttons[j].__value, buttons);
      });
      buttons.push(b);
      container.appendChild(b);
    });
    return buttons;
  }

  function buildItem(item) {
    var card = el("div", "item");
    item.card = card;
    var ident = el("div", "ident");
    if (item.logo) {
      var box = el("div", "logo");
      var img = el("img");
      img.src = item.logo;
      img.alt = "";
      img.decoding = "async";
      img.draggable = false;
      img.addEventListener("load", function () { fitLogo(img); updateHeight(); });
      state.logoImgs.push(img);
      box.appendChild(img);
      ident.appendChild(box);
    }
    ident.appendChild(el("div", "name", item.name));
    card.appendChild(ident);

    if (state.mode === "familiarity") {
      card.classList.add("binary");
      var choices = el("div", "choices");
      buildGroup(item, choices, state.options, "opt", "Do you know " + item.name + "?");
      card.appendChild(choices);
    } else {
      var scale = el("div", "scale");
      var pts = state.options.map(function (o) {
        return { value: o.value, label: String(o.value), aria: o.label };
      });
      buildGroup(item, scale, pts, "pt", item.name + ": " + state.question);
      card.appendChild(scale);
      var anchors = el("div", "anchors");
      anchors.setAttribute("aria-hidden", "true");
      // End labels under the first / last points, the middle one centred
      // under the centre point (odd scales only), never overlapping.
      var n = state.options.length;
      var hasMid = n % 2 === 1 && n >= 3 && !!state.anchors[1];
      var side = hasMid ? Math.max(1, (n - 3) / 2) : Math.max(1, Math.floor(n / 2));
      var left = el("span", "a-left", state.anchors[0] || "");
      left.style.gridColumn = "1 / span " + side;
      anchors.appendChild(left);
      if (hasMid) {
        var mid = el("span", "a-mid", state.anchors[1]);
        mid.style.gridColumn = (side + 1) + " / span " + (n - 2 * side);
        anchors.appendChild(mid);
      }
      var right = el("span", "a-right", state.anchors[2] || "");
      right.style.gridColumn = (n - side + 1) + " / span " + side;
      anchors.appendChild(right);
      card.appendChild(anchors);
    }
    return card;
  }

  function buildPage(args) {
    state = {
      t0: performance.now(), enterTs: iso(), submitted: false,
      mode: args.mode, question: args.question || "",
      options: args.options || [], anchors: args.anchors || [],
      items: (args.items || []).map(function (it) {
        return { key: String(it.key), name: it.name || "", logo: it.logo || null };
      }),
      answers: {}, history: [], logoImgs: []
    };
    root.textContent = "";
    if (args.logo_height) root.style.setProperty("--logo-h", args.logo_height + "px");
    if (typeof args.logo_radius === "number") root.style.setProperty("--logo-r", args.logo_radius + "px");
    root.style.setProperty("--points", String(Math.max(1, state.options.length)));

    if (args.step_label) root.appendChild(el("p", "step", args.step_label));
    root.appendChild(el("h1", null, args.title || ""));
    if (args.lead) root.appendChild(el("p", "lead", args.lead));

    var list = el("div", "list");
    state.items.forEach(function (item) { list.appendChild(buildItem(item)); });
    root.appendChild(list);

    var actions = el("div", "actions");
    state.count = el("p", "count");
    state.count.setAttribute("aria-live", "polite");
    state.next = el("button", "next", args.button_label || "Continue");
    state.next.type = "button";
    state.next.addEventListener("click", submit);
    actions.appendChild(state.count);
    actions.appendChild(state.next);
    root.appendChild(actions);

    refresh();
    if (args.warn_on_leave) installLeaveGuard();
    scrollParentToTop();
    updateHeight();
  }

  function submit() {
    if (state.submitted) return;
    var missing = state.items.filter(function (it) { return state.answers[it.key] === undefined; });
    if (missing.length) {
      // Point to the first outlet still without an answer.
      missing.forEach(function (it) { it.card.classList.add("missing"); });
      state.count.textContent = missing.length === 1
        ? "Please answer for 1 more outlet."
        : "Please answer for " + missing.length + " more outlets.";
      state.count.classList.add("warn");
      try { missing[0].card.scrollIntoView({ behavior: "smooth", block: "center" }); } catch (e) {}
      return;
    }
    state.submitted = true;
    state.next.setAttribute("aria-disabled", "true");
    state.next.textContent = "One moment…";
    send("streamlit:setComponentValue", { dataType: "json", value: {
      page_id: currentPage,
      mode: state.mode,
      enter_ts: state.enterTs,
      exit_ts: iso(),
      duration_ms: ms(),
      order: state.items.map(function (it) { return it.key; }),
      answers: state.answers,
      history: state.history
    } });
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
"""

_SURVEY_COMPONENT_DIR = os.path.join(
    tempfile.gettempdir(),
    "nps_survey_" + hashlib.sha1(SURVEY_COMPONENT_HTML.encode("utf-8")).hexdigest()[:12],
)


def _ensure_survey_component_files():
    path = os.path.join(_SURVEY_COMPONENT_DIR, "index.html")
    if not os.path.exists(path):
        os.makedirs(_SURVEY_COMPONENT_DIR, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{_rng.randint(0, 10**9)}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(SURVEY_COMPONENT_HTML)
        os.replace(tmp, path)


_ensure_survey_component_files()
survey_component = components.declare_component("outlet_survey", path=_SURVEY_COMPONENT_DIR)

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
li.nps-row { display: grid; grid-template-columns: 200px 1fr; gap: 22px; align-items: center;
             margin: 0 !important; padding: 0 !important; }
li.nps-row::marker { content: none; }
.nps-row p { margin: 0; font-size: 16px; line-height: 1.55; color: var(--ink-2); }
.nps-row b { color: var(--ink); font-weight: 600; }
.nps-token { display: flex; align-items: center; justify-content: flex-start; }
/* Logos row: the logos get the full width, the text goes below them. */
li.nps-row.nps-row-logos { grid-template-columns: 1fr; gap: 14px; }
.tok-logos { display: flex !important; flex-wrap: wrap; gap: 12px 14px;
             justify-content: flex-start; align-items: center; }
.tok-logos figure.tok-logo { margin: 0; display: flex; flex-direction: column;
                             align-items: flex-start; gap: 6px; max-width: 100%; }
.tok-logos img { display: block; height: var(--intro-logo-h, 64px); width: auto; max-width: 100%;
                 object-fit: contain; border-radius: 12px; }
.tok-logos figcaption { font-size: 14px; font-weight: 600; line-height: 1.25; color: var(--ink); }
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
.tok-swipe {
  display: inline-flex; align-items: center; gap: 8px; height: 34px; padding: 0 14px;
  border-radius: 10px; border: 1px solid var(--line); background: var(--surface);
  color: var(--ink); font-weight: 600; font-size: 14px; box-shadow: 0 4px 12px rgba(25, 28, 32, .08);
}
.tok-next {
  display: inline-flex; align-items: center; height: 34px; padding: 0 18px; border-radius: 17px;
  background: var(--ink); color: #FFFFFF; font-weight: 600; font-size: 14px;
}
.nps-plain { font-size: 16px; line-height: 1.55; color: var(--ink-2); margin: 0 0 40px; }

/* ---------- Continue / Finish bar, fixed at the bottom during a feed ---------- */
#nps-feedbar {
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 999990;
  padding: 40px 16px calc(16px + env(safe-area-inset-bottom));
  background: linear-gradient(to bottom, rgba(236, 238, 241, 0), var(--canvas) 42%);
  pointer-events: none;
}
.nps-feedbar-inner {
  max-width: 700px; margin: 0 auto; padding: 0 1.25rem;
  display: flex; flex-direction: column; align-items: center; gap: 10px;
}
/* min-height keeps the button still when the hint fades in. */
.nps-feedbar-hint {
  margin: 0; font-size: 13.5px; line-height: 1.4; min-height: 1.4em;
  color: var(--ink-3); text-align: center;
  opacity: 0; transition: opacity .35s ease; pointer-events: none;
}
.nps-feedbar-hint.on { opacity: 1; }
.nps-feedbar-btn {
  pointer-events: auto; appearance: none; border: 0; height: 48px; padding: 0 30px;
  border-radius: 24px; background: var(--ink); color: #FFFFFF;
  font: 600 16px/1 var(--sans); cursor: pointer; font-variant-numeric: tabular-nums;
  min-width: 215px; text-align: center;
  box-shadow: 0 6px 18px rgba(25, 28, 32, .18); transition: background-color .15s;
}
.nps-feedbar-btn[aria-disabled="true"], .nps-feedbar-btn:disabled {
  background: #8B9097; cursor: default; box-shadow: none;
}
.nps-feedbar-btn:focus-visible { outline: 2px solid var(--ink); outline-offset: 3px; }
body:has(#nps-feedbar) [data-testid="stMainBlockContainer"],
body:has(#nps-feedbar) .block-container { padding-bottom: 160px !important; }

/* ---------- wait between the feeds ---------- */
.nps-wait { padding: 10vh 0 0; }
.nps-wait .nps-sub { margin: 16px 0 0; }
.nps-loading {
  display: flex; align-items: center; gap: 12px; min-height: 48px;
  font-size: 17px; color: var(--ink-2);
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
  .tok-logos img { height: calc(var(--intro-logo-h, 64px) * 0.8); }
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
        # Read once from the URL (?PROLIFIC_PID=...&STUDY_ID=...&SESSION_ID=...)
        # and kept in the session from here on.
        "prolific_pid": qp.get("PROLIFIC_PID", "").strip(),
        "study_id": qp.get("STUDY_ID", "").strip(),
        "prolific_session_id": qp.get("SESSION_ID", "").strip(),
        "pid_status": None,
        "user_agent": user_agent,
        "stage": "intro",
        "opened_at": utc_iso(),
        "started_at": None,
        "started_ts": None,
        "completed_at": None,
        "condition": None,
        "assignment_method": None,
        "logo_files": {},
        "intro_logos": None,
        "survey_items": [],
        "familiarity_result": None,
        "preference_result": None,
        "preference_items": None,
        "article_order": {},
        "split": {},
        "pages": [],
        "page_idx": 0,
        "processed": [],
        "page_server_start": {},
        "first_feed_shown": False,
        "break_ready": False,
        "responses": [],
        "save_status": None,
        "saved_parts": [],
    })


def outlet_display_name(outlet):
    """Name shown next to the logo in the feeds: the name given in
    SURVEY_OUTLETS for that domain (e.g. rt.com -> "RussiaToday"), or the
    domain itself when the outlet is not listed there."""
    key = _logo_key(outlet)
    for o in SURVEY_OUTLETS:
        if _logo_key(o["domain"]) == key:
            return o.get("name") or outlet
    for domain, name in EXTRA_OUTLET_NAMES.items():
        if _logo_key(domain) == key:
            return name
    return outlet


def preference_options():
    """[(value, label), ...] for the Likert scale: 1..PREFERENCE_POINTS, the
    ends and the centre labelled with PREFERENCE_ANCHORS."""
    n = PREFERENCE_POINTS
    left, mid, right = (list(PREFERENCE_ANCHORS) + ["", "", ""])[:3]
    out = []
    for v in range(1, n + 1):
        if v == 1:
            label = left
        elif v == n:
            label = right
        elif n % 2 == 1 and v == (n + 1) // 2:
            label = mid
        else:
            label = ""
        out.append((v, f"{v} ({label})" if label else str(v)))
    return out


def survey_outlets(data):
    """The survey outlets as dicts with their logo, plus whether each one
    also appears in the feeds."""
    feed_keys = {_logo_key(o) for o in data["outlets"]}
    out = []
    for i, o in enumerate(SURVEY_OUTLETS):
        domain = o["domain"]
        out.append({
            "key": f"o{i + 1}",
            "domain": domain,
            "name": o.get("name") or domain,
            "group": o.get("group"),
            "ng_rating": o.get("ng_rating"),
            "ng_score": o.get("ng_score"),
            "in_feed": _logo_key(domain) in feed_keys,
            "logo_file": (data["logos"].get(domain) or {}).get("file"),
        })
    return out


def start_study(data):
    """Randomise everything for this participant: condition, phase split,
    feed orders and the order of the survey outlets."""
    ss = st.session_state
    topics, outlets = list(data["topics"]), list(data["outlets"])

    # Phase split: one of the balanced splits (see compute_split_options),
    # drawn at random for this participant. It says how many articles of each
    # topic x outlet cell go to phase 1 (phase 2 gets the rest); which ones
    # is random as well. Default: 18 + 18 articles, 4 or 5 per outlet.
    counts1 = _rng.choice(data["split_options"])
    split = {ph: {t: [] for t in topics} for ph in ("1", "2")}
    for (t, o), k1 in zip(data["split_cells"], counts1):
        uids = list(data["cells"][t][o])
        _rng.shuffle(uids)
        split["1"][t].extend(uids[:k1])
        split["2"][t].extend(uids[k1:])

    # One feed per phase with all its articles; the order mixes topics and
    # outlets and is drawn independently for each phase.
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
            "repeat_rounds": REPEAT_ROUNDS,
            "min_seconds": MIN_FEED_SECONDS,
        })

    items = survey_outlets(data)
    if SHUFFLE_SURVEY_OUTLETS:
        _rng.shuffle(items)

    logo_files = {o: data["logos"][o]["file"] for o in outlets}

    ss.update({
        "condition": condition, "assignment_method": method, "logo_files": logo_files,
        "survey_items": items,
        "split": split, "article_order": order, "pages": pages, "page_idx": 0,
        "stage": "familiarity", "started_at": utc_iso(), "started_ts": time.time(),
    })
    log(f"session {ss.session_id} started: condition {condition} ({method}), "
        f"{len(order['1'])} + {len(order['2'])} articles")


def _view_stats(views):
    """Summary of the swipe-mode views of one article (empty in scroll mode)."""
    if FEED_STYLE != "swipe":
        return {}
    views = views or []
    dwell = [max(0, (v.get("exit_ms") or 0) - (v.get("enter_ms") or 0)) for v in views]
    return {
        "n_views": len(views),
        "dwell_total_ms": sum(dwell),
        "dwell_first_ms": dwell[0] if dwell else None,
        "n_swipe_left": sum(1 for v in views if v.get("dir") == -1),
        "n_swipe_right": sum(1 for v in views if v.get("dir") == 1),
        "views": json.dumps(views),
    }


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
            "logo_file": ss.logo_files.get(art["outlet"]) if page["source_visible"] else None,
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
            **_view_stats(item.get("views")),
        })


def build_session_row():
    ss = st.session_state
    resp = ss.responses
    duration = round((time.time() - ss.started_ts) / 60, 2) if ss.started_ts else None
    fam, pref = ss.familiarity_result, ss.preference_result
    domains = {it["key"]: it["domain"] for it in ss.survey_items}

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
        "logo_files": json.dumps(ss.logo_files),
        "intro_logos": json.dumps(ss.intro_logos or []),
        "phase1_order": json.dumps(ss.article_order.get("1", [])),
        "phase2_order": json.dumps(ss.article_order.get("2", [])),
        "article_split": json.dumps(ss.split),
        "familiarity_order": json.dumps([domains.get(k, k) for k in (fam or {}).get("order", [])]),
        "preference_order": json.dumps([domains.get(k, k) for k in (pref or {}).get("order", [])]),
        "n_outlets_known": sum(1 for v in ((fam or {}).get("answers") or {}).values() if v == "yes"),
        "familiarity_duration_ms": (fam or {}).get("duration_ms"),
        "preference_duration_ms": (pref or {}).get("duration_ms"),
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
        "feed_style": FEED_STYLE,
        "app_version": APP_VERSION,
    }


def _answer_stats(result, key):
    """(first_ms, last_ms, n_changes) of the answers given to one outlet."""
    times = [h.get("ms") for h in ((result or {}).get("history") or []) if h.get("key") == key]
    if not times:
        return None, None, 0
    return times[0], times[-1], len(times) - 1


def build_outlet_rows():
    """One row per survey outlet: familiarity and, if known, preference."""
    ss = st.session_state
    fam = ss.familiarity_result or {}
    pref = ss.preference_result or {}
    fam_pos = {k: i + 1 for i, k in enumerate(fam.get("order") or [])}
    pref_pos = {k: i + 1 for i, k in enumerate(pref.get("order") or [])}
    labels = dict(preference_options())
    rows = []
    for it in sorted(ss.survey_items, key=lambda x: fam_pos.get(x["key"], 999)):
        k = it["key"]
        f_first, f_last, f_changes = _answer_stats(fam, k)
        p_first, p_last, p_changes = _answer_stats(pref, k)
        value = (pref.get("answers") or {}).get(k)
        rows.append({
            "session_id": ss.session_id,
            "prolific_pid": ss.prolific_pid,
            "condition": ss.condition,
            "condition_label": CONDITIONS[ss.condition]["label"],
            "outlet": it["domain"],
            "outlet_name": it["name"],
            "outlet_group": it["group"],
            "in_feed": it["in_feed"],
            "newsguard_rating": it["ng_rating"],
            "newsguard_score": it["ng_score"],
            "logo_file": it.get("logo_file"),
            "familiarity_position": fam_pos.get(k),
            "familiar": (fam.get("answers") or {}).get(k),
            "familiarity_first_ms": f_first,
            "familiarity_last_ms": f_last,
            "familiarity_changes": f_changes,
            "preference_shown": k in pref_pos,
            "preference_position": pref_pos.get(k),
            "preference": value,
            "preference_label": labels.get(value),
            "preference_first_ms": p_first,
            "preference_last_ms": p_last,
            "preference_changes": p_changes,
            "familiarity_enter_ts": fam.get("enter_ts"),
            "familiarity_exit_ts": fam.get("exit_ts"),
            "preference_enter_ts": pref.get("enter_ts"),
            "preference_exit_ts": pref.get("exit_ts"),
        })
    return rows


def save_results():
    """Single write at the end: responses + session row to Google Sheets,
    plus a local CSV/JSON backup."""
    ss = st.session_state
    session_row = build_session_row()
    response_rows = [[cell(r.get(h)) for h in RESPONSE_HEADERS] for r in ss.responses]
    outlet_dicts = build_outlet_rows()
    outlet_rows = [[cell(r.get(h)) for h in OUTLET_HEADERS] for r in outlet_dicts]
    session_values = [cell(session_row.get(h)) for h in SESSION_HEADERS]

    try:
        os.makedirs(LOCAL_BACKUP_DIR, exist_ok=True)
        pd.DataFrame(ss.responses, columns=RESPONSE_HEADERS).to_csv(
            os.path.join(LOCAL_BACKUP_DIR, f"{ss.session_id}_responses.csv"), index=False)
        pd.DataFrame(outlet_dicts, columns=OUTLET_HEADERS).to_csv(
            os.path.join(LOCAL_BACKUP_DIR, f"{ss.session_id}_outlets.csv"), index=False)
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
            if "outlets" not in ss.saved_parts:
                store.append(SHEET_OUTLETS, OUTLET_HEADERS, outlet_rows)
                ss.saved_parts.append("outlets")
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


def render_access_blocked(reason):
    """Shown instead of the study when the participant did not come from
    Prolific (no PROLIFIC_PID in the URL) or has already completed it."""
    if reason == "missing_pid":
        title = "Please open this study from Prolific"
        text = ("This study can only be taken through Prolific. Please go back to "
                "Prolific and open the study from the link on the study page.")
    else:  # duplicate
        title = "You have already taken part"
        text = ("Our records show that this Prolific ID has already completed the study, "
                "so it can't be taken again. If you think this is a mistake, please "
                "contact the researcher through Prolific.")
    st.html(f"""
<section>
  <h1 class="nps-title">{title}</h1>
  <p class="nps-lead">{text}</p>
</section>""")


def render_intro(data):
    ss = st.session_state
    n_feeds = 2
    # Example logos: drawn once per participant from every logo in LOGO_DIR.
    pool = {p["file"]: p for p in load_logo_pool(_logo_index())}
    if ss.intro_logos is None or any(f not in pool for f in ss.intro_logos):
        ss.intro_logos = _rng.sample(sorted(pool), min(INTRO_LOGO_COUNT, len(pool)))
    logos = "".join(
        f'<figure class="tok-logo"><img src="{pool[f]["src"]}" alt="">'
        f'<figcaption>{html_lib.escape(outlet_display_name(pool[f]["key"]))}</figcaption></figure>'
        for f in ss.intro_logos
    )
    if MIN_FEED_SECONDS > 0:
        wait_note = (" It becomes active once you've spent a little time on the feed, "
                     "so take your time.")
    else:
        wait_note = ""
    if FEED_STYLE == "swipe":
        feed_desc = ("Each one is a stream of real news articles from different outlets, shown "
                     "one at a time, and you can go through it for as long as you like.")
        swipe_row = ('<li class="nps-row"><div class="nps-token"><span class="tok-swipe">'
                     '<span aria-hidden="true">&larr;</span> Swipe <span aria-hidden="true">&rarr;</span>'
                     '</span></div><p>Swipe the article card <b>left or right</b> to see the next one. '
                     'On a computer you can also drag it with the mouse, use the arrow keys, or click '
                     '<b>Next</b>.</p></li>')
    else:
        feed_desc = ("Each one is a stream of real news articles from different outlets, and you "
                     "can scroll through it for as long as you like.")
        swipe_row = ""
    st.html(f"""
<section class="nps-intro" style="--intro-logo-h: {INTRO_LOGO_HEIGHT_PX}px">
  <h1 class="nps-title">News Perception Study</h1>
  <p class="nps-lead">You'll start with a few quick questions about some news outlets. Then you'll browse {number_word(n_feeds)} news feeds, one after the other. {feed_desc}</p>
  <p class="nps-lead">The articles are there to be used, not just looked at: open the ones that catch your eye, read as much or as little of them as you want, and like the ones you enjoy. Browse them the way you'd browse news online.</p>
  <p class="nps-sub">It takes about {ESTIMATED_MINUTES} minutes. Please finish in one sitting and don't refresh the page.</p>

  <h2 class="nps-h2">How it works</h2>
  <ul class="nps-rows">
    {swipe_row}
    <li class="nps-row">
      <div class="nps-token"><span class="tok-more">…&nbsp;Read more</span></div>
      <p>Each article opens with a short preview. Click <b>Read more</b> to keep reading. After that, <b>Read full article</b> shows the whole text.</p>
    </li>
    <li class="nps-row nps-row-logos">
      <div class="nps-token tok-logos">{logos}</div>
      <p>The logo and name at the top of each article show which news outlet published it.</p>
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


def render_survey(data, mode):
    """Outlet survey: mode "familiarity" (all outlets) or "preference" (only
    the outlets the participant knows)."""
    ss = st.session_state
    page_id = f"{ss.session_id}-{mode}"
    if mode == "familiarity":
        items = ss.survey_items
        options = [{"value": v, "label": l} for v, l in FAMILIARITY_OPTIONS]
        title, lead, step = FAMILIARITY_TITLE, FAMILIARITY_LEAD, "News outlets · 1 of 2"
    else:
        known = {k for k, v in ss.familiarity_result["answers"].items() if v == "yes"}
        items = [it for it in ss.survey_items if it["key"] in known]
        if SHUFFLE_SURVEY_OUTLETS:
            # A new order for this step, kept across reruns.
            if ss.preference_items is None:
                ss.preference_items = _rng.sample(items, len(items))
            items = ss.preference_items
        options = [{"value": v, "label": l} for v, l in preference_options()]
        title, lead, step = PREFERENCE_TITLE, PREFERENCE_LEAD, "News outlets · 2 of 2"

    result = survey_component(
        page_id=page_id,
        mode=mode,
        step_label=step,
        title=title,
        lead=lead,
        question=PREFERENCE_TITLE,
        options=options,
        anchors=list(PREFERENCE_ANCHORS),
        items=[{"key": it["key"], "name": it["name"],
                "logo": data["logos"][it["domain"]]["src"]} for it in items],
        logo_height=SURVEY_LOGO_HEIGHT_PX,
        logo_radius=LOGO_RADIUS_PX,
        button_label="Continue",
        warn_on_leave=WARN_ON_LEAVE,
        key=f"survey-{page_id}",
        default=None,
    )

    if not (isinstance(result, dict) and result.get("page_id") == page_id
            and page_id not in ss.processed):
        return
    ss.processed.append(page_id)
    if mode == "familiarity":
        ss.familiarity_result = result
        n_known = sum(1 for v in (result.get("answers") or {}).values() if v == "yes")
        # Nobody to rate: straight on to the first feed.
        ss.stage = "preference" if n_known else "break"
    else:
        ss.preference_result = result
        ss.stage = "break"
    ss.break_ready = False
    st.rerun()


def render_feed(data):
    ss = st.session_state
    page = ss.pages[ss.page_idx]
    page_id = f"{ss.session_id}-p{page['page_number']}"
    ss.page_server_start.setdefault(page_id, time.time())

    rewritten = page["text_version"] == "rewritten"
    # Neutral keys (L1, L2, ...) instead of outlet names in the card data.
    logo_keys = {o: f"L{i + 1}" for i, o in enumerate(data["outlets"])}
    logos = ({logo_keys[o]: {"src": data["logos"][o]["src"], "alt": o,
                             "name": outlet_display_name(o)} for o in data["outlets"]}
             if page["source_visible"] else {})
    progress_label = f"Feed {page['phase']} of {len(ss.pages)}"
    cards = []
    for uid in page["articles"]:
        art = data["articles"][uid]
        cards.append({
            "uid": uid,
            # Logo key, sent only when the source is visible: with the source
            # hidden nothing in the page tells the outlets apart.
            "logo": logo_keys[art["outlet"]] if page["source_visible"] else None,
            # Headline and body always come from the same version: Title +
            # clean_text for the original, LLM_title + LLM_text for the rewrite.
            "title": art["title_rewritten"] if rewritten else art["title"],
            "sentences": art["sentences_rewritten"] if rewritten else art["sentences_original"],
        })

    result = feed_component(
        page_id=page_id,
        page_number=page["page_number"],
        total_pages=len(ss.pages),
        is_last=page["page_number"] == len(ss.pages),
        progress_label=progress_label,
        articles=cards,
        logos=logos,
        logo_height=LOGO_HEIGHT_PX,
        logo_radius=LOGO_RADIUS_PX,
        repeat_rounds=page["repeat_rounds"],
        preview_chars=PREVIEW_CHARS,
        readmore_chars=READMORE_CHARS,
        min_seconds=page["min_seconds"],
        countdown_visible_seconds=COUNTDOWN_VISIBLE_SECONDS,
        feed_style=FEED_STYLE,
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
    """"Loading the feed" screen before each feed, framed only as loading: the
    participant is never told to take a break. Before the first feed it also
    recalls how the feed works (the intro was a few screens ago). The pause
    length can be computed from the data as page_enter_ts of the feed minus
    the exit time of the screen before it."""
    ss = st.session_state
    slot = st.empty()

    upcoming = ss.pages[ss.page_idx]["phase"]
    first = ss.page_idx == 0
    ordinal = {1: "first", 2: "second", 3: "third"}.get(upcoming, "next")
    wait = FIRST_FEED_LOADING_SECONDS if first else BREAK_LOADING_SECONDS

    if not ss.break_ready and wait > 0:
        slot.html(f"""
<section class="nps-wait">
  <div class="nps-loading" role="status">
    <span class="nps-spinner" aria-hidden="true"></span>
    <span>Loading the {ordinal} feed…</span>
  </div>
  <p class="nps-sub">Please don't refresh or close this page.</p>
</section>""")
        time.sleep(wait)
    ss.break_ready = True

    if first and FEED_STYLE == "swipe":
        note = ("The articles come one at a time: swipe the card left or right to see the next "
                "one. Open the ones that catch your eye with <b>Read more</b>, like the ones you "
                "enjoy, and click <b>Continue</b> at the bottom of the screen when you want to "
                "move on.")
    elif first:
        note = ("Browse it the way you'd browse news online: open the articles that catch "
                "your eye with <b>Read more</b>, like the ones you enjoy, and click "
                "<b>Continue</b> at the bottom of the screen when you want to move on.")
    else:
        note = "It works the same way as the first."
    slot.html(f"""
<section class="nps-wait">
  <div class="nps-loading">
    <span>The {ordinal} feed is ready.</span>
  </div>
  <p class="nps-sub">{note} Please don't refresh or close this page.</p>
</section>""")

    if st.button("Continue", type="primary", key=f"break_button_{ss.page_idx}"):
        ss.stage = "feed"
        st.rerun()


def render_end():
    ss = st.session_state
    if ss.save_status is None:
        with st.spinner("Saving your responses…"):
            save_results()

    # Completion code: from secrets ([prolific] completion_code), otherwise
    # the PROLIFIC_COMPLETION_CODE fallback set at the top of the file.
    code = str(get_secret("prolific", "completion_code", "") or PROLIFIC_COMPLETION_CODE).strip()
    if not code:
        log("WARNING: no Prolific completion code configured")
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

    ss = st.session_state

    # Prolific access check, done once per session: no PROLIFIC_PID in the
    # URL, or a PROLIFIC_PID that already completed the study, stops here.
    if ss.pid_status is None:
        ss.pid_status = check_access()
    if ss.pid_status in ("missing_pid", "duplicate"):
        st.html("<style>#nps-feedbar { display: none !important; }</style>")
        render_access_blocked(ss.pid_status)
        return

    stage = ss.stage
    if stage != "feed":
        # The Continue bar belongs to the feed only.
        st.html("<style>#nps-feedbar { display: none !important; }</style>")

    mtime = os.path.getmtime(ARTICLES_CSV) if os.path.exists(ARTICLES_CSV) else 0.0
    data, errors = load_study_data(ARTICLES_CSV, mtime)
    if errors:
        render_setup_error(errors)
        return

    logo_outlets = list(data["outlets"]) + [o["domain"] for o in SURVEY_OUTLETS
                                            if o["domain"] not in data["outlets"]]
    logos, missing = load_logos(tuple(logo_outlets), _logo_index())
    if missing:
        present = [f for f, _, _ in _logo_index()]
        render_setup_error(
            [f"No logo for outlet '{o}': add the file data/logos/{logo_hint(o)} "
             f"(www.{logo_hint(o)} and .svg, .webp, .jpg, .gif, .avif also work)."
             for o in missing]
            + [f"Logo folder: {LOGO_DIR}",
               "Files found in it: " + (", ".join(present) if present else "none")])
        return
    data = dict(data, logos=logos)

    if stage == "intro":
        render_intro(data)
    elif stage in ("familiarity", "preference"):
        render_survey(data, stage)
    elif stage == "feed":
        render_feed(data)
    elif stage == "break":
        render_break()
    else:
        render_end()


main()