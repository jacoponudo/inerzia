import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import random
import string
import time
import hashlib

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="News Perception Study",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# SETUP DIRECTORIES
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESPONSES_DIR = os.path.join(DATA_DIR, "responses")
ARTICLES_CSV = os.path.join(DATA_DIR, "articoli_selezionati - articoli scelti.csv")

os.makedirs(RESPONSES_DIR, exist_ok=True)

# ============================================================================
# GENERATE SESSION ID (timestamp + random)
# ============================================================================
def generate_session_id():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{timestamp}_{random_str}"

# ============================================================================
# GENERATE COLOR FOR SOURCE (consistent hash-based color)
# ============================================================================
def get_source_color(source):
    """Generate a consistent color for each source based on its name"""
    hash_obj = hashlib.md5(source.encode())
    hash_hex = hash_obj.hexdigest()[:6]
    return f"#{hash_hex}"

# ============================================================================
# LOAD ARTICLES
# ============================================================================
@st.cache_data(show_spinner=False)
def load_articles():
    if not os.path.exists(ARTICLES_CSV):
        st.error(f"❌ File articoli_selezionati - articoli scelti.csv non trovato in {ARTICLES_CSV}")
        st.info("Assicurati di avere il file nella cartella ./data/")
        st.stop()

    df = pd.read_csv(ARTICLES_CSV)
    return df

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "session_initialized" not in st.session_state:
    session_id = generate_session_id()
    condition = random.randint(1, 4)  # Casuale non bilanciato: 1-4

    st.session_state.update({
        "session_initialized": True,
        "session_id": session_id,
        "condition": condition,
        "phase": 0,
        "articles_df": load_articles(),
        "article_index": 0,
        "responses": [],
        "page_load_time": time.time(),
        "expanded_snippet_count": 0,  # Traccia quante volte ho cliccato "mostra altro"
        "expanded_full": False,
        "current_reaction": None,
        "final_reaction": None,
    })

session_id = st.session_state.session_id
condition = st.session_state.condition
articles_df = st.session_state.articles_df

# ============================================================================
# CONDITION MAPPING
# ============================================================================
CONDITION_MAP = {
    1: {"name": "Con fonti visibili + normali", "with_source": True, "rewritten": False},
    2: {"name": "Senza fonti visibili + normali", "with_source": False, "rewritten": False},
    3: {"name": "Con fonti visibili + riscritti", "with_source": True, "rewritten": True},
    4: {"name": "Senza fonti visibili + riscritti", "with_source": False, "rewritten": True},
}

condition_info = CONDITION_MAP[condition]

# ============================================================================
# PHASE 0 — WELCOME & CONSENT
# ============================================================================
if st.session_state.phase == 0:
    st.markdown("## Welcome to the News Perception Study")
    st.markdown(f"""
You are about to participate in a study about how people perceive and evaluate news articles.

**What will you do?**
- Read 32 news articles
- For each article, you can expand it progressively (testo completo clickabile)
- Leave a reaction (👍) at the end of reading each article
- Your feedback helps us understand how people evaluate news content
- **You are completely free to ignore articles and scroll to the next one** ✓

**Estimated time:** 20-30 minutes

**Your Session ID:** `{session_id}`
**Your Condition:** {condition_info['name']}

By clicking "Start", you confirm that you have read this information and agree to participate.
""")

    if st.button("Start Study"):
        st.session_state.phase = 1
        st.rerun()

# ============================================================================
# PHASE 1 — ARTICLE READING LOOP
# ============================================================================
elif st.session_state.phase == 1:

    article_index = st.session_state.article_index
    total_articles = len(articles_df)

    if article_index >= total_articles:
        # Move to completion phase
        st.session_state.phase = 2
        st.rerun()

    # Get current article
    current_article = articles_df.iloc[article_index]

    # Extract based on condition
    with_source = condition_info["with_source"]

    # Use clean_text as the full article text
    article_text = current_article['clean_text']

    # Determine if we show source
    if with_source:
        source = current_article['domain']
        source_color = get_source_color(source)
        source_display = f"<span style='display: inline-block; width: 12px; height: 12px; background-color: {source_color}; border-radius: 50%; margin-right: 8px; vertical-align: middle;'></span>**{source}**"
    else:
        source_display = None

    # Display progress
    st.markdown(f"**Article {article_index + 1} of {total_articles}**")
    st.progress((article_index + 1) / total_articles)
    st.markdown("---")

    # Display headline (always visible)
    st.markdown(f"### {current_article['topic']}")

    # Display source if applicable
    if source_display:
        st.markdown(source_display)

    st.markdown("---")

    # Display snippet (always visible)
    st.markdown(f"*{current_article['clean_text'][:200]}...*")

    st.markdown("")

    # Split text into sentences for progressive expansion
    import re
    sentences = re.split(r'(?<=[.!?])\s+', article_text)

    # Calculate how many sentences to show based on expansion count
    base_sentences = 3
    expanded_snippet_count = st.session_state.expanded_snippet_count
    sentences_to_show = base_sentences + (expanded_snippet_count * 3)

    # Show initial text + clickable "mostra altro"
    text_so_far = ' '.join(sentences[:min(sentences_to_show, len(sentences))])

    st.markdown(text_so_far)

    # Show "mostra altro" link if there's more text
    if sentences_to_show < len(sentences):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("📖 Mostra altro", key=f"expand_more_{article_index}"):
                st.session_state.expanded_snippet_count += 1
                st.rerun()

        with col2:
            if st.button("📄 Vedi tutto", key=f"expand_full_{article_index}"):
                st.session_state.expanded_full = True
                st.rerun()
    else:
        # All sentences shown, only "Vedi tutto" available
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📄 Vedi tutto", key=f"expand_full_{article_index}"):
                st.session_state.expanded_full = True
                st.rerun()

    # Show full text if expanded_full
    if st.session_state.expanded_full:
        st.markdown("---")
        st.markdown("**Full Article:**")
        st.markdown(article_text)

    st.markdown("---")

    # Info message
    st.info("💡 **Libertà di scelta:** Puoi completamente ignorare questo articolo e passare al successivo se preferisci!")

    st.markdown("")

    # Reaction buttons (always available during interaction)
    st.markdown("**Your reaction:**")
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        if st.button("👍 Like", key=f"reaction_{article_index}", use_container_width=True):
            st.session_state.final_reaction = "like"

    st.markdown("")

    # Save response button (only when reaction is set)
    if st.session_state.final_reaction:
        st.success(f"✅ Reaction saved: 👍")

        if st.button("Continue to next article", key=f"continue_{article_index}", use_container_width=True):
            # Save response
            response = {
                "article_id": article_index,
                "topic": current_article["topic"],
                "domain": current_article["domain"],
                "year": current_article["year"],
                "month": current_article["month"],
                "rating": current_article["rating"],
                "reaction": st.session_state.final_reaction,
                "expansion_clicks": st.session_state.expanded_snippet_count,
                "timestamp": datetime.now().isoformat(),
            }
            st.session_state.responses.append(response)

            # Reset for next article
            st.session_state.article_index += 1
            st.session_state.expanded_snippet_count = 0
            st.session_state.expanded_full = False
            st.session_state.final_reaction = None
            st.rerun()
    else:
        st.info("👉 Please leave a reaction (👍) before continuing.")

    # Alternative: Skip button
    st.markdown("")
    if st.button("⏭️ Skip to next article (no reaction)", key=f"skip_{article_index}", use_container_width=False):
        # Save response with no reaction (skip)
        response = {
            "article_id": article_index,
            "topic": current_article["topic"],
            "domain": current_article["domain"],
            "year": current_article["year"],
            "month": current_article["month"],
            "rating": current_article["rating"],
            "reaction": "skipped",
            "expansion_clicks": st.session_state.expanded_snippet_count,
            "timestamp": datetime.now().isoformat(),
        }
        st.session_state.responses.append(response)

        # Reset for next article
        st.session_state.article_index += 1
        st.session_state.expanded_snippet_count = 0
        st.session_state.expanded_full = False
        st.session_state.final_reaction = None
        st.rerun()

# ============================================================================
# PHASE 2 — COMPLETION & SAVE
# ============================================================================
elif st.session_state.phase == 2:
    st.markdown("## Thank you!")
    st.markdown(f"""
You have completed the study. Your responses have been recorded.

**Session Summary:**
- Session ID: `{session_id}`
- Condition: {condition_info['name']}
- Articles evaluated: {len(st.session_state.responses)}
- Duration: {round((time.time() - st.session_state.page_load_time) / 60, 2)} minutes

Your data is being saved locally.
""")

    # Save to CSV
    if st.session_state.responses:
        responses_df = pd.DataFrame(st.session_state.responses)

        # Add metadata columns
        responses_df.insert(0, "session_id", session_id)
        responses_df.insert(1, "condition", condition)
        responses_df.insert(2, "condition_name", condition_info["name"])
        responses_df["session_date"] = datetime.now().isoformat()
        responses_df["session_duration_min"] = round((time.time() - st.session_state.page_load_time) / 60, 2)

        # Save to CSV
        csv_filename = f"{session_id}_responses.csv"
        csv_path = os.path.join(RESPONSES_DIR, csv_filename)
        responses_df.to_csv(csv_path, index=False)

        st.success(f"✅ Data saved to: {csv_filename}")
        st.markdown(f"📁 Location: `{RESPONSES_DIR}`")

    st.markdown("---")
    st.markdown("You may now close this window.")