"""
Reel Transcriber — Streamlit UI

Launch:
    streamlit run app.py

Defaults to local offline transcription via mlx-whisper (Apple Silicon).
First-time use of a model triggers a download from HuggingFace (~1.5GB for
large-v3-turbo). After that, everything runs offline and free.
"""
from __future__ import annotations

import os
import platform
import tempfile
import time
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# mlx-whisper only works on Apple Silicon Macs. On any other platform (e.g. Streamlit
# Cloud's Linux containers) we hide the Local backend option entirely.
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

from reel_agent import (
    MLX_MODEL_REPOS,
    ReelData,
    fetch_pending_rows,
    get_url_from_page,
    load_local_model,
    parse_notion_db_id,
    resolve_data_source_id,
    scrape_and_download,
    transcribe_with_groq,
    transcribe_with_local,
    transcribe_with_mlx,
    update_notion_row,
)

# Faster-whisper model sizes that fit in Streamlit Cloud's 1 GB RAM.
FW_MODEL_SIZES = ["tiny", "base", "small"]

load_dotenv()

st.set_page_config(
    page_title="Reel Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Notion-inspired styling — softer surfaces, document-like layout, quieter buttons.
st.markdown(
    """
<style>
/* Inter / system font everywhere */
html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, .stButton,
.stSelectbox, .stRadio, .stCheckbox {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #37352f;
}

/* Document-like centered content area */
.block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 4rem !important;
    max-width: 900px !important;
}

/* Headings: tighter letter-spacing, Notion weights */
h1 {
    font-size: 40px !important;
    font-weight: 700 !important;
    color: #37352f !important;
    letter-spacing: -0.018em !important;
    line-height: 1.2 !important;
    margin-bottom: 0.25rem !important;
}
h2 {
    font-size: 28px !important;
    font-weight: 600 !important;
    color: #37352f !important;
    letter-spacing: -0.01em !important;
}
h3 {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #37352f !important;
}
h4, h5 {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #37352f !important;
    text-transform: none !important;
    margin-top: 1.25rem !important;
    margin-bottom: 0.5rem !important;
}

/* Captions in Notion's secondary text gray */
.stCaption, [data-testid="stCaptionContainer"], small {
    color: #787774 !important;
    font-size: 13px !important;
}

/* Sidebar — soft off-white background, subtle right border */
[data-testid="stSidebar"] {
    background: #fbfbfa !important;
    border-right: 1px solid #ececea !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 2rem !important;
}
[data-testid="stSidebar"] hr {
    margin: 1.25rem 0 !important;
    border-color: #ececea !important;
}

/* Buttons: subtle by default, dark filled for primary (Notion action style) */
.stButton > button {
    background: #ffffff !important;
    color: #37352f !important;
    border: 1px solid #e9e8e3 !important;
    box-shadow: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: background 0.08s ease, border-color 0.08s ease !important;
}
.stButton > button:hover {
    background: #f4f3ef !important;
    border-color: #d9d8d3 !important;
    color: #37352f !important;
}
.stButton > button[kind="primary"] {
    background: #37352f !important;
    color: #ffffff !important;
    border: 1px solid #37352f !important;
}
.stButton > button[kind="primary"]:hover {
    background: #2c2a26 !important;
    border-color: #2c2a26 !important;
    color: #ffffff !important;
}

/* Inputs: thin border, no fill */
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {
    border: 1px solid #e9e8e3 !important;
    background: #ffffff !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    color: #37352f !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #2383e2 !important;
    box-shadow: 0 0 0 1px #2383e2 !important;
    outline: none !important;
}

/* Tabs: underline-only style (no boxy backgrounds) */
.stTabs [data-baseweb="tab-list"] {
    gap: 28px !important;
    border-bottom: 1px solid #ececea !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 0 !important;
    font-weight: 500 !important;
    color: #787774 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #37352f !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background: #37352f !important;
    height: 2px !important;
}

/* Subtle dividers */
hr {
    border-color: #ececea !important;
    margin: 1.5rem 0 !important;
}

/* Expanders: thin bordered cards */
.streamlit-expanderHeader, [data-testid="stExpander"] details summary {
    background: transparent !important;
    border: 1px solid #ececea !important;
    border-radius: 6px !important;
    color: #37352f !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] details {
    border: 1px solid #ececea !important;
    border-radius: 6px !important;
    background: #ffffff !important;
}

/* Metric cards: soft gray fill, no border */
[data-testid="stMetric"] {
    background: #fbfbfa !important;
    padding: 14px 18px !important;
    border-radius: 6px !important;
    border: 1px solid #ececea !important;
}
[data-testid="stMetricLabel"] {
    color: #787774 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: #37352f !important;
    font-weight: 600 !important;
}

/* Status / progress widgets */
[data-testid="stStatus"], [data-testid="stStatusWidget"] {
    background: #fbfbfa !important;
    border: 1px solid #ececea !important;
    border-radius: 6px !important;
}

/* Info / warning / success banners — softer */
[data-testid="stAlert"] {
    border-radius: 6px !important;
    border: 1px solid transparent !important;
    padding: 12px 16px !important;
}

/* Dataframes: cleaner borders */
[data-testid="stDataFrame"] {
    border: 1px solid #ececea !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

/* Code blocks and inline code */
code {
    background: #f4f3ef !important;
    color: #eb5757 !important;
    padding: 1px 5px !important;
    border-radius: 3px !important;
    font-size: 0.88em !important;
}
</style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Cached resources
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_groq_client(api_key: str):
    from groq import Groq
    return Groq(api_key=api_key)


@st.cache_resource(show_spinner="Loading the Whisper model on the server (one-time, ~30s)…")
def get_faster_whisper_model(model_size: str):
    return load_local_model(model_size)


@st.cache_resource
def get_notion_client(token: str):
    from notion_client import Client as NotionClient
    return NotionClient(auth=token)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — settings
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Setup")
    st.caption("Tweak how the app transcribes and where it gets data.")

    st.markdown("**Where to transcribe**")
    if IS_APPLE_SILICON:
        backend_options = {
            "mlx": "On your Mac (free, offline, fast)",
            "groq": "Groq cloud (fast, needs API key)",
        }
        default_idx = 0
    else:
        backend_options = {
            "fw": "On the server (free, no key, slower)",
            "groq": "Groq cloud (fast, needs API key)",
        }
        default_idx = 0

    backend_id = st.radio(
        "backend",
        options=list(backend_options.keys()),
        format_func=lambda k: backend_options[k],
        index=default_idx,
        label_visibility="collapsed",
        help="Pick where the audio gets transcribed. The free options work without any signup but are slower than Groq.",
    )
    is_mlx = backend_id == "mlx"
    is_fw = backend_id == "fw"
    is_groq = backend_id == "groq"

    if is_mlx:
        model_size = st.selectbox(
            "Model quality",
            list(MLX_MODEL_REPOS.keys()),
            index=4,  # large-v3-turbo
            help="turbo = best speed/quality balance (recommended). large-v3 = highest accuracy. tiny/base = fastest.",
        )
        st.caption("First time you pick a model, it downloads (~1.5 GB for turbo). After that, everything stays on your Mac.")
    elif is_fw:
        model_size = st.selectbox(
            "Model quality",
            FW_MODEL_SIZES,
            index=2,  # small
            help="small = best balance for the free tier. base = faster, less accurate. tiny = fastest, lowest accuracy.",
        )
        st.caption("Runs entirely on the server — no signup, no key. ~30-90 sec per reel. The model loads once when you first transcribe (~30s).")
    else:
        model_size = None

    if IS_APPLE_SILICON:
        st.markdown("---")
        st.markdown("**Browser login (for gated content)**")
        browser = st.selectbox(
            "Pull cookies from",
            ["none", "chrome", "safari", "firefox", "edge", "brave"],
            index=1,
            help=(
                "If you're logged into Instagram, TikTok, Twitter, etc. in this browser, "
                "the app borrows those cookies. Skips rate limits and unlocks private/gated videos."
            ),
        )
        cookies_from_browser = None if browser == "none" else browser
    else:
        # On a server, there's no logged-in browser to borrow cookies from.
        browser = "none"
        cookies_from_browser = None

    # Groq API key — only shown when Groq backend is selected.
    if is_groq:
        st.markdown("---")
        st.markdown("**Groq API key**")
        groq_api_key = st.text_input(
            "Groq API key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Free tier at console.groq.com — sign up, create an API key, paste here. Each user brings their own.",
            placeholder="gsk_…",
            label_visibility="collapsed",
        )
    else:
        groq_api_key = ""

    st.markdown("---")
    st.markdown("**Notion**")
    notion_token = st.text_input(
        "Integration token",
        type="password",
        value=os.getenv("NOTION_TOKEN", ""),
        help="From notion.so/profile/integrations. The same token works for any database you've shared with the integration.",
        placeholder="ntn_…",
    )

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_transcribe_fn():
    if is_mlx:
        return lambda path: transcribe_with_mlx(path, model_size=model_size)
    if is_fw:
        model = get_faster_whisper_model(model_size)
        return lambda path: transcribe_with_local(model, path)
    # is_groq
    if not groq_api_key:
        raise RuntimeError("No Groq API key set. Add one in the sidebar.")
    groq = get_groq_client(groq_api_key)
    return lambda path: transcribe_with_groq(groq, path)


def _fmt(n: Optional[int]) -> str:
    return f"{n:,}" if isinstance(n, int) else "—"


def process_url(url: str, log) -> ReelData:
    """Scrape + transcribe a single URL. log is a callable for progress messages."""
    transcribe_fn = get_transcribe_fn()
    with tempfile.TemporaryDirectory() as tmp:
        log("Grabbing audio…")
        t0 = time.time()
        data = scrape_and_download(url, tmp, None, cookies_from_browser)
        log(f"   {_fmt(data.likes)} likes · {_fmt(data.views)} views · {_fmt(data.comments)} comments · @{data.username or '?'}  _({time.time()-t0:.1f}s)_")
        log("Transcribing…")
        t1 = time.time()
        data.transcript = transcribe_fn(data.video_path)
        log(f"   {len(data.transcript or ''):,} characters  _({time.time()-t1:.1f}s)_")
    return data


def render_result_card(url: str, data: Optional[ReelData], error: Optional[str]):
    if error:
        with st.expander(f"⚠️  Couldn't process  ·  {url}", expanded=False):
            st.error(error)
            err_lower = error.lower()
            # YouTube on cloud servers is famously flaky thanks to Google's anti-bot
            # measures. Give the user a concrete next step instead of a cryptic error.
            if "youtube" in url.lower() and (
                "format is not available" in err_lower
                or "video unavailable" in err_lower
                or "sign in" in err_lower
                or "bot" in err_lower
            ):
                st.info(
                    "🤖 **YouTube tip:** YouTube actively blocks scraping from "
                    "cloud server IPs (which is what this deployed app runs on). "
                    "Instagram, TikTok, and most other sites work fine from the cloud — "
                    "YouTube is the stubborn one.\n\n"
                    "**To transcribe YouTube reliably**, run the app on your own "
                    "machine — see the **Run offline** tab. From a home IP, "
                    "YouTube works for nearly every public video."
                )
        return
    d = data
    header_metrics = []
    if d.views: header_metrics.append(f"{_fmt(d.views)} views")
    if d.likes: header_metrics.append(f"{_fmt(d.likes)} likes")
    if d.comments: header_metrics.append(f"{_fmt(d.comments)} comments")
    summary = "  ·  ".join(header_metrics) if header_metrics else "no metrics"
    header = f"@{d.username or 'unknown'}  ·  {summary}"
    with st.expander(header, expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Views", _fmt(d.views))
        c2.metric("Likes", _fmt(d.likes))
        c3.metric("Comments", _fmt(d.comments))
        st.markdown(f"[Open reel →]({url})")
        if d.caption:
            st.markdown(f"_{d.caption}_")
        st.markdown("**Transcript**")
        st.write(d.transcript or "_(empty)_")


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.title("Reel Transcriber")
st.write(
    "Pull metrics and transcribe short videos from Instagram, YouTube, TikTok, "
    "Twitter, and [1,000+ other sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md) — "
    "paste a few links or sync a whole Notion database."
)

if is_mlx:
    backend_label = f"on your Mac · `{model_size}`"
elif is_fw:
    backend_label = f"on the server · `{model_size}`"
else:
    backend_label = "via Groq cloud"
st.caption(f"Transcribing **{backend_label}** · browser cookies from **{browser}**.")

st.write("")  # breathing room

tab_paste, tab_notion, tab_offline = st.tabs([
    "  Try a few links  ",
    "  Sync with Notion  ",
    "  Run offline  ",
])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Quick Paste
# ──────────────────────────────────────────────────────────────────────────────

with tab_paste:
    st.markdown("##### Drop in some video links")
    st.caption(
        "One per line. Works with Instagram reels, YouTube videos, TikToks, X posts, "
        "and most other public video sites. Hit transcribe and you'll get the metrics + full transcript for each."
    )

    urls_text = st.text_area(
        "Video URLs",
        height=140,
        placeholder=(
            "https://www.instagram.com/reel/…\n"
            "https://www.youtube.com/watch?v=…\n"
            "https://www.tiktok.com/@user/video/…\n"
            "https://x.com/user/status/…"
        ),
        label_visibility="collapsed",
    )

    if st.button("Transcribe", type="primary", key="paste_run"):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if not urls:
            st.warning("Paste at least one link first.")
        else:
            results = []
            label = "Working on 1 reel…" if len(urls) == 1 else f"Working through {len(urls)} reels…"
            with st.status(label, expanded=True) as status:
                log = lambda msg: st.write(msg)
                for i, url in enumerate(urls, 1):
                    st.write(f"---\n**Reel {i} of {len(urls)}** · `{url}`")
                    try:
                        data = process_url(url, log)
                        results.append((url, data, None))
                    except Exception as e:
                        results.append((url, None, str(e)))
                        st.write(f"   ⚠️ {e}")
                ok = sum(1 for _, d, _ in results if d is not None)
                if ok == len(urls):
                    final = f"All {ok} done." if ok > 1 else "Done."
                else:
                    final = f"{ok} of {len(urls)} done — {len(urls) - ok} hit a snag."
                status.update(label=final, state="complete" if ok == len(urls) else "error")

            st.write("")
            st.markdown("##### Results")
            for url, data, err in results:
                render_result_card(url, data, err)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: From Notion
# ──────────────────────────────────────────────────────────────────────────────

with tab_notion:
    if not notion_token:
        st.info("Add your Notion integration token in the sidebar to get going.")
    else:
        st.markdown("##### Which Notion database?")
        st.caption("Paste any link to your database, or just the ID. The app remembers what you've connected to in this session.")

        default_db_input = st.session_state.get(
            "notion_db_input", os.getenv("NOTION_DATABASE_ID", "")
        )
        db_col, btn_col = st.columns([4, 1])
        with db_col:
            db_input = st.text_input(
                "Notion database URL or ID",
                value=default_db_input,
                placeholder="https://www.notion.so/your-workspace/Reels-a585678b…",
                help="Make sure you've shared the database with your integration via … → Connections.",
                label_visibility="collapsed",
            )
        with btn_col:
            connect = st.button("Connect", type="primary", use_container_width=True)

        if connect:
            db_id = parse_notion_db_id(db_input)
            if not db_id:
                st.error("That doesn't look like a Notion link or ID. Try pasting the database URL from your browser.")
            else:
                try:
                    notion = get_notion_client(notion_token)
                    db = notion.databases.retrieve(database_id=db_id)
                    title_arr = db.get("title", [])
                    db_name = title_arr[0]["plain_text"] if title_arr else "(untitled)"
                    ds_id = resolve_data_source_id(notion, db_id)
                    total = 0
                    with_transcript = 0
                    cursor = None
                    while True:
                        kwargs = {"data_source_id": ds_id, "page_size": 100}
                        if cursor:
                            kwargs["start_cursor"] = cursor
                        resp = notion.data_sources.query(**kwargs)
                        for p in resp["results"]:
                            total += 1
                            tprop = p["properties"].get("Transcript", {})
                            if tprop.get("rich_text"):
                                with_transcript += 1
                        if not resp.get("has_more"):
                            break
                        cursor = resp.get("next_cursor")
                    st.session_state["notion_db_id"] = db_id
                    st.session_state["notion_db_name"] = db_name
                    st.session_state["notion_db_input"] = db_input
                    st.session_state["notion_db_total"] = total
                    st.session_state["notion_db_with_transcript"] = with_transcript
                    st.session_state.pop("notion_rows", None)
                except Exception as e:
                    st.error(f"Couldn't connect — {e}")

        if "notion_db_id" in st.session_state:
            total = st.session_state["notion_db_total"]
            done = st.session_state["notion_db_with_transcript"]
            empty = total - done

            # Friendly status banner with metrics
            st.write("")
            name_c, total_c, done_c, empty_c = st.columns([2.5, 1, 1, 1])
            name_c.markdown(f"**📁 {st.session_state['notion_db_name']}**\n\n_connected_")
            total_c.metric("Rows", _fmt(total))
            done_c.metric("Transcribed", _fmt(done))
            empty_c.metric("Still to do", _fmt(empty))

            st.write("")
            st.markdown("##### Pick which rows to run")

            c1, c2, c3 = st.columns([1.8, 1, 2])
            with c1:
                do_force = st.checkbox(
                    "Re-run rows that already have a transcript",
                    value=(empty == 0),
                    help="Off = only empty rows. On = every row, even ones already done.",
                )
            with c2:
                limit = st.number_input("Cap at", min_value=0, value=0, help="0 = run them all")
            with c3:
                title_filter = st.text_input(
                    "Search by name",
                    placeholder="e.g. 'too expensive'",
                )

            if st.button("Show matching rows", key="notion_load"):
                try:
                    notion = get_notion_client(notion_token)
                    rows = list(fetch_pending_rows(
                        notion, st.session_state["notion_db_id"], do_force,
                        title_contains=title_filter or None,
                    ))
                    st.session_state["notion_rows"] = rows
                except Exception as e:
                    st.error(f"Couldn't load rows — {e}")

            if "notion_rows" in st.session_state:
                rows = st.session_state["notion_rows"]
                if not rows:
                    if not do_force and empty == 0:
                        st.info(
                            "Nothing to do — every row already has a transcript. "
                            "Tick **Re-run rows that already have a transcript** above if you want to refresh them."
                        )
                    elif title_filter:
                        st.info(f"No rows match `{title_filter}`. Try a different word.")
                    else:
                        st.info("No rows to show.")
                else:
                    preview = []
                    for p in rows:
                        title_arr = p["properties"].get("Name", {}).get("title", []) or []
                        preview.append({
                            "Name": title_arr[0]["plain_text"] if title_arr else "(no title)",
                            "Reel": get_url_from_page(p) or "(missing)",
                        })
                    st.caption(f"Found {len(rows)} row(s).")
                    st.dataframe(preview, use_container_width=True, hide_index=True)

                    count = min(len(rows), limit) if limit else len(rows)
                    label = f"Run on {count} row" + ("" if count == 1 else "s")
                    if st.button(label, type="primary", key="notion_run"):
                        notion = get_notion_client(notion_token)
                        ok = failed = 0
                        status_label = "Working on 1 row…" if count == 1 else f"Working through {count} rows…"
                        with st.status(status_label, expanded=True) as status:
                            log = lambda msg: st.write(msg)
                            for i, page in enumerate(rows, 1):
                                if limit and i > limit:
                                    break
                                url = get_url_from_page(page)
                                if not url:
                                    st.write(f"**Row {i}** · skipping (no link)")
                                    continue
                                st.write(f"---\n**Row {i} of {count}** · `{url}`")
                                try:
                                    data = process_url(url, log)
                                    update_notion_row(notion, page["id"], data, status="Done")
                                    ok += 1
                                    st.write("   Saved to Notion ✓")
                                except Exception as e:
                                    failed += 1
                                    st.write(f"   ⚠️ {e}")
                            if failed == 0:
                                final = f"All {ok} saved." if ok != 1 else "Done — saved to Notion."
                            else:
                                final = f"{ok} saved, {failed} hit a snag."
                            status.update(label=final, state="complete" if failed == 0 else "error")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: Run offline
# ──────────────────────────────────────────────────────────────────────────────

with tab_offline:
    st.markdown("##### Run the whole thing on your own computer")
    st.caption(
        "Useful if you want truly offline transcription, faster results, "
        "better accuracy with the larger Whisper models, or full privacy."
    )

    st.markdown("**Why bother?**")
    st.markdown(
        """
- 🔒 **Privacy** — audio never leaves your machine
- ⚡ **Speed** — on Apple Silicon, mlx-whisper transcribes a 30-sec reel in ~3 sec
- 🎯 **Accuracy** — run `large-v3` (the best Whisper model), not just `small`
- 📵 **Offline** — works on a plane, in the woods, without WiFi
- ♾️ **No rate limits** — process hundreds of reels in a row
        """
    )

    st.markdown("**Setup — pick your operating system**")

    os_mac, os_win, os_linux = st.tabs(["  🍎 macOS  ", "  🪟 Windows  ", "  🐧 Linux  "])

    # ── macOS ──────────────────────────────────────────────────────────────
    with os_mac:
        st.markdown("**1. Install Python 3.12 + ffmpeg**")
        st.caption("Requires [Homebrew](https://brew.sh). If you don't have it: paste their one-liner first.")
        st.code("brew install python@3.12 ffmpeg git", language="bash")

        st.markdown("**2. Clone and install**")
        st.code(
            """git clone https://github.com/Vijyone/reel-transcriber.git
cd reel-transcriber
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt""",
            language="bash",
        )

        st.markdown("**3. Add your secrets**")
        st.code("cp .env.example .env && open .env", language="bash")
        st.caption("Paste your Notion token + database ID into the file that opens. Groq key is optional.")

        st.markdown("**4. Run it**")
        st.code("./run.sh", language="bash")
        st.markdown("The app opens at **http://localhost:8501**.")
        st.caption(
            "On Apple Silicon Macs you also get the `mlx-whisper` backend — "
            "5-10× faster than the server-side `small` model and supports `large-v3-turbo`."
        )

    # ── Windows ────────────────────────────────────────────────────────────
    with os_win:
        st.markdown("**1. Install Python 3.12 + ffmpeg + Git**")
        st.caption("Open **PowerShell** (search `PowerShell` in the Start menu).")
        st.code(
            "winget install Python.Python.3.12 Gyan.FFmpeg Git.Git",
            language="powershell",
        )
        st.caption("Close PowerShell and reopen it after install so the new programs are on your PATH.")

        st.markdown("**2. Clone and install**")
        st.code(
            """git clone https://github.com/Vijyone/reel-transcriber.git
cd reel-transcriber
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt""",
            language="powershell",
        )

        st.markdown("**3. Add your secrets**")
        st.code("copy .env.example .env\nnotepad .env", language="powershell")
        st.caption("Paste your Notion token + database ID into Notepad and save. Groq key is optional.")

        st.markdown("**4. Run it**")
        st.code("run.bat", language="powershell")
        st.markdown("The app opens at **http://localhost:8501**. You can also double-click `run.bat` from File Explorer.")
        st.caption(
            "Windows uses `faster-whisper` for the local backend (CPU). For a meaningful speedup, "
            "pick a small model — `base` or `small` — in the sidebar."
        )

    # ── Linux ──────────────────────────────────────────────────────────────
    with os_linux:
        st.markdown("**1. Install Python 3.12 + ffmpeg + Git**")
        st.caption("Debian/Ubuntu:")
        st.code(
            "sudo apt update && sudo apt install -y python3.12 python3.12-venv ffmpeg git",
            language="bash",
        )
        st.caption("Fedora: `sudo dnf install python3.12 ffmpeg git` · Arch: `sudo pacman -S python ffmpeg git`")

        st.markdown("**2. Clone and install**")
        st.code(
            """git clone https://github.com/Vijyone/reel-transcriber.git
cd reel-transcriber
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt""",
            language="bash",
        )

        st.markdown("**3. Add your secrets**")
        st.code("cp .env.example .env && nano .env", language="bash")
        st.caption("Paste your Notion token + database ID. Save with `Ctrl+O`, `Enter`, `Ctrl+X`. Groq key is optional.")

        st.markdown("**4. Run it**")
        st.code("./run.sh", language="bash")
        st.markdown("The app opens at **http://localhost:8501**.")

    st.divider()

    st.markdown("##### Just want the Whisper model file?")
    st.caption(
        "If you're using a different transcription tool (e.g. [Vibe](https://thewh1teagle.github.io/vibe/) "
        "or [whisper.cpp](https://github.com/ggerganov/whisper.cpp)) and just need the model:"
    )
    st.markdown(
        """
| Model | Size | Best for | Direct download |
|---|---|---|---|
| `tiny` | 75 MB | Speed over accuracy | [HuggingFace ↗](https://huggingface.co/Systran/faster-whisper-tiny) |
| `base` | 150 MB | Quick, decent quality | [HuggingFace ↗](https://huggingface.co/Systran/faster-whisper-base) |
| `small` | 500 MB | Balanced (this app's default) | [HuggingFace ↗](https://huggingface.co/Systran/faster-whisper-small) |
| `medium` | 1.5 GB | Strong accuracy | [HuggingFace ↗](https://huggingface.co/Systran/faster-whisper-medium) |
| `large-v3-turbo` | 1.5 GB | Best speed/quality balance | [HuggingFace ↗](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo) |
| `large-v3` | 3 GB | Highest accuracy | [HuggingFace ↗](https://huggingface.co/Systran/faster-whisper-large-v3) |

The local install above downloads these automatically the first time you pick a model — you don't need to fetch them manually.
        """
    )
