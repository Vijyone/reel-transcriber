"""
Reel Transcriber — Streamlit UI (local Mac only)

Launch:
    ./run.sh

Transcribes via mlx-whisper on Apple Silicon. Everything stays on your machine.
First time you pick a model, it downloads from HuggingFace (~1.5 GB for turbo).
After that, runs offline and free.
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from reel_agent import (
    MLX_MODEL_REPOS,
    ReelData,
    fetch_pending_rows,
    get_available_notion_columns,
    get_url_from_page,
    parse_notion_db_id,
    resolve_data_source_id,
    scrape_and_download,
    transcribe_with_mlx_stream,
    update_notion_row,
)

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
h2 { font-size: 28px !important; font-weight: 600 !important; color: #37352f !important; letter-spacing: -0.01em !important; }
h3 { font-size: 20px !important; font-weight: 600 !important; color: #37352f !important; }
h4, h5 {
    font-size: 14px !important; font-weight: 600 !important; color: #37352f !important;
    text-transform: none !important; margin-top: 1.25rem !important; margin-bottom: 0.5rem !important;
}

/* Captions in Notion's secondary text gray */
.stCaption, [data-testid="stCaptionContainer"], small { color: #787774 !important; font-size: 13px !important; }

/* Sidebar — soft off-white background, subtle right border */
[data-testid="stSidebar"] { background: #fbfbfa !important; border-right: 1px solid #ececea !important; }
[data-testid="stSidebar"] .block-container { padding-top: 2rem !important; }
[data-testid="stSidebar"] hr { margin: 1.25rem 0 !important; border-color: #ececea !important; }

/* Buttons: subtle by default, dark filled for primary (Notion action style) */
.stButton > button {
    background: #ffffff !important; color: #37352f !important;
    border: 1px solid #e9e8e3 !important; box-shadow: none !important;
    border-radius: 6px !important; font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: background 0.08s ease, border-color 0.08s ease !important;
}
.stButton > button:hover { background: #f4f3ef !important; border-color: #d9d8d3 !important; }
.stButton > button[kind="primary"] { background: #37352f !important; color: #ffffff !important; border: 1px solid #37352f !important; }
.stButton > button[kind="primary"]:hover { background: #2c2a26 !important; border-color: #2c2a26 !important; color: #ffffff !important; }

/* Inputs: thin border, no fill */
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {
    border: 1px solid #e9e8e3 !important; background: #ffffff !important;
    border-radius: 6px !important; box-shadow: none !important; color: #37352f !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #2383e2 !important; box-shadow: 0 0 0 1px #2383e2 !important; outline: none !important;
}

/* Tabs: underline-only style (no boxy backgrounds) */
.stTabs [data-baseweb="tab-list"] { gap: 28px !important; border-bottom: 1px solid #ececea !important; margin-bottom: 1.5rem !important; }
.stTabs [data-baseweb="tab"] { padding: 8px 0 !important; font-weight: 500 !important; color: #787774 !important; background: transparent !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #37352f !important; }
.stTabs [data-baseweb="tab-highlight"] { background: #37352f !important; height: 2px !important; }

/* Subtle dividers */
hr { border-color: #ececea !important; margin: 1.5rem 0 !important; }

/* Expanders */
.streamlit-expanderHeader, [data-testid="stExpander"] details summary {
    background: transparent !important; border: 1px solid #ececea !important;
    border-radius: 6px !important; color: #37352f !important; font-weight: 500 !important;
}
[data-testid="stExpander"] details { border: 1px solid #ececea !important; border-radius: 6px !important; background: #ffffff !important; }

/* Metric cards: soft gray fill */
[data-testid="stMetric"] { background: #fbfbfa !important; padding: 14px 18px !important; border-radius: 6px !important; border: 1px solid #ececea !important; }
[data-testid="stMetricLabel"] { color: #787774 !important; font-size: 12px !important; font-weight: 500 !important; }
[data-testid="stMetricValue"] { color: #37352f !important; font-weight: 600 !important; }

/* Status / progress widgets */
[data-testid="stStatus"], [data-testid="stStatusWidget"] {
    background: #fbfbfa !important; border: 1px solid #ececea !important; border-radius: 6px !important;
}

/* Alerts softer */
[data-testid="stAlert"] { border-radius: 6px !important; border: 1px solid transparent !important; padding: 12px 16px !important; }

/* Dataframes: cleaner borders */
[data-testid="stDataFrame"] { border: 1px solid #ececea !important; border-radius: 6px !important; overflow: hidden !important; }

/* Inline code */
code { background: #f4f3ef !important; color: #eb5757 !important; padding: 1px 5px !important; border-radius: 3px !important; font-size: 0.88em !important; }
</style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Cached resources
# ──────────────────────────────────────────────────────────────────────────────

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

    st.markdown("**Model quality**")
    model_size = st.selectbox(
        "Model quality",
        list(MLX_MODEL_REPOS.keys()),
        index=4,  # large-v3-turbo
        label_visibility="collapsed",
        help="turbo = best speed/quality balance (recommended). large-v3 = highest accuracy. tiny/base = fastest.",
    )
    st.caption(
        "First time you pick a model, it downloads from HuggingFace (~1.5 GB for turbo). "
        "After that everything stays on your Mac — no internet, no API keys, no rate limits."
    )

    st.markdown("---")
    use_existing_captions = st.checkbox(
        "Use existing captions when available",
        value=True,
        help=(
            "If the video already has captions (e.g. YouTube auto-captions or "
            "uploaded subtitles), use those instead of running Whisper. Much "
            "faster — turns a 5-min transcription on a long video into ~2 sec. "
            "Quality is slightly lower than Whisper on technical content."
        ),
    )

    st.markdown("---")
    st.markdown("**Browser login (for gated content)**")
    browser = st.selectbox(
        "Pull cookies from",
        ["none", "chrome", "safari", "firefox", "edge", "brave"],
        index=1,
        help=(
            "If you're logged into Instagram, TikTok, YouTube, etc. in this browser, "
            "the app borrows those cookies. Skips rate limits and unlocks private/gated videos."
        ),
    )
    cookies_from_browser = None if browser == "none" else browser

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

def _fmt(n: Optional[int]) -> str:
    return f"{n:,}" if isinstance(n, int) else "—"


def process_url(url: str, log) -> ReelData:
    """Scrape + transcribe a single URL, with live progress in the UI.

    Fast path: if the source platform already has captions (e.g. YouTube),
    those are used directly and Whisper is skipped — instant transcript.
    Slow path: stream Whisper segments with a live progress bar.

    `log` writes to the surrounding status block."""
    with tempfile.TemporaryDirectory() as tmp:
        log("Grabbing audio + checking for captions…")
        t0 = time.time()
        data = scrape_and_download(
            url, tmp, None, cookies_from_browser, try_subtitles=use_existing_captions
        )
        log(
            f"   {_fmt(data.likes)} likes · {_fmt(data.views)} views · "
            f"{_fmt(data.comments)} comments · @{data.username or '?'}  _({time.time()-t0:.1f}s)_"
        )

        # Fast path — the platform already has captions, no need to transcribe.
        if use_existing_captions and data.subtitle_text:
            src = data.subtitle_source or "captions"
            label = "uploaded subtitles" if src == "manual" else "auto-captions"
            log(f"Using existing {label} — skipping Whisper.")
            data.transcript = data.subtitle_text
            log(f"   {len(data.transcript):,} characters from {label}")
            return data

        # Slow path — run Whisper with live progress.
        duration = data.duration or 0.0
        if duration:
            log(f"Transcribing with Whisper… (audio is {duration:.0f}s long)")
        else:
            log("Transcribing with Whisper…")

        bar = st.progress(0.0, text="0% · waiting for first segment…")
        live_box = st.empty()
        parts = []
        t1 = time.time()
        for seg_text, end_sec in transcribe_with_mlx_stream(
            data.video_path, model_size=model_size
        ):
            parts.append(seg_text)
            if duration > 0:
                p = min(1.0, end_sec / duration)
                bar.progress(p, text=f"{int(p * 100)}% · {end_sec:.0f}s of {duration:.0f}s")
            live_box.markdown("> " + " ".join(parts))
        bar.progress(1.0, text="100% · done")
        data.transcript = " ".join(parts).strip()
        log(f"   {len(data.transcript):,} characters  _({time.time()-t1:.1f}s)_")
    return data


def render_result_card(url: str, data: Optional[ReelData], error: Optional[str]):
    if error:
        with st.expander(f"⚠️  Couldn't process  ·  {url}", expanded=False):
            st.error(error)
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
    "Pull metrics and transcribe short videos from Instagram, YouTube, TikTok, Twitter, "
    "and most other video sites. Everything runs on your Mac — no API keys, no internet needed after first model download."
)
st.caption(f"Transcribing on your Mac · `{model_size}` · IG cookies from **{browser}**.")

st.write("")

tab_paste, tab_notion = st.tabs(["  Try a few links  ", "  Sync with Notion  "])


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
            label = "Working on 1 link…" if len(urls) == 1 else f"Working through {len(urls)} links…"
            with st.status(label, expanded=True) as status:
                log = lambda msg: st.write(msg)
                for i, url in enumerate(urls, 1):
                    st.write(f"---\n**Link {i} of {len(urls)}** · `{url}`")
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

        default_db_input = st.session_state.get("notion_db_input", os.getenv("NOTION_DATABASE_ID", ""))
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
                    available_cols = get_available_notion_columns(notion, db_id)
                    st.session_state["notion_db_id"] = db_id
                    st.session_state["notion_db_name"] = db_name
                    st.session_state["notion_db_input"] = db_input
                    st.session_state["notion_db_total"] = total
                    st.session_state["notion_db_with_transcript"] = with_transcript
                    st.session_state["notion_available_cols"] = available_cols
                    st.session_state.pop("notion_rows", None)
                except Exception as e:
                    st.error(f"Couldn't connect — {e}")

        if "notion_db_id" in st.session_state:
            total = st.session_state["notion_db_total"]
            done = st.session_state["notion_db_with_transcript"]
            empty = total - done

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
                title_filter = st.text_input("Search by name", placeholder="e.g. 'too expensive'")

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
                        available_cols = st.session_state.get("notion_available_cols")
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
                                data = None
                                try:
                                    data = process_url(url, log)
                                except Exception as e:
                                    failed += 1
                                    st.write(f"   ⚠️ scrape/transcribe failed: {e}")
                                    continue
                                try:
                                    update_notion_row(
                                        notion, page["id"], data,
                                        status="Done", available_props=available_cols,
                                    )
                                    ok += 1
                                    st.write("   Saved to Notion ✓")
                                except Exception as e:
                                    failed += 1
                                    st.write(f"   ⚠️ Notion write failed: {e}")
                                    with st.expander(f"📝 Transcript (Notion write failed — copy from here)", expanded=False):
                                        st.markdown(
                                            f"**@{data.username or '?'}** · "
                                            f"{_fmt(data.likes)} likes · "
                                            f"{_fmt(data.views)} views · "
                                            f"{_fmt(data.comments)} comments"
                                        )
                                        if data.caption:
                                            st.markdown(f"_{data.caption}_")
                                        st.markdown("**Transcript**")
                                        st.write(data.transcript or "_(empty)_")
                            if failed == 0:
                                final = f"All {ok} saved." if ok != 1 else "Done — saved to Notion."
                            else:
                                final = f"{ok} saved, {failed} hit a snag."
                            status.update(label=final, state="complete" if failed == 0 else "error")
