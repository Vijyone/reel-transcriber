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
        st.markdown("**Instagram login**")
        browser = st.selectbox(
            "Pull cookies from",
            ["none", "chrome", "safari", "firefox", "edge", "brave"],
            index=1,
            help="If you're logged into Instagram in this browser, the app borrows those cookies. Skips rate limits and unlocks more metadata.",
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
st.write("Pull metrics and transcribe Instagram reels — paste a few links or sync a whole Notion database.")

if is_mlx:
    backend_label = f"on your Mac · `{model_size}`"
elif is_fw:
    backend_label = f"on the server · `{model_size}`"
else:
    backend_label = "via Groq cloud"
st.caption(f"Transcribing **{backend_label}** · IG cookies from **{browser}**.")

st.write("")  # breathing room

tab_paste, tab_notion = st.tabs(["  Try a few reels  ", "  Sync with Notion  "])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Quick Paste
# ──────────────────────────────────────────────────────────────────────────────

with tab_paste:
    st.markdown("##### Drop in some Instagram reel links")
    st.caption("One per line. Hit transcribe and you'll get the metrics + a full transcript for each.")

    urls_text = st.text_area(
        "Reel URLs",
        height=140,
        placeholder="https://www.instagram.com/reel/…\nhttps://www.instagram.com/reel/…",
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
