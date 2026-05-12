"""
Reel Transcriber — Streamlit UI (local Mac only)

Launch:
    ./run.sh

Transcribes via mlx-whisper on Apple Silicon. Everything stays on your machine.
First time you pick a model, it downloads from HuggingFace (~1.5 GB for turbo).
After that, runs offline and free.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

from reel_agent import (
    MLX_MODEL_REPOS,
    ReelData,
    VideoListing,
    create_notion_row,
    detect_creator_source,
    enumerate_youtube,
    fetch_pending_rows,
    get_available_notion_columns,
    get_notion_title_column,
    get_url_from_page,
    list_existing_urls,
    parse_notion_db_id,
    resolve_data_source_id,
    scrape_and_download,
    transcribe_with_mlx_stream,
    update_notion_row,
)

load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# Multi-integration storage
#   Persisted at ~/.config/reel-transcriber/integrations.json (chmod 600).
#   Each entry: {"name": "Personal", "token": "ntn_..."}.
# ──────────────────────────────────────────────────────────────────────────────

INTEGRATIONS_FILE = Path.home() / ".config" / "reel-transcriber" / "integrations.json"


def load_integrations() -> List[dict]:
    if not INTEGRATIONS_FILE.exists():
        return []
    try:
        data = json.loads(INTEGRATIONS_FILE.read_text())
        if isinstance(data, list):
            return [{"name": str(d.get("name", "")), "token": str(d.get("token", ""))}
                    for d in data if isinstance(d, dict)]
    except Exception:
        pass
    return []


def save_integrations(items: List[dict]) -> None:
    INTEGRATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    INTEGRATIONS_FILE.write_text(json.dumps(items, indent=2))
    try:
        os.chmod(INTEGRATIONS_FILE, 0o600)  # owner-only read/write
    except OSError:
        pass


def init_integrations() -> None:
    """Load saved integrations into session_state. On first launch, seed from
    the NOTION_TOKEN env var if present so existing users don't have to retype."""
    if "integrations" in st.session_state:
        return
    items = load_integrations()
    if not items and os.getenv("NOTION_TOKEN"):
        items = [{"name": "Default", "token": os.getenv("NOTION_TOKEN")}]
        save_integrations(items)
    st.session_state["integrations"] = items


def resolve_notion_for_db(
    db_id: str, integrations: List[dict]
) -> Tuple[object, str, dict]:
    """Probe each saved token until one can access the database. Returns
    (notion_client, integration_name, db_metadata). Raises if none work."""
    from notion_client import Client as NotionClient
    last_err: Optional[Exception] = None
    tried = 0
    for item in integrations:
        token = (item.get("token") or "").strip()
        if not token:
            continue
        tried += 1
        try:
            client = NotionClient(auth=token)
            db = client.databases.retrieve(database_id=db_id)
            return client, item.get("name", "(unnamed)"), db
        except Exception as e:
            last_err = e
            continue
    if tried == 0:
        raise RuntimeError(
            "No Notion integrations configured. Add one in the sidebar first."
        )
    raise RuntimeError(
        f"None of your {tried} integration(s) can access this database. "
        f"Make sure the integration is added to the database via ··· → "
        f"Connections in Notion. Last error: {last_err}"
    )


init_integrations()

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

/* Use the full width — no more 900px corridor with empty margins */
.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 100% !important;
}

/* Hide Streamlit's "Made with Streamlit" footer and the menu for a cleaner header */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }

/* Compact headings */
h1 {
    font-size: 28px !important; font-weight: 700 !important; color: #37352f !important;
    letter-spacing: -0.015em !important; line-height: 1.2 !important;
    margin: 0 !important; padding: 0 !important;
}
h2 { font-size: 22px !important; font-weight: 600 !important; color: #37352f !important; letter-spacing: -0.01em !important; }
h3 { font-size: 17px !important; font-weight: 600 !important; color: #37352f !important; }
h4, h5 {
    font-size: 13px !important; font-weight: 600 !important; color: #787774 !important;
    text-transform: uppercase !important; letter-spacing: 0.04em !important;
    margin-top: 0.75rem !important; margin-bottom: 0.5rem !important;
}

/* Captions tighter */
.stCaption, [data-testid="stCaptionContainer"], small { color: #787774 !important; font-size: 12px !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #fbfbfa !important; border-right: 1px solid #ececea !important; }
[data-testid="stSidebar"] .block-container { padding-top: 1.25rem !important; padding-bottom: 1.25rem !important; }
[data-testid="stSidebar"] hr { margin: 1rem 0 !important; border-color: #ececea !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: 0.4rem !important; }

/* Buttons: subtle by default, dark for primary */
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

/* Inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {
    border: 1px solid #e9e8e3 !important; background: #ffffff !important;
    border-radius: 6px !important; box-shadow: none !important; color: #37352f !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #2383e2 !important; box-shadow: 0 0 0 1px #2383e2 !important; outline: none !important;
}

/* Tabs: underline-only */
.stTabs [data-baseweb="tab-list"] { gap: 24px !important; border-bottom: 1px solid #ececea !important; margin-top: 0.5rem !important; margin-bottom: 1rem !important; }
.stTabs [data-baseweb="tab"] { padding: 8px 0 !important; font-weight: 500 !important; color: #787774 !important; background: transparent !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #37352f !important; }
.stTabs [data-baseweb="tab-highlight"] { background: #37352f !important; height: 2px !important; }

/* Subtle dividers */
hr { border-color: #ececea !important; margin: 1rem 0 !important; }

/* Expander cards: tighter */
[data-testid="stExpander"] details summary {
    background: transparent !important; border: 1px solid #ececea !important;
    border-radius: 6px !important; color: #37352f !important; font-weight: 500 !important;
    padding: 8px 12px !important;
}
[data-testid="stExpander"] details { border: 1px solid #ececea !important; border-radius: 6px !important; background: #ffffff !important; margin-bottom: 8px !important; }
[data-testid="stExpander"] details[open] summary { border-bottom: 1px solid #ececea !important; border-radius: 6px 6px 0 0 !important; }

/* Metric cards: tighter */
[data-testid="stMetric"] { background: #fbfbfa !important; padding: 10px 14px !important; border-radius: 6px !important; border: 1px solid #ececea !important; }
[data-testid="stMetricLabel"] { color: #787774 !important; font-size: 11px !important; font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.04em !important; }
[data-testid="stMetricValue"] { color: #37352f !important; font-weight: 600 !important; font-size: 18px !important; }

/* Status / progress widgets */
[data-testid="stStatus"], [data-testid="stStatusWidget"] {
    background: #fbfbfa !important; border: 1px solid #ececea !important; border-radius: 6px !important;
}

[data-testid="stAlert"] { border-radius: 6px !important; border: 1px solid transparent !important; padding: 10px 14px !important; }

[data-testid="stDataFrame"] { border: 1px solid #ececea !important; border-radius: 6px !important; overflow: hidden !important; }

code { background: #f4f3ef !important; color: #eb5757 !important; padding: 1px 5px !important; border-radius: 3px !important; font-size: 0.88em !important; }

/* Compact vertical gaps between widgets */
.element-container { margin-bottom: 0.5rem !important; }
.stMarkdown p { margin-bottom: 0.4rem !important; }

/* Hero strip — title + status on one row */
.hero-row { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; margin-bottom: 0; }
.hero-status { color: #787774; font-size: 12px; }
.hero-status code { background: #f4f3ef; color: #37352f !important; padding: 1px 5px; border-radius: 3px; font-size: 11px; }
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
    model_size = st.selectbox(
        "Whisper model",
        list(MLX_MODEL_REPOS.keys()),
        index=4,  # large-v3-turbo
        help="turbo = best speed/quality balance. large-v3 = highest accuracy. tiny/base = fastest.",
    )

    use_existing_captions = st.checkbox(
        "Skip Whisper if captions exist",
        value=True,
        help=(
            "Use existing captions (e.g. YouTube auto-captions) when available — "
            "turns a 5-min transcription on a long video into ~2 sec. Quality slightly "
            "lower than Whisper on technical content."
        ),
    )

    browser = st.selectbox(
        "Browser cookies",
        ["none", "chrome", "safari", "firefox", "edge", "brave"],
        index=1,
        help=(
            "If you're logged into Instagram, TikTok, YouTube, etc. in this browser, "
            "the app borrows those cookies to skip rate limits and unlock gated videos."
        ),
    )
    cookies_from_browser = None if browser == "none" else browser

    st.markdown("---")
    st.markdown("**Notion integrations**")
    st.caption(
        "Add one or more. When you paste a database URL, the app probes each "
        "token to find which one has access."
    )

    integrations = st.session_state["integrations"]

    # Edit existing integrations in-place
    for i, item in enumerate(integrations):
        c_name, c_tok, c_del = st.columns([1.2, 2.5, 0.4])
        new_name = c_name.text_input(
            "Name", value=item.get("name", ""), key=f"int_name_{i}",
            label_visibility="collapsed", placeholder="Name",
        )
        new_token = c_tok.text_input(
            "Token", value=item.get("token", ""), key=f"int_tok_{i}",
            type="password", label_visibility="collapsed", placeholder="ntn_…",
        )
        delete = c_del.button("✕", key=f"int_del_{i}", help="Remove this integration")
        if delete:
            integrations.pop(i)
            save_integrations(integrations)
            # Invalidate cached DB→integration mapping since tokens changed.
            st.session_state.pop("notion_db_integration", None)
            st.rerun()
        if new_name != item.get("name", "") or new_token != item.get("token", ""):
            integrations[i] = {"name": new_name, "token": new_token}
            save_integrations(integrations)
            st.session_state.pop("notion_db_integration", None)

    if st.button("+ Add integration", use_container_width=True):
        integrations.append({"name": "", "token": ""})
        save_integrations(integrations)
        st.rerun()

    st.caption(
        "🔒 Tokens are stored locally at `~/.config/reel-transcriber/integrations.json` "
        "(owner-only). Anyone with a token can access whatever the integration "
        "has been added to — share carefully."
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
        if use_existing_captions:
            log("Checking for existing captions…")
        else:
            log("Grabbing audio…")
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
# Header — compact hero with title + live status on one row
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    f"""<div class='hero-row'>
        <h1>Reel Transcriber</h1>
        <div class='hero-status'>
            on your Mac · <code>{model_size}</code> · cookies from <code>{browser}</code>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

tab_paste, tab_notion, tab_creator = st.tabs([
    "  Try a few links  ",
    "  Sync with Notion  ",
    "  Pull from a creator  ",
])


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Quick Paste
# ──────────────────────────────────────────────────────────────────────────────

with tab_paste:
    # Two-pane layout: input + button on the left, live progress + results on the right.
    pane_in, pane_out = st.columns([1, 2], gap="large")

    with pane_in:
        st.markdown("##### Paste video links")
        urls_text = st.text_area(
            "Video URLs",
            height=240,
            placeholder=(
                "https://www.instagram.com/reel/…\n"
                "https://www.youtube.com/watch?v=…\n"
                "https://www.tiktok.com/@user/video/…\n"
                "https://x.com/user/status/…"
            ),
            label_visibility="collapsed",
        )
        run_clicked = st.button("Transcribe", type="primary", key="paste_run", use_container_width=True)
        st.caption("Instagram · YouTube · TikTok · X · Vimeo · Reddit · 1000+ sites")

    with pane_out:
        if not run_clicked:
            st.markdown("##### Results")
            st.caption("Hit **Transcribe** on the left and results will appear here.")
        else:
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

                # 2-up grid of result cards
                st.markdown("##### Results")
                grid = [st.columns(2, gap="medium") for _ in range((len(results) + 1) // 2)]
                for idx, (url, data, err) in enumerate(results):
                    with grid[idx // 2][idx % 2]:
                        render_result_card(url, data, err)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: From Notion
# ──────────────────────────────────────────────────────────────────────────────

with tab_notion:
    valid_integrations = [it for it in integrations if (it.get("token") or "").strip()]
    if not valid_integrations:
        st.info("Add a Notion integration in the sidebar to get going.")
    else:
        # Compact connect bar — paste a DB URL, the app auto-picks the right integration.
        default_db_input = st.session_state.get("notion_db_input", os.getenv("NOTION_DATABASE_ID", ""))
        db_col, btn_col = st.columns([5, 1])
        with db_col:
            db_input = st.text_input(
                "Notion database URL or ID",
                value=default_db_input,
                placeholder="Paste your Notion database URL or ID…",
                help=(
                    f"The app will probe your {len(valid_integrations)} saved "
                    "integration(s) and use whichever one has access."
                ),
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
                    notion, integration_name, db = resolve_notion_for_db(db_id, valid_integrations)
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
                    st.session_state["notion_db_integration"] = integration_name
                    st.session_state.pop("notion_rows", None)
                except Exception as e:
                    st.error(f"Couldn't connect — {e}")

        if "notion_db_id" in st.session_state:
            total = st.session_state["notion_db_total"]
            done = st.session_state["notion_db_with_transcript"]
            empty = total - done
            via = st.session_state.get("notion_db_integration", "")

            # Single-line status banner — DB name on the left, inline counts on the right
            st.markdown(
                f"""<div style='display:flex;align-items:center;justify-content:space-between;
                    padding:10px 14px;background:#fbfbfa;border:1px solid #ececea;
                    border-radius:6px;margin:0.5rem 0;'>
                    <div><strong>📁 {st.session_state['notion_db_name']}</strong>
                         <span style='color:#787774;margin-left:0.5rem;'>via <strong style='color:#37352f'>{via}</strong></span></div>
                    <div style='color:#787774;font-size:13px;'>
                        <strong style='color:#37352f'>{_fmt(total)}</strong> rows ·
                        <strong style='color:#37352f'>{_fmt(done)}</strong> transcribed ·
                        <strong style='color:#37352f'>{_fmt(empty)}</strong> to do
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Tight filters row — checkbox + cap + search + button on a single row
            c1, c2, c3, c4 = st.columns([2.5, 1, 2.5, 1])
            with c1:
                do_force = st.checkbox(
                    "Include already-done rows",
                    value=(empty == 0),
                    help="Off = only rows with empty Transcript. On = every row.",
                )
            with c2:
                limit = st.number_input("Cap", min_value=0, value=0, help="0 = run them all", label_visibility="visible")
            with c3:
                title_filter = st.text_input("Search by name", placeholder="e.g. 'too expensive'")
            with c4:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)  # align with input baseline
                load_clicked = st.button("Show rows", key="notion_load", use_container_width=True)

            if load_clicked:
                try:
                    # Use the integration that matched this DB during connect.
                    matched = next(
                        (it for it in integrations
                         if it.get("name") == st.session_state.get("notion_db_integration")),
                        None,
                    )
                    if not matched:
                        raise RuntimeError("Reconnect — the matching integration was removed.")
                    notion = get_notion_client(matched["token"])
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
                        st.info("Every row already has a transcript. Tick **Include already-done rows** to refresh them.")
                    elif title_filter:
                        st.info(f"No rows match `{title_filter}`.")
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

                    count = min(len(rows), limit) if limit else len(rows)
                    label = f"Run on {count} row" + ("" if count == 1 else "s")

                    # Preview rows on the left, run button on the right — same baseline
                    preview_col, run_col = st.columns([4, 1])
                    with preview_col:
                        st.caption(f"Found **{len(rows)}** row{'' if len(rows) == 1 else 's'} matching.")
                    with run_col:
                        run_clicked = st.button(label, type="primary", key="notion_run", use_container_width=True)
                    st.dataframe(preview, use_container_width=True, hide_index=True, height=min(35 * (len(rows) + 1) + 3, 280))

                    if run_clicked:
                        matched = next(
                            (it for it in integrations
                             if it.get("name") == st.session_state.get("notion_db_integration")),
                            None,
                        )
                        if not matched:
                            st.error("Reconnect — the matching integration was removed.")
                            st.stop()
                        notion = get_notion_client(matched["token"])
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


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: Pull from a creator
# ──────────────────────────────────────────────────────────────────────────────

with tab_creator:
    if "notion_db_id" not in st.session_state:
        st.info(
            "Connect a Notion database in the **Sync with Notion** tab first — "
            "that's where new rows will be created."
        )
    else:
        via = st.session_state.get("notion_db_integration", "")
        st.caption(
            f"New rows will be created in **{st.session_state['notion_db_name']}** "
            f"(via _{via}_)."
        )

        # URL input + Enumerate
        url_col, btn_col = st.columns([5, 1])
        with url_col:
            creator_url = st.text_input(
                "Creator URL",
                placeholder="https://www.youtube.com/@channel  or  https://www.youtube.com/playlist?list=…",
                label_visibility="collapsed",
                key="creator_url",
            )
        with btn_col:
            enumerate_clicked = st.button(
                "Find videos", type="primary", use_container_width=True, key="creator_enum"
            )

        # Filter row — date range, shorts toggle, limit
        c1, c2, c3, c4 = st.columns([1.4, 1.4, 1.5, 1])
        with c1:
            from_date = st.date_input(
                "From date", value=None, key="creator_from",
                help="Leave blank for no lower bound.",
            )
        with c2:
            to_date = st.date_input(
                "To date", value=None, key="creator_to",
                help="Leave blank for no upper bound.",
            )
        with c3:
            include_shorts = st.checkbox(
                "Include YouTube Shorts",
                value=False, key="creator_shorts",
                help="Pull Shorts in addition to long-form videos.",
            )
        with c4:
            enum_limit = st.number_input(
                "Max", min_value=0, value=200, key="creator_max",
                help="Hard cap on enumeration. 0 = no cap. Keep low for big channels.",
            )

        # Enumerate handler
        if enumerate_clicked:
            url = creator_url.strip()
            if not url:
                st.warning("Paste a YouTube channel or playlist URL first.")
            else:
                source = detect_creator_source(url)
                if source == "instagram":
                    st.error(
                        "Instagram profile enumeration isn't supported here — IG actively blocks "
                        "third-party profile scraping and the workarounds aren't worth the hassle. "
                        "Paste individual reel URLs in the **Try a few links** tab instead — that "
                        "works without any auth."
                    )
                elif source != "youtube":
                    st.error("Use a YouTube channel or playlist URL.")
                else:
                    matched = next(
                        (it for it in integrations
                         if it.get("name") == st.session_state.get("notion_db_integration")),
                        None,
                    )
                    if not matched:
                        st.error("Reconnect — the Notion integration that opened this DB was removed.")
                    else:
                        try:
                            has_date_filter = bool(from_date or to_date)
                            if has_date_filter:
                                spinner_msg = (
                                    f"Fetching dates for up to {enum_limit or '∞'} videos — "
                                    "~1.5 sec each, so this can take a few minutes for big channels."
                                )
                            else:
                                spinner_msg = "Enumerating channel…"
                            with st.spinner(spinner_msg):
                                listings = enumerate_youtube(
                                    url,
                                    limit=(enum_limit or None),
                                    include_shorts=include_shorts,
                                    cookies_from_browser=cookies_from_browser,
                                    need_dates=has_date_filter,
                                )

                            # Apply date filter (yt-dlp / instaloader both expose YYYY-MM-DD via VideoListing)
                            if from_date or to_date:
                                from_str = from_date.strftime("%Y-%m-%d") if from_date else None
                                to_str = to_date.strftime("%Y-%m-%d") if to_date else None
                                listings = [
                                    v for v in listings
                                    if (not from_str or (v.upload_date and v.upload_date >= from_str))
                                       and (not to_str or (v.upload_date and v.upload_date <= to_str))
                                ]

                            # Dedup against the target DB
                            notion_client = get_notion_client(matched["token"])
                            existing = list_existing_urls(notion_client, st.session_state["notion_db_id"])

                            st.session_state["creator_listings"] = listings
                            st.session_state["creator_existing_urls"] = existing
                            st.session_state["creator_source"] = source
                            st.session_state.pop("creator_run_done", None)
                        except Exception as e:
                            st.error(f"Enumeration failed — {e}")

        # Show preview if we have a result
        if "creator_listings" in st.session_state:
            listings = st.session_state["creator_listings"]
            existing = st.session_state["creator_existing_urls"]

            new_count = sum(1 for v in listings if v.url not in existing)
            already_count = len(listings) - new_count

            st.write("")
            st.caption(
                f"Found **{len(listings)}** video{'' if len(listings) == 1 else 's'}  ·  "
                f"**{new_count}** new  ·  **{already_count}** already in this DB"
            )

            if not listings:
                st.info("No videos match the filter. Try widening the date range, raising the cap, or check the URL.")
            else:
                # Build a DataFrame for the editable preview
                import pandas as pd

                def _fmt_dur(secs):
                    if not secs:
                        return "—"
                    m, s = divmod(int(secs), 60)
                    return f"{m}:{s:02d}"

                rows = []
                for v in listings:
                    is_new = v.url not in existing
                    rows.append({
                        "Run": is_new,  # pre-check new, leave existing unchecked
                        "Date": v.upload_date or "?",
                        "Title": (v.title or "")[:120] or "(no title)",
                        "Length": _fmt_dur(v.duration),
                        "In DB": "—" if is_new else "yes",
                        "_url": v.url,
                    })
                df = pd.DataFrame(rows)

                edited = st.data_editor(
                    df,
                    column_config={
                        "Run":    st.column_config.CheckboxColumn("Run", width="small", default=True),
                        "Date":   st.column_config.TextColumn("Date", width="small"),
                        "Title":  st.column_config.TextColumn("Title", width="large"),
                        "Length": st.column_config.TextColumn("Length", width="small"),
                        "In DB":  st.column_config.TextColumn("In DB", width="small"),
                        "_url":   None,  # hidden
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=min(35 * (len(rows) + 1) + 3, 420),
                    key="creator_table",
                )

                selected = edited[edited["Run"] == True]  # noqa: E712
                count = len(selected)

                # Process button on the right, with count label
                _, run_col = st.columns([4, 1])
                with run_col:
                    process_clicked = st.button(
                        f"Run on {count} video{'' if count == 1 else 's'}",
                        type="primary",
                        disabled=(count == 0),
                        use_container_width=True,
                        key="creator_run",
                    )

                if count > already_count and already_count > 0:
                    # User checked some 'In DB' rows → will create duplicates.
                    st.caption("⚠️ Some checked rows are already in your DB; processing them will create duplicate rows.")

                if process_clicked:
                    matched = next(
                        (it for it in integrations
                         if it.get("name") == st.session_state.get("notion_db_integration")),
                        None,
                    )
                    if not matched:
                        st.error("Reconnect — Notion integration is missing.")
                    else:
                        notion = get_notion_client(matched["token"])
                        available_cols = st.session_state.get("notion_available_cols")
                        try:
                            title_col = get_notion_title_column(notion, st.session_state["notion_db_id"])
                        except Exception:
                            title_col = "Name"

                        urls_to_process = list(selected["_url"])
                        ok = failed = 0
                        status_label = (
                            "Working on 1 video…" if count == 1
                            else f"Working through {count} videos…"
                        )
                        with st.status(status_label, expanded=True) as status:
                            log = lambda msg: st.write(msg)
                            for i, vid_url in enumerate(urls_to_process, 1):
                                st.write(f"---\n**Video {i} of {count}** · `{vid_url}`")
                                try:
                                    data = process_url(vid_url, log)
                                except Exception as e:
                                    failed += 1
                                    st.write(f"   ⚠️ scrape/transcribe failed: {e}")
                                    continue
                                try:
                                    create_notion_row(
                                        notion, st.session_state["notion_db_id"], vid_url, data,
                                        available_props=available_cols,
                                        title_column=title_col,
                                    )
                                    ok += 1
                                    st.write("   Created in Notion ✓")
                                except Exception as e:
                                    failed += 1
                                    st.write(f"   ⚠️ Notion create failed: {e}")
                                    with st.expander("📝 Transcript (Notion create failed — copy from here)", expanded=False):
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

                            final = (
                                f"All {ok} created."
                                if failed == 0
                                else f"{ok} created, {failed} hit a snag."
                            )
                            status.update(label=final, state="complete" if failed == 0 else "error")
                        st.session_state["creator_run_done"] = True
