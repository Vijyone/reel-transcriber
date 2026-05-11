"""
Reel Transcriber Agent  (yt-dlp + Groq | local faster-whisper)
--------------------------------------------------------------
Reads Instagram reel URLs from a Notion database, scrapes engagement metrics
(likes / views / comments / caption) via yt-dlp, transcribes the audio either
with Groq's hosted whisper-large-v3 (default) OR locally via faster-whisper
(--local), and writes everything back to the same Notion row.

Pipeline:
    Notion (URLs)  ->  yt-dlp (metrics + video download)
                    ->  Whisper transcription (Groq or local)
                    ->  Notion (write back)

Run on demand:
    python reel_agent.py                       # default: Groq cloud transcription
    python reel_agent.py --local               # use faster-whisper locally (free, offline)
    python reel_agent.py --local --model medium  # smaller model = faster, less accurate
    python reel_agent.py --limit 5             # only process 5 rows
    python reel_agent.py --force                # re-process rows that already have a transcript
    python reel_agent.py --dry-run              # show what would happen, write nothing
    python reel_agent.py --cookies ig.txt       # use exported cookies for gated reels
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import tempfile
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from notion_client import Client as NotionClient
import yt_dlp


# --------------------------------------------------------------------------- #
# Config — customize the property names below to match your Notion database.  #
# --------------------------------------------------------------------------- #

# Notion property names (case-sensitive, must match exactly). Set any to None to skip.
PROP_URL        = "Reference"    # type: URL
PROP_LIKES      = "Likes"        # type: Number
PROP_VIEWS      = "Views"        # type: Number
PROP_COMMENTS   = "Comments"     # type: Number
PROP_CAPTION    = "Caption "     # type: Rich text  (note the trailing space)
PROP_TRANSCRIPT = "Transcript"   # type: Rich text
PROP_USERNAME   = None           # type: Rich text  (column not present in DB)
PROP_STATUS     = None           # type: Select     (column not present in DB)
PROP_LAST_RUN   = None           # type: Date       (column not present in DB)

GROQ_MODEL = "whisper-large-v3"   # alt: "whisper-large-v3-turbo" (faster, slightly less accurate)
GROQ_MAX_MB = 24                   # Groq limit is 25MB per request; leave a buffer

# faster-whisper defaults for --local mode.
# Model sizes (RAM):  tiny~75MB  base~150MB  small~500MB  medium~1.5GB  large-v3~3GB
LOCAL_MODEL_DEFAULT = "large-v3"
LOCAL_DEVICE = "auto"              # "cuda" / "cpu" / "auto"  (auto picks GPU when available)
LOCAL_COMPUTE = "default"          # "int8" for low memory, "float16" for GPU, "default" lets CT2 pick

# Notion rich-text fields cap at 2000 chars per block.
NOTION_TEXT_BLOCK = 2000


# --------------------------------------------------------------------------- #
# Data containers                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class ReelData:
    likes: Optional[int] = None
    views: Optional[int] = None
    comments: Optional[int] = None
    caption: Optional[str] = None
    username: Optional[str] = None
    video_path: Optional[str] = None
    transcript: Optional[str] = None


# --------------------------------------------------------------------------- #
# Notion helpers                                                               #
# --------------------------------------------------------------------------- #

_NOTION_ID_RE = re.compile(r"[a-f0-9]{32}|[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", re.IGNORECASE)


def parse_notion_db_id(value: str) -> Optional[str]:
    """Extract a 32-char Notion database ID from a URL, share link, or raw ID.
    Returns the ID without hyphens, or None if no valid ID is found."""
    if not value:
        return None
    matches = _NOTION_ID_RE.findall(value)
    if not matches:
        return None
    # First hex match = database ID. (Anything after `?v=` would be a view ID.)
    return matches[0].replace("-", "").lower()


def resolve_data_source_id(notion: NotionClient, db_id: str) -> str:
    """notion-client >= 3.0 requires querying a data source, not the database directly."""
    db = notion.databases.retrieve(database_id=db_id)
    data_sources = db.get("data_sources") or []
    if not data_sources:
        raise RuntimeError(
            f"Database {db_id} has no data sources. "
            "Make sure the database is shared with your integration."
        )
    return data_sources[0]["id"]


def fetch_pending_rows(notion: NotionClient, db_id: str, force: bool, title_contains: Optional[str] = None):
    """Yield Notion pages that still need processing."""
    data_source_id = resolve_data_source_id(notion, db_id)
    cursor = None
    while True:
        kwargs = {"data_source_id": data_source_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor

        filters = []
        if not force and not title_contains:
            filters.append({"property": PROP_TRANSCRIPT, "rich_text": {"is_empty": True}})
        if title_contains:
            filters.append({"property": "Name", "title": {"contains": title_contains}})
        if len(filters) == 1:
            kwargs["filter"] = filters[0]
        elif len(filters) > 1:
            kwargs["filter"] = {"and": filters}

        resp = notion.data_sources.query(**kwargs)
        for page in resp["results"]:
            yield page

        if not resp.get("has_more"):
            return
        cursor = resp.get("next_cursor")


def get_url_from_page(page) -> Optional[str]:
    prop = page["properties"].get(PROP_URL)
    if not prop:
        return None
    if prop["type"] == "url":
        return prop["url"]
    if prop["type"] == "rich_text" and prop["rich_text"]:
        return prop["rich_text"][0]["plain_text"]
    return None


def chunk_rich_text(text: str):
    if not text:
        return [{"text": {"content": ""}}]
    return [
        {"text": {"content": text[i : i + NOTION_TEXT_BLOCK]}}
        for i in range(0, len(text), NOTION_TEXT_BLOCK)
    ]


def update_notion_row(notion: NotionClient, page_id: str, data: ReelData, status: str = "Done"):
    props = {}
    if PROP_LIKES and data.likes is not None:
        props[PROP_LIKES] = {"number": data.likes}
    if PROP_VIEWS and data.views is not None:
        props[PROP_VIEWS] = {"number": data.views}
    if PROP_COMMENTS and data.comments is not None:
        props[PROP_COMMENTS] = {"number": data.comments}
    if PROP_CAPTION and data.caption is not None:
        props[PROP_CAPTION] = {"rich_text": chunk_rich_text(data.caption)}
    if PROP_TRANSCRIPT and data.transcript is not None:
        props[PROP_TRANSCRIPT] = {"rich_text": chunk_rich_text(data.transcript)}
    if PROP_USERNAME and data.username:
        props[PROP_USERNAME] = {"rich_text": [{"text": {"content": data.username}}]}
    if PROP_STATUS:
        props[PROP_STATUS] = {"select": {"name": status}}
    if PROP_LAST_RUN:
        props[PROP_LAST_RUN] = {"date": {"start": time.strftime("%Y-%m-%dT%H:%M:%S")}}

    notion.pages.update(page_id=page_id, properties=props)


# --------------------------------------------------------------------------- #
# yt-dlp scrape + download                                                     #
# --------------------------------------------------------------------------- #

def scrape_and_download(
    reel_url: str,
    out_dir: str,
    cookies_file: Optional[str],
    cookies_from_browser: Optional[str] = None,
) -> ReelData:
    """Fetch reel metadata and download the video to out_dir."""
    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")
    is_youtube = "youtube.com" in reel_url.lower() or "youtu.be" in reel_url.lower()

    def _build_opts(use_cookies: bool) -> dict:
        # Audio-only: ~10x smaller than full video, fits under Groq's 24MB limit, transcription
        # quality is identical (Whisper only uses the audio track anyway).
        opts = {
            "outtmpl": out_template,
            "format": "bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio/best",
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "retries": 3,
        }
        if use_cookies:
            if cookies_file:
                opts["cookiefile"] = cookies_file
            if cookies_from_browser:
                # yt-dlp wants a tuple: (browser_name,) or (browser_name, profile, keyring, container)
                opts["cookiesfrombrowser"] = (cookies_from_browser,)
        return opts

    def _run(use_cookies: bool):
        with yt_dlp.YoutubeDL(_build_opts(use_cookies)) as ydl:
            info = ydl.extract_info(reel_url, download=True)
            video_path = ydl.prepare_filename(info)
            if not os.path.exists(video_path):
                base = os.path.splitext(video_path)[0]
                for ext in (".mp4", ".mkv", ".webm", ".m4a"):
                    if os.path.exists(base + ext):
                        video_path = base + ext
                        break
            return info, video_path

    using_cookies = bool(cookies_file or cookies_from_browser)
    try:
        info, video_path = _run(use_cookies=using_cookies)
    except yt_dlp.utils.DownloadError as e:
        # YouTube's anti-bot can filter ALL formats when the request is authenticated
        # (an "audio_ext=m4a / vcodec=none" listing comes back empty). Cookies still help
        # IG/TikTok, so we only retry anonymously for YouTube.
        msg = str(e).lower()
        anti_bot_signal = (
            "format is not available" in msg
            or "sign in to confirm" in msg
            or "po token" in msg
            or "forbidden" in msg
        )
        if is_youtube and using_cookies and anti_bot_signal:
            info, video_path = _run(use_cookies=False)
        else:
            raise

    views = info.get("view_count") or info.get("play_count")
    # yt-dlp's IG extractor doesn't return view_count for reels — fall back to instaloader.
    if views is None and "instagram.com" in reel_url.lower():
        views = fetch_instagram_view_count(reel_url)

    return ReelData(
        likes=info.get("like_count"),
        views=views,
        comments=info.get("comment_count"),
        caption=info.get("description") or info.get("title") or "",
        username=info.get("uploader_id") or info.get("uploader") or info.get("channel"),
        video_path=video_path,
    )


_IG_SHORTCODE_RE = re.compile(r"/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)")


def fetch_instagram_view_count(reel_url: str) -> Optional[int]:
    """Use instaloader to get view counts for IG reels (yt-dlp doesn't expose them)."""
    m = _IG_SHORTCODE_RE.search(reel_url)
    if not m:
        return None
    shortcode = m.group(1)
    try:
        import instaloader
    except ImportError:
        print("     (instaloader not installed — skipping view count)")
        return None
    try:
        L = instaloader.Instaloader(
            quiet=True,
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
        )
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        return post.video_view_count or post.video_play_count
    except Exception as e:
        print(f"     (instaloader view-count lookup failed: {e})")
        return None


# --------------------------------------------------------------------------- #
# Whisper transcription — two backends                                         #
# --------------------------------------------------------------------------- #

def transcribe_with_groq(groq, video_path: str) -> str:
    """Cloud transcription via Groq (whisper-large-v3)."""
    if not video_path or not os.path.exists(video_path):
        return ""

    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if size_mb > GROQ_MAX_MB:
        raise RuntimeError(
            f"Video is {size_mb:.1f}MB which exceeds Groq's {GROQ_MAX_MB}MB limit"
        )

    with open(video_path, "rb") as f:
        result = groq.audio.transcriptions.create(
            model=GROQ_MODEL,
            file=(os.path.basename(video_path), f.read()),
            response_format="text",
        )
    return result if isinstance(result, str) else getattr(result, "text", str(result))


def transcribe_with_local(model, video_path: str) -> str:
    """Local transcription via faster-whisper. No file size limit, no quotas."""
    if not video_path or not os.path.exists(video_path):
        return ""

    segments, _info = model.transcribe(video_path, beam_size=5, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()


def load_local_model(model_size: str):
    """Lazy-import faster-whisper so users on Groq don't need it installed."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit(
            "faster-whisper is not installed. Install it with:\n"
            "    pip install faster-whisper"
        )
    print(f"Loading local Whisper model '{model_size}' (device={LOCAL_DEVICE})...")
    return WhisperModel(model_size, device=LOCAL_DEVICE, compute_type=LOCAL_COMPUTE)


# mlx-whisper: Apple Silicon native, ~5-10x faster than faster-whisper on M-series Macs.
MLX_MODEL_REPOS = {
    "tiny":              "mlx-community/whisper-tiny",
    "base":              "mlx-community/whisper-base",
    "small":             "mlx-community/whisper-small",
    "medium":            "mlx-community/whisper-medium",
    "large-v3-turbo":    "mlx-community/whisper-large-v3-turbo",
    "large-v3":          "mlx-community/whisper-large-v3-mlx",
}


def transcribe_with_mlx(video_path: str, model_size: str = "large-v3-turbo") -> str:
    """Local transcription via mlx-whisper (Apple Silicon native, very fast)."""
    if not video_path or not os.path.exists(video_path):
        return ""
    try:
        import mlx_whisper
    except ImportError:
        sys.exit(
            "mlx-whisper is not installed. Install it with:\n"
            "    pip install mlx-whisper"
        )
    repo = MLX_MODEL_REPOS.get(model_size, model_size)
    result = mlx_whisper.transcribe(video_path, path_or_hf_repo=repo)
    return (result.get("text") or "").strip()


# --------------------------------------------------------------------------- #
# Main loop                                                                    #
# --------------------------------------------------------------------------- #

def process_row(
    page,
    transcribe_fn,
    notion,
    dry_run: bool,
    cookies_file: Optional[str],
    cookies_from_browser: Optional[str] = None,
) -> str:
    page_id = page["id"]
    url = get_url_from_page(page)
    if not url:
        return "skip (no url)"

    with tempfile.TemporaryDirectory() as tmp:
        print(f"  -> scraping + downloading  {url}")
        data = scrape_and_download(url, tmp, cookies_file, cookies_from_browser)
        print(
            f"     likes={data.likes}  views={data.views}  comments={data.comments}  "
            f"@{data.username}"
        )

        print(f"  -> transcribing")
        data.transcript = transcribe_fn(data.video_path)
        print(f"     transcript: {len(data.transcript or '')} chars")

    if dry_run:
        return "dry-run"

    update_notion_row(notion, page_id, data, status="Done")
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process")
    parser.add_argument("--force", action="store_true", help="Re-process rows even if Transcript is filled")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen but don't write to Notion")
    parser.add_argument("--cookies", default=None, help="Path to a Netscape-format cookies file (for gated reels)")
    parser.add_argument(
        "--browser",
        default=None,
        help="Pull Instagram cookies from a logged-in browser to unlock view counts. "
             "e.g. chrome, safari, firefox, edge, brave. You must be logged into instagram.com in that browser.",
    )
    parser.add_argument("--title", default=None, help="Only process rows whose Name (title) contains this substring")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Transcribe locally with faster-whisper instead of calling Groq",
    )
    parser.add_argument(
        "--model",
        default=LOCAL_MODEL_DEFAULT,
        help=f"faster-whisper model size when using --local (default: {LOCAL_MODEL_DEFAULT}). "
             "Options: tiny, base, small, medium, large-v3, large-v3-turbo",
    )
    args = parser.parse_args()

    load_dotenv()
    notion_token = os.environ["NOTION_TOKEN"]
    notion_db_id = os.environ["NOTION_DATABASE_ID"]
    notion = NotionClient(auth=notion_token)

    # Wire up the chosen transcription backend.
    if args.local:
        local_model = load_local_model(args.model)
        transcribe_fn = lambda path: transcribe_with_local(local_model, path)
        backend = f"local faster-whisper ({args.model})"
    else:
        from groq import Groq
        groq = Groq(api_key=os.environ["GROQ_API_KEY"])
        transcribe_fn = lambda path: transcribe_with_groq(groq, path)
        backend = f"Groq cloud ({GROQ_MODEL})"
    print(f"Backend: {backend}")

    processed = ok = failed = skipped = 0
    for page in fetch_pending_rows(notion, notion_db_id, args.force, title_contains=args.title):
        if args.limit and processed >= args.limit:
            break
        processed += 1

        url = get_url_from_page(page) or "(no url)"
        print(f"\n[{processed}] {url}")

        try:
            outcome = process_row(page, transcribe_fn, notion, args.dry_run, args.cookies, args.browser)
            if outcome in ("ok", "dry-run"):
                ok += 1
            else:
                skipped += 1
                print(f"     {outcome}")
        except Exception as exc:
            failed += 1
            print(f"     FAILED: {exc}", file=sys.stderr)
            if not args.dry_run and PROP_STATUS:
                try:
                    notion.pages.update(
                        page_id=page["id"],
                        properties={PROP_STATUS: {"select": {"name": "Error"}}},
                    )
                except Exception:
                    pass

    print(
        f"\nDone. processed={processed}  ok={ok}  skipped={skipped}  failed={failed}"
    )


if __name__ == "__main__":
    main()
