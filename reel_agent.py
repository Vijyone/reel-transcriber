"""
Reel Transcriber Agent  (yt-dlp + mlx-whisper, Apple Silicon)
-------------------------------------------------------------
Reads video URLs from a Notion database, scrapes engagement metrics
(likes / views / comments / caption) via yt-dlp, transcribes the audio locally
on Apple Silicon via mlx-whisper, and writes everything back to the same Notion row.

Pipeline:
    Notion (URLs)  ->  yt-dlp (metrics + audio download)
                    ->  mlx-whisper transcription (on your Mac)
                    ->  Notion (write back)

Most people use the Streamlit UI (`./run.sh`). This CLI is here if you'd
rather script it.

Run on demand:
    python reel_agent.py                        # transcribe rows missing a transcript
    python reel_agent.py --model large-v3       # higher accuracy (slower, bigger)
    python reel_agent.py --limit 5              # only process 5 rows
    python reel_agent.py --force                # re-process rows already done
    python reel_agent.py --dry-run              # show what would happen, write nothing
    python reel_agent.py --browser chrome       # pull cookies from a logged-in browser
    python reel_agent.py --title "expensive"    # only rows whose Name contains this
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

# Default mlx-whisper model. Override with --model on the CLI.
# Sizes:  tiny~75MB  base~150MB  small~500MB  medium~1.5GB  large-v3-turbo~1.5GB  large-v3~3GB
MLX_MODEL_DEFAULT = "large-v3-turbo"

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


def get_available_notion_columns(notion: NotionClient, db_id: str) -> set:
    """Return the set of property (column) names that exist in this database's
    data source. Used to skip writing to columns the user hasn't created."""
    ds_id = resolve_data_source_id(notion, db_id)
    ds = notion.data_sources.retrieve(data_source_id=ds_id)
    return set(ds.get("properties", {}).keys())


def update_notion_row(
    notion: NotionClient,
    page_id: str,
    data: ReelData,
    status: str = "Done",
    available_props: Optional[set] = None,
):
    """Write the scraped/transcribed data back to a Notion row.

    If `available_props` is provided, only writes to columns that actually
    exist in the database — silently skipping the rest. This lets the app
    work against Notion DBs with different schemas (some users have a
    Comments column, others don't, etc.)."""

    def _ok(name: Optional[str]) -> bool:
        if not name:
            return False
        if available_props is None:
            return True  # caller didn't filter — trust their config
        return name in available_props

    props = {}
    if _ok(PROP_LIKES) and data.likes is not None:
        props[PROP_LIKES] = {"number": data.likes}
    if _ok(PROP_VIEWS) and data.views is not None:
        props[PROP_VIEWS] = {"number": data.views}
    if _ok(PROP_COMMENTS) and data.comments is not None:
        props[PROP_COMMENTS] = {"number": data.comments}
    if _ok(PROP_CAPTION) and data.caption is not None:
        props[PROP_CAPTION] = {"rich_text": chunk_rich_text(data.caption)}
    if _ok(PROP_TRANSCRIPT) and data.transcript is not None:
        props[PROP_TRANSCRIPT] = {"rich_text": chunk_rich_text(data.transcript)}
    if _ok(PROP_USERNAME) and data.username:
        props[PROP_USERNAME] = {"rich_text": [{"text": {"content": data.username}}]}
    if _ok(PROP_STATUS):
        props[PROP_STATUS] = {"select": {"name": status}}
    if _ok(PROP_LAST_RUN):
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
        # Audio-only: ~10x smaller than full video. Whisper only reads the audio track
        # anyway, so transcription quality is identical.
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
# Whisper transcription (mlx-whisper on Apple Silicon)                         #
# --------------------------------------------------------------------------- #

# Apple Silicon native, ~5-10x faster than CPU-based alternatives on M-series Macs.
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
    parser.add_argument("--cookies", default=None, help="Path to a Netscape-format cookies file (for gated content)")
    parser.add_argument(
        "--browser",
        default=None,
        help="Pull cookies from a logged-in browser to skip rate limits / unlock gated content. "
             "e.g. chrome, safari, firefox, edge, brave.",
    )
    parser.add_argument("--title", default=None, help="Only process rows whose Name contains this substring")
    parser.add_argument(
        "--model",
        default=MLX_MODEL_DEFAULT,
        help=f"mlx-whisper model size (default: {MLX_MODEL_DEFAULT}). "
             "Options: tiny, base, small, medium, large-v3, large-v3-turbo",
    )
    args = parser.parse_args()

    load_dotenv()
    notion_token = os.environ["NOTION_TOKEN"]
    notion_db_id = os.environ["NOTION_DATABASE_ID"]
    notion = NotionClient(auth=notion_token)

    transcribe_fn = lambda path: transcribe_with_mlx(path, model_size=args.model)
    print(f"Backend: mlx-whisper ({args.model}) on your Mac")

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
