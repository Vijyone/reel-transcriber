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
import contextlib
import glob
import json
import os
import queue
import re
import sys
import threading
import time
import tempfile
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

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
    duration: Optional[float] = None  # seconds; used to scale transcription progress
    video_path: Optional[str] = None
    transcript: Optional[str] = None
    # If the source platform (e.g. YouTube) had existing captions, they land here
    # so the caller can skip Whisper entirely. `subtitle_source` is "manual" for
    # uploaded subs or "auto" for auto-generated.
    subtitle_text: Optional[str] = None
    subtitle_source: Optional[str] = None


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
    """Split text into Notion rich_text blocks of ≤ 2000 UTF-16 code units each.

    Notion's 2000-char limit is measured in UTF-16 code units, not Python
    characters. An emoji (surrogate pair) counts as 2, so naïvely slicing by
    Python char index can push a chunk to 2001+ on the wire. We walk the
    string and split on UTF-16 boundaries instead."""
    if not text:
        return [{"text": {"content": ""}}]

    chunks = []
    current = []
    current_u16 = 0
    for ch in text:
        # encode each char to UTF-16-LE; len // 2 gives the code-unit count (1 or 2)
        ch_u16 = len(ch.encode("utf-16-le")) // 2
        if current_u16 + ch_u16 > NOTION_TEXT_BLOCK:
            chunks.append({"text": {"content": "".join(current)}})
            current = [ch]
            current_u16 = ch_u16
        else:
            current.append(ch)
            current_u16 += ch_u16
    if current:
        chunks.append({"text": {"content": "".join(current)}})
    return chunks


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


def get_notion_title_column(notion: NotionClient, db_id: str) -> str:
    """Return the name of the title-type column (every Notion DB has exactly one)."""
    ds_id = resolve_data_source_id(notion, db_id)
    ds = notion.data_sources.retrieve(data_source_id=ds_id)
    for name, prop in ds.get("properties", {}).items():
        if prop.get("type") == "title":
            return name
    return "Name"  # sensible fallback


def list_existing_urls(notion: NotionClient, db_id: str, url_column: str = "Reference") -> set:
    """Return the set of URLs already present in the given DB's URL column.
    Used for dedup when bulk-adding videos from a channel/profile."""
    ds_id = resolve_data_source_id(notion, db_id)
    urls: set = set()
    cursor = None
    while True:
        kwargs = {"data_source_id": ds_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        resp = notion.data_sources.query(**kwargs)
        for page in resp["results"]:
            prop = page["properties"].get(url_column, {})
            if prop.get("type") == "url" and prop.get("url"):
                urls.add(prop["url"])
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")
    return urls


def create_notion_row(
    notion: NotionClient,
    db_id: str,
    url: str,
    data: ReelData,
    available_props: Optional[set] = None,
    title_column: str = "Name",
) -> str:
    """Create a new row in the Notion database for `url`, populating whatever
    metric/transcript columns exist in the data source. Returns the new page id."""
    ds_id = resolve_data_source_id(notion, db_id)

    # Build a reasonable title from caption / username / URL.
    title_text = (data.caption or data.username or url)
    title_text = title_text.strip().splitlines()[0] if title_text else url
    if len(title_text) > 200:
        title_text = title_text[:200] + "…"

    properties = {
        title_column: {"title": [{"text": {"content": title_text}}]},
    }

    def _ok(name: Optional[str]) -> bool:
        if not name:
            return False
        if available_props is None:
            return True
        return name in available_props

    if _ok(PROP_URL):
        properties[PROP_URL] = {"url": url}
    if _ok(PROP_LIKES) and data.likes is not None:
        properties[PROP_LIKES] = {"number": data.likes}
    if _ok(PROP_VIEWS) and data.views is not None:
        properties[PROP_VIEWS] = {"number": data.views}
    if _ok(PROP_COMMENTS) and data.comments is not None:
        properties[PROP_COMMENTS] = {"number": data.comments}
    if _ok(PROP_CAPTION) and data.caption is not None:
        properties[PROP_CAPTION] = {"rich_text": chunk_rich_text(data.caption)}
    if _ok(PROP_TRANSCRIPT) and data.transcript is not None:
        properties[PROP_TRANSCRIPT] = {"rich_text": chunk_rich_text(data.transcript)}
    if _ok(PROP_USERNAME) and data.username:
        properties[PROP_USERNAME] = {"rich_text": [{"text": {"content": data.username}}]}
    if _ok(PROP_STATUS):
        properties[PROP_STATUS] = {"select": {"name": "Done"}}
    if _ok(PROP_LAST_RUN):
        properties[PROP_LAST_RUN] = {"date": {"start": time.strftime("%Y-%m-%dT%H:%M:%S")}}

    page = notion.pages.create(
        parent={"data_source_id": ds_id},
        properties=properties,
    )
    return page["id"]


# --------------------------------------------------------------------------- #
# Channel / profile enumeration                                                #
# --------------------------------------------------------------------------- #

@dataclass
class VideoListing:
    """Lightweight metadata for a video found via channel/profile enumeration.
    Use the URL to feed into scrape_and_download for the full pipeline."""
    url: str
    title: str
    upload_date: Optional[str]  # YYYY-MM-DD
    duration: Optional[float]   # seconds
    id: str = ""


def _ydmd_to_iso(s: Optional[str]) -> Optional[str]:
    """yt-dlp returns dates as 'YYYYMMDD'. Convert to 'YYYY-MM-DD'."""
    if not s or len(s) != 8 or not s.isdigit():
        return None
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


_YT_CHANNEL_RE = re.compile(
    r"^https?://(?:www\.)?youtube\.com/(@[^/?#]+|c/[^/?#]+|user/[^/?#]+|channel/[^/?#]+)/?(?:[?#].*)?$"
)


def _normalize_youtube_url(url: str) -> str:
    """Bare channel URLs (youtube.com/@channel) make yt-dlp emit a nested
    list of TABS (Videos, Shorts, Live…), so playlistend=N caps at N tabs,
    not N videos. Rewriting to /videos gives a flat list of videos where
    playlistend works the way the caller expects."""
    s = url.strip()
    if "list=" in s:
        return s  # leave playlist URLs alone
    m = _YT_CHANNEL_RE.match(s)
    if m:
        return f"https://www.youtube.com/{m.group(1)}/videos"
    return s


def enumerate_youtube(
    url: str,
    limit: Optional[int] = None,
    include_shorts: bool = False,
    cookies_from_browser: Optional[str] = None,
    need_dates: bool = False,
) -> List[VideoListing]:
    """Enumerate videos from a YouTube channel or playlist URL.

    Two modes:
      - `need_dates=False` (default): yt-dlp's flat extraction. Fast (~1 sec
        for any channel size) but `upload_date` comes back as None for
        YouTube channel entries.
      - `need_dates=True`: full extraction. Returns real upload dates but
        costs ~1.5 sec per video. Use when the user has a date filter set —
        the `limit` keeps the total wait time bounded."""
    url = _normalize_youtube_url(url)

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    if not need_dates:
        opts["extract_flat"] = "in_playlist"
    if limit:
        opts["playlistend"] = limit
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    entries = info.get("entries") or []
    results: List[VideoListing] = []

    def _add(entry: dict) -> bool:
        """Return True if we hit the user-requested limit."""
        _maybe_add_yt_entry(entry, include_shorts, results)
        return bool(limit and len(results) >= limit)

    for e in entries:
        if not e:
            continue
        # Defensive: handle nested tabs in case yt-dlp returns the channel-tab
        # structure despite the URL rewrite above.
        nested = e.get("entries")
        if nested:
            done = False
            for inner in nested:
                if not inner:
                    continue
                if _add(inner):
                    done = True
                    break
            if done:
                break
        else:
            if _add(e):
                break

    return results


def _maybe_add_yt_entry(e: dict, include_shorts: bool, out: List[VideoListing]) -> None:
    entry_url = e.get("url") or e.get("webpage_url") or ""
    if not entry_url:
        vid = e.get("id")
        if vid:
            entry_url = f"https://www.youtube.com/watch?v={vid}"
        else:
            return

    duration = e.get("duration")
    # Detect Shorts: explicit URL hint, OR a duration ≤ 61s (YT's cutoff).
    # Channel enumeration often rewrites /shorts/ URLs to /watch?v= form, so
    # the URL hint alone misses many of them — the duration check catches the rest.
    is_short = "/shorts/" in entry_url
    if not is_short and isinstance(duration, (int, float)) and 0 < duration <= 61:
        is_short = True

    if is_short and not include_shorts:
        return

    out.append(VideoListing(
        url=entry_url,
        title=e.get("title", "") or "",
        upload_date=_ydmd_to_iso(e.get("upload_date")),
        duration=duration,
        id=e.get("id", "") or "",
    ))


def enumerate_instagram(
    profile_url: str,
    limit: Optional[int] = None,
    cookies_from_browser: Optional[str] = None,  # accepted for symmetry; instaloader uses its own session
) -> List[VideoListing]:
    """Enumerate video/reel posts from an Instagram profile URL via instaloader.

    Only includes video posts (skips image posts and carousels-without-video).
    Note: IG rate-limits hard on big accounts; anonymous enumeration may stop
    after ~50–100 posts. For large accounts the user should authenticate
    instaloader separately (see instaloader docs)."""
    try:
        import instaloader
    except ImportError:
        sys.exit("instaloader is not installed.  pip install instaloader")

    # Extract username from URL.
    m = re.search(r"instagram\.com/(?:@?)([^/?#]+)", profile_url)
    if not m:
        raise ValueError(f"Couldn't extract an Instagram username from {profile_url!r}")
    username = m.group(1)
    if username.lower() in {"reel", "reels", "p", "tv", "stories"}:
        raise ValueError(
            f"That looks like a post URL, not a profile. Paste the creator's profile page instead "
            f"(e.g. https://www.instagram.com/their_username)."
        )

    L = instaloader.Instaloader(
        quiet=True,
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
    )
    profile = instaloader.Profile.from_username(L.context, username)

    results: List[VideoListing] = []
    for post in profile.get_posts():
        if not post.is_video:
            continue
        shortcode = post.shortcode
        results.append(VideoListing(
            url=f"https://www.instagram.com/p/{shortcode}/",
            title=(post.caption or "").strip().splitlines()[0][:200] if post.caption else "",
            upload_date=post.date_local.strftime("%Y-%m-%d") if post.date_local else None,
            duration=getattr(post, "video_duration", None),
            id=shortcode,
        ))
        if limit and len(results) >= limit:
            break
    return results


def detect_creator_source(url: str) -> str:
    """Return 'youtube', 'instagram', or 'unknown' based on the URL."""
    u = url.lower()
    if "youtube.com" in u or "youtu.be" in u:
        return "youtube"
    if "instagram.com" in u:
        return "instagram"
    return "unknown"


# --------------------------------------------------------------------------- #
# yt-dlp scrape + download                                                     #
# --------------------------------------------------------------------------- #

_ANTI_BOT_PHRASES = (
    "format is not available",
    "sign in to confirm",
    "po token",
    "forbidden",
)


def _is_youtube_anti_bot(err_msg: str) -> bool:
    msg = err_msg.lower()
    return any(p in msg for p in _ANTI_BOT_PHRASES)


def scrape_and_download(
    reel_url: str,
    out_dir: str,
    cookies_file: Optional[str],
    cookies_from_browser: Optional[str] = None,
    try_subtitles: bool = True,
) -> ReelData:
    """Fetch metadata and prepare a transcript source for `reel_url`.

    Two phases:
      1. If `try_subtitles=True`: a cheap metadata-only request that *also* pulls
         any existing English captions (manual or auto). No audio is downloaded.
         If captions are found, we stop here — the caller uses `data.subtitle_text`.
      2. Only if no captions were available, a second request downloads the audio
         for Whisper. This is the path for IG / TikTok / etc., and YouTube videos
         that genuinely have no captions.

    On YouTube anti-bot errors (often triggered when authenticated), each phase
    retries once without cookies."""
    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")
    is_youtube = "youtube.com" in reel_url.lower() or "youtu.be" in reel_url.lower()
    using_cookies = bool(cookies_file or cookies_from_browser)

    def _common_cookie_opts(opts: dict, use_cookies: bool) -> dict:
        if use_cookies:
            if cookies_file:
                opts["cookiefile"] = cookies_file
            if cookies_from_browser:
                opts["cookiesfrombrowser"] = (cookies_from_browser,)
        return opts

    def _subs_opts(use_cookies: bool) -> dict:
        # Metadata + subtitle files only — no audio/video download.
        # `skip_download=True` is what stops the audio download; the subtitle
        # writers still fire and drop .json3/.vtt files into out_dir.
        return _common_cookie_opts({
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "retries": 3,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-GB", "en-orig"],
            "subtitlesformat": "json3/srv3/vtt/best",
        }, use_cookies)

    def _audio_opts(use_cookies: bool) -> dict:
        # Audio-only download for Whisper. ~10x smaller than full video; Whisper
        # only reads the audio track anyway, so transcription quality is identical.
        return _common_cookie_opts({
            "outtmpl": out_template,
            "format": "bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio/best",
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "retries": 3,
        }, use_cookies)

    def _run(opts: dict, want_path: bool):
        """Single yt-dlp run with YouTube anti-bot retry. Returns (info, video_path).
        video_path is None when want_path=False (subs phase)."""

        def _once(o):
            with yt_dlp.YoutubeDL(o) as ydl:
                info = ydl.extract_info(reel_url, download=True)
                return info, (ydl.prepare_filename(info) if want_path else None)

        try:
            return _once(opts)
        except yt_dlp.utils.DownloadError as e:
            if is_youtube and using_cookies and _is_youtube_anti_bot(str(e)):
                clean = {**opts}
                clean.pop("cookiefile", None)
                clean.pop("cookiesfrombrowser", None)
                return _once(clean)
            raise

    # ── Phase 1: try to grab existing captions (no audio download) ────────────
    info = None
    video_path = None
    subtitle_text, subtitle_source = (None, None)

    if try_subtitles:
        try:
            info, _ = _run(_subs_opts(using_cookies), want_path=False)
            subtitle_text, subtitle_source = _find_and_parse_subs(out_dir, info)
        except yt_dlp.utils.DownloadError:
            # If even the metadata-only call fails, fall through to phase 2 —
            # the audio-download call will surface a real error there.
            info = None

    # ── Phase 2: only download audio if we don't already have a transcript ───
    if not subtitle_text:
        info, video_path = _run(_audio_opts(using_cookies), want_path=True)
        if video_path and not os.path.exists(video_path):
            base = os.path.splitext(video_path)[0]
            for ext in (".mp4", ".mkv", ".webm", ".m4a"):
                if os.path.exists(base + ext):
                    video_path = base + ext
                    break

    # Metadata (from whichever phase produced `info`)
    views = info.get("view_count") or info.get("play_count")
    if views is None and "instagram.com" in reel_url.lower():
        views = fetch_instagram_view_count(reel_url)

    return ReelData(
        likes=info.get("like_count"),
        views=views,
        comments=info.get("comment_count"),
        caption=info.get("description") or info.get("title") or "",
        username=info.get("uploader_id") or info.get("uploader") or info.get("channel"),
        duration=info.get("duration"),
        video_path=video_path,
        subtitle_text=subtitle_text,
        subtitle_source=subtitle_source,
    )


# --------------------------------------------------------------------------- #
# Subtitle helpers — use existing captions instead of running Whisper          #
# --------------------------------------------------------------------------- #

def _find_and_parse_subs(out_dir: str, info: dict) -> Tuple[Optional[str], Optional[str]]:
    """Look for any subtitle files yt-dlp dropped into out_dir and parse the
    best one. Returns (text, source) where source is 'manual', 'auto', or None."""
    EN_KEYS = ("en", "en-US", "en-GB", "en-orig")
    has_manual = any(info.get("subtitles", {}).get(k) for k in EN_KEYS)
    has_auto = any(info.get("automatic_captions", {}).get(k) for k in EN_KEYS)
    if not (has_manual or has_auto):
        return None, None
    source = "manual" if has_manual else "auto"

    # Priority: json3 → srv3 → vtt → srt. json3 is the cleanest YouTube format.
    for ext, parser in (
        ("json3", _parse_json3),
        ("srv3", _parse_srv3),
        ("vtt", _parse_vtt),
        ("srt", _parse_srt),
    ):
        for path in sorted(glob.glob(os.path.join(out_dir, f"*.{ext}"))):
            try:
                text = parser(path)
            except Exception:
                continue
            if text and len(text) > 10:
                return text, source
    return None, None


def _parse_json3(path: str) -> str:
    """YouTube's `json3` caption format. Cleanest parse — no rolling-caption mess."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    parts = []
    for event in data.get("events", []):
        for seg in event.get("segs", []):
            t = seg.get("utf8", "")
            if t and t != "\n":
                parts.append(t)
    return _tidy(" ".join(parts))


def _parse_srv3(path: str) -> str:
    """YouTube's `srv3` XML caption format. Similar to TTML."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()
    parts = []
    # srv3 has <p> elements with text inside (sometimes split across <s> spans)
    for p in root.iter():
        if p.tag.endswith("}p") or p.tag == "p":
            text = "".join(p.itertext())
            if text and text.strip():
                parts.append(text.strip())
    return _tidy(" ".join(parts))


_VTT_TIMESTAMP_RE = re.compile(r"^\d+:\d+:\d+\.\d+\s+-->")
_VTT_INLINE_TAG_RE = re.compile(r"<[^>]+>")


def _parse_vtt(path: str) -> str:
    """WebVTT format. Strips headers, cue timestamps, inline tags. Dedupes
    consecutive duplicate lines (YouTube auto-captions often have rolling
    repetition)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    lines = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT"):
            continue
        if line.startswith(("Kind:", "Language:", "NOTE")):
            continue
        if _VTT_TIMESTAMP_RE.match(line):
            continue
        line = _VTT_INLINE_TAG_RE.sub("", line)
        if line:
            lines.append(line)
    # Dedupe consecutive identical lines (rolling captions)
    deduped: list = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    return _tidy(" ".join(deduped))


def _parse_srt(path: str) -> str:
    """SubRip format. Strip indices and timestamps; keep just text."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    lines = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        line = _VTT_INLINE_TAG_RE.sub("", line)
        if line:
            lines.append(line)
    return _tidy(" ".join(lines))


def _tidy(text: str) -> str:
    """Collapse runs of whitespace; strip."""
    return re.sub(r"\s+", " ", text).strip()


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
    """Local transcription via mlx-whisper (Apple Silicon native, very fast).
    Blocking — returns the full text. For live progress, use transcribe_with_mlx_stream."""
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


# mlx-whisper's verbose output emits one line per segment like:
#   [00:00.000 --> 00:05.200]  Hello world
# We parse these on-the-fly so callers can show live progress.
_SEGMENT_LINE_RE = re.compile(
    r"\[(\d+):(\d+(?:\.\d+)?)\s*-->\s*(\d+):(\d+(?:\.\d+)?)\]\s*(.+)"
)


class _SegmentParser:
    """File-like stdout sink that fires a callback for each segment line."""

    def __init__(self, on_segment):
        self.on_segment = on_segment
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            m = _SEGMENT_LINE_RE.search(line)
            if not m:
                continue
            end_sec = int(m.group(3)) * 60 + float(m.group(4))
            text = m.group(5).strip()
            if text:
                self.on_segment(text, end_sec)
        return len(s)

    def flush(self):
        pass


def transcribe_with_mlx_stream(
    video_path: str,
    model_size: str = "large-v3-turbo",
) -> Iterator[Tuple[str, float]]:
    """Generator yielding `(segment_text, end_timestamp_seconds)` tuples as
    mlx-whisper transcribes the audio. The full transcript is the
    concatenation of all yielded segment_text values.

    Runs mlx-whisper in a background thread and captures its verbose stdout.
    The caller (main thread) is free to update Streamlit widgets between yields."""
    if not video_path or not os.path.exists(video_path):
        return
    try:
        import mlx_whisper
    except ImportError:
        sys.exit(
            "mlx-whisper is not installed. Install it with:\n"
            "    pip install mlx-whisper"
        )
    repo = MLX_MODEL_REPOS.get(model_size, model_size)

    q: "queue.Queue" = queue.Queue()
    DONE = object()
    err_holder = {"error": None}

    def _worker():
        try:
            parser = _SegmentParser(lambda text, ts: q.put((text, ts)))
            # redirect_stdout is process-global, but during transcription nothing
            # else meaningful is printing on this Python process.
            with contextlib.redirect_stdout(parser):
                mlx_whisper.transcribe(video_path, path_or_hf_repo=repo, verbose=True)
        except Exception as e:
            err_holder["error"] = e
        finally:
            q.put(DONE)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is DONE:
            break
        yield item  # (text, end_sec)

    t.join()
    if err_holder["error"]:
        raise err_holder["error"]


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
