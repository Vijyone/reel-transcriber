# Reel Transcriber Agent

An on-demand Python script that walks a Notion database of Instagram reel links, scrapes engagement metrics (likes / views / comments / caption / username), transcribes the audio, and writes everything back to the same Notion row.

## Stack

```
Notion (URLs)  ->  yt-dlp (free, no API key)
                ->  Whisper transcription:
                       Groq cloud (default, fast, ~$0/free tier)
                       OR faster-whisper local (--local, $0 forever, offline)
                ->  Notion (write back)
```

No paid scraping APIs. Two transcription backends — pick per run:

| Backend | Cost | Speed (30s reel) | Offline | Setup |
|---|---|---|---|---|
| **Groq cloud** (default) | $0 free tier (~120min/day), then ~$0.04/hour | 1–2s | no | API key |
| **Local faster-whisper** (`--local`) | $0 forever, no quotas | 3–8s on Apple Silicon, slower on CPU | yes | one-time ~3GB model download |

Real cost when using Groq: around **$0.001 per reel** (~$1 per 1,000). Local mode is free forever.

## Why this and not Sort Feed alone

Sort Feed is great for *discovery* (browse a profile, sort reels by views, find winners) — keep using it for that. Where it falls short:

- 150 min/month transcription cap (~300 reels)
- Transcription has to be triggered manually per reel
- No Notion sync; you're left exporting CSVs and pasting

This agent fixes those: unattended batch processing on a Notion DB you curate, no monthly cap, transcript + metrics + status + last-synced timestamp written back to the same row.

## What it fills in per row

| Notion property | Type      | Source                     |
| --------------- | --------- | -------------------------- |
| URL             | URL       | input (you provide)        |
| Likes           | Number    | yt-dlp `like_count`        |
| Views           | Number    | yt-dlp `view_count`        |
| Comments        | Number    | yt-dlp `comment_count`     |
| Caption         | Rich text | yt-dlp `description`       |
| Username        | Rich text | yt-dlp `uploader_id`       |
| Transcript      | Rich text | Groq `whisper-large-v3`    |
| Status          | Select    | `Done` / `Error`           |
| Last Synced     | Date      | timestamp of the run       |

Property names live in the `PROP_*` constants at the top of `reel_agent.py`. Rename them to match your Notion column names; set the optional ones (`PROP_USERNAME`, `PROP_STATUS`, `PROP_LAST_RUN`) to `None` if you don't want them.

## Setup

### 1. Install dependencies

```bash
cd "reel-transcriber-agent"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need **ffmpeg** on your system (yt-dlp uses it for some merges):

```bash
# macOS
brew install ffmpeg
# Ubuntu / Debian
sudo apt install ffmpeg
```

### 2. Get your API keys

**Notion** — https://www.notion.so/profile/integrations

1. Create a new internal integration, copy the token.
2. Open your reel database -> `...` menu -> Connections -> add the integration. Without this step Notion returns "object_not_found".
3. Grab the database ID from the URL: `notion.so/<workspace>/<DATABASE_ID>?v=...` — it's a 32-char hex string.

**Groq** — https://console.groq.com

1. Sign up, free tier is enough to start.
2. API Keys -> Create API Key.

### 3. Configure

```bash
cp .env.example .env
# fill in NOTION_TOKEN, NOTION_DATABASE_ID, GROQ_API_KEY
```

### 4. Make sure your Notion database has the columns

Add columns matching the table above, or rename the `PROP_*` constants in the script. Minimum required: a `URL` column.

## Run it

```bash
# Cloud (default) — fast, uses Groq
python reel_agent.py
python reel_agent.py --limit 5
python reel_agent.py --force
python reel_agent.py --dry-run
python reel_agent.py --cookies ig.txt

# Local — fully offline, no quotas, no Groq key needed
pip install faster-whisper                    # one-time install
python reel_agent.py --local                  # default model: large-v3 (~3GB)
python reel_agent.py --local --model medium   # smaller model, faster, slightly less accurate
python reel_agent.py --local --model tiny     # tiny: ~75MB, fast on any machine
```

### Choosing a local model

| Size | RAM | Quality | Best for |
|---|---|---|---|
| `tiny` | ~75MB | Rough | Quick test on a slow laptop |
| `base` | ~150MB | OK for clear English | Lightweight runs |
| `small` | ~500MB | Good for most reels | Balanced default |
| `medium` | ~1.5GB | Near-large quality | Mac M-series, balanced |
| `large-v3` | ~3GB | Best | Apple Silicon Mac or GPU |
| `large-v3-turbo` | ~1.6GB | Near-large, ~2x faster | Best speed/quality on Apple Silicon |

The model auto-downloads from HuggingFace on first use into `~/.cache/huggingface`.

### About the `--cookies` flag

Most public reels work fine without cookies. If you start hitting "Login required" errors, export your Instagram cookies once and pass them in:

1. Install the **Get cookies.txt** Chrome extension.
2. Visit instagram.com, click the extension, export to `ig.txt`.
3. Pass `--cookies ig.txt` on each run.

## Cost per reel

- yt-dlp scrape + download: $0
- Groq transcription (20s reel): ~$0.0002 (paid) or free on free tier
- Local transcription: $0 forever (electricity only)
- Notion API: free
- **Total: <$0.001 per reel on Groq, $0 on `--local`**

## Limits and gotchas

- **Groq has a 25MB per-request limit.** Reels longer than ~5 minutes can exceed this. The script catches it and reports a clean error. **Local mode has no size limit.**
- **yt-dlp can break for a day or two** when Instagram changes things; updates land fast (`pip install -U yt-dlp`).
- **Private accounts** still won't work — there's no public data to read.
- **Rate limits** on yt-dlp aren't formal but heavy hammering can earn you a soft block. For >100 reels in one run, consider splitting with `--limit`.
- **Notion text limit** is 2000 chars per rich-text block; long transcripts are auto-split into multiple blocks.
- **TOS note.** Scraping competitor Instagram content is against Meta's TOS even when public — same caveat as Sort Feed and any other tool in this category. Use at your own risk.

## Optional next step: add Claude analysis

The transcript by itself is just text. The strategic value comes from a second pass that extracts:

- **Hook** — the first 3-5 seconds, pulled from the transcript
- **Structure** — problem / solution / CTA, or whatever pattern the reel uses
- **CTA** — what the creator asks the viewer to do
- **Repurposing angle** — how you'd remix this for your audience

That's a 5-line addition: send each transcript to `claude-haiku-4-5` with a prompt template, write the result to a `Claude Analysis` column. Costs about $0.001/reel. Ask me when you want it wired in.
