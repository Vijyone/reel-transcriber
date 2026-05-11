# Reel Transcriber

A small local Mac app for pulling metrics and transcribing short videos —
Instagram reels, YouTube videos, TikToks, X posts, and most other public
video sites — and saving the results to a Notion database.

Everything runs **on your Mac**:
- 🔒 Privacy — audio never leaves your machine
- ⚡ Fast — `mlx-whisper` on Apple Silicon transcribes a 30-sec reel in ~3 sec
- 🎯 Accurate — runs Whisper `large-v3-turbo` (or `large-v3` if you want)
- 📵 Offline — works without internet after the first model download
- ♾️ No rate limits, no API keys, no cost

## Requirements

- macOS with Apple Silicon (M1 / M2 / M3 / M4)
- Python 3.12
- ffmpeg

## Setup

```bash
brew install python@3.12 ffmpeg git
git clone https://github.com/Vijyone/reel-transcriber.git
cd reel-transcriber
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
open .env  # paste your Notion token + database ID
./run.sh
```

The app opens at <http://localhost:8501>.

First time you hit Transcribe, it downloads the Whisper model from HuggingFace
(~1.5 GB for `large-v3-turbo`). After that, everything runs offline.

## Two ways to use it

- **Try a few links** — paste a list of URLs, see metrics + transcripts inline
- **Sync with Notion** — connect a Notion database, process all rows missing
  a transcript, write back automatically

The Notion DB needs at minimum a `Reference` URL column. Any of these
columns will be auto-populated if they exist:

| Column | Type |
|---|---|
| `Reference` | URL |
| `Likes` | Number |
| `Views` | Number |
| `Comments` | Number |
| `Caption ` | Rich text *(note: trailing space in the column name)* |
| `Transcript` | Rich text |

Columns the app doesn't recognize are left alone. Missing columns are
silently skipped — they don't error.

## CLI usage

The Streamlit app is the main interface, but you can also script the
agent directly:

```bash
python reel_agent.py                        # transcribe all rows missing transcripts
python reel_agent.py --limit 5              # only 5
python reel_agent.py --force                # re-process even completed rows
python reel_agent.py --title "expensive"    # only rows matching this substring
python reel_agent.py --browser chrome       # borrow IG/TikTok cookies from Chrome
python reel_agent.py --dry-run              # show what would happen, write nothing
```
