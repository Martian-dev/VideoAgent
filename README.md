# montage

Cut a finished video out of one montage, from an instruction.

```bash
montage edit source.mp4 "a 60 second trailer that sets up the conflict" \
  --duration 60 --subtitles
```

You give it a single stitched-together source — footage, voice-over, assets, all
in one file — and say what you want. It works out what is in the video, decides
which spans deliver the brief, and renders the cut with ffmpeg.

No video generation. Nothing is synthesised: the picture and the voice in the
output are the ones you supplied.

## Why this exists

ffmpeg driven by an LLM is a very good editing *typist* — it can execute any cut
you describe, but it cannot see what is at 04:32, so a human has to watch the
source and decide every beat.

The missing piece is **grounding**: a transcript with timecodes, plus a
description of what is on screen in each span. Once the model has that, choosing
*what* to cut becomes a text problem it can reason about, and the human goes from
specifying cuts to specifying intent.

## Install

```bash
make setup          # .venv + three dependencies, ~30 seconds
echo 'META_AI_API_KEY=...' >> .env
make models         # whisper.cpp model, ~0.5 GB
make doctor         # verifies everything, including one live API call
```

Needs Python 3.10+, ffmpeg, and about 1 GB of disk. **No GPU, no torch, no CUDA.**
The only local compute is whisper.cpp and ffmpeg.

## How it works

Five stages. Each is a module you can read in one sitting and run on its own.

| Stage | Does | Runs on |
|---|---|---|
| `segment` | splits the source at its own scene changes (ffmpeg `scdet`) | local |
| `transcribe` | what is said, with timecodes | local, whisper.cpp |
| `describe` | what is seen, per segment | Muse |
| `plan` | instruction + index → edit decision list | Muse |
| `render` | executes the EDL: cut, join, titles, subtitles | local, ffmpeg |

The **edit decision list** is the contract between deciding and doing — a list of
source spans in playback order, each optionally carrying on-screen text. `render`
never makes a creative choice; `plan` never touches a file. You can read
`.montage/<source>/edl.json`, edit it by hand, and re-render.

Indexing is the slow part and is cached per source, so trying a second
instruction against the same montage costs one planning call.

## Commands

```bash
montage index source.mp4                    # build the index only, then inspect it
montage edit source.mp4 "instruction"       # index (or reuse) + plan + render
montage doctor --api                        # environment check
```

Useful flags on `edit`: `--duration` (target length in seconds), `--subtitles`
(burn in open captions), `--output`, `--width`, `--workers` (parallel description
calls, default 4), `--force` (rebuild a cached index).

## Configuration

Everything is environment-driven, read from `.env` in the project root. **`.env`
overrides the ambient environment** on purpose — a stale global export of the same
name is otherwise invisible and silently beats the key you just set.

| Variable | Default | Notes |
|---|---|---|
| `META_AI_API_KEY` | — | required |
| `MONTAGE_MODEL` | `muse-spark-1.2-contributor` | see the region note below |
| `MONTAGE_WHISPER_MODEL` | `small.en` | see the speed table below |

### The model is region-limited

The Meta Model API does not serve every model everywhere. From an EU host,
`/v1/models` lists only 1.2 and below, and asking for 1.3 returns
`model_not_found` — with a key that works fine for 1.3 elsewhere. `1.2-contributor`
is the portable default. Set `MONTAGE_MODEL` where 1.3 is offered.

### Whisper model choice is a 26x decision on CPU

Measured on a 16-core EPYC-Rome (avx2, no avx512) over 122 s of speech:

| Model | Time | Throughput | 1 hour of source |
|---|---|---|---|
| `large-v3-turbo-q5_0` | 291.2 s | 0.4x realtime | ~2.5 hours |
| `small.en` (default) | 39.0 s | 3.1x realtime | ~19 min |
| `base.en` | 11.6 s | 10.6x realtime | ~6 min |

`large-v3-turbo` is a GPU-era choice and not viable here. Drop the `.en` suffix
for non-English audio.

## Cost

Description is one call per segment; planning is one call over the whole index.
At contributor pricing ($0.10/M in, $0.20/M out) a 1-hour source costs a few
cents to index and well under a cent to re-plan.

## Layout

```
montage/
  config.py      paths, credentials, model ids
  llm.py         one entry point to Muse: ask() and ask_json()
  segment.py     scene detection
  transcribe.py  whisper.cpp
  describe.py    per-segment description -> the index
  plan.py        index + instruction -> EDL
  render.py      EDL -> mp4
  cli.py         the three commands
```

Derived from [HKUDS/VideoAgent](https://github.com/HKUDS/VideoAgent), reduced to
the montage-editing path and retargeted at the Meta Model API and CPU-only hosts.
