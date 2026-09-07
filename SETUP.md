# VideoAgent (lean fork) — setup

Video **understanding and editing**, driven by Meta's Muse Spark 1.3 Contributor.
No generation, no TTS/voice-cloning stack.

## Quickstart

```bash
git clone <your-fork> && cd VideoAgent
make setup PROFILE=cpu        # .venv on py3.11 + all deps (~5 min)
echo 'META_AI_API_KEY=...' >> .env
make models                   # ~5 GB of weights, resumable
make doctor                   # verify, including one live API call
make run
```

## What runs where

| Piece | Model | Where |
|---|---|---|
| Agentic router, editing, QA, summarization, frame analysis, captioning | `muse-spark-1.3-contributor` | Meta Model API |
| Speech-to-text | whisper.cpp `small.en` | local, `tools/whispercpp-models/` |
| Visual retrieval (text→clip search) | ImageBind huge | local, `.checkpoints/imagebind_huge.pth` |

Three retrieval channels feed `VideoSearcher`: the whisper.cpp **transcript** (what
was said), Muse Spark **captions** (described semantics), and ImageBind **visual
vectors** (raw appearance). ImageBind is what makes "find the shot where the car
pulls away" work when nobody narrates it.

## CPU-only deployment

This fork targets CPU-only servers.

- **Transcription runs on whisper.cpp** (`pywhispercpp`, prebuilt wheels, no cmake),
  not the transformers pipeline. Everything transcribes through
  `environment/config/transcribe.py`; `transformers` and `accelerate` are gone.
- **torch is profile-selected.** `make setup PROFILE=cpu` pulls the ~200 MB CPU
  wheels instead of ~2.5 GB of CUDA ones. `PROFILE=auto` (the default) picks CUDA
  only when `nvidia-smi` is present.
- **torch is only still needed for ImageBind.** It is the one remaining GPU-shaped
  component, and the slowest thing on a CPU box - see the note below.
### Whisper model choice matters a lot on CPU

Measured on the deployment target (16-core EPYC-Rome, avx2, no avx512),
transcribing 122 s of speech:

| Model | Time | Throughput | Segments | 1 hour of source |
|---|---|---|---|---|
| `large-v3-turbo-q5_0` | 291.2 s | 0.4x realtime | 15 | ~2.5 hours |
| `small.en` (default) | 39.0 s | 3.1x realtime | 32 | ~19 min |
| `base.en` | 11.6 s | 10.6x realtime | 32 | ~6 min |

`large-v3-turbo` is a GPU-era choice and not viable here. Override with
`VIDEOAGENT_WHISPER_MODEL`; drop the `.en` suffix for non-English audio.

## Configuration

Keys live in `.env`, and **`.env` overrides the ambient environment**. That is
deliberate: a stale global export of the same name is otherwise invisible and
silently beats the key you just set, which is how you end up debugging a 404
against the wrong account. `make doctor` prints which source the key came from.
`make setup` copies `.env.example` to `.env` for you.

Model routing lives in `environment/config/config.yml`. The shared block sets
everything; per-role `*_model` / `*_api_key` / `*_base_url` override one role:

```yaml
llm:
  base_url: "https://api.meta.ai/v1"
  model: "muse-spark-1.3-contributor"
  gemini_model: ""     # role used for captioning + frame analysis
```

The four role names (`claude`, `gpt`, `gemini`, `deepseek`) are upstream's, and are
now just labels for *slots* — all four point at Muse Spark by default:

| Slot | Drives |
|---|---|
| `claude` | agentic graph router (`environment/agents/multi.py`) |
| `gpt` | QA, summarization, commentary |
| `gemini` | frame analysis (`vid_editor.py`) + captioning (`videorag/.../caption.py`) |
| `deepseek` | JSON-mode conversions |

Muse Spark is a reasoning model and spends 200–380 reasoning tokens even on trivial
prompts. Cost is negligible at contributor pricing, but `caption.py` calls it once
per video segment — if indexing feels slow, point `gemini_model` at a
non-reasoning model.

## Notes and gotchas

- **Model id has no vendor prefix.** Meta's own API wants
  `muse-spark-1.3-contributor`. The aggregator-style `meta/muse-spark-1.3-contributor`
  returns 404 here.
- **torch is pinned to cu128, not upstream's cu121.** Blackwell GPUs (RTX 50xx) are
  `sm_120`; a cu121 build has no kernels for them and fails at the first `.cuda()`
  call *after* installing cleanly. `make doctor` checks the GPU arch against
  `torch.cuda.get_arch_list()` and tells you if they disagree.
- **No GPU?** Everything falls back to CPU. Force it with `VIDEOAGENT_DEVICE=cpu`.
- **ImageBind is cached per process.** Upstream reloaded the 4.5 GB checkpoint inside
  both `upsert()` and `query()`; it now loads once (`environment/config/runtime.py`).
- **Startup lists unavailable roles.** On a lean install the TTS/SVC/stand-up roles
  are withheld and summarised in one line. Two reasons appear: roles that cannot
  *import* (no `cosyvoice`, no `mido`), and roles that import fine but cannot
  *execute* without the voice models. The second group is gated deliberately by
  `VOICE_ROLES` in `environment/config/runtime.py` - otherwise the router plans a
  workflow around a tool that dies partway through. Install the voice models and
  they reappear automatically.
- **`make setup` patches one dependency.** `pytorchvideo` imports
  `torchvision.transforms.functional_tensor`, removed in torchvision 0.17, and we need
  0.22 for `sm_120`. `scripts/patch_deps.py` redirects that import (the module only
  uses `F_t.affine`, still public). Idempotent, re-run safe.

## Available intents

Working: `Video Edit`, `Video QA`, `Audio Overview`, `Rhythm-cut`, `Add BGM`,
`Video to Audio`, `Loudness Normalization`, `Audio Resample`, `Vocal Separation`,
`Audio Optimization`, `Format Conversion`.

Needs the voice stack (not installed): `Text-to-Speech`, `Singing Voice Conversion`,
`Stand-up Comedy`, `CrossTalk`, `Commentary`, `News`. `Audio Transcription` needs
`funasr` for the `fap transcribe` path — note the Whisper above is a separate ASR
used by the VideoRAG indexer.

## Added in this fork

Four roles upstream does not have:

| Role | Does |
|---|---|
| `VoiceoverTimeline` | Transcribes the source's own narration, picks the lines that serve the requested edit, trims the original audio to them, and writes the scene + cut-point files. No speech synthesis. |
| `SubtitleGenerator` | Media → `.srt` via Whisper |
| `SubtitleBurner` | Burns an `.srt` into the picture (open captions) |
| `TextOverlay` | Timed titles, lower thirds, chapter cards, watermarks via ffmpeg `drawtext` |

New intents: `Trailer`, `Subtitles`, `Text Overlay`, `Captioned Edit`.

### Why `VoiceoverTimeline` exists

`VideoEditor` is spine-driven: an audio track plus a cut-point file define the
timeline, and footage is matched to it. Upstream produces that spine either from
music (`RhythmDetector`) or from *synthesised* narration (`VoiceGenerator` and the
`*_conversion` roles). Neither fits a source video that already carries its own
voice-over: `RhythmDetector` finds **zero** cut points in speech, and the synthesis
roles need the voice stack.

`VoiceoverTimeline` fills that hole using the narration already in the footage.
It writes the same two formats the existing code reads — `{"segment_scene": "…/////…"}`
for `VideoSearcher` and `{"sentence_data": {"chunks": [...]}}` for `VideoEditor` —
so nothing downstream changed.

### Trailer pipeline

```
VoiceoverTimeline -> VideoPreloader -> VideoSearcher -> VideoEditor
                  -> SubtitleGenerator -> SubtitleBurner -> TextOverlay
```
