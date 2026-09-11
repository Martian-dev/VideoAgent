# videoagent-harness

A minimal agent harness for measuring a model's **video editing capability**.

It puts the model in a container with `ffmpeg`, a shell, and a way to see, hands
it a task instruction, and gets out of the way. Every editorial decision belongs
to the model; the harness contributes no domain knowledge at all.

```bash
python -m harness.run \
  --source-dir /app/data/source_pack \
  --instruction /app/instruction.md \
  --output     /app/output/final_video.mp4 \
  --runs-dir   /app/runs
```

## What this is for

This is an **instrument, not a product**. A harness that improves the output by
doing some of the thinking is a worse instrument, because you can no longer tell
whose judgement produced the result.

The sibling `montage/` pipeline is the opposite: it segments, samples, captions
and plans according to a strategy a human wrote. It makes good videos. It is a
poor measurement, because a good score might be the prompt engineering rather
than the model.

Run both against the same task and the **gap between them** is the finding.

| | `montage` (control) | `harness` (experiment) |
|---|---|---|
| Segmentation | scdet thresholds chosen by a human | model decides |
| What the model sees | 3 stills per segment, chosen for it | model chooses, can watch clips |
| Editorial strategy | a written `PLAN_PROMPT` | the model's own |
| Reviews its own output | never | if it thinks to |
| Shape of the run | 63 fixed API calls | whatever it does |

## Setup

Needs Python 3.10+, `ffmpeg`, `ffprobe`, a font, and the whisper.cpp model.

```bash
docker build -t videoagent-harness harness/      # everything baked in
```

Or on a host directly:

```bash
pip install openai pywhispercpp
apt install ffmpeg fonts-dejavu-core
python -c "from pywhispercpp.model import Model; Model('medium.en-q5_0', models_dir='/opt/whisper-models')"
install -m755 harness/bin/transcribe /usr/local/bin/transcribe
export META_AI_API_KEY=...
```

The container is `python:3.11-slim` plus ffmpeg, fonts, two pip packages and a
~539 MB whisper model. **No torch, no CUDA, no GPU.**

### Whisper model choice

`medium.en-q5_0` by default. Measured on a 16-core EPYC-Rome (avx2, no avx512)
over 122 s of real narration:

| Model | Time | Throughput | First segment |
|---|---|---|---|
| `medium.en-q5_0` | 80.4 s | 1.5x realtime | `In 1971, William Friedkin directed "The French Conne...` |
| `small.en` | 35.2 s | 3.5x realtime | `[Music]` |

`small.en` is more than twice as fast and missed the opening narration entirely,
labelling it `[Music]`. On a narration-driven edit that is the difference between
having the script and not, so the slower model is the default.

Budget accordingly: at 1.5x realtime a one-hour source costs ~40 minutes of
transcription, which is most of the default 3600 s wall clock. Override with
`HARNESS_WHISPER_MODEL` — the model only pays this cost for the spans it chooses
to transcribe, not the whole source, unless it asks for the whole source.

## The tool surface

Three tools. Only one is a capability; the other two are protocol.

| Tool | Returns |
|---|---|
| `shell(command)` | stdout, stderr, exit code |
| `view(paths)` | image or video files as content blocks in context |
| `submit(path)` | accepts or rejects the finished file |

`view` is the one irreducible affordance — a model cannot perceive a file, so
something has to turn bytes into content blocks. Note the split: **the model
extracts what it wants to look at using ffmpeg; the harness only delivers it.**
`view` knows nothing about scenes, cuts, or edits.

### The system prompt

166 words. A factual inventory of tools, available commands, paths and limits.
It says nothing about editing, what to look at, how to sample, or what makes a
good cut. The task instruction supplies the goal; every step toward it is the
thing being measured.

Two things unavoidably leak: the output path implies a video is wanted, and
handing someone `ffmpeg` implies the domain. Everything else comes from the task.

## API constraints

These were established by probing `muse-spark-1.2-contributor` directly, not
from documentation, and they shape the loop:

| | |
|---|---|
| `tools` / `tool_choice` | works |
| Media inside a `role:"tool"` result | **rejected (400)** |
| `video_url` mid-conversation | works, and perceives motion across frames |
| Images per request | **at most 50 (400 above that)** |
| Videos per request | **at most 5 (400 above that)** |
| Audio input | **not perceived** |

Because tool results are text-only, a `view` costs two messages: a text stub as
the tool result, then a user message carrying the actual media.

Both caps are per request, and history is resent every turn, so they bound
everything in context, not just the latest look — four views of 15 frames is 60
images. The harness keeps every request within `--max-images` (default 45,
headroom under 50) and `--max-videos` (default 5), evicting the oldest first,
and enforces both where the request is built. A `view` of more than that is not
refused: its files are queued and delivered one batch per turn, and the tool
result says how many turns that takes. If a request is still refused for either
— the limit differs by model, or counts something the harness doesn't — the
harness lowers that bound to the number the API reported and retries, then
retries once more with all media dropped, leaving a marker where each file was.
The run never ends on this error; the `media_limit_retry` trace event records
that it happened.

The audio finding is the dangerous one. `audio_url` is rejected outright, but
`input_audio` and `file` blocks are **accepted and silently ignored** — the model
then answers as though it heard silence. So `view()` refuses audio loudly and
points at `transcribe`.

**Picture perception is measured. Speech perception is provided.**

### When a request fails

Anything other than a media-cap refusal — a 500, a 400 the harness cannot fix, a
dropped connection, a timeout — is retried unchanged up to `--api-retries` times
(default 5), waiting 15s, 30s, 60s, 120s, then 240s in between (`--retry-backoff`
sets the first wait). A retry that succeeds carries the run on as if nothing
happened; only when all of them fail does the error end the run, with
termination `error:<ExceptionType>`. Each retry is an `api_retry` trace event,
and the `end` event counts them. The OpenAI SDK's own retries are off, so this
is the only count — left on, they would silently triple it.

A retry the remaining wall clock cannot wait for is not attempted: the run ends
as `TIMEOUT`, with an `api_retry_out_of_time` trace event saying which failure
it was waiting out.

## What bounds a run

No turn cap. A turn count encodes an assumption about how many steps the work
should take — a source that legitimately needs 120 turns should not be killed for
it. Terminal-Bench bounds on wall-clock alone for the same reason.

| Bound | Default | Flag |
|---|---|---|
| Wall clock | 3600s | `--max-seconds` |
| Cost | $6.00 | `--max-cost` |
| Consecutive tool errors | 10 | `--max-errors` |
| Stall | see below | `--no-rewind` to disable recovery |

`safety_turns=500` exists only as a runaway guard and is not a capability limit.

### Finishing in time

The wall clock is a promise to whatever runs the harness, which kills it at its
own limit — taking the trajectory with it. So nothing inside a run may outlast
`--max-seconds`:

| | Ceiling | Flag | Also capped at |
|---|---|---|---|
| One shell command | 1800s | `--command-timeout` | the time left |
| One model request | 600s | `--request-timeout` | the time left |
| A retry's wait | 15s doubling | `--retry-backoff` | not attempted if it would not fit |

A command's timeout kills its whole process group, so an ffmpeg or `transcribe`
it started does not run on. Once less than a second is left, a command is not
started at all. What stays uncapped is small: ffprobe on a view or a submission
(60s at most each) and writing the trajectory. Leave a margin for those between
`--max-seconds` and the outer limit — the Harbor adapter leaves 600s.

### Stall detection

Grounded in the filesystem, **never** in the model's own account of its progress.
arXiv 2607.25152 measured a frontier agent claiming improvement on all 54 cycles
while 56% showed zero or negative change, and an in-band self-judge rejecting 38%
of genuine improvements while accepting 44% of real regressions. So the oracle
here is the output file's hash, its ffprobe duration, and the work-dir file count.

Two detectors, both deliberately lenient:

- **tight loop** — the same action returning the same observation three times.
  Not "the same command three times": `ffmpeg -ss 10/20/30` is three actions, and
  a repeated command whose output changed is new information. Threshold follows
  WebArena (arXiv 2307.13854, App. A.6).
- **no progress** — nothing observable changed for 12 turns. Extracting frames,
  viewing a new path, or trying a new command all count as progress; a real run
  spent its first 24 turns purely perceiving, which is legitimate work.

An 8-turn grace period precedes both, because a run needs room to orient.

On a stall the run **rewinds once** rather than dying: history truncates back to
where the loop began, following OpenHands. That adds no information — it only
removes the context the model got stuck in — so it stays clean for a capability
measurement. One rewind per episode, recorded in the trace.

## Outputs

The **deliverable directory holds only the deliverable.** Everything else goes to
`--runs-dir`.

```
/app/output/final_video.mp4     the submission, verified with ffprobe
/app/runs/trace.jsonl           one JSON line per event, appended as it happens
/app/runs/trajectory.json       the full conversation, media redacted
/app/work/                      the model's own scratch: frames, scripts, cuts
```

`trace.jsonl` is written open-append-close per event, so a run that is killed or
loses its stdout still leaves a complete record. That is not theoretical — a
17-minute run lost its entire stdout to an ssh timeout and lost nothing that
mattered. **Drive automation off the trace file, not stdout.**

`trajectory.json` is written in a `finally` block, so a crashed run still leaves a
readable transcript. Media blocks become `[image_url, 30017 bytes, omitted]`
placeholders — a single 30s clip is several megabytes of base64 and is not what
anyone reads a trajectory for.

### Outcome reporting

Three fields, never merged, following Terminal-Bench's `is_resolved` +
`FailureMode`:

- `submitted` — boolean
- `termination` — `SUBMITTED` · `TIMEOUT` · `COST` · `TOO_MANY_ERRORS` · `STALLED` · `ABANDONED`
- `delivered_elsewhere` — boolean

A run that timed out is not a bad edit. Pooling those makes the numbers lie.
A run the API gave out on ends as `error:<ExceptionType>` and says nothing about
the model at all.

`delivered_elsewhere` is the same argument one level down. Given `--output`, an
accepted submission is copied there if it is not already there — the grader
looks at one agreed location, and a good edit abandoned in the scratch dir would
score as no edit at all. But when the task instruction names that path, putting
the file there is part of what was asked, so a run the harness had to rescue is
not the same as one that landed it. The copy happens; the flag says it happened.

## A verified run

Sorcerer trailer task, 259.8s source, isolated environment, CPU-only host:

```
termination: SUBMITTED     rewinds: 0
42 turns · 1022s · $0.24 · 41 distinct actions in 41 turns
output: 64.7s, 1280x720, h264 + aac
```

Unprompted, it ran `transcribe`, discovered scene detection on its own
(`select='gt(scene,0.4)',showinfo`), sampled adaptively, cut at sub-second
boundaries, split one shot in two, added an on-screen graphic drawn from the
narration, viewed its own output, and revised before submitting.

For reference: the human reference edit is 60.7s; the scaffolded `montage`
pipeline produced 76.4s.

## Known limits

- **n=1.** Single runs. Two harness runs on the same task produced 64.7s and
  81.9s — that spread is run-to-run variance, not evidence about anything.
- **Nothing here has been rubric-scored.** Every number above is mechanical:
  duration, cost, gates. "Closest to the reference length" is not "best edit".
- **A large view is paged, not seen at once.** The model never has more than
  `--max-images` images or `--max-videos` videos in context together, so files
  further apart than one batch cannot be compared side by side unless the model
  combines them first.
- **Shell output is not stored in the trace**, only its length. The trajectory
  has the conversation; reconstructing exactly what a command printed needs the
  work dir.
- Audio findings are specific to `muse-spark-1.2-contributor`. Documentation
  claims 1.3 accepts audio, and 1.3 is not served in every region.
