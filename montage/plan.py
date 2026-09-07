"""Turn an instruction plus the segment index into an edit decision list.

The EDL is the contract between thinking and doing: a list of source spans in
playback order, each optionally carrying on-screen text. `render` executes it
with ffmpeg and never makes creative choices of its own.
"""
import json

from montage.llm import ask_json

PLAN_PROMPT = """You are an editor cutting a finished piece from one source video.

THE BRIEF: {instruction}
{length_note}
Below is every segment of the source, with its timecode, the narration over it,
and what is visible on screen.

{catalogue}

Choose the segments that deliver the brief and put them in the order they should
play. You may reorder freely and you may use a segment only once.

Rules:
- Use only the segment indices listed above.
- Prefer segments whose narration carries the story; use visual-only segments to
  open, to close, or to break up talking.
- The result should flow as one piece, not a list of clips.
- Add "text" to a cut only where an on-screen title genuinely helps - a title
  card, a name, a place. Most cuts should have no text. Keep any text under 40
  characters.

Reply with ONLY a JSON object:
{{"cuts": [{{"index": <segment index>, "text": "<optional on-screen text>"}}],
  "reasoning": "<one sentence on the shape you chose>"}}"""


def format_catalogue(entries):
    rows = []
    for entry in entries:
        speech = entry.get("speech") or ""
        speech = f'  SAID: "{speech}"' if speech else "  SAID: (no narration)"
        rows.append(
            f"[{entry['index']}] {entry['start']:.1f}-{entry['end']:.1f}s "
            f"({entry['duration']:.1f}s)\n"
            f"  SEEN: {entry['description']}\n{speech}"
        )
    return "\n\n".join(rows)


def plan_edit(entries, instruction, target_duration=None):
    """Ask Muse for an EDL. Returns (cuts, reasoning).

    Each cut is {index, start, end, text}, resolved against the segment index so
    render never has to look anything up.
    """
    length_note = ""
    if target_duration:
        length_note = (
            f"\nTARGET LENGTH: about {target_duration:.0f} seconds. The chosen "
            f"segment durations should add up to roughly that.\n"
        )

    reply = ask_json(PLAN_PROMPT.format(
        instruction=instruction,
        length_note=length_note,
        catalogue=format_catalogue(entries),
    ))

    by_index = {e["index"]: e for e in entries}
    cuts = []
    seen = set()
    for cut in reply.get("cuts", []):
        index = cut.get("index")
        if index not in by_index or index in seen:
            continue  # hallucinated or duplicated index
        seen.add(index)
        entry = by_index[index]
        text = (cut.get("text") or "").strip()
        cuts.append({
            "index": index,
            "start": entry["start"],
            "end": entry["end"],
            "text": text or None,
        })

    if not cuts:
        raise ValueError("The planner returned no usable cuts.")
    return cuts, reply.get("reasoning", "")


def save(cuts, reasoning, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"reasoning": reasoning, "cuts": cuts}, handle, indent=2,
                  ensure_ascii=False)


def load(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data["cuts"], data.get("reasoning", "")
