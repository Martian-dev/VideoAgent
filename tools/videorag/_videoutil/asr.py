import os

from tqdm import tqdm

from environment.config.transcribe import transcribe_segments


def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    """Transcribe each cached audio segment via whisper.cpp.

    The deployment target is CPU-only, so this runs through whisper.cpp rather
    than the transformers pipeline the upstream project used. The returned shape
    is unchanged: {segment_index: "[0.00s -> 1.23s] text\\n..."}.
    """
    cache_path = os.path.join(working_dir, '_cache', video_name)

    transcripts = {}
    for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")

        try:
            lines = transcribe_segments(audio_file)
        except Exception as e:
            print(f"Transcription failed for {segment_name}: {e}")
            lines = []

        transcripts[index] = "".join(
            f"[{start:.2f}s -> {end:.2f}s] {text}\n" for start, end, text in lines
        )

    return transcripts
