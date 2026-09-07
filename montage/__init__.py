"""montage - cut a finished video out of one montage, driven by an instruction.

Five stages, each independently useful:
    segment    -> spans at the source's own scene changes
    transcribe -> what is said, with timecodes (whisper.cpp, CPU)
    describe   -> what is seen, per segment (Muse)
    plan       -> instruction + index -> edit decision list (Muse)
    render     -> ffmpeg executes the EDL
"""
__version__ = "0.1.0"
