"""A minimal agent harness for measuring a model's video editing capability.

The environment supplies raw capability (ffmpeg, a shell, a way to see and to
transcribe). The model supplies all judgement. The harness supplies neither -
it only moves messages, executes what is asked, and records what happened.
"""
__version__ = "0.1.0"
