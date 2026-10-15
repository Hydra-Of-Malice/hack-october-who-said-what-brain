"""The "who said what" engine: live audio -> VAD -> ASR + speaker embeddings -> online clustering -> Utterance."""

from .audio import WavAppender, decode_file, float_to_pcm16, pcm16_to_float, write_wav
from .factory import get_asr, get_speaker_embedder
from .pipeline import process_file
from .session import LiveSession
from .speakers import OnlineClusterer

__all__ = [
    "LiveSession",
    "OnlineClusterer",
    "WavAppender",
    "decode_file",
    "float_to_pcm16",
    "get_asr",
    "get_speaker_embedder",
    "pcm16_to_float",
    "process_file",
    "write_wav",
]
