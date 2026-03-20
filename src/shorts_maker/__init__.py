"""
Shorts Maker GPU: A high-performance video processing library.
"""
import logging

from .config import ProcessingConfig
from .core.processor import VideoProcessor
from .io.streamer import GPUVideoStreamer

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "GPUVideoStreamer",
    "VideoProcessor",
    "ProcessingConfig",
]
