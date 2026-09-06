"""
AuraDSP: Binary RIFF WAV Audio Codec.
Encodes and decodes uncompressed 16-bit linear PCM audio containers from first principles
using the pure Python standard library struct module.
"""

import struct
from typing import List, Tuple, Union


def encode_wav_bytes(
    samples: Union[List[float], Tuple[List[float], List[float]]],
    sample_rate: int = 44100
) -> bytes:
    """
    Encodes audio samples into a standard 16-bit linear PCM RIFF WAVE byte stream.
    Supports mono (List[float]) or stereo (Tuple[left_channel, right_channel]).
    Audio samples are expected in the normalized floating-point range [-1.0, 1.0].
    """
    if isinstance(samples, tuple):
        # Stereo
        left, right = samples
        if len(left) != len(right):
            raise ValueError("Left and right channels must have identical sample counts")
        num_channels = 2
        num_frames = len(left)
        interleaved_ints: List[int] = []
        for l_samp, r_samp in zip(left, right):
            # Clamp to [-1.0, 1.0] and scale to 16-bit signed integer
            l_clamped = max(-1.0, min(1.0, l_samp))
            r_clamped = max(-1.0, min(1.0, r_samp))
            interleaved_ints.append(int(round(l_clamped * 32767.0)))
            interleaved_ints.append(int(round(r_clamped * 32767.0)))
        raw_pcm = struct.pack(f"<{len(interleaved_ints)}h", *interleaved_ints)
    else:
        # Mono
        num_channels = 1
        num_frames = len(samples)
        clamped_ints = [
            int(round(max(-1.0, min(1.0, s)) * 32767.0))
            for s in samples
        ]
        raw_pcm = struct.pack(f"<{len(clamped_ints)}h", *clamped_ints)

    bits_per_sample = 16
    block_align = num_channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    subchunk2_size = len(raw_pcm)
    chunk_size = 36 + subchunk2_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,                 # Subchunk1Size (16 for PCM)
        1,                  # AudioFormat (1 for uncompressed PCM)
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        subchunk2_size
    )

    return header + raw_pcm


def decode_wav_bytes(data: bytes) -> Tuple[List[float], int]:
    """
    Decodes an uncompressed 16-bit linear PCM RIFF WAVE byte stream.
    Returns:
      (samples: List[float] in range [-1.0, 1.0], sample_rate: int)
    """
    if len(data) < 44:
        raise ValueError("WAV data too short for valid RIFF header")

    riff, chunk_size, wave = struct.unpack("<4sI4s", data[:12])
    if riff != b"RIFF" or wave != b"WAVE":
        raise ValueError("Invalid RIFF/WAVE header identifier")

    offset = 12
    fmt_found = False
    data_found = False
    audio_format = 1
    num_channels = 1
    sample_rate = 44100
    bits_per_sample = 16
    pcm_bytes = b""

    # Walk chunks
    while offset + 8 <= len(data):
        chunk_id, sub_size = struct.unpack("<4sI", data[offset:offset + 8])
        chunk_data_start = offset + 8
        chunk_data_end = chunk_data_start + sub_size

        if chunk_id == b"fmt ":
            fmt_found = True
            (
                audio_format,
                num_channels,
                sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
            ) = struct.unpack("<HHIIHH", data[chunk_data_start:chunk_data_start + 16])
            if audio_format != 1:
                raise ValueError(f"Only uncompressed PCM (format 1) supported; got {audio_format}")
            if bits_per_sample != 16:
                raise ValueError(f"Only 16-bit PCM supported; got {bits_per_sample}")
        elif chunk_id == b"data":
            data_found = True
            pcm_bytes = data[chunk_data_start:chunk_data_end]
            break

        offset = chunk_data_end

    if not fmt_found or not data_found:
        raise ValueError("Missing 'fmt ' or 'data' chunk in WAV stream")

    num_samples = len(pcm_bytes) // 2
    raw_shorts = struct.unpack(f"<{num_samples}h", pcm_bytes)

    if num_channels == 1:
        float_samples = [s / 32767.0 for s in raw_shorts]
    else:
        # Average stereo to mono for uniform 1D representation
        mono_samples = []
        for i in range(0, len(raw_shorts) - 1, 2):
            left = raw_shorts[i] / 32767.0
            right = raw_shorts[i + 1] / 32767.0
            mono_samples.append(0.5 * (left + right))
        float_samples = mono_samples

    return float_samples, sample_rate


def write_wav(
    filename: str,
    samples: Union[List[float], Tuple[List[float], List[float]]],
    sample_rate: int = 44100
) -> None:
    """Writes audio samples to a .wav file on disk."""
    data = encode_wav_bytes(samples, sample_rate)
    with open(filename, "wb") as f:
        f.write(data)


def read_wav(filename: str) -> Tuple[List[float], int]:
    """Reads a .wav file from disk and returns (samples, sample_rate)."""
    with open(filename, "rb") as f:
        data = f.read()
    return decode_wav_bytes(data)
