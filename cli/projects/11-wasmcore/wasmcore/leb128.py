"""
WasmCore: LEB128 (Little-Endian Base 128) Variable-Length Integer Codecs.
Spec-compliant decoding and encoding for signed and unsigned 32-bit and 64-bit integers.
"""

import io
from typing import Union, List, Callable, TypeVar
from .types import WasmValidationError

T = TypeVar("T")


def decode_u32(stream: Union[io.BytesIO, bytes, bytearray]) -> int:
    """Decode an unsigned 32-bit integer from LEB128 format."""
    if isinstance(stream, (bytes, bytearray)):
        stream = io.BytesIO(stream)

    result = 0
    shift = 0
    count = 0

    while True:
        byte_data = stream.read(1)
        if not byte_data:
            raise WasmValidationError("Unexpected end of LEB128 u32 stream")
        byte = byte_data[0]
        count += 1

        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
        if shift >= 35:
            raise WasmValidationError("LEB128 u32 integer overflow (exceeds 5 bytes)")

    return result


def decode_i32(stream: Union[io.BytesIO, bytes, bytearray]) -> int:
    """Decode a signed 32-bit integer from LEB128 format with sign extension."""
    if isinstance(stream, (bytes, bytearray)):
        stream = io.BytesIO(stream)

    result = 0
    shift = 0
    count = 0

    while True:
        byte_data = stream.read(1)
        if not byte_data:
            raise WasmValidationError("Unexpected end of LEB128 i32 stream")
        byte = byte_data[0]
        count += 1

        result |= (byte & 0x7F) << shift
        shift += 7

        if (byte & 0x80) == 0:
            # Sign extend if sign bit (0x40) of final 7-bit chunk is set
            if shift < 32 and (byte & 0x40) != 0:
                result |= (~0 << shift)
            break
        if shift >= 35:
            raise WasmValidationError("LEB128 i32 integer overflow (exceeds 5 bytes)")

    # Normalize to signed 32-bit two's complement integer
    result = result & 0xFFFFFFFF
    if result & 0x80000000:
        result -= 0x100000000
    return result


def decode_u64(stream: Union[io.BytesIO, bytes, bytearray]) -> int:
    """Decode an unsigned 64-bit integer from LEB128 format."""
    if isinstance(stream, (bytes, bytearray)):
        stream = io.BytesIO(stream)

    result = 0
    shift = 0

    while True:
        byte_data = stream.read(1)
        if not byte_data:
            raise WasmValidationError("Unexpected end of LEB128 u64 stream")
        byte = byte_data[0]

        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
        if shift >= 70:
            raise WasmValidationError("LEB128 u64 integer overflow (exceeds 10 bytes)")

    return result


def decode_i64(stream: Union[io.BytesIO, bytes, bytearray]) -> int:
    """Decode a signed 64-bit integer from LEB128 format with sign extension."""
    if isinstance(stream, (bytes, bytearray)):
        stream = io.BytesIO(stream)

    result = 0
    shift = 0

    while True:
        byte_data = stream.read(1)
        if not byte_data:
            raise WasmValidationError("Unexpected end of LEB128 i64 stream")
        byte = byte_data[0]

        result |= (byte & 0x7F) << shift
        shift += 7

        if (byte & 0x80) == 0:
            if shift < 64 and (byte & 0x40) != 0:
                result |= (~0 << shift)
            break
        if shift >= 70:
            raise WasmValidationError("LEB128 i64 integer overflow (exceeds 10 bytes)")

    result = result & 0xFFFFFFFFFFFFFFFF
    if result & 0x8000000000000000:
        result -= 0x10000000000000000
    return result


def encode_u32(val: int) -> bytes:
    """Encode an unsigned 32-bit integer into LEB128 format."""
    val = val & 0xFFFFFFFF
    out = bytearray()
    while True:
        byte = val & 0x7F
        val >>= 7
        if val != 0:
            byte |= 0x80
        out.append(byte)
        if val == 0:
            break
    return bytes(out)


def encode_i32(val: int) -> bytes:
    """Encode a signed 32-bit integer into LEB128 format."""
    out = bytearray()
    more = True
    while more:
        byte = val & 0x7F
        val >>= 7
        # Check if sign bit of current 7-bit chunk matches remaining bits
        sign_bit = (byte & 0x40) != 0
        if (val == 0 and not sign_bit) or (val == -1 and sign_bit):
            more = False
        else:
            byte |= 0x80
        out.append(byte)
    return bytes(out)


def encode_u64(val: int) -> bytes:
    """Encode an unsigned 64-bit integer into LEB128 format."""
    val = val & 0xFFFFFFFFFFFFFFFF
    out = bytearray()
    while True:
        byte = val & 0x7F
        val >>= 7
        if val != 0:
            byte |= 0x80
        out.append(byte)
        if val == 0:
            break
    return bytes(out)


def encode_i64(val: int) -> bytes:
    """Encode a signed 64-bit integer into LEB128 format."""
    out = bytearray()
    more = True
    while more:
        byte = val & 0x7F
        val >>= 7
        sign_bit = (byte & 0x40) != 0
        if (val == 0 and not sign_bit) or (val == -1 and sign_bit):
            more = False
        else:
            byte |= 0x80
        out.append(byte)
    return bytes(out)


def decode_vec(stream: io.BytesIO, decode_elem: Callable[[io.BytesIO], T]) -> List[T]:
    """Decode a vector prefixed by count as LEB128 u32."""
    count = decode_u32(stream)
    return [decode_elem(stream) for _ in range(count)]


def encode_vec(items: List[T], encode_elem: Callable[[T], bytes]) -> bytes:
    """Encode a vector prefixed by count as LEB128 u32."""
    out = bytearray(encode_u32(len(items)))
    for item in items:
        out.extend(encode_elem(item))
    return bytes(out)


def decode_name(stream: io.BytesIO) -> str:
    """Decode a UTF-8 string prefixed by byte length as LEB128 u32."""
    length = decode_u32(stream)
    raw = stream.read(length)
    if len(raw) != length:
        raise WasmValidationError(f"Expected {length} string bytes, got {len(raw)}")
    return raw.decode("utf-8", errors="replace")


def encode_name(name: str) -> bytes:
    """Encode a UTF-8 string prefixed by byte length as LEB128 u32."""
    raw = name.encode("utf-8")
    return encode_u32(len(raw)) + raw
