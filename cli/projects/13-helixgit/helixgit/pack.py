"""
HelixGit: Packfile Serialization and Delta Compression Engine.
Implements the Git Packfile v2 format, copy/insert delta compression,
sliding window block matching, and delta reconstitution.
"""

import zlib
import struct
import binascii
import hashlib
from typing import List, Tuple, Dict, Optional, Any
from .objects import GitObject, ObjectType, Blob, Tree, Commit, Tag, parse_raw_object

PACK_MAGIC = b"PACK"
PACK_VERSION = 2

# Pack object type codes
OBJ_COMMIT = 1
OBJ_TREE = 2
OBJ_BLOB = 3
OBJ_TAG = 4
OBJ_OFS_DELTA = 6
OBJ_REF_DELTA = 7

TYPE_TO_OBJ = {
    ObjectType.COMMIT: OBJ_COMMIT,
    ObjectType.TREE: OBJ_TREE,
    ObjectType.BLOB: OBJ_BLOB,
    ObjectType.TAG: OBJ_TAG
}

OBJ_TO_TYPE = {
    OBJ_COMMIT: ObjectType.COMMIT,
    OBJ_TREE: ObjectType.TREE,
    OBJ_BLOB: ObjectType.BLOB,
    OBJ_TAG: ObjectType.TAG
}


def encode_size(size: int) -> bytes:
    """Encodes an integer into Git variable-length LEB128 format."""
    out = bytearray()
    while True:
        byte = size & 0x7F
        size >>= 7
        if size > 0:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


def decode_size(data: bytes, offset: int) -> Tuple[int, int]:
    """Decodes an integer from Git variable-length LEB128 format at offset."""
    val = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        val |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            break
    return val, offset


def create_delta(base: bytes, target: bytes) -> bytes:
    """
    Computes a Git-compatible copy/insert delta from base to target.
    Uses a 16-byte sliding window hash index to identify repetitive chunks.
    Emits Git delta opcodes:
      - Copy opcode (0x80 | flags): copies [offset, offset + size] from base.
      - Insert opcode (0 < len <= 127): literal byte insertion.
    """
    out = bytearray()
    # Header: base size and target size in LEB128
    out.extend(encode_size(len(base)))
    out.extend(encode_size(len(target)))

    # Index base in 16-byte blocks
    block_size = 16
    base_index: Dict[bytes, List[int]] = {}
    for i in range(0, len(base) - block_size + 1, block_size):
        chunk = base[i:i + block_size]
        if chunk not in base_index:
            base_index[chunk] = []
        base_index[chunk].append(i)

    t_idx = 0
    t_len = len(target)
    pending_insert = bytearray()

    def flush_insert():
        nonlocal pending_insert
        while pending_insert:
            chunk_len = min(len(pending_insert), 127)
            out.append(chunk_len)
            out.extend(pending_insert[:chunk_len])
            pending_insert = pending_insert[chunk_len:]

    while t_idx < t_len:
        best_offset = -1
        best_match_len = 0

        if t_idx + block_size <= t_len:
            probe = target[t_idx:t_idx + block_size]
            candidates = base_index.get(probe, [])
            for cand in candidates:
                # Extend match as far as possible
                match_len = 0
                while (
                    cand + match_len < len(base) and
                    t_idx + match_len < t_len and
                    base[cand + match_len] == target[t_idx + match_len] and
                    match_len < 65536
                ):
                    match_len += 1

                if match_len > best_match_len:
                    best_match_len = match_len
                    best_offset = cand

        if best_match_len >= block_size:
            # We have an efficient copy match
            flush_insert()

            # Encode copy opcode: 0x80 | flags
            # Bits 0-3: presence of offset bytes
            # Bits 4-5: presence of size bytes
            flags = 0x80
            off_bytes = bytearray()
            for shift in (0, 8, 16, 24):
                b = (best_offset >> shift) & 0xFF
                if b > 0 or shift == 0:
                    flags |= (1 << (shift // 8))
                    off_bytes.append(b)

            sz_bytes = bytearray()
            # If size == 65536 (0x10000), Git convention sets size bytes to empty
            if best_match_len != 0x10000:
                for shift in (0, 8):
                    b = (best_match_len >> shift) & 0xFF
                    if b > 0 or shift == 0:
                        flags |= (1 << (4 + shift // 8))
                        sz_bytes.append(b)

            out.append(flags)
            out.extend(off_bytes)
            out.extend(sz_bytes)

            t_idx += best_match_len
        else:
            pending_insert.append(target[t_idx])
            t_idx += 1

    flush_insert()
    return bytes(out)


def apply_delta(base: bytes, delta: bytes) -> bytes:
    """
    Reconstructs target data from base bytes and Git copy/insert delta bytecode.
    """
    offset = 0
    base_size, offset = decode_size(delta, offset)
    target_size, offset = decode_size(delta, offset)

    if len(base) != base_size:
        raise ValueError(f"Delta application error: base size mismatch (expected {base_size}, got {len(base)})")

    out = bytearray()
    delta_len = len(delta)

    while offset < delta_len:
        opcode = delta[offset]
        offset += 1

        if opcode & 0x80:
            # Copy opcode
            copy_offset = 0
            copy_size = 0

            # Parse offset
            for i in range(4):
                if opcode & (1 << i):
                    copy_offset |= delta[offset] << (i * 8)
                    offset += 1

            # Parse size
            for i in range(2):
                if opcode & (1 << (4 + i)):
                    copy_size |= delta[offset] << (i * 8)
                    offset += 1

            if copy_size == 0:
                copy_size = 0x10000  # Git convention: 65536 bytes

            if copy_offset + copy_size > len(base):
                raise ValueError(f"Delta copy out of bounds ({copy_offset} + {copy_size} > {len(base)})")

            out.extend(base[copy_offset:copy_offset + copy_size])
        elif opcode > 0:
            # Literal insert opcode (length: 1..127)
            insert_len = opcode
            out.extend(delta[offset:offset + insert_len])
            offset += insert_len
        else:
            raise ValueError("Corrupt Git delta: zero opcode encountered")

    if len(out) != target_size:
        raise ValueError(f"Delta application error: target size mismatch (expected {target_size}, got {len(out)})")

    return bytes(out)


class PackObjectInfo:
    """Metadata for an object written to or read from a packfile."""
    __slots__ = ("sha1", "type_code", "offset", "size", "crc32", "base_sha1")

    def __init__(
        self,
        sha1: str,
        type_code: int,
        offset: int,
        size: int,
        crc32: int,
        base_sha1: Optional[str] = None
    ) -> None:
        self.sha1 = sha1.lower()
        self.type_code = type_code
        self.offset = offset
        self.size = size
        self.crc32 = crc32 & 0xFFFFFFFF
        self.base_sha1 = base_sha1.lower() if base_sha1 else None


class PackWriter:
    """
    Serializes a set of GitObjects into standard Git Packfile v2 format.
    Generates object metadata (offsets, CRC32) suitable for .idx generation.
    """

    def __init__(self) -> None:
        self.objects: List[GitObject] = []

    def add_object(self, obj: GitObject) -> None:
        self.objects.append(obj)

    def write_pack(self) -> Tuple[bytes, List[PackObjectInfo]]:
        """
        Builds packfile byte stream and returns (pack_bytes, list_of_pack_object_infos).
        """
        out = bytearray()
        # 12-byte header: 'PACK' + version 2 + object count
        out.extend(PACK_MAGIC)
        out.extend(struct.pack(">2I", PACK_VERSION, len(self.objects)))

        infos: List[PackObjectInfo] = []

        for obj in self.objects:
            obj_offset = len(out)
            sha1 = obj.sha1()
            type_code = TYPE_TO_OBJ[obj.object_type]
            payload = obj.serialize()
            size = len(payload)

            # Object header byte 1: MSB, 3-bit type, 4-bit size
            byte1 = ((type_code & 0x7) << 4) | (size & 0x0F)
            size >>= 4
            if size > 0:
                byte1 |= 0x80
            entry_header = bytearray([byte1])

            while size > 0:
                byte = size & 0x7F
                size >>= 7
                if size > 0:
                    byte |= 0x80
                entry_header.append(byte)

            # Compress payload with zlib
            compressed_payload = zlib.compress(payload, level=zlib.Z_BEST_SPEED)

            entry_bytes = bytes(entry_header) + compressed_payload
            crc = zlib.crc32(entry_bytes) & 0xFFFFFFFF

            out.extend(entry_bytes)

            infos.append(PackObjectInfo(
                sha1=sha1,
                type_code=type_code,
                offset=obj_offset,
                size=len(payload),
                crc32=crc
            ))

        # Trailing 20-byte SHA-1 of all preceding pack bytes
        pack_sha1 = hashlib.sha1(bytes(out)).digest()
        out.extend(pack_sha1)

        return bytes(out), infos


class PackReader:
    """
    Parses Git Packfile v2 byte stream and reconstructs contained GitObjects.
    Handles standard objects and delta-compressed objects.
    """

    def __init__(self, pack_data: bytes) -> None:
        self.data = pack_data
        self.objects: Dict[str, GitObject] = {}
        self._parse()

    def _parse(self) -> None:
        if len(self.data) < 32:
            raise ValueError("Packfile too small: missing header or checksum")

        # Verify trailing checksum
        body = self.data[:-20]
        expected_checksum = self.data[-20:]
        calc_checksum = hashlib.sha1(body).digest()
        if expected_checksum != calc_checksum:
            raise ValueError("Corrupt Packfile: checksum mismatch")

        magic = self.data[:4]
        if magic != PACK_MAGIC:
            raise ValueError(f"Invalid packfile magic: {magic!r}")

        version, num_objects = struct.unpack(">2I", self.data[4:12])
        if version != PACK_VERSION:
            raise ValueError(f"Unsupported packfile version: {version}")

        offset = 12
        pending_deltas: List[Tuple[str, int, str, bytes]] = []  # (type, offset, base_sha, delta_bytes)

        for _ in range(num_objects):
            entry_start = offset
            byte = self.data[offset]
            offset += 1

            type_code = (byte >> 4) & 0x7
            size = byte & 0x0F
            shift = 4

            while byte & 0x80:
                byte = self.data[offset]
                offset += 1
                size |= (byte & 0x7F) << shift
                shift += 7

            base_sha1: Optional[str] = None
            if type_code == OBJ_REF_DELTA:
                raw_base_sha = self.data[offset:offset + 20]
                base_sha1 = binascii.hexlify(raw_base_sha).decode("ascii")
                offset += 20
            elif type_code == OBJ_OFS_DELTA:
                # Offset delta: decode variable-length negative offset
                b = self.data[offset]
                offset += 1
                neg_offset = b & 0x7F
                while b & 0x80:
                    b = self.data[offset]
                    offset += 1
                    neg_offset = ((neg_offset + 1) << 7) | (b & 0x7F)
                # Compute base entry offset
                base_entry_offset = entry_start - neg_offset

            # Decompress zlib stream
            decompressor = zlib.decompressobj()
            payload = decompressor.decompress(self.data[offset:])
            consumed = len(self.data[offset:]) - len(decompressor.unused_data)
            offset += consumed

            if type_code in OBJ_TO_TYPE:
                obj_type = OBJ_TO_TYPE[type_code]
                if obj_type == ObjectType.BLOB:
                    obj = Blob.deserialize(payload)
                elif obj_type == ObjectType.TREE:
                    obj = Tree.deserialize(payload)
                elif obj_type == ObjectType.COMMIT:
                    obj = Commit.deserialize(payload)
                elif obj_type == ObjectType.TAG:
                    obj = Tag.deserialize(payload)
                self.objects[obj.sha1()] = obj
            elif type_code == OBJ_REF_DELTA and base_sha1:
                pending_deltas.append(("ref", entry_start, base_sha1, payload))

        # Resolve any pending ref-deltas
        for delta_type, _, base_sha, delta_payload in pending_deltas:
            if base_sha in self.objects:
                base_obj = self.objects[base_sha]
                reconstructed = apply_delta(base_obj.serialize(), delta_payload)
                # Parse reconstructed object based on base type
                if base_obj.object_type == ObjectType.BLOB:
                    new_obj = Blob.deserialize(reconstructed)
                elif base_obj.object_type == ObjectType.TREE:
                    new_obj = Tree.deserialize(reconstructed)
                elif base_obj.object_type == ObjectType.COMMIT:
                    new_obj = Commit.deserialize(reconstructed)
                elif base_obj.object_type == ObjectType.TAG:
                    new_obj = Tag.deserialize(reconstructed)
                self.objects[new_obj.sha1()] = new_obj
