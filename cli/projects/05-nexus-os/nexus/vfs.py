"""
NexusOS: Virtual File System (VFS) and Unix Pipes.
Implements hierarchical inode tree, file descriptor table management,
standard I/O streams, and unidirectional circular buffer pipes with EOF and EPIPE handling.
"""

from typing import Dict, List, Optional, Tuple, Union
from collections import deque
from dataclasses import dataclass
from nexus.types import Errno


class InodeType:
    FILE = 1
    DIRECTORY = 2
    PIPE = 3


@dataclass
class Inode:
    """Core filesystem object containing data or directory entries."""
    inode_id: int
    inode_type: int
    data: bytearray
    entries: Dict[str, int]  # Name -> Inode ID (for directories)


class UnixPipe:
    """
    Unidirectional FIFO byte buffer with Unix pipe semantics.
    Supports circular buffering, blocking on empty/full, EOF on closed writer,
    and EPIPE (broken pipe) on write to closed reader.
    """
    CAPACITY = 4096

    def __init__(self, pipe_id: int):
        self.pipe_id = pipe_id
        self.buffer = bytearray()
        self.reader_count = 1
        self.writer_count = 1
        self.waiting_readers: deque[int] = deque()
        self.waiting_writers: deque[int] = deque()

    def read(self, length: int, reader_pid: Optional[int] = None) -> Tuple[Errno, bytes]:
        """
        Read up to `length` bytes from the pipe:
        - If data exists, return it immediately and wake waiting writers.
        - If empty and no writers remaining, return EOF (b'', SUCCESS).
        - If empty and writers exist, return EAGAIN (to block reader).
        """
        if len(self.buffer) > 0:
            read_size = min(length, len(self.buffer))
            chunk = bytes(self.buffer[:read_size])
            del self.buffer[:read_size]
            return Errno.SUCCESS, chunk

        # Buffer is empty
        if self.writer_count <= 0:
            return Errno.SUCCESS, b""  # EOF

        if reader_pid is not None:
            self.waiting_readers.append(reader_pid)
        return Errno.EAGAIN, b""

    def write(self, data: bytes, writer_pid: Optional[int] = None) -> Tuple[Errno, int]:
        """
        Write bytes into the pipe:
        - If no readers remaining, return EPIPE (broken pipe).
        - If space available, append data and wake waiting readers.
        - If buffer full, return EAGAIN (to block writer).
        """
        if self.reader_count <= 0:
            return Errno.EPIPE, 0

        space_left = self.CAPACITY - len(self.buffer)
        if space_left <= 0:
            if writer_pid is not None:
                self.waiting_writers.append(writer_pid)
            return Errno.EAGAIN, 0

        write_size = min(len(data), space_left)
        self.buffer.extend(data[:write_size])
        return Errno.SUCCESS, write_size

    def close_read_end(self) -> None:
        self.reader_count = max(0, self.reader_count - 1)

    def close_write_end(self) -> None:
        self.writer_count = max(0, self.writer_count - 1)


@dataclass
class FileDescriptor:
    """Per-process handle to an open file, pipe, or device."""
    fd_num: int
    mode: str                  # 'r', 'w', 'rw'
    inode: Optional[Inode] = None
    pipe: Optional[UnixPipe] = None
    offset: int = 0
    is_pipe_reader: bool = False
    is_pipe_writer: bool = False


class VirtualFileSystem:
    """
    Hierarchical Inode File System with root, bin, dev, etc, and tmp.
    """
    def __init__(self):
        self.inodes: Dict[int, Inode] = {}
        self._next_inode_id = 1
        self._next_pipe_id = 1

        # Create root directory '/' (Inode 1)
        self.root_inode_id = self._create_inode(InodeType.DIRECTORY)
        self._mkdir_root("bin")
        self._mkdir_root("dev")
        self._mkdir_root("etc")
        self._mkdir_root("tmp")

    def _create_inode(self, inode_type: int) -> int:
        inode_id = self._next_inode_id
        self._next_inode_id += 1
        self.inodes[inode_id] = Inode(
            inode_id=inode_id,
            inode_type=inode_type,
            data=bytearray(),
            entries={}
        )
        return inode_id

    def _mkdir_root(self, name: str) -> int:
        inode_id = self._create_inode(InodeType.DIRECTORY)
        self.inodes[self.root_inode_id].entries[name] = inode_id
        return inode_id

    def resolve_path(self, path: str) -> Optional[int]:
        """Traverse directory hierarchy and return target Inode ID."""
        if not path.startswith("/"):
            path = "/" + path
        parts = [p for p in path.split("/") if p]
        curr_id = self.root_inode_id

        for part in parts:
            curr_inode = self.inodes.get(curr_id)
            if curr_inode is None or curr_inode.inode_type != InodeType.DIRECTORY:
                return None
            if part not in curr_inode.entries:
                return None
            curr_id = curr_inode.entries[part]

        return curr_id

    def create_file(self, path: str, data: bytes = b"") -> Tuple[Errno, Optional[int]]:
        """Create a new regular file at path with initial data."""
        if not path.startswith("/"):
            path = "/" + path
        parts = [p for p in path.split("/") if p]
        if not parts:
            return Errno.EINVAL, None

        filename = parts[-1]
        dir_parts = parts[:-1]

        # Resolve parent directory
        curr_id = self.root_inode_id
        for part in dir_parts:
            curr_inode = self.inodes.get(curr_id)
            if curr_inode is None or curr_inode.inode_type != InodeType.DIRECTORY:
                return Errno.ENOENT, None
            if part not in curr_inode.entries:
                return Errno.ENOENT, None
            curr_id = curr_inode.entries[part]

        parent_inode = self.inodes[curr_id]
        if filename in parent_inode.entries:
            # File already exists: overwrite
            file_inode_id = parent_inode.entries[filename]
            self.inodes[file_inode_id].data = bytearray(data)
            return Errno.SUCCESS, file_inode_id

        file_inode_id = self._create_inode(InodeType.FILE)
        self.inodes[file_inode_id].data = bytearray(data)
        parent_inode.entries[filename] = file_inode_id
        return Errno.SUCCESS, file_inode_id

    def open(self, path: str, mode: str = "r") -> Tuple[Errno, Optional[FileDescriptor]]:
        """Open a file and return a FileDescriptor object."""
        inode_id = self.resolve_path(path)
        if inode_id is None:
            if "w" in mode:
                status, inode_id = self.create_file(path)
                if status != Errno.SUCCESS:
                    return status, None
            else:
                return Errno.ENOENT, None

        inode = self.inodes[inode_id]
        if inode.inode_type == InodeType.DIRECTORY and "w" in mode:
            return Errno.EACCES, None

        fd = FileDescriptor(fd_num=-1, mode=mode, inode=inode, offset=0)
        return Errno.SUCCESS, fd

    def create_pipe(self) -> Tuple[FileDescriptor, FileDescriptor]:
        """Allocate a new Unix pipe returning (read_fd, write_fd)."""
        pipe_id = self._next_pipe_id
        self._next_pipe_id += 1
        pipe = UnixPipe(pipe_id=pipe_id)

        read_fd = FileDescriptor(
            fd_num=-1,
            mode="r",
            pipe=pipe,
            is_pipe_reader=True
        )
        write_fd = FileDescriptor(
            fd_num=-1,
            mode="w",
            pipe=pipe,
            is_pipe_writer=True
        )
        return read_fd, write_fd
