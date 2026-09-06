"""
Unit tests for NexusOS Virtual File System (VFS) and Unix Pipes.
Tests directory traversal, file read/write, pipe buffering, EOF, and EPIPE.
"""

import unittest
from nexus.types import Errno
from nexus.vfs import VirtualFileSystem, InodeType, UnixPipe


class TestVFS(unittest.TestCase):
    def setUp(self):
        self.vfs = VirtualFileSystem()

    def test_directory_and_file_creation(self):
        status, inode_id = self.vfs.create_file("/etc/config.json", b'{"debug": true}')
        self.assertEqual(status, Errno.SUCCESS)
        self.assertIsNotNone(inode_id)

        # Open and read
        status, fd = self.vfs.open("/etc/config.json", "r")
        self.assertEqual(status, Errno.SUCCESS)
        self.assertEqual(fd.inode.data, bytearray(b'{"debug": true}'))

    def test_unix_pipe_lifecycle(self):
        r_fd, w_fd = self.vfs.create_pipe()
        pipe = r_fd.pipe

        # Write data into pipe
        status, written = pipe.write(b"HELLO_PIPE")
        self.assertEqual(status, Errno.SUCCESS)
        self.assertEqual(written, 10)

        # Read data from pipe
        status, data = pipe.read(64)
        self.assertEqual(status, Errno.SUCCESS)
        self.assertEqual(data, b"HELLO_PIPE")

    def test_unix_pipe_eof(self):
        r_fd, w_fd = self.vfs.create_pipe()
        pipe = r_fd.pipe

        pipe.write(b"FINAL_DATA")
        # Close write end
        pipe.close_write_end()

        # Read remaining data
        status, data = pipe.read(64)
        self.assertEqual(status, Errno.SUCCESS)
        self.assertEqual(data, b"FINAL_DATA")

        # Next read should return EOF (empty bytes, SUCCESS)
        status, eof_data = pipe.read(64)
        self.assertEqual(status, Errno.SUCCESS)
        self.assertEqual(eof_data, b"")

    def test_unix_pipe_broken_pipe_epipe(self):
        r_fd, w_fd = self.vfs.create_pipe()
        pipe = r_fd.pipe

        # Reader closes read end
        pipe.close_read_end()

        # Writer attempts to write: should return EPIPE
        status, written = pipe.write(b"DEAD_DATA")
        self.assertEqual(status, Errno.EPIPE)
        self.assertEqual(written, 0)


if __name__ == "__main__":
    unittest.main()
