"""
HelixGit: Performance Benchmark Suite.
Measures throughput of Object Hashing, Loose Storage, Binary Index Codec,
Myers Diff Engine, Packfile Delta Compression, and .idx Fan-Out Lookups.
"""

import os
import sys
import time
import shutil
import tempfile

# Ensure helixgit is importable when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helixgit.objects import Blob, Tree, TreeEntry, Commit
from helixgit.storage import LooseObjectStore, ObjectDatabase
from helixgit.index import Index, IndexEntry
from helixgit.diff import myers_diff
from helixgit.pack import create_delta, apply_delta, PackWriter, PackReader
from helixgit.idx import PackIndexWriter, PackIndexReader


def benchmark_object_hashing_and_storage():
    temp_dir = tempfile.mkdtemp()
    try:
        store = LooseObjectStore(temp_dir)
        num_objects = 2000
        payload = b"print('High performance Git implementation in pure Python standard library')\n"

        start = time.perf_counter()
        shas = []
        for i in range(num_objects):
            blob = Blob(payload + f"# line {i}\n".encode("ascii"))
            sha = store.write(blob)
            shas.append(sha)
        write_time = time.perf_counter() - start
        write_rate = num_objects / write_time

        # Read back
        start = time.perf_counter()
        for sha in shas:
            _ = store.read(sha)
        read_time = time.perf_counter() - start
        read_rate = num_objects / read_time

        print(f"[*] Object Store Write: {write_rate:,.0f} objects/s ({write_time*1000/num_objects:.3f} ms/obj)")
        print(f"[*] Object Store Read : {read_rate:,.0f} objects/s ({read_time*1000/num_objects:.3f} ms/obj)")
    finally:
        shutil.rmtree(temp_dir)


def benchmark_binary_index_codec():
    temp_dir = tempfile.mkdtemp()
    try:
        index_path = os.path.join(temp_dir, "index")
        num_entries = 5000

        index = Index()
        for i in range(num_entries):
            entry = IndexEntry(
                path=f"src/module_{i // 100}/sub_{i // 10}/file_{i}.py",
                sha1=f"{i:040x}",
                mode=0o100644,
                file_size=1024 + (i % 500)
            )
            index.add_entry(entry)

        # Measure serialization
        start = time.perf_counter()
        index.write(index_path)
        write_time = time.perf_counter() - start
        write_rate = num_entries / write_time

        # Measure deserialization
        loaded_index = Index()
        start = time.perf_counter()
        loaded_index.read(index_path)
        read_time = time.perf_counter() - start
        read_rate = num_entries / read_time

        print(f"[*] Binary Index DIRC v2 Write: {write_rate:,.0f} entries/s")
        print(f"[*] Binary Index DIRC v2 Read : {read_rate:,.0f} entries/s")
    finally:
        shutil.rmtree(temp_dir)


def benchmark_myers_diff():
    # 1000 lines with 50 edits
    base_lines = [f"// line {i}: standard repository function definition" for i in range(1000)]
    target_lines = list(base_lines)
    for i in range(0, 1000, 20):
        target_lines[i] = f"// line {i}: MODIFIED by feature branch"

    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        _ = myers_diff(base_lines, target_lines)
    elapsed = time.perf_counter() - start
    diff_rate = (iterations * len(base_lines)) / elapsed

    print(f"[*] Myers O((N+M)D) Diff Engine: {diff_rate:,.0f} lines/s ({elapsed*1000/iterations:.2f} ms/diff)")


def benchmark_delta_compression_and_idx():
    base_code = (
        b"class NeuralNetwork:\n"
        b"    def __init__(self, layers):\n"
        b"        self.layers = layers\n"
        b"    def forward(self, x):\n"
        b"        return x\n"
    ) * 100

    target_code = base_code.replace(b"NeuralNetwork", b"OptimizedNeuralNetwork")

    iterations = 200
    start = time.perf_counter()
    for _ in range(iterations):
        delta = create_delta(base_code, target_code)
    delta_time = time.perf_counter() - start
    delta_rate = iterations / delta_time

    # Measure reconstitution
    start = time.perf_counter()
    for _ in range(iterations):
        _ = apply_delta(base_code, delta)
    apply_time = time.perf_counter() - start
    apply_rate = iterations / apply_time

    print(f"[*] Delta Compression  : {delta_rate:,.0f} deltas/s (size: {len(target_code)} -> {len(delta)} bytes)")
    print(f"[*] Delta Reconstitution: {apply_rate:,.0f} applies/s")

    # Pack index lookup benchmark
    writer = PackIndexWriter(b"\x12" * 20)
    for i in range(2000):
        from helixgit.pack import PackObjectInfo
        writer.add_entry(PackObjectInfo(
            sha1=f"{i:040x}",
            type_code=3,
            offset=i * 100,
            size=100,
            crc32=i * 7
        ))
    idx_bytes = writer.write_index()
    reader = PackIndexReader(idx_bytes)

    queries = 50000
    start = time.perf_counter()
    for i in range(queries):
        target_sha = f"{(i % 2000):040x}"
        _ = reader.lookup(target_sha)
    lookup_time = time.perf_counter() - start
    lookup_rate = queries / lookup_time

    print(f"[*] Pack Index v2 Lookups: {lookup_rate:,.0f} queries/s (O(log N) fan-out binary search)")


def run_all_benchmarks():
    print("=" * 70)
    print("  HELIXGIT PERFORMANCE BENCHMARK SUITE (PURE PYTHON STD LIB)")
    print("=" * 70)
    benchmark_object_hashing_and_storage()
    benchmark_binary_index_codec()
    benchmark_myers_diff()
    benchmark_delta_compression_and_idx()
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
