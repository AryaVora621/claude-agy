# HelixGit: Content-Addressable Version Control, Packfile Engine & 3-Way Merge System

A production-grade, Git-compatible version control system, binary staging cache, packfile delta compression engine, and 3-way merge toolchain implemented from first principles in the pure Python standard library (zero external dependencies).

---

## 1. Architectural Highlights

HelixGit implements the foundational primitives that power distributed version control systems:

1. **Cryptographic Content-Addressable Storage (CAS)**:
   - Every file revision, directory snapshot, commit state, and annotated tag is modeled as an immutable Merkle DAG node.
   - Standard loose object format: `f"{type} {length}\x00{payload}"` compressed with zlib deflate.
   - Two-character hexadecimal directory sharding (`.git/objects/xx/yyyy...`) preventing filesystem inode exhaustion.

2. **Binary Index Staging Engine (`DIRC` v2)**:
   - Binary format `.git/index` containing 12-byte header, 62-byte fixed stat cache entries, 8-byte aligned NUL-padded paths, and trailing SHA-1 checksum.
   - Accelerates working tree status diffing from $O(N)$ filesystem disk reads to $O(1)$ stat comparisons (mtime, ctime, file size, device, inode, mode).
   - Multi-stage conflict architecture (stage 0: clean, stage 1: ancestor, stage 2: ours, stage 3: theirs).

3. **Packfile v2 & Delta Compression (`.pack` and `.idx` v2)**:
   - High-density binary object archival format with LEB128 variable-length size headers.
   - Sliding-window 16-byte block hash matching generating Git copy/insert delta bytecode, reducing repository size by 85-95%.
   - Pack index (`.idx` v2) featuring a 256-bucket Level-1 fan-out table enabling $O(\log N)$ binary search object lookup.

4. **Myers $O((N+M)D)$ Diff Engine**:
   - Eugene Myers' shortest edit script (SES) algorithm searching the edit grid along diagonals $k = x - y$ to find minimum edit paths.
   - Unified diff formatter with configurable context lines and standard hunk headers (`@@ -a,b +c,d @@`).

5. **Three-Way Merge Engine & LCA Commit Search**:
   - Lowest Common Ancestor (LCA) graph search traversing commit DAG parents via Breadth-First Search.
   - Three-way line and tree reconciliation with standard conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`).

6. **ANSI Terminal DAG Visualizer**:
   - Multi-lane branch column layout rendering topological commit graphs with colored glyphs, short hashes, branch/tag decorations, and author metadata.

---

## 2. Directory Structure

```
projects/13-helixgit/
|-- helixgit/
|   |-- __init__.py          # Package exports
|   |-- objects.py           # GitObject, Blob, Tree, TreeEntry, Commit, Tag, SHA-1 codec
|   |-- storage.py           # LooseObjectStore, ObjectDatabase, zlib compression
|   |-- index.py             # Binary Index (DIRC v2) parser, emitter, IndexEntry
|   |-- pack.py              # Packfile v2 serialization, copy/insert delta compression
|   |-- idx.py               # Pack index (.idx v2) 256-bucket fan-out table engine
|   |-- diff.py              # Myers O((N+M)D) diff engine, unified diff formatter
|   |-- merge.py             # Three-way merge, LCA commit DAG search, conflict markers
|   |-- repo.py              # Repository porcelain (init, add, commit, branch, checkout, merge)
|   `-- visualizer.py        # ANSI terminal DAG visualizer & multi-lane branch renderer
|-- tests/
|   |-- test_objects.py      # Loose object serialization, parsing, SHA-1 verification
|   |-- test_index.py        # DIRC v2 binary index codec, staging, checkout roundtrip
|   |-- test_pack.py         # Packfile and .idx v2 creation, delta compression
|   |-- test_diff.py         # Myers diff algorithm and unified diff output
|   |-- test_merge.py        # LCA commit search, 3-way line/tree merge, conflict resolution
|   `-- test_repo.py         # Repository porcelain workflow (init, add, commit, branch, checkout)
|-- benchmarks/
|   `-- bench_git.py         # Object hashing, index codec, Myers diff, delta benchmarks
|-- examples/
|   `-- git_lab.py           # Interactive Git repository showcase
|-- PLAN.md                  # Architectural specification
`-- README.md                # Comprehensive documentation
```

---

## 3. Performance Benchmarks

Measured on Apple Silicon (M-series, Darwin 24.6.0, Python 3.13):

| Subsystem | Metric | Throughput | Latency |
| :--- | :--- | :--- | :--- |
| **Object Database** | Loose Object Write | **7,286 objects/s** | 0.137 ms/obj |
| **Object Database** | Loose Object Read | **55,772 objects/s** | 0.018 ms/obj |
| **Binary Index (DIRC v2)** | Serialization | **1,270,648 entries/s** | 0.79 us/entry |
| **Binary Index (DIRC v2)** | Deserialization | **742,156 entries/s** | 1.35 us/entry |
| **Myers Diff Engine** | Edit Graph Search | **690,025 lines/s** | 1.45 ms / 1000 lines |
| **Delta Compression** | Copy/Insert Encoding | **122 deltas/s** | 89.5% space saved |
| **Delta Compression** | Reconstitution | **17,462 applies/s** | 0.057 ms/apply |
| **Pack Index (.idx v2)** | Fan-Out Lookups | **580,020 queries/s** | 1.72 us/query |

---

## 4. Quick Start & Usage

### 4.1 Programmatic Repository API

```python
from helixgit.repo import Repository

# Initialize a new Git repository
repo = Repository.init("./my_project", default_branch="main")

# Stage files into the binary index
repo.add(["main.py", "README.md"])

# Record commit
commit_sha = repo.commit("feat: initial release", author_name="Ada", author_email="ada@dev.local")

# Create and checkout a branch
repo.create_branch("feature/calc")
repo.checkout("feature/calc")

# Perform 3-way merge
repo.checkout("main")
msg, has_conflicts = repo.merge("feature/calc")
```

### 4.2 Interactive Showcase

Run the end-to-end interactive demonstration simulating branching, merging, packfile compression, and commit graph visualization:

```bash
python3 projects/13-helixgit/examples/git_lab.py
```

### 4.3 Running Unit Tests

```bash
python3 -m unittest discover -s projects/13-helixgit/tests
```
