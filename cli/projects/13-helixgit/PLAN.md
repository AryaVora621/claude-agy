# HelixGit: Content-Addressable Version Control, Packfile Engine & 3-Way Merge System
## Technical Specification & Architecture Plan

---

## 1. Executive Overview

HelixGit is a production-grade, zero-dependency Git-compatible version control system, content-addressable storage engine, and repository toolchain implemented entirely from first principles in the pure Python standard library.

Git is the foundation of modern software engineering, yet its inner mechanics are widely considered opaque:
1. **Cryptographic Content-Addressable Storage (CAS)**: Every filesystem revision is modeled as an immutable Merkle DAG of SHA-1/SHA-256 hashed loose objects (`blob`, `tree`, `commit`, `tag`) compressed via zlib deflate.
2. **Binary Index Staging Area (`DIRC` v2)**: The intermediate staging cache (`.git/index`) stores stat cache metadata (ctime, mtime, dev, inode, mode, size) and multi-stage conflict slots (0: normal, 1: ancestor, 2: ours, 3: theirs) to accelerate working tree diffing from $O(N)$ filesystem disk reads to $O(1)$ stat comparisons.
3. **Packfile Serialization & Delta Compression (`.pack` and `.idx` v2)**: High-density object archival format utilizing 256-bucket fan-out tables, CRC32 chunk verification, and sliding-window copy/insert delta compression to reduce repository disk footprints by 85-95%.
4. **Myers $O((N+M)D)$ Diff Algorithm**: Computes the optimal Shortest Edit Script (SES) between arbitrary sequences by finding the maximum-reach path along $k$-diagonals in the edit grid.
5. **Three-Way Tree Merge & Conflict Resolution**: Automatically finds the Lowest Common Ancestor (LCA) in arbitrary branching commit DAGs, performs 3-way file and directory reconciliation, and emits standard conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) when modifications overlap.
6. **ANSI Terminal DAG Visualizer**: Interactive commit graph renderer with multi-lane branch rendering, topological ordering, and commit inspection.

---

## 2. Technical Specifications

### 2.1 Object Format & Loose Storage
Loose Git objects are stored at `.git/objects/{hash[:2]}/{hash[2:]}` as zlib-compressed streams:

```
+--------+---+----------------+------+-------------------------+
|  type  | ' '| size_in_bytes  | \x00 |    raw object content   |
| (ASCII)|   | (ASCII decimal)|      |         (binary)        |
+--------+---+----------------+------+-------------------------+
```

1. **Blob**: Raw binary payload.
2. **Tree**: Binary list of directory entries, sorted lexicographically:
   `f"{mode:o} {name}\x00{20-byte-sha1}"`
   - `100644`: standard file
   - `100755`: executable file
   - `040000`: subdirectory tree
   - `120000`: symbolic link
3. **Commit**: Key-value header block and message:
   ```
   tree <40-hex-tree-sha>
   parent <40-hex-parent-sha> [optional, multiple for merges]
   author Name <email> <unix_timestamp> <tz_offset>
   committer Name <email> <unix_timestamp> <tz_offset>

   <commit message>
   ```
4. **Tag**: Annotated tag pointing to an object with tagger and signature.

### 2.2 Binary Index Staging Format (`DIRC` v2)
Binary index `.git/index` layout:
- **Header (12 bytes)**:
  - 4 bytes: Magic string `DIRC`
  - 4 bytes: Version (0x00000002)
  - 4 bytes: Total number of index entries
- **Entries (62 bytes fixed + path + padding)**:
  - `ctime_seconds` (4 bytes, unsigned int, big-endian)
  - `ctime_nanoseconds` (4 bytes, unsigned int, big-endian)
  - `mtime_seconds` (4 bytes, unsigned int, big-endian)
  - `mtime_nanoseconds` (4 bytes, unsigned int, big-endian)
  - `dev` (4 bytes), `ino` (4 bytes), `mode` (4 bytes)
  - `uid` (4 bytes), `gid` (4 bytes), `file_size` (4 bytes)
  - `sha1` (20 bytes binary)
  - `flags` (2 bytes):
    - Bit 15: assume-valid flag
    - Bit 14: extended flag (must be 0 in v2)
    - Bits 13-12: stage (0: normal, 1: base, 2: ours, 3: theirs)
    - Bits 11-0: path string length (clamped to 0xFFF)
  - `path`: UTF-8 path string, NUL-terminated, padded with 1-8 NUL bytes so total entry size is a multiple of 8 bytes.
- **Checksum (20 bytes)**:
  - SHA-1 hash of all preceding bytes in the index file.

### 2.3 Packfile & Index Format v2
- **`.pack` file**:
  - 4 bytes: Magic `PACK`
  - 4 bytes: Version (0x00000002)
  - 4 bytes: Object count
  - Object records:
    - Byte 1: MSB (continuation), 3-bit type, 4-bit size chunk.
    - Subsequent bytes: 7-bit size chunks (LEB128).
    - If `OBJ_OFS_DELTA` (type 6): negative offset to base.
    - If `OBJ_REF_DELTA` (type 7): 20-byte base object SHA-1.
    - Zlib compressed payload (or delta copy/insert bytecode).
  - 20-byte trailing SHA-1 of all preceding bytes.
- **`.idx` file v2**:
  - 4 bytes magic: `\xfftOc`
  - 4 bytes version: `0x00000002`
  - 256-entry Level-1 Fan-out Table (each 4 bytes, count of objects with first byte <= $i$)
  - Table of 20-byte SHA-1s in lexicographical order
  - Table of 4-byte CRC32 checksums
  - Table of 4-byte packfile offsets
  - 20-byte packfile SHA-1 checksum + 20-byte index SHA-1 checksum.

### 2.4 Myers Diff Algorithm
Constructs an edit graph from sequence $A$ of length $N$ to sequence $B$ of length $M$.
Horizontal edges represent deletions (cost 1).
Vertical edges represent insertions (cost 1).
Diagonal edges represent matching elements (cost 0).
The algorithm iteratively expands the furthest-reaching paths along diagonals $k = x - y$ for edit distance $D = 0, 1, 2, \dots$ until reaching $(N, M)$.

### 2.5 Three-Way Merge & Lowest Common Ancestor
1. Traverses the commit DAG backward from both branch heads using BFS to identify the set of common ancestors.
2. Finds the topological Lowest Common Ancestor (LCA) with maximal depth.
3. Performs 3-way tree reconciliation between $O$ (base), $A$ (ours), and $B$ (theirs).
4. For files modified in both $A$ and $B$, runs 3-way chunk reconciliation. If disjoint, merges automatically; if overlapping, writes standard conflict markers.

---

## 3. Directory Layout & Module Structure

```
projects/13-helixgit/
|-- helixgit/
|   |-- __init__.py          # Package exports
|   |-- objects.py           # GitObject, Blob, Tree, TreeEntry, Commit, Tag
|   |-- storage.py           # LooseObjectStore, ObjectDatabase, SHA1 hashing
|   |-- index.py             # Binary Index (DIRC v2) parser, emitter, IndexEntry
|   |-- pack.py              # Packfile generator, parser, and Delta compression
|   |-- idx.py               # Pack index (.idx v2) fan-out table generator/parser
|   |-- diff.py              # Myers O((N+M)D) diff engine, unified diff formatter
|   |-- merge.py             # 3-Way merge, LCA graph search, conflict markers
|   |-- repo.py              # Repository manager (init, add, commit, branch, checkout)
|   `-- visualizer.py        # ANSI terminal DAG graph visualizer & branch lane renderer
|-- tests/
|   |-- test_objects.py      # Loose object serialization, parsing, hashing
|   |-- test_index.py        # Binary index format v2 parsing, staging, roundtrip
|   |-- test_pack.py         # Packfile and .idx v2 creation, delta compression
|   |-- test_diff.py         # Myers diff algorithm and unified diff output
|   |-- test_merge.py        # LCA commit search, 3-way merge, conflict resolution
|   `-- test_repo.py         # End-to-end repository workflow (init, commit, branch)
|-- benchmarks/
|   `-- bench_git.py         # Object hashing, pack generation, diff throughput
|-- examples/
|   `-- git_lab.py           # Interactive Git repository showcase
|-- PLAN.md                  # Architectural specification
`-- README.md                # Comprehensive documentation
```

---

## 4. Implementation Phasing

1. **Phase 1 (Task #64)**: Object store, loose object serialization (`objects.py`, `storage.py`), SHA-1 hashing, and unit tests (`test_objects.py`).
2. **Phase 2 (Task #65)**: Binary index staging area (`index.py`), DIRC v2 format, stat cache, working tree checkout, and unit tests (`test_index.py`).
3. **Phase 3 (Task #66)**: Packfile and index v2 (`pack.py`, `idx.py`), copy/insert delta compression, and unit tests (`test_pack.py`).
4. **Phase 4 (Task #67)**: Myers diff algorithm (`diff.py`), commit DAG LCA traversal, 3-way merge engine (`merge.py`), and unit tests (`test_diff.py`, `test_merge.py`).
5. **Phase 5 (Task #68)**: Repository porcelain (`repo.py`), branch/rebase engine, ANSI DAG visualizer (`visualizer.py`), and interactive lab (`examples/git_lab.py`).
6. **Phase 6 (Task #69)**: Comprehensive test suite and benchmarks (`benchmarks/bench_git.py`).
7. **Phase 7 (Task #70)**: Integration into `showcase.py`, `projects.md`, `tracker/data.json`, `CHECKPOINT_LAST.md`, and `README.md`.
