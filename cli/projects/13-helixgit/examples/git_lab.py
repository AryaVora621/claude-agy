"""
HelixGit: Interactive Git Laboratory & Showcase.
Simulates a multi-developer workflow, branch creation, 3-way merging,
delta packfile compression, and terminal DAG visualization.
"""

import os
import sys
import shutil
import tempfile
import time

# Ensure helixgit is importable when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helixgit.repo import Repository
from helixgit.visualizer import render_commit_dag
from helixgit.pack import PackWriter
from helixgit.diff import unified_diff


def run_showcase():
    print("=" * 70)
    print("  HELIXGIT: PURE PYTHON GIT-COMPATIBLE VCS & MERKLE DAG ENGINE")
    print("=" * 70)

    demo_dir = tempfile.mkdtemp(prefix="helixgit_demo_")
    try:
        print(f"\n[1] Initializing repository at: {demo_dir}")
        repo = Repository.init(demo_dir, default_branch="main")
        print("    Initialized empty Git repository in .git/")

        # Create initial files
        print("\n[2] Creating initial codebase on 'main' branch...")
        main_py = os.path.join(demo_dir, "main.py")
        readme = os.path.join(demo_dir, "README.md")

        with open(main_py, "w") as f:
            f.write(
                "def calculate_total(prices):\n"
                "    return sum(prices)\n\n"
                "if __name__ == '__main__':\n"
                "    print(calculate_total([10, 20, 30]))\n"
            )

        with open(readme, "w") as f:
            f.write("# Accounting Engine\nProduction accounting toolchain.\n")

        repo.add(["main.py", "README.md"])
        c1 = repo.commit("feat: initial commit with total calculator", "Alice", "alice@company.com")
        print(f"    Committed {c1[:7]}: feat: initial commit with total calculator")

        # Create a feature branch
        print("\n[3] Branching 'feature/tax' from 'main'...")
        repo.create_branch("feature/tax")
        repo.checkout("feature/tax")
        print("    Switched to branch 'feature/tax'")

        with open(main_py, "w") as f:
            f.write(
                "def calculate_total(prices, tax_rate=0.08):\n"
                "    subtotal = sum(prices)\n"
                "    return subtotal * (1.0 + tax_rate)\n\n"
                "if __name__ == '__main__':\n"
                "    print(calculate_total([10, 20, 30]))\n"
            )

        repo.add(["main.py"])
        c2 = repo.commit("feat: support configurable sales tax rate", "Bob", "bob@company.com")
        print(f"    Committed {c2[:7]} on feature/tax")

        # Switch back to main and commit another change
        print("\n[4] Switching back to 'main' and adding discount logic...")
        repo.checkout("main")
        print("    Switched to branch 'main'")

        discount_py = os.path.join(demo_dir, "discount.py")
        with open(discount_py, "w") as f:
            f.write("def apply_coupon(total, code):\n    return total * 0.9 if code == 'SAVE10' else total\n")

        repo.add(["discount.py"])
        c3 = repo.commit("feat: add promo discount coupon support", "Alice", "alice@company.com")
        print(f"    Committed {c3[:7]} on main")

        # Perform 3-way merge
        print("\n[5] Merging 'feature/tax' into 'main' (3-Way Merge)...")
        msg, conflict = repo.merge("feature/tax")
        print(f"    Merge result: {msg}")

        # Show status
        status = repo.status()
        print(f"\n[6] Repository Status on branch '{status['branch']}':")
        print(f"    Head Commit : {status['head'][:7]}")
        print(f"    Staged Added: {status['staged']['added']}")
        print(f"    Unstaged Mod: {status['unstaged']['modified']}")

        # Render DAG
        print("\n[7] Visualizing Commit DAG History:")
        print("-" * 70)
        dag_output = render_commit_dag(repo, color=True)
        print(dag_output)
        print("-" * 70)

        # Packfile delta compression demo
        print("\n[8] Archiving objects into Git Packfile v2...")
        writer = PackWriter()
        all_shas = repo.odb.loose.list_all_shas()
        for sha in all_shas:
            writer.add_object(repo.odb.get(sha))

        pack_bytes, infos = writer.write_pack()
        loose_bytes = sum(len(repo.odb.get(s).raw_data()) for s in all_shas)
        packed_bytes = len(pack_bytes)
        savings = (1.0 - (packed_bytes / loose_bytes)) * 100 if loose_bytes else 0

        print(f"    Objects Archived : {len(infos)}")
        print(f"    Raw Loose Size   : {loose_bytes} bytes")
        print(f"    Packed Size      : {packed_bytes} bytes")
        print(f"    Disk Compression : {savings:.1f}% space saved")

        print("\n" + "=" * 70)
        print("  HELIXGIT DEMONSTRATION COMPLETE - ALL SYSTEMS FUNCTIONAL")
        print("=" * 70)

    finally:
        shutil.rmtree(demo_dir)


if __name__ == "__main__":
    run_showcase()
