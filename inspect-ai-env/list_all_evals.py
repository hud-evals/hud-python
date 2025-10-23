#!/usr/bin/env python3
"""
One-off script to download inspect_evals and list all available evals.

This clones the inspect_evals repository and lists all eval folders
found in src/inspect_evals/.
"""

import shutil
import subprocess
import sys
from pathlib import Path


def main():
    repo_url = "https://github.com/UKGovernmentBEIS/inspect_evals.git"
    repo_dir = Path("inspect_evals_full")
    cleanup_needed = False

    try:
        # Clone or update the repository
        if repo_dir.exists():
            print(f"📂 Repository already exists at {repo_dir}")
            print("   Updating...")
            try:
                subprocess.run(
                    ["git", "-C", str(repo_dir), "pull"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print("   ✅ Updated successfully")
            except subprocess.CalledProcessError as e:
                print(f"   ⚠️  Update failed: {e.stderr}")
                print("   Continuing with existing repo...")
        else:
            print(f"📥 Cloning inspect_evals from {repo_url}...")
            cleanup_needed = True
            try:
                subprocess.run(
                    ["git", "clone", repo_url, str(repo_dir)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print("   ✅ Cloned successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Clone failed: {e.stderr}")
                sys.exit(1)

        # List all evals in src/inspect_evals/
        evals_dir = repo_dir / "src" / "inspect_evals"

        if not evals_dir.exists():
            print(f"❌ Expected directory not found: {evals_dir}")
            sys.exit(1)

        # Find all directories (excluding __pycache__ and hidden dirs)
        eval_dirs = [
            d.name for d in evals_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith('_')
            and not d.name.startswith('.')
        ]

        eval_dirs.sort()

        print(f"\n📋 Found {len(eval_dirs)} evals in inspect_evals:\n")
        print("=" * 60)

        for i, eval_name in enumerate(eval_dirs, 1):
            # Check if there's a README or description
            eval_path = evals_dir / eval_name
            readme = eval_path / "README.md"

            description = ""
            if readme.exists():
                # Try to extract first line of description
                try:
                    with open(readme) as f:
                        lines = f.readlines()
                        # Skip title line, get first paragraph
                        for line in lines[1:]:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                description = line[:70]
                                if len(line) > 70:
                                    description += "..."
                                break
                except Exception:
                    pass

            print(f"{i:3}. {eval_name:<30} {description}")

        print("=" * 60)
        print(f"\n💡 Usage:")
        print(f"   uv run python prepare_dataset.py --eval <eval_name> --limit 1")
        print(f"\nExample:")
        print(f"   uv run python prepare_dataset.py --eval mbpp --limit 1")
        print(f"   uv run python prepare_dataset.py --eval swe_bench --limit 1")

        # Create a simple text file with the list
        output_file = "available_evals.txt"
        with open(output_file, "w") as f:
            f.write("Available inspect_evals:\n")
            f.write("=" * 60 + "\n")
            for eval_name in eval_dirs:
                f.write(f"{eval_name}\n")

        print(f"\n📝 List saved to: {output_file}")

    finally:
        # Clean up the cloned repository if we created it
        if cleanup_needed and repo_dir.exists():
            print(f"\n🧹 Cleaning up: removing {repo_dir}...")
            try:
                shutil.rmtree(repo_dir)
                print("   ✅ Cleanup complete")
            except Exception as e:
                print(f"   ⚠️  Cleanup failed: {e}")


if __name__ == "__main__":
    main()
