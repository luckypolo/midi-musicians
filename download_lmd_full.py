from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path


LMD_FULL_URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"


def download_with_progress(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def report(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_count * block_size
        percent = min(100.0, downloaded * 100.0 / total_size)
        mib_done = downloaded / (1024 * 1024)
        mib_total = total_size / (1024 * 1024)
        print(f"\rDownloading {mib_done:.1f}/{mib_total:.1f} MiB ({percent:.1f}%)", end="")

    urllib.request.urlretrieve(url, output_path, reporthook=report)
    print()


def safe_extract_tar_gz(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (output_dir / member.name).resolve()
            if output_root != target and output_root not in target.parents:
                raise RuntimeError(f"Refusing to extract unsafe archive member: {member.name}")
        archive.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and optionally extract LMD-full for MidiCaps paths such as lmd_full/1/...."
    )
    parser.add_argument("--output-dir", default="data", help="Directory where lmd_full should be available.")
    parser.add_argument("--archive", default="data/lmd_full.tar.gz", help="Download archive path.")
    parser.add_argument("--url", default=LMD_FULL_URL)
    parser.add_argument("--download-only", action="store_true", help="Download the archive but do not extract it.")
    parser.add_argument("--skip-download", action="store_true", help="Use an existing archive and only extract it.")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the tar.gz after extraction.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm downloading/extracting the large LMD-full dataset without an interactive prompt.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    archive_path = Path(args.archive)
    lmd_dir = output_dir / "lmd_full"

    if lmd_dir.exists():
        print(f"LMD-full already exists at {lmd_dir.resolve()}")
        return

    if not args.yes:
        print(
            "This downloads LMD-full, a large archive of about 1.6GB compressed "
            "and several GB extracted."
        )
        response = input("Continue? [y/N] ").strip().lower()
        if response not in {"y", "yes"}:
            print("Cancelled.")
            sys.exit(1)

    if not args.skip_download:
        print(f"Downloading {args.url}")
        download_with_progress(args.url, archive_path)
    elif not archive_path.exists():
        raise FileNotFoundError(f"--skip-download was set but archive does not exist: {archive_path}")

    if args.download_only:
        print(f"Archive saved to {archive_path.resolve()}")
        return

    print(f"Extracting {archive_path} into {output_dir}")
    safe_extract_tar_gz(archive_path, output_dir)
    if not lmd_dir.exists():
        print(
            "Extraction completed, but data/lmd_full was not found. "
            "Check the archive contents and pass the extracted folder to --midi-root."
        )
    else:
        print(f"LMD-full ready at {lmd_dir.resolve()}")

    if not args.keep_archive:
        archive_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
