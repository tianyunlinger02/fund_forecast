"""Download signed-repository Ubuntu libraries into project temp; never install system packages.

Run with python -I. Each archive and extraction has its own initially empty directory.
Only dpkg-deb extracts data; maintainer scripts are never run.
"""

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / ".claude-tmp" / "sessions" / "fund-build"
TEMP.mkdir(parents=True, exist_ok=True)
run = Path(tempfile.mkdtemp(prefix="browser-libraries-", dir=TEMP))
packages = ["libatk1.0-0", "libatk-bridge2.0-0", "libxkbcommon0", "libasound2",
            "libgbm1", "libxdamage1", "libatspi2.0-0", "libdrm2", "libwayland-server0"]
paths = []
for package in packages:
    download = run / package / "download"
    extracted = run / package / "extracted"
    download.mkdir(parents=True)
    extracted.mkdir()
    subprocess.run(["apt-get", "-o", "Acquire::Retries=0", "-o", "Acquire::http::Timeout=20",
                    "download", package], cwd=download, check=True)
    archives = list(download.glob("*.deb"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one archive for {package}")
    subprocess.run(["dpkg-deb", "--extract", str(archives[0]), str(extracted)], cwd=ROOT, check=True)
    for candidate in (extracted / "usr/lib/x86_64-linux-gnu", extracted / "lib/x86_64-linux-gnu"):
        if candidate.is_dir():
            paths.append(str(candidate))
value = os.pathsep.join(paths)
(run / "library-path.txt").write_text(value)
print(f"BROWSER_LIBRARY_PATH_FILE={run / 'library-path.txt'}")
