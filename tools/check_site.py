#!/usr/bin/env python3
"""Static-site sanity check for the PointGT project page.

Verifies:
1. Every local asset referenced by index.html exists on disk
   (except an allowlist of intentionally-reserved paths).
2. No leftover template placeholder tokens remain in index.html.
3. No file is dangerously large for GitHub (100 MB hard limit;
   web videos must stay under 20 MB).

Exit code 0 = all checks pass, 1 = failure.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)            # tools/ -> repo root
INDEX = os.path.join(ROOT, "index.html")

# Referenced in HTML but intentionally absent until the user uploads it.
RESERVED_MISSING = {"static/pdfs/paper.pdf"}

PLACEHOLDER_TOKENS = [
    "PAPER_TITLE", "AUTHOR_NAMES", "BRIEF_DESCRIPTION", "YOUR_DOMAIN",
    "KEYWORD1", "FIRST_AUTHOR", "SECOND_AUTHOR", "CONFERENCE_NAME",
    "INSTITUTION_OR_LAB_NAME", "INSTITUTION_NAME", "RESEARCH_AREA",
    "YOUR_TWITTER_HANDLE", "YOUR_GITHUB_USERNAME", "ARXIV PAPER ID",
    "YOUR REPO HERE", "Lorem ipsum", "Aliquam vitae", "YourPaperKey",
    "FULL_ABSTRACT_TEXT_HERE", "BIBTEX_CITATION_HERE", "PAPER_ID_",
    "Academic Project Page</h1>", "banner_video", "carousel1",
]


def main():
    with open(INDEX, encoding="utf-8") as f:
        html = f.read()

    failures = []

    # 1. local asset references resolve
    for ref in re.findall(r'(?:src|href)="([^"]+)"', html):
        if ref.startswith(("http://", "https://", "#", "mailto:", "data:")):
            continue
        if ref in RESERVED_MISSING:
            continue
        if not os.path.exists(os.path.join(ROOT, ref)):
            failures.append(f"Missing asset: {ref}")

    # 2. no leftover template placeholders
    for tok in PLACEHOLDER_TOKENS:
        if tok in html:
            failures.append(f"Leftover placeholder token: {tok!r}")

    # 3. file sizes
    for dirpath, _dirnames, filenames in os.walk(ROOT):
        if ".git" in dirpath.split(os.sep):
            continue
        for name in filenames:
            fp = os.path.join(dirpath, name)
            size = os.path.getsize(fp)
            rel = os.path.relpath(fp, ROOT)
            if size > 100 * 1024 * 1024:
                failures.append(f"File over 100MB GitHub limit: {rel} ({size // 1048576}MB)")
            elif rel.startswith("static/videos" + os.sep) and size > 20 * 1024 * 1024:
                failures.append(f"Web video over 20MB: {rel} ({size // 1048576}MB)")

    if failures:
        print("FAIL")
        for fl in failures:
            print("  -", fl)
        return 1
    print("PASS - all asset references resolve, no placeholders, sizes OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
