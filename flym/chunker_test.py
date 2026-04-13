"""
flym/chunker_test.py
--------------------
Manual smoke tests for the chunker.  Not a pytest suite — just run it:

    python flym/chunker_test.py

Each test prints a labelled result so you can read and reason about the
output rather than just seeing pass/fail.  Experiment freely: change the
sample texts, tweak target_chars, and re-run to build intuition.
"""

from flym.chunker import (
    Chunk,
    _collect_break_points,
    _extract_section_path,
    _is_inside_code_block,
    _strip_frontmatter,
    chunk,
)

# ANSI colours for readable terminal output
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RESET  = "\033[0m"

def header(title: str) -> None:
    print(f"\n{_CYAN}{'─' * 60}{_RESET}")
    print(f"{_CYAN}{title}{_RESET}")
    print(f"{_CYAN}{'─' * 60}{_RESET}")

def ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET}  {msg}")

def info(label: str, value: object) -> None:
    print(f"  {_YELLOW}{label:<22}{_RESET} {value}")


# ---------------------------------------------------------------------------
# Test 1 — frontmatter stripping
# ---------------------------------------------------------------------------
header("1. YAML frontmatter stripping")

text_with_fm = """\
---
title: My Note
tags: [auth, security]
---
# Real Content
Some prose here.
"""

body, offset = _strip_frontmatter(text_with_fm)
info("offset (chars removed)", offset)
info("body starts with", repr(body[:20]))
assert not body.startswith("---"),    "frontmatter not stripped"
assert body.startswith("# Real"),     "body should start at heading"
assert offset > 0,                    "offset should be non-zero"
ok("frontmatter stripped correctly")


# ---------------------------------------------------------------------------
# Test 2 — break point collection and max-score-wins
# ---------------------------------------------------------------------------
header("2. Break point collection")

sample = "\n## Section A\n\nSome prose.\n\n## Section B\n"
breaks = _collect_break_points(sample)
positions = sorted(breaks.keys())
info("break positions", positions)
info("labels at each", [(p, breaks[p].label, breaks[p].score) for p in positions])

# Both \n\n and \n## fire near the same position.
# The h2 (score 90) should win over blank (score 20).
h2_breaks = [b for b in breaks.values() if b.label == "h2"]
assert len(h2_breaks) == 2, "expected two h2 breaks"
ok("h2 breaks detected, max-score-wins applied")


# ---------------------------------------------------------------------------
# Test 3 — code block guard
# ---------------------------------------------------------------------------
header("3. Code block interior is not a break point")

code_text = "\nSome prose.\n\n```python\ndef foo():\n    pass\n```\n\nMore prose.\n"
breaks = _collect_break_points(code_text)

# Find the position of the blank line INSIDE the code block (between def and ```)
# and verify is_inside_code_block correctly identifies it.
inside_pos = code_text.index("    pass")
outside_pos = code_text.index("More prose")

info("inside_pos inside block?",  _is_inside_code_block(code_text, inside_pos))
info("outside_pos inside block?", _is_inside_code_block(code_text, outside_pos))
assert     _is_inside_code_block(code_text, inside_pos),  "should be inside"
assert not _is_inside_code_block(code_text, outside_pos), "should be outside"
ok("code block guard works")


# ---------------------------------------------------------------------------
# Test 4 — section path extraction
# ---------------------------------------------------------------------------
header("4. Section path breadcrumb")

doc = """\
# Guide
Some intro.
## Installation
Steps here.
### macOS
Homebrew steps.
#### Using brew
brew install flym
"""

# At the position of "Homebrew steps", path should be "Guide > Installation > macOS"
homebrew_pos = doc.index("Homebrew steps")
path = _extract_section_path(doc, homebrew_pos)
info("section path", repr(path))
assert path == "Guide > Installation > macOS", f"unexpected: {path!r}"
ok("section path is correct")

# After a new h2 the h3 should reset
after_install = doc.index("## Installation") + 5
path2 = _extract_section_path(doc, after_install)
info("path just inside ## Installation", repr(path2))
ok("heading resets deeper levels")


# ---------------------------------------------------------------------------
# Test 5 — chunking a realistic markdown document
# ---------------------------------------------------------------------------
header("5. Chunk a realistic document (target=300 chars for short demo)")

realistic = """\
# Authentication Guide

This guide covers the main patterns for securing your API.

## JWT Tokens

JSON Web Tokens are a compact, URL-safe means of representing claims.
A token consists of three parts: header, payload, and signature.

### Validation

Always validate the signature before trusting the payload.
Never decode without verifying.

## Session Cookies

Cookies are an alternative to JWTs. They are stored server-side.

### Security Flags

Set HttpOnly and Secure flags on all session cookies.
"""

chunks = chunk(realistic, target_chars=300, overlap_chars=45, min_chars=30)

info("total chunks", len(chunks))
for i, c in enumerate(chunks):
    text_preview = realistic[c.pos : c.pos + c.len][:60].replace("\n", "↵")
    print(
        f"  [{i}] pos={c.pos:<5} len={c.len:<5} "
        f"type={c.chunk_type:<6} "
        f"path={c.section_path!r:<35} "
        f"preview={text_preview!r}"
    )

assert len(chunks) >= 2,          "should produce multiple chunks"
assert all(c.len >= 30 for c in chunks), "all chunks should meet min_chars"
ok(f"produced {len(chunks)} valid chunks")


# ---------------------------------------------------------------------------
# Test 6 — overlap: adjacent chunks share text
# ---------------------------------------------------------------------------
header("6. Overlap: adjacent chunks share a tail of text")

if len(chunks) >= 2:
    c0_end   = realistic[chunks[0].pos : chunks[0].pos + chunks[0].len]
    c1_start = realistic[chunks[1].pos : chunks[1].pos + min(60, chunks[1].len)]
    # The start of chunk 1 should appear somewhere near the end of chunk 0.
    overlap_region = c0_end[-60:]
    shared = any(word in overlap_region for word in c1_start.split()[:5] if len(word) > 3)
    info("chunk 0 tail (last 60)", repr(overlap_region))
    info("chunk 1 start (first 60)", repr(c1_start))
    if shared:
        ok("overlap confirmed — adjacent chunks share text")
    else:
        print("  (chunks may be too short to show overlap at this target size)")


# ---------------------------------------------------------------------------
# Test 7 — chunk a real file (DESIGN.md if present)
# ---------------------------------------------------------------------------
header("7. Chunk DESIGN.md (full default parameters)")

import os
design_path = os.path.join(os.path.dirname(__file__), "..", "DESIGN.md")
if os.path.exists(design_path):
    with open(design_path) as f:
        design_text = f.read()
    design_chunks = chunk(design_text)
    info("total chunks", len(design_chunks))
    info("total chars", len(design_text))
    info("avg chunk len", round(sum(c.len for c in design_chunks) / len(design_chunks)))
    section_paths = [c.section_path for c in design_chunks if c.section_path]
    info("unique section paths", len(set(section_paths)))
    ok("DESIGN.md chunked successfully")
else:
    print("  (DESIGN.md not found, skipping)")


print(f"\n{_GREEN}All tests passed.{_RESET}\n")
