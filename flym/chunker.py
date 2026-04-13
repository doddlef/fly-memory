"""
flym.chunker
------------
Splits a markdown document into overlapping chunks at semantic breakpoints.

The core idea
~~~~~~~~~~~~~
Instead of cutting text at a fixed character count, we scan for natural
break points (headings, paragraph boundaries, code fences) and pick the
best one near the target size boundary.  This keeps chunks semantically
coherent — a chunk rarely starts mid-sentence or mid-paragraph.

Scoring formula
~~~~~~~~~~~~~~~
Each break candidate is scored by combining its *type quality* with its
*proximity* to the ideal cut position:

    final_score = type_score × (1 - normalised_distance² × 0.7)

    type_score         : quality of the break type (h1=100 … newline=1)
    normalised_distance: how far the break is from the target boundary,
                         expressed as a fraction of the chunk start→target
                         span.  0.0 = exactly at target, 1.0 = at start.

The quadratic distance penalty strongly prefers nearby breaks, but a
high-quality heading far back can still beat a low-quality break nearby.
Example: h1 (100) at distance 0.9 → 100 × (1 - 0.81×0.7) = 43.3
         blank (20) at distance 0.1 → 20 × (1 - 0.01×0.7) = 19.9
The heading wins even though it's much further away.

Overlap
~~~~~~~
After each chunk boundary is chosen, the next chunk starts at:
    next_start = chunk_end - overlap_chars
    (snapped forward to the next word boundary, clamped to prevent
     backtracking into the previous chunk)

This means adjacent chunks share a small tail of text, so a sentence
split across a boundary appears in both chunks and is never missed.

Key constraints
~~~~~~~~~~~~~~~
- Never cut inside a fenced code block  (tracked via fence parity)
- Never cut inside a markdown table     (tracked via table state)
- Strip YAML frontmatter before chunking (closing --- would score as hr)
- Discard chunks shorter than min_chars
"""

import re
from dataclasses import dataclass, field
from typing import Iterator

# ---------------------------------------------------------------------------
# Break patterns
# ---------------------------------------------------------------------------
# Each entry: (compiled regex, integer score, label)
# Scores are intentionally spread wide so headings decisively beat paragraph
# or sentence breaks.  max-score-wins when two patterns hit the same position.

BREAK_PATTERNS: list[tuple[re.Pattern[str], int, str]] = [
    (re.compile(r"\n#{1}(?!#)"),              100, "h1"),
    (re.compile(r"\n#{2}(?!#)"),               90, "h2"),
    (re.compile(r"\n#{3}(?!#)"),               80, "h3"),
    (re.compile(r"\n#{4}(?!#)"),               70, "h4"),
    (re.compile(r"\n(?:---|\*\*\*|___)\s*\n"), 65, "hr"),
    (re.compile(r"\n```"),                     65, "codeblock"),
    (re.compile(r"\n#{5}(?!#)"),               60, "h5"),
    (re.compile(r"\n#{6}(?!#)"),               50, "h6"),
    (re.compile(r"[.!?]\s+(?=[A-Z])"),         30, "sentence"),
    (re.compile(r"\n\n+"),                     20, "blank"),
    (re.compile(r"\n[-*]\s"),                   5, "list"),
    (re.compile(r"\n\d+\.\s"),                  5, "numlist"),
    (re.compile(r"\n"),                         1, "newline"),
]

# Regex to detect and strip YAML frontmatter at the start of a document.
_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BreakPoint:
    """A candidate position where the chunker may cut the text."""
    pos: int            # character offset in the (stripped) text
    score: int          # type quality score (from BREAK_PATTERNS)
    label: str          # human-readable type name, e.g. "h2"


@dataclass
class Chunk:
    """
    A single chunk produced by the chunker.

    pos and len refer to character offsets in the *original* text passed
    to chunk() — i.e. after frontmatter stripping, before any other changes.
    Chunk text is recovered with: text[chunk.pos : chunk.pos + chunk.len]
    """
    pos: int                        # start offset in the stripped body
    len: int                        # character length
    section_path: str               # e.g. "Installation > MacOS"
    chunk_type: str = "prose"       # "prose" | "code" | "mixed"
    language: str | None = None     # set for code chunks with a known language


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_frontmatter(text: str) -> tuple[str, int]:
    """
    Remove YAML frontmatter and return (body, offset).

    offset is the number of characters removed from the start.  All chunk
    positions are relative to the *body*, so callers that need positions in
    the original full text must add offset back.

    Why strip before chunking?
        The closing '---' of frontmatter matches the hr pattern (score 65).
        Without stripping, the chunker would treat it as a break point and
        might place it in section_path or cut there unnecessarily.
    """
    m = _FRONTMATTER_RE.match(text)
    if m:
        return text[m.end():], m.end()
    return text, 0


def _collect_break_points(text: str) -> dict[int, BreakPoint]:
    """
    Scan *text* with every pattern and return a position → BreakPoint map.

    When two patterns match at the same position, the higher score wins
    (max-score-wins).  This is the merge step that reconciles overlapping
    pattern matches without double-counting.
    """
    breaks: dict[int, BreakPoint] = {}
    for pattern, score, label in BREAK_PATTERNS:
        for m in pattern.finditer(text):
            pos = m.start()
            if pos not in breaks or score > breaks[pos].score:
                breaks[pos] = BreakPoint(pos=pos, score=score, label=label)
    return breaks


def _is_inside_code_block(text: str, pos: int) -> bool:
    """
    Return True if character position *pos* is inside a fenced code block.

    We count the number of ``` fences before *pos*.  An odd count means we
    are inside an open fence; an even count means we are outside.

    Limitation: this is a heuristic — it doesn't handle nested fences or
    fences inside block quotes.  Sufficient for standard markdown documents.
    """
    fence_count = len(re.findall(r"\n```", text[:pos]))
    return fence_count % 2 == 1


def _is_inside_table(text: str, pos: int) -> bool:
    """
    Return True if *pos* is on a line that looks like a markdown table row.

    We check the current line: if it starts with '|' it is part of a table.
    We do NOT split inside tables — only at row boundaries (blank lines
    after the table).
    """
    line_start = text.rfind("\n", 0, pos) + 1
    line = text[line_start:text.find("\n", pos)]
    return line.strip().startswith("|")


def _extract_section_path(text: str, up_to: int) -> str:
    """
    Build the heading breadcrumb at position *up_to* in *text*.

    Scans backwards through all ATX headings seen before *up_to* and
    keeps only the deepest heading at each level, building a trail like:
        "Installation > MacOS > Using Homebrew"

    Returns an empty string if no headings have been seen yet.
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    # heading levels currently "active" (index 0 = h1 … index 5 = h6)
    active: list[str | None] = [None] * 6

    for m in heading_re.finditer(text, 0, up_to):
        level = len(m.group(1)) - 1          # 0-based index
        active[level] = m.group(2).strip()
        # Clear all deeper levels — a new h2 resets h3, h4, h5, h6.
        for deeper in range(level + 1, 6):
            active[deeper] = None

    parts = [h for h in active if h is not None]
    return " > ".join(parts)


def _detect_chunk_type(text: str) -> tuple[str, str | None]:
    """
    Determine whether a chunk is 'prose', 'code', or 'mixed',
    and extract the language if it is pure code.

    A chunk is 'code' if every non-blank line is inside a fenced block.
    It is 'mixed' if it contains both prose and a code fence.
    Otherwise 'prose'.

    Returns (chunk_type, language).
    language is extracted from the opening fence annotation, e.g. ```python.
    """
    has_fence = "```" in text
    if not has_fence:
        return "prose", None

    # Count characters inside vs outside fences.
    inside_fence = False
    inside_chars = 0
    outside_chars = 0
    language: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if not inside_fence:
                # Opening fence — extract language annotation.
                lang = stripped[3:].strip()
                if lang and language is None:
                    language = lang
            inside_fence = not inside_fence
            continue
        if inside_fence:
            inside_chars += len(stripped)
        else:
            outside_chars += len(stripped)

    if outside_chars == 0 and inside_chars > 0:
        return "code", language
    if inside_chars > 0:
        return "mixed", language
    return "prose", None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk(
    text: str,
    target_chars: int = 1500,
    overlap_chars: int = 225,
    min_chars: int = 100,
) -> list[Chunk]:
    """
    Split *text* into a list of overlapping Chunk objects.

    Parameters
    ----------
    text          : full document body (may include YAML frontmatter)
    target_chars  : ideal chunk size in characters
    overlap_chars : characters of overlap between adjacent chunks (≈15%)
    min_chars     : discard chunks shorter than this

    Returns a list of Chunk objects.  Positions are relative to the
    *stripped body* (frontmatter removed).  If you need positions in the
    original text, add the frontmatter offset returned by _strip_frontmatter.

    Algorithm
    ~~~~~~~~~
    1. Strip frontmatter.
    2. Collect all break points across the entire text (one pass per pattern).
    3. Walk forward from the current start position:
       a. The ideal end is start + target_chars.
       b. Filter breaks in (start, ideal_end] that are not inside a code
          block or table.
       c. Score each candidate and pick the highest.
       d. If no candidate found, fall back to ideal_end (hard cut).
       e. Record the chunk, advance start with overlap.
    4. Emit any remaining text as a final chunk.
    """
    body, _fm_offset = _strip_frontmatter(text)
    n = len(body)

    if n == 0:
        return []

    # Pre-compute all break points once (not re-scanned per chunk).
    all_breaks = _collect_break_points(body)

    chunks: list[Chunk] = []
    start = 0

    while start < n:
        ideal_end = min(start + target_chars, n)

        if ideal_end == n:
            # Reached the end of the document — emit the remainder as-is.
            _emit(chunks, body, start, n, min_chars)
            break

        # --- Find the best break point in (start, ideal_end] ----------------
        span = ideal_end - start  # used to normalize distances

        best: BreakPoint | None = None
        best_final_score: float = -1.0

        for pos, bp in all_breaks.items():
            if pos >= ideal_end:
                break                                      # past ideal_end
            if pos <= start:
                continue                                   # outside window
            if _is_inside_code_block(body, pos):
                continue                                   # never cut mid-block
            if _is_inside_table(body, pos):
                continue                                   # never cut mid-table

            # normalised_distance: 0.0 at ideal_end, 1.0 at start
            norm_dist = (ideal_end - pos) / span
            final_score = bp.score * (1.0 - norm_dist ** 2 * 0.7)

            if final_score > best_final_score:
                best_final_score = final_score
                best = bp

        end = best.pos if best is not None else ideal_end

        _emit(chunks, body, start, end, min_chars)

        # --- Advance start with overlap --------------------------------------
        # next_start = end - overlap_chars, but:
        #   • must not go back into the previous chunk's start
        #   • snap forward to the next word boundary
        next_start = end - overlap_chars
        prev_start = chunks[-1].pos if chunks else 0

        if next_start <= prev_start:
            next_start = end           # no room for overlap
        else:
            # Snap forward to word boundary (space or newline).
            while next_start < end and body[next_start] not in (" ", "\n"):
                next_start += 1

        start = next_start

    return chunks


def _emit(
    chunks: list[Chunk],
    body: str,
    start: int,
    end: int,
    min_chars: int,
) -> None:
    """Append a Chunk to *chunks* if it meets the minimum length threshold."""
    length = end - start
    if length < min_chars:
        return

    chunk_text = body[start:end]
    section_path = _extract_section_path(body, start)
    chunk_type, language = _detect_chunk_type(chunk_text)

    chunks.append(Chunk(
        pos=start,
        len=length,
        section_path=section_path,
        chunk_type=chunk_type,
        language=language,
    ))
