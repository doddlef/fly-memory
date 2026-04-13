"""
flym.chunker_code
-----------------
AST-aware breakpoint extraction for fenced code blocks in markdown.

Why AST breakpoints?
~~~~~~~~~~~~~~~~~~~~
The regex chunker ignores content inside fenced code blocks (the code-block
guard in _is_inside_code_block prevents cuts there).  This is correct for
prose documents, but means a large code block is never split — it becomes
one giant chunk regardless of target_chars.

With tree-sitter we parse the code inside each fence, find the boundaries of
top-level declarations (functions, classes, imports), and inject those
positions as high-score break points.  The main chunker then treats them
exactly like heading breaks — it can cut there, and will prefer them over
lower-quality breaks.

Scores for AST nodes
~~~~~~~~~~~~~~~~~~~~
    class_definition / class_declaration   : 92   (above h2=90)
    function_definition / declaration      : 88   (above h3=80)
    import block                           : 38   (above sentence=30)

These values sit between heading scores deliberately: an AST boundary should
outcompete a blank line or sentence break, but a real heading in the document
still wins over a function boundary inside a code block.

Integration with chunker.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
chunk() calls _collect_break_points() (regex) and then merges the result
with code_break_points() (AST) using max-score-wins at each position.
If tree-sitter packages are not installed, code_break_points() returns {}
and the chunker behaves exactly as before (graceful degradation).

Byte vs character offsets
~~~~~~~~~~~~~~~~~~~~~~~~~
tree-sitter works in bytes; Python strings are Unicode characters.  For
source code that is entirely ASCII (the common case) bytes == characters.
For code containing non-ASCII (Unicode identifiers, string literals) the
positions would diverge.

We sidestep this by using start_point.row (line number) rather than
start_byte, then locating the line's character offset in the original
document.  Line-based positioning is immune to the byte/char mismatch.

Supported languages
~~~~~~~~~~~~~~~~~~~
    python      — tree-sitter-python
    javascript  — tree-sitter-javascript
    typescript  — tree-sitter-javascript  (JS grammar covers TS subset)

Other languages fall back to regex-only silently.
"""

from __future__ import annotations

import re

# Deferred imports — tree-sitter packages are optional.
# If they are missing, code_break_points() returns {} and the caller
# continues with regex-only breakpoints.
try:
    from tree_sitter import Language, Parser

    import tree_sitter_python     as _tspython
    import tree_sitter_javascript as _tsjavascript

    _LANGUAGES: dict[str, Language] = {
        "python":     Language(_tspython.language()),
        "javascript": Language(_tsjavascript.language()),
        "typescript": Language(_tsjavascript.language()),  # JS grammar covers TS
    }
    _AVAILABLE = True
except ImportError:
    Language   = None   # type: ignore[assignment, misc]
    Parser     = None   # type: ignore[assignment, misc]
    _tspython  = None   # type: ignore[assignment, misc]
    _tsjavascript = None
    _AVAILABLE = False
    _LANGUAGES = {}


# ---------------------------------------------------------------------------
# Node-type → break score per language
# ---------------------------------------------------------------------------

# Each entry: set of node type strings that mark a good cut point, and the
# score to assign.  Checked in order; first match wins for a given node.
_NODE_SCORES: dict[str, list[tuple[set[str], int]]] = {
    "python": [
        ({"class_definition"},                                   92),
        ({"function_definition", "decorated_definition"},        88),
        ({"import_statement", "import_from_statement"},          38),
    ],
    "javascript": [
        ({"class_declaration"},                                  92),
        ({"function_declaration", "export_statement"},           88),
        ({"import_declaration"},                                 38),
    ],
    "typescript": [
        ({"class_declaration"},                                  92),
        ({"function_declaration", "export_statement"},           88),
        ({"import_declaration"},                                 38),
    ],
}


# ---------------------------------------------------------------------------
# Regex to find fenced code blocks in markdown
# ---------------------------------------------------------------------------

# Captures: (opening_fence_line, language_tag, code_body)
_FENCE_RE = re.compile(
    r"(```([a-zA-Z]*)\n)"   # group 1 = opening line, group 2 = language tag
    r"(.*?)"                # group 3 = code body (non-greedy)
    r"```",                  # closing fence
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def code_break_points(text: str) -> "dict[int, object]":
    """
    Return AST-derived break points for fenced code blocks in *text*.

    Returns a dict compatible with the output of _collect_break_points():
        position (int) → BreakPoint

    If tree-sitter is not installed, returns {}.

    Parameters
    ----------
    text : full document body (frontmatter already stripped)
    """
    if not _AVAILABLE:
        return {}

    # Import here to avoid circular import (chunker imports us).
    from flym.chunker import BreakPoint

    breaks: dict[int, BreakPoint] = {}

    for m in _FENCE_RE.finditer(text):
        lang_tag  = m.group(2).lower()
        code_body = m.group(3)

        if lang_tag not in _LANGUAGES:
            continue   # unsupported language — skip

        # Character position where the code body starts in the document.
        code_start = m.start(3)

        # Pre-compute a line-start index for fast row → char-offset lookup.
        line_starts = _build_line_starts(code_body)

        parser = Parser(_LANGUAGES[lang_tag])
        tree   = parser.parse(code_body.encode())

        for node in tree.root_node.children:
            if node.is_named and node.start_point[0] > 0:
                # row > 0 only: skip nodes at the very first line of the
                # fence (nothing useful to cut before them).
                score = _node_score(node.type, lang_tag)
                if score is None:
                    continue

                # Map tree-sitter row to document character position.
                row       = node.start_point[0]
                doc_pos   = code_start + line_starts[row] - 1  # -1 → the \n before
                doc_pos   = max(code_start, doc_pos)

                if doc_pos not in breaks or score > breaks[doc_pos].score:
                    breaks[doc_pos] = BreakPoint(
                        pos   = doc_pos,
                        score = score,
                        label = f"ast:{node.type}",
                    )

    return breaks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_line_starts(code: str) -> list[int]:
    """
    Return a list where index i is the character offset of line i in *code*.

    line_starts[0] = 0  (first line starts at 0)
    line_starts[1] = position of the character after the first '\n'
    …
    """
    starts = [0]
    for i, ch in enumerate(code):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _node_score(node_type: str, lang: str) -> int | None:
    """Return the break score for *node_type* in *lang*, or None if unscored."""
    for type_set, score in _NODE_SCORES.get(lang, []):
        if node_type in type_set:
            return score
    return None
