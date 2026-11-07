"""Heading-aware markdown chunking. Pure functions: no I/O, no model, no network.

Token counting is injected (`count_tokens`), so the same code runs with the real embedding
tokenizer in production and with a trivial word counter in tests.

Pipeline
1. Clean: strip YAML/TOML front matter (reading `title`), Hugo shortcodes, HTML comments.
2. Split on ATX headings (ignoring `#` inside fenced code). Each section gets a breadcrumb
   `Title > H2 > H3` and its body includes its own heading line.
3. Every chunk is embedded as `breadcrumb + "\\n\\n" + content`, so the breadcrumb is part of the
   token budget: budget = max_tokens - tokens(breadcrumb + "\\n\\n").
4. Tiny sections (< min_tokens) are merged forward while the result fits the budget. A tiny
   final section is merged backward into the previous one if it fits.
5. A section that does not fit is packed from blank-line separated units (fenced code blocks
   are atomic). Oversized units are split by lines, then sentences, then words (hard split).
6. Consecutive chunks of the same split section overlap by ~overlap_ratio * max_tokens, made of
   whole trailing units when possible, else the last few words. No overlap across sections.

`Chunk.token_count` counts the embedding input (breadcrumb + content).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

CountTokens = Callable[[str], int]

# Overlap may exceed its target by at most this factor when built from whole units.
_OVERLAP_SLACK = 1.15


class ChunkingError(RuntimeError):
    """An internal chunking invariant was violated (a chunk exceeded max_tokens)."""


@dataclass(frozen=True)
class Chunk:
    source_path: str
    chunk_index: int
    section: str
    url: str
    content: str
    token_count: int


class _Unit(NamedTuple):
    text: str
    joiner: str  # string placed between the previous unit and this one
    tokens: int


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_FRONT_MATTER = re.compile(
    r"\A(---|\+\+\+)[ \t]*\n(?:(.*?)\n)?\1[ \t]*(?:\n|\Z)",
    re.DOTALL,
)
_TITLE = re.compile(r"^title\s*[:=]\s*(.+?)\s*$", re.MULTILINE)
_SHORTCODE = re.compile(r"\{\{[<%].*?[%>]\}\}", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def _strip_front_matter(text: str) -> tuple[str, str]:
    """Return (body, title). Title is '' when absent."""
    m = _FRONT_MATTER.match(text)
    if not m:
        return text, ""
    title = ""
    tm = _TITLE.search(m.group(2) or "")
    if tm:
        title = tm.group(1).strip().strip("\"'").strip()
    return text[m.end() :], title


def _clean(text: str) -> tuple[str, str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    body, title = _strip_front_matter(text)
    body = _HTML_COMMENT.sub("", body)
    body = _SHORTCODE.sub("", body)
    return body, title


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_ANCHOR = re.compile(r"\s*\{#[^}]*\}\s*$")


def _clean_heading(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    text = _ANCHOR.sub("", text)
    text = re.sub(r"[*_`]+", "", text)
    return " ".join(text.split())


def _fallback_title(source_path: str) -> str:
    parts = [p for p in source_path.replace("\\", "/").split("/") if p]
    if not parts:
        return "Document"
    stem = parts[-1].rsplit(".", 1)[0]
    if stem == "_index" and len(parts) > 1:
        stem = parts[-2]
    return " ".join(stem.replace("-", " ").replace("_", " ").split()).title() or "Document"


def _parse_sections(body: str) -> tuple[list[tuple[tuple[str, ...], str]], str]:
    """Split on ATX headings. Returns ([(heading_path, section_text)], first_h1_or_empty)."""
    sections: list[tuple[tuple[str, ...], str]] = []
    stack: list[tuple[int, str]] = []
    lines: list[str] = []
    first_h1 = ""
    fence_char = ""
    fence_len = 0

    def flush() -> None:
        text = "\n".join(lines).strip("\n")
        if text.strip():
            sections.append((tuple(t for _, t in stack), text))

    for line in body.split("\n"):
        fm = _FENCE.match(line)
        if fence_char:
            lines.append(line)
            if (
                fm
                and fm.group(1)[0] == fence_char
                and len(fm.group(1)) >= fence_len
                and not fm.group(2).strip()
            ):
                fence_char = ""
            continue
        if fm and not (fm.group(1)[0] == "`" and "`" in fm.group(2)):
            fence_char, fence_len = fm.group(1)[0], len(fm.group(1))
            lines.append(line)
            continue
        hm = _HEADING.match(line)
        if hm:
            title = _clean_heading(hm.group(2))
            if not title:
                lines.append(line)
                continue
            flush()
            level = len(hm.group(1))
            if level == 1 and not first_h1:
                first_h1 = title
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            lines = [line]
        else:
            lines.append(line)
    flush()
    return sections, first_h1


def _breadcrumb(root: str, path: tuple[str, ...], limit: int, count: CountTokens) -> str:
    """`Title > H2 > H3`, clipped so that breadcrumb + "\n\n" is at most `limit` tokens.

    Clipping drops the leading (outermost) parts first, keeping the most specific ones; if the
    last part alone is still too long it is cut to its first words. The clipped string is what
    gets stored in `section` and embedded, so the two always match.
    """
    parts: list[str] = []
    for part in (root, *path):
        if not parts or parts[-1].casefold() != part.casefold():
            parts.append(part)
    while len(parts) > 1 and count(" > ".join(parts) + "\n\n") > limit:
        parts.pop(0)
    crumb = " > ".join(parts)
    if count(crumb + "\n\n") > limit:
        crumb = (_hard_split(crumb, max(limit, 1), count) or [crumb[:1]])[0]
    return crumb


# ---------------------------------------------------------------------------
# Splitting oversized text
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\S+")


def _split_blocks(text: str) -> list[str]:
    """Blank-line separated blocks; fenced code blocks stay in one piece."""
    blocks: list[str] = []
    cur: list[str] = []
    fence_char = ""
    fence_len = 0
    for line in text.split("\n"):
        fm = _FENCE.match(line)
        if fence_char:
            cur.append(line)
            if (
                fm
                and fm.group(1)[0] == fence_char
                and len(fm.group(1)) >= fence_len
                and not fm.group(2).strip()
            ):
                fence_char = ""
            continue
        if fm and not (fm.group(1)[0] == "`" and "`" in fm.group(2)):
            fence_char, fence_len = fm.group(1)[0], len(fm.group(1))
        if not line.strip() and not fence_char:
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return [b for b in blocks if b.strip()]


def _hard_split(text: str, limit: int, count: CountTokens) -> list[str]:
    """Last resort: pack whitespace-separated words, bisecting a single over-long word."""
    pieces: list[str] = []
    start = end = -1
    used = 0
    for m in _WORD.finditer(text):
        n = count(m.group())
        if n > limit:
            if start >= 0:
                pieces.append(text[start:end])
                start, used = -1, 0
            pieces.extend(_split_chars(m.group(), limit, count))
            continue
        if start >= 0 and used + n > limit:
            pieces.append(text[start:end])
            start, used = -1, 0
        if start < 0:
            start = m.start()
        end = m.end()
        used += n
    if start >= 0:
        pieces.append(text[start:end])
    return pieces


def _split_chars(word: str, limit: int, count: CountTokens) -> list[str]:
    out: list[str] = []
    rest = word
    while rest:
        if count(rest) <= limit:
            out.append(rest)
            break
        lo, hi = 1, len(rest)  # largest prefix length whose count fits (at least 1 char)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count(rest[:mid]) <= limit:
                lo = mid
            else:
                hi = mid - 1
        out.append(rest[:lo])
        rest = rest[lo:]
    return out


_SPLITTERS: tuple[tuple[str, Callable[[str], list[str]]], ...] = (
    ("\n", lambda t: t.split("\n")),
    (" ", lambda t: _SENTENCE_END.split(t)),
)


def _fit(text: str, limit: int, count: CountTokens, level: int = 0) -> list[tuple[str, str]]:
    """Break text into (piece, joiner) pairs, each piece within `limit` tokens."""
    if count(text) <= limit:
        return [(text, "")]
    if level == len(_SPLITTERS):
        return [(p, " " if i else "") for i, p in enumerate(_hard_split(text, limit, count))]
    sep, split = _SPLITTERS[level]
    parts = [p for p in split(text) if p.strip()]
    if len(parts) <= 1:
        return _fit(text, limit, count, level + 1)
    out: list[tuple[str, str]] = []
    for i, part in enumerate(parts):
        sub = _fit(part, limit, count, level + 1)
        sub[0] = (sub[0][0], sep if i else "")
        out.extend(sub)
    return out


def _make_units(body: str, limit: int, count: CountTokens) -> list[_Unit]:
    units: list[_Unit] = []
    for block in _split_blocks(body):
        for i, (piece, joiner) in enumerate(_fit(block, limit, count)):
            units.append(_Unit(piece, joiner if i else "\n\n", count(piece)))
    return units


def _render(units: list[_Unit]) -> str:
    return "".join((u.joiner if i else "") + u.text for i, u in enumerate(units))


def _tail_words(text: str, target: int, cap: int, count: CountTokens) -> str:
    """Last few words of text totalling >= target tokens (never more than cap if avoidable)."""
    words = list(_WORD.finditer(text))
    total = 0
    start = len(text)
    for m in reversed(words):
        n = count(m.group())
        if total + n > cap:
            break
        total += n
        start = m.start()
        if total >= target:
            break
    return text[start:]


def _overlap(prev: list[_Unit], target: int, cap: int, count: CountTokens) -> str:
    acc: list[_Unit] = []
    total = 0
    for unit in reversed(prev):
        if total + unit.tokens > cap:
            break
        acc.insert(0, unit)
        total += unit.tokens
        if total >= target:
            return _render(acc)
    return _tail_words(_render(prev), target, cap, count)


def _split_section(
    body: str,
    budget: int,
    overlap_target: int,
    count: CountTokens,
) -> list[str]:
    """Split an oversized section body into chunk texts (<= budget tokens each)."""
    cap = max(overlap_target, int(overlap_target * _OVERLAP_SLACK))
    reserve = cap if overlap_target > 0 and cap * 2 <= budget else 0
    target = overlap_target if reserve else 0

    units = _make_units(body, budget - reserve, count)
    groups: list[list[_Unit]] = []
    cur: list[_Unit] = []
    used = 0
    for unit in units:
        room = budget if not groups else budget - reserve
        if cur and used + unit.tokens > room:
            groups.append(cur)
            cur, used = [], 0
        cur.append(unit)
        used += unit.tokens
    if cur:
        groups.append(cur)

    texts: list[str] = []
    for i, group in enumerate(groups):
        text = _render(group)
        if i and target:
            prefix = _overlap(groups[i - 1], target, cap, count)
            if prefix:
                text = prefix + group[0].joiner + text
        texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_markdown(
    text: str,
    source_path: str,
    url: str,
    count_tokens: CountTokens,
    min_tokens: int,
    max_tokens: int,
    overlap_ratio: float,
) -> list[Chunk]:
    body, title = _clean(text)
    if not body.strip():
        return []
    raw_sections, first_h1 = _parse_sections(body)
    if not raw_sections:
        return []
    root = title or first_h1 or _fallback_title(source_path)

    # Breadcrumbs are clipped to half the budget, so the body always keeps at least half and
    # breadcrumb + body never exceeds max_tokens.
    crumb_limit = max(max_tokens // 2, 1)

    def budget_for(crumb: str) -> int:
        return max(max_tokens - count_tokens(crumb + "\n\n"), 1)

    # (breadcrumb, body, tokens)
    merged: list[tuple[str, str, int]] = []
    items = [
        (_breadcrumb(root, path, crumb_limit, count_tokens), sec, count_tokens(sec))
        for path, sec in raw_sections
    ]
    i = 0
    while i < len(items):
        crumb, sec, tokens = items[i]
        i += 1
        budget = budget_for(crumb)
        # Merge forward: the merged chunk keeps only the first section's breadcrumb; later
        # sections stay identifiable through their heading lines in the body.
        while tokens < min_tokens and i < len(items) and tokens + items[i][2] <= budget:
            sec = sec + "\n\n" + items[i][1]
            tokens += items[i][2]
            i += 1
        merged.append((crumb, sec, tokens))
    if len(merged) > 1:
        last_crumb, last_sec, last_tokens = merged[-1]
        prev_crumb, prev_sec, prev_tokens = merged[-2]
        if last_tokens < min_tokens and prev_tokens + last_tokens <= budget_for(prev_crumb):
            merged[-2:] = [(prev_crumb, prev_sec + "\n\n" + last_sec, prev_tokens + last_tokens)]

    overlap_target = round(overlap_ratio * max_tokens)
    chunks: list[Chunk] = []
    for crumb, sec, tokens in merged:
        budget = budget_for(crumb)
        texts = (
            [sec] if tokens <= budget else _split_section(sec, budget, overlap_target, count_tokens)
        )
        for content in texts:
            content = content.strip()
            if not content:
                continue
            token_count = count_tokens(f"{crumb}\n\n{content}")
            if token_count > max_tokens:  # invariant: CHUNK_MAX_TOKENS includes the breadcrumb
                raise ChunkingError(
                    f"{source_path}: chunk of {token_count} tokens exceeds max_tokens={max_tokens}"
                )
            chunks.append(
                Chunk(
                    source_path=source_path,
                    chunk_index=len(chunks),
                    section=crumb,
                    url=url,
                    content=content,
                    token_count=token_count,
                )
            )
    return chunks
