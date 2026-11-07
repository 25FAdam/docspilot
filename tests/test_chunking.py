import pytest

from app.rag.chunking import chunk_markdown


def count(s: str) -> int:
    return len(s.split())


def run(text, min_tokens=20, max_tokens=100, overlap=0.12, path="content/handbook/a/b.md"):
    return chunk_markdown(
        text,
        source_path=path,
        url="https://example.test/a/b/",
        count_tokens=count,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_ratio=overlap,
    )


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def sentences(n_sentences, words_per=10, prefix="s"):
    return " ".join(
        " ".join(f"{prefix}{s}x{w}" for w in range(words_per)) + "." for s in range(n_sentences)
    )


def test_empty_and_front_matter_only():
    assert run("") == []
    assert run("   \n\n  ") == []
    assert run("---\ntitle: Only\n---\n") == []
    assert run("+++\ntitle = 'Only'\n+++\n\n<!-- nothing -->\n{{< foo >}}") == []


def test_front_matter_shortcodes_comments_stripped_and_title_used():
    text = (
        '---\ntitle: "My Page"\ndescription: x\n---\n'
        "{{% alert %}}\nIntro text here.\n<!-- hidden comment -->\n{{< /alert >}}\n"
    )
    chunks = run(text)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.section == "My Page"
    assert "title:" not in c.content
    assert "{{" not in c.content
    assert "hidden" not in c.content
    assert "Intro text here." in c.content


def test_toml_front_matter_title():
    chunks = run("+++\ntitle = 'Toml Title'\n+++\nBody text.")
    assert chunks[0].section == "Toml Title"


def test_title_falls_back_to_h1_then_path():
    assert run("# Heading One\n\ntext")[0].section == "Heading One"
    assert run("just text", path="content/handbook/team-ops/_index.md")[0].section == "Team Ops"
    assert run("just text", path="content/x/my-page.md")[0].section == "My Page"


def test_breadcrumbs_follow_heading_hierarchy():
    text = (
        "---\ntitle: Doc\n---\n"
        "Preamble with enough words to stay alone.\n\n"
        f"## Alpha\n{words(30)}\n\n"
        f"### Alpha One\n{words(30, 'b')}\n\n"
        f"### Alpha Two\n{words(30, 'c')}\n\n"
        f"## Beta\n{words(30, 'd')}\n"
    )
    sections = [c.section for c in run(text, min_tokens=5, max_tokens=60)]
    assert sections[0] == "Doc"
    assert "Doc > Alpha" in sections
    assert "Doc > Alpha > Alpha One" in sections
    assert "Doc > Alpha > Alpha Two" in sections
    assert "Doc > Beta" in sections


def test_h1_equal_to_title_is_not_duplicated():
    text = f"---\ntitle: Same\n---\n# Same\n{words(30)}\n\n## Sub\n{words(30, 'b')}"
    sections = {c.section for c in run(text, min_tokens=5, max_tokens=50)}
    assert "Same" in sections
    assert "Same > Sub" in sections


def test_hash_in_code_fence_is_not_a_heading():
    text = "# Title\n\nIntro.\n\n```bash\n# not a heading\necho hi\n```\n\nAfter."
    chunks = run(text)
    assert len(chunks) == 1
    assert chunks[0].section == "Title"
    assert "# not a heading" in chunks[0].content


def test_heading_with_trailing_hash_and_link_markup():
    chunks = run("# T\n\n## Learning C# [docs](http://x.y) ##\n\ntext " + words(30), min_tokens=2)
    assert any(c.section == "T > Learning C# docs" for c in chunks)


def test_small_document_is_one_chunk():
    chunks = run("# Title\n\nShort doc.\n\n## Sub\n\nAlso short.")
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].source_path == "content/handbook/a/b.md"
    assert chunks[0].url == "https://example.test/a/b/"
    assert "## Sub" in chunks[0].content  # section heading line stays in the body


def test_oversized_section_splits_within_budget_with_overlap():
    max_tokens, ratio = 100, 0.12
    paragraphs = "\n\n".join(words(25, f"p{i}_") for i in range(12))
    text = f"# Doc\n\n## Big\n\n{paragraphs}"
    chunks = [c for c in run(text, 20, max_tokens, ratio) if c.section == "Doc > Big"]
    assert len(chunks) > 2
    for c in chunks:
        assert c.token_count <= max_tokens
    target = ratio * max_tokens
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        prev_words = prev.content.split()
        nxt_words = nxt.content.split()
        shared = 0
        for k in range(1, min(len(prev_words), len(nxt_words)) + 1):
            if prev_words[-k:] == nxt_words[:k]:
                shared = k
        assert target * 0.99 <= shared <= target * 1.15 + 1, shared
        assert 0.10 <= shared / max_tokens <= 0.15


def test_overlap_not_applied_across_sections():
    text = f"# Doc\n\n## A\n{words(70, 'a')}\n\n## B\n{words(70, 'b')}"
    chunks = run(text, min_tokens=2, max_tokens=80)
    a = [c for c in chunks if c.section.endswith("> A")]
    b = [c for c in chunks if c.section.endswith("> B")]
    assert a and b
    assert not any(w.startswith("a") and w[1:].isdigit() for c in b for w in c.content.split())


def test_no_words_lost_when_splitting():
    text = "# Doc\n\n" + "\n\n".join(sentences(4, 10, f"q{i}") for i in range(10))
    chunks = run(text, 20, 90)
    src = {w for w in text.split() if w not in {"#"}}
    got = {w for c in chunks for w in c.content.split()}
    assert src <= got


def test_giant_paragraph_is_hard_split_without_loss():
    big = words(500)  # one paragraph, no sentence boundaries
    chunks = run(f"# Doc\n\n{big}", 20, 100)
    assert len(chunks) >= 5
    assert all(c.token_count <= 100 for c in chunks)
    assert set(big.split()) <= {w for c in chunks for w in c.content.split()}


def test_oversized_paragraph_split_on_sentences():
    para = sentences(30, 10)
    chunks = run(f"# Doc\n\n{para}", 20, 100)
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)
    # sentence-aligned: every chunk body except possibly overlap tail ends with a full stop
    assert all(c.content.rstrip().endswith(".") for c in chunks)


def test_code_fence_kept_atomic_when_it_fits():
    code = "```python\nline1\n\nline2 = 1\n```"
    text = f"# Doc\n\n{words(60, 'a')}\n\n{code}\n\n{words(60, 'b')}"
    chunks = run(text, 20, 100)
    assert any(code in c.content for c in chunks)


def test_tiny_sections_are_merged_forward_and_keep_first_breadcrumb():
    text = "# Doc\n\n## One\nfew words\n\n## Two\nmore words\n\n## Three\n" + words(15)
    chunks = run(text, min_tokens=20, max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].section == "Doc"
    assert "## Two" in chunks[0].content and "## Three" in chunks[0].content


def test_merge_stops_when_combined_exceeds_budget():
    text = f"# Doc\n\n## One\n{words(15, 'a')}\n\n## Two\n{words(95, 'b')}"
    chunks = run(text, min_tokens=20, max_tokens=100)
    assert [c.section for c in chunks][-1] == "Doc > Two"
    assert all(c.token_count <= 100 for c in chunks)
    assert len(chunks) >= 2


def test_trailing_tiny_section_merges_into_previous_if_it_fits():
    text = f"# Doc\n\n## One\n{words(40, 'a')}\n\n## Two\n{words(40, 'b')}\n\n## End\nbye"
    chunks = run(text, min_tokens=30, max_tokens=100)
    assert "bye" in chunks[-1].content
    assert not any(
        c.content.strip().endswith("## End\nbye") and c.section.endswith("End") for c in chunks
    )
    assert all(c.token_count <= 100 for c in chunks)


def test_chunk_indexes_contiguous_and_deterministic():
    text = "# Doc\n\n" + "\n\n".join(f"## S{i}\n{words(70, f'x{i}_')}" for i in range(6))
    first = run(text, 20, 80)
    second = run(text, 20, 80)
    assert first == second
    assert [c.chunk_index for c in first] == list(range(len(first)))
    assert len(first) >= 6


def test_crlf_input_is_normalised():
    chunks = run("# T\r\n\r\ntext line\r\n\r\n## S\r\nmore text")
    assert len(chunks) == 1
    assert "\r" not in chunks[0].content


def test_token_count_covers_breadcrumb_and_content():
    c = run("# Title\n\nsome body words")[0]
    assert c.token_count == count(f"{c.section}\n\n{c.content}")


def test_very_long_heading_is_clipped_and_never_exceeds_max():
    long_title = words(90, "t")
    chunks = run(f"# {long_title}\n\n## Sub\n{words(150, 'b')}", 20, 100)
    assert chunks
    for c in chunks:
        assert c.token_count <= 100
        assert count(c.section + "\n\n") <= 50  # breadcrumb limited to max // 2
        assert c.token_count == count(f"{c.section}\n\n{c.content}")  # stored == embedded


def test_clipping_drops_outermost_parts_first_and_keeps_deepest():
    text = f"---\ntitle: {words(30, 'root')}\n---\n## {words(30, 'mid')}\n\n### Leaf\n{words(60)}"
    chunks = run(text, 5, 80)
    deepest = [c for c in chunks if c.section.endswith("Leaf")]
    assert deepest
    assert all("root0" not in c.section for c in deepest)
    assert all(c.token_count <= 80 for c in chunks)


def test_single_giant_heading_is_truncated_to_its_first_words():
    chunks = run(f"# {words(300, 'h')}\n\ntext body here", 5, 40)
    assert chunks
    assert all(c.token_count <= 40 for c in chunks)
    assert chunks[0].section.startswith("h0 h1")


def test_guard_raises_if_a_chunk_would_exceed_max():
    def lying_counter(s):
        # Honest while sizing; inflates only the final breadcrumb + content embedding input.
        return len(s.split()) + (1000 if s.startswith("Doc\n\n#") else 0)

    with pytest.raises(ValueError, match="exceeds max_tokens"):
        chunk_markdown("# Doc\n\nbody", "a.md", "u", lying_counter, 5, 50, 0.1)
