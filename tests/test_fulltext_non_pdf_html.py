"""Regression tests for #265: local fulltext for non-PDF/HTML attachments.

Previously ``_extract_fulltext_for_item`` only collected PDF and HTML
attachments as fulltext candidates. Any other attachment (``.txt``,
``.vtt``, ``.srt``, transcripts, captions, plain markdown notes) was
silently dropped — the local API can't serve raw bytes for them, so the
upstream fallback also failed and the tool returned a misleading 404.

Behavior we want: the gate accepts any plain-text attachment too;
the existing ``_extract_text_from_file`` fallback already handles it.
"""

from pathlib import Path

from conftest import extracted_doc

from zotero_mcp.extract import DEFAULT_ATTACHMENT_PRIORITY, is_extractable
from zotero_mcp.local_db import LocalZoteroReader


class _Reader(LocalZoteroReader):
    """LocalZoteroReader stub with no DB and configurable attachments."""

    def __init__(self, attachments, fake_path_for=None):
        self.db_path = "/dev/null"
        self._connection = None
        self.pdf_max_pages = 10
        self.attachment_priority = DEFAULT_ATTACHMENT_PRIORITY
        self._attachments = attachments
        self._fake_path_for = fake_path_for or {}

    def _iter_parent_attachments(self, parent_item_id: int):
        yield from self._attachments

    def _resolve_attachment_path(self, attachment_key: str, zotero_path: str):
        return self._fake_path_for.get(attachment_key)


# ---------------------------------------------------------------------------
# is_extractable classifier
# ---------------------------------------------------------------------------


class TestIsExtractableAttachment:
    def test_text_plain_ctype(self):
        assert is_extractable(Path("a.txt"), "text/plain")

    def test_vtt_ctype(self):
        assert is_extractable(Path("a.vtt"), "text/vtt")

    def test_srt_extension_without_ctype(self):
        assert is_extractable(Path("captions.srt"), None)

    def test_unknown_text_subtype(self):
        # Any ``text/*`` MIME is accepted (covers obscure transcript types).
        assert is_extractable(Path("a.txt"), "text/x-asm")

    def test_docx_rejected(self):
        """Binary office formats are not accepted — read_text would return
        garbage and pollute the semantic index."""
        assert not is_extractable(
            Path("paper.docx"),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    def test_video_rejected(self):
        assert not is_extractable(
            Path("talk.mp4"), "video/mp4"
        )

    def test_pdf_returns_true_via_extension_set(self):
        """PDFs are not on the textual allowlist — the PDF path is handled
        before this classifier ever runs. False here is correct."""
        assert not is_extractable(
            Path("a.pdf"), "application/pdf"
        )


# ---------------------------------------------------------------------------
# _extract_fulltext_for_item end-to-end
# ---------------------------------------------------------------------------


def test_extract_fulltext_uses_vtt_when_only_attachment(tmp_path):
    """A solo .vtt attachment must yield fulltext (#265 main repro)."""
    vtt = tmp_path / "lecture.vtt"
    vtt.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nIntroduction to the topic.\n"
    )
    reader = _Reader(
        attachments=[("ATTKEY01", "storage:lecture.vtt", "text/vtt")],
        fake_path_for={"ATTKEY01": vtt},
    )
    result = reader._extract_fulltext_for_item(item_id=1)

    assert result is not None
    text, source = result
    assert "Introduction to the topic" in text
    assert source == "file"


def test_extract_fulltext_prefers_pdf_over_textual(tmp_path):
    """When both a PDF and a textual attachment exist, the PDF still wins."""
    pdf = tmp_path / "paper.pdf"
    pdf.touch()
    txt = tmp_path / "notes.txt"
    txt.write_text("plaintext fallback content")

    class _ReaderWithPdfText(_Reader):
        def _extract_doc_from_file(self, file_path):
            # Asserting the suffix is the point of the test: the chooser must
            # hand us the PDF, not the .txt sitting alongside it.
            assert file_path.suffix == ".pdf"
            return extracted_doc("PDF content extracted")

    reader = _ReaderWithPdfText(
        attachments=[
            ("PDF0001", "storage:paper.pdf", "application/pdf"),
            ("TXT0001", "storage:notes.txt", "text/plain"),
        ],
        fake_path_for={"PDF0001": pdf, "TXT0001": txt},
    )
    text, source = reader._extract_fulltext_for_item(item_id=1)
    assert text == "PDF content extracted"
    assert source == "pdf"


def test_extract_fulltext_falls_back_to_textual_when_no_pdf_html(tmp_path):
    txt = tmp_path / "transcript.txt"
    txt.write_text("transcript body")
    reader = _Reader(
        attachments=[("T0001", "storage:transcript.txt", "text/plain")],
        fake_path_for={"T0001": txt},
    )
    text, source = reader._extract_fulltext_for_item(item_id=1)
    assert text == "transcript body"
    assert source == "file"


def test_extract_fulltext_ignores_binary_attachment(tmp_path):
    """A .docx-only item still returns None — we don't have a docx extractor
    here and the read_text fallback would emit garbage."""
    binary = tmp_path / "paper.docx"
    binary.write_bytes(b"\x50\x4b\x03\x04binary garbage")
    reader = _Reader(
        attachments=[
            (
                "D0001",
                "storage:paper.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        ],
        fake_path_for={"D0001": binary},
    )
    assert reader._extract_fulltext_for_item(item_id=1) is None

class _AllReader(_Reader):
    """Stub that reads every attachment and records cache calls."""

    def __init__(self, attachments, fake_path_for=None, texts=None):
        super().__init__(attachments, fake_path_for=fake_path_for)
        self.fulltext_cache_enabled = False
        self._texts = texts or {}

    def _cache_lookup(self, target, attachment_key):
        return None

    def _cache_store(self, *args, **kwargs):
        return None

    def _extract_doc_from_file(self, path):
        return extracted_doc(self._texts[Path(path).suffix.lower()])


def test_extract_fulltext_all_reads_every_readable_attachment(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    html = tmp_path / "snapshot.html"
    html.write_text("<p>transcript</p>")
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x89PNG\r\n")

    reader = _AllReader(
        attachments=[
            ("HTM", "storage:snapshot.html", "text/html"),
            ("IMG", "storage:scan.png", "image/png"),
            ("PDF", "storage:paper.pdf", "application/pdf"),
        ],
        fake_path_for={"HTM": html, "IMG": image, "PDF": pdf},
        texts={".pdf": "PDF body.", ".html": "Transcript body."},
    )

    parts = reader.extract_fulltext_all(item_id=1)

    # Default priority is PDF before HTML; the image is not text-bearing.
    assert [(key, text) for key, text, _source in parts] == [
        ("PDF", "PDF body."),
        ("HTM", "Transcript body."),
    ]


def test_extract_fulltext_all_honours_priority(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    html = tmp_path / "snapshot.html"
    html.write_text("<p>transcript</p>")

    reader = _AllReader(
        attachments=[
            ("PDF", "storage:paper.pdf", "application/pdf"),
            ("HTM", "storage:snapshot.html", "text/html"),
        ],
        fake_path_for={"HTM": html, "PDF": pdf},
        texts={".pdf": "PDF body.", ".html": "Transcript body."},
    )
    reader.attachment_priority = ("html", "pdf", "other")

    parts = reader.extract_fulltext_all(item_id=1)
    assert [p[0] for p in parts] == ["HTM", "PDF"]
