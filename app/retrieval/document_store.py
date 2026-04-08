from pathlib import Path

from pypdf import PdfReader

from app.models import DocumentInfo, DocumentLoadIssue, DocumentLoadResult, DocumentRecord


class DocumentStore:
    DISCOVERABLE_EXTENSIONS = (".md", ".txt", ".pdf")
    RETRIEVAL_EXTENSIONS = (".md", ".txt", ".pdf")
    MARKDOWN_EXTENSION = ".md"
    TEXT_EXTENSION = ".txt"
    PDF_EXTENSION = ".pdf"

    def __init__(self, docs_path: Path, pdf_min_text_characters: int = 40) -> None:
        self.docs_path = docs_path
        self.pdf_min_text_characters = pdf_min_text_characters

    def list_documents(self) -> list[DocumentInfo]:
        if not self.docs_path.exists() or not self.docs_path.is_dir():
            return []

        documents: list[DocumentInfo] = []
        for path in sorted(self.docs_path.iterdir()):
            if not path.is_file() or path.suffix.lower() not in self.DISCOVERABLE_EXTENSIONS:
                continue

            documents.append(
                DocumentInfo(
                    file_name=path.name,
                    path=path,
                    extension=path.suffix.lower(),
                    size_bytes=path.stat().st_size,
                    source_type=self._source_type_for_extension(path.suffix.lower()),
                )
            )

        return documents

    def load_retrieval_documents(self) -> list[DocumentRecord]:
        return self.load_retrieval_bundle().documents

    def get_document_load_issues(self) -> list[DocumentLoadIssue]:
        return self.load_retrieval_bundle().issues

    def load_retrieval_bundle(self) -> DocumentLoadResult:
        documents, issues = self._load_retrieval_documents_with_issues()
        return DocumentLoadResult(documents=documents, issues=issues)

    def _load_retrieval_documents_with_issues(
        self,
    ) -> tuple[list[DocumentRecord], list[DocumentLoadIssue]]:
        if not self.docs_path.exists() or not self.docs_path.is_dir():
            return [], []

        documents: list[DocumentRecord] = []
        issues: list[DocumentLoadIssue] = []
        for path in sorted(self.docs_path.iterdir()):
            if not path.is_file() or path.suffix.lower() not in self.RETRIEVAL_EXTENSIONS:
                continue

            if path.suffix.lower() in {self.MARKDOWN_EXTENSION, self.TEXT_EXTENSION}:
                text = self._read_plain_text(path)
                if text is None:
                    issues.append(
                        DocumentLoadIssue(
                            file_name=path.name,
                            path=path,
                            reason=f"{self._source_type_for_extension(path.suffix.lower()).title()} file could not be read.",
                            source_type=self._source_type_for_extension(path.suffix.lower()) or "text",
                        )
                    )
                    continue

                documents.append(
                    DocumentRecord(
                        document_id=path.stem,
                        file_name=path.name,
                        path=path,
                        text=text,
                        source_type=self._source_type_for_extension(path.suffix.lower()) or "text",
                    )
                )
                continue

            pdf_text = self._extract_pdf_text(path)
            if pdf_text is None:
                issues.append(
                    DocumentLoadIssue(
                        file_name=path.name,
                        path=path,
                        reason="PDF text extraction failed.",
                        source_type="pdf",
                    )
                )
                continue

            normalized_text = self._normalize_extracted_text(pdf_text)
            if len(normalized_text) < self.pdf_min_text_characters:
                issues.append(
                    DocumentLoadIssue(
                        file_name=path.name,
                        path=path,
                        reason="PDF extracted too little usable text for retrieval.",
                        source_type="pdf",
                    )
                )
                continue

            documents.append(
                DocumentRecord(
                    document_id=path.stem,
                    file_name=path.name,
                    path=path,
                    text=normalized_text,
                    source_type="pdf",
                )
            )

        return documents, issues

    def _read_plain_text(self, path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def _extract_pdf_text(self, path: Path) -> str | None:
        try:
            reader = PdfReader(str(path))
        except Exception:
            return None

        pages: list[str] = []
        try:
            for page_number, page in enumerate(reader.pages, start=1):
                extracted = page.extract_text() or ""
                extracted = extracted.strip()
                if extracted:
                    pages.append(f"## Page {page_number}\n{extracted}")
        except Exception:
            return None

        if not pages:
            return None

        return "\n\n".join(pages)

    def _normalize_extracted_text(self, text: str) -> str:
        return "\n".join(line.rstrip() for line in text.splitlines()).strip()

    def _source_type_for_extension(self, extension: str) -> str | None:
        if extension == self.MARKDOWN_EXTENSION:
            return "markdown"
        if extension == self.TEXT_EXTENSION:
            return "text"
        if extension == self.PDF_EXTENSION:
            return "pdf"
        return None
