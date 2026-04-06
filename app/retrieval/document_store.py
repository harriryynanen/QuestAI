from pathlib import Path

from models import DocumentInfo


class DocumentStore:
    SUPPORTED_EXTENSIONS = (".md", ".txt", ".pdf")

    def __init__(self, docs_path: Path) -> None:
        self.docs_path = docs_path

    def list_documents(self) -> list[DocumentInfo]:
        if not self.docs_path.exists() or not self.docs_path.is_dir():
            return []

        documents: list[DocumentInfo] = []
        for path in sorted(self.docs_path.iterdir()):
            if not path.is_file() or path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            documents.append(
                DocumentInfo(
                    file_name=path.name,
                    path=path,
                    extension=path.suffix.lower(),
                    size_bytes=path.stat().st_size,
                )
            )

        return documents
