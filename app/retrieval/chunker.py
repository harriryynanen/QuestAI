from app.models import DocumentChunk, DocumentRecord


class MarkdownChunker:
    def __init__(self, max_characters: int) -> None:
        self.max_characters = max_characters

    def chunk_documents(self, documents: list[DocumentRecord]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for document in documents:
            chunks.extend(self._chunk_document(document))
        return chunks

    def _chunk_document(self, document: DocumentRecord) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        current_heading: str | None = None
        buffer: list[str] = []
        chunk_index = 1

        for line in document.text.splitlines():
            stripped_line = line.strip()

            if stripped_line.startswith("#"):
                chunk_index = self._flush_buffer(
                    chunks=chunks,
                    document=document,
                    section_heading=current_heading,
                    buffer=buffer,
                    chunk_index=chunk_index,
                )
                current_heading = stripped_line.lstrip("#").strip() or None
                continue

            if not stripped_line:
                chunk_index = self._flush_buffer(
                    chunks=chunks,
                    document=document,
                    section_heading=current_heading,
                    buffer=buffer,
                    chunk_index=chunk_index,
                )
                continue

            buffer.append(stripped_line)
            if len(" ".join(buffer)) >= self.max_characters:
                chunk_index = self._flush_buffer(
                    chunks=chunks,
                    document=document,
                    section_heading=current_heading,
                    buffer=buffer,
                    chunk_index=chunk_index,
                )

        self._flush_buffer(
            chunks=chunks,
            document=document,
            section_heading=self._resolved_heading(document, current_heading),
            buffer=buffer,
            chunk_index=chunk_index,
        )
        return chunks

    def _flush_buffer(
        self,
        chunks: list[DocumentChunk],
        document: DocumentRecord,
        section_heading: str | None,
        buffer: list[str],
        chunk_index: int,
    ) -> int:
        if not buffer:
            return chunk_index

        chunk_text = " ".join(buffer).strip()
        while chunk_text:
            piece = chunk_text[: self.max_characters].strip()
            if len(chunk_text) > self.max_characters:
                split_at = piece.rfind(" ")
                if split_at > 0:
                    piece = piece[:split_at].strip()

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.document_id}-chunk-{chunk_index}",
                    document_id=document.document_id,
                    file_name=document.file_name,
                    text=piece,
                    section_heading=self._resolved_heading(document, section_heading),
                )
            )
            chunk_index += 1
            chunk_text = chunk_text[len(piece):].strip()

        buffer.clear()
        return chunk_index

    def _resolved_heading(
        self,
        document: DocumentRecord,
        section_heading: str | None,
    ) -> str | None:
        if section_heading:
            return section_heading
        if document.source_type == "pdf":
            return "Extracted text"
        return None
