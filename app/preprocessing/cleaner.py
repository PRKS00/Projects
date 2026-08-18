import re
import hashlib
from typing import List, Set
from langchain_core.documents import Document


class DocumentCleaner:
    """
    Cleans raw document text, strips boilerplate headers/footers,
    normalizes whitespace, and deduplicates identical passages.
    """

    def __init__(self, deduplicate: bool = True):
        self.deduplicate = deduplicate
        self._seen_hashes: Set[str] = set()

    def clean_text(self, text: str) -> str:
        """Applies normalization rules to a single text string."""
        if not text:
            return ""

        # Remove control characters (except newline, tab)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Standardize markdown horizontal rules and separators
        text = re.sub(r"^[ \t]*[-*_]{3,}[ \t]*$", "---", text, flags=re.MULTILINE)

        # Remove repeated page header/footer patterns like "Page 1 of 12"
        text = re.sub(r"(?i)\bpage\s+\d+\s+(?:of|\/)\s+\d+\b", "", text)
        text = re.sub(r"(?i)confidential\s+-\s+internal\s+use\s+only", "", text)

        # Collapse excessive blank lines to max 2 newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Normalize trailing spaces on lines
        text = "\n".join(line.strip() for line in text.splitlines())

        return text.strip()

    def clean_document(self, doc: Document) -> Document:
        """Cleans the page_content of a single LangChain Document."""
        cleaned_content = self.clean_text(doc.page_content)
        return Document(
            page_content=cleaned_content,
            metadata=dict(doc.metadata)
        )

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """Cleans and optionally deduplicates a batch of documents."""
        cleaned_docs: List[Document] = []

        for doc in documents:
            cleaned_doc = self.clean_document(doc)
            if not cleaned_doc.page_content:
                continue

            if self.deduplicate:
                content_hash = hashlib.sha256(cleaned_doc.page_content.encode("utf-8")).hexdigest()
                if content_hash in self._seen_hashes:
                    continue
                self._seen_hashes.add(content_hash)

            cleaned_docs.append(cleaned_doc)

        return cleaned_docs
