import re
from typing import List, Tuple
from langchain_core.documents import Document


class PIIMasker:
    """
    Identifies and masks Personally Identifiable Information (PII)
    prior to vector embedding and storage.
    """

    PATTERNS = [
        # Email addresses
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL_REDACTED]"),
        # Phone numbers (US/Intl formats)
        (r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]"),
        # Social Security Numbers (SSN)
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),
        # Credit / Debit Card Numbers (13 to 19 digits with optional hyphens/spaces)
        (r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{15,16}\b", "[CREDIT_CARD_REDACTED]"),
        # Staff / Employee ID codes (e.g., Staff ID: AJ-4421, ID: EMP12345)
        (r"(?i)\b(?:staff\s*id|employee\s*id|emp\s*id)[\s:]+([A-Z0-9-]+)\b", "Staff ID: [ID_REDACTED]"),
        # IPv4 addresses
        (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP_REDACTED]"),
    ]

    def mask_text(self, text: str) -> Tuple[str, bool]:
        """
        Masks PII patterns in text. Returns (masked_text, pii_detected).
        """
        if not text:
            return "", False

        masked = text
        pii_detected = False

        for pattern, replacement in self.PATTERNS:
            new_text, count = re.subn(pattern, replacement, masked)
            if count > 0:
                pii_detected = True
                masked = new_text

        return masked, pii_detected

    def mask_document(self, doc: Document) -> Document:
        """Masks PII inside a single Document and attaches audit metadata."""
        masked_content, detected = self.mask_text(doc.page_content)
        metadata = dict(doc.metadata)
        if detected:
            metadata["has_masked_pii"] = True

        return Document(page_content=masked_content, metadata=metadata)

    def mask_documents(self, documents: List[Document]) -> List[Document]:
        """Masks PII across a list of Documents."""
        return [self.mask_document(doc) for doc in documents]
