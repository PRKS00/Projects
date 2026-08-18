import json
from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, CSVLoader


class UniversalDocumentLoader:
    """
    Universal document loader supporting PDF, Markdown, TXT, CSV, and JSON files
    with standardized metadata extraction.
    """

    def __init__(self, raw_data_dir: Union[str, Path] = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)

    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Loads a single file and converts it into LangChain Document objects."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        documents: List[Document] = []

        if ext == ".pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(path))
                documents = loader.load()
            except ImportError:
                # Fallback if pypdf is not available
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                documents = [Document(page_content=content, metadata={"source": str(path), "page": 1})]

        elif ext in [".txt", ".md", ".markdown"]:
            loader = TextLoader(str(path), encoding="utf-8")
            documents = loader.load()

        elif ext == ".csv":
            loader = CSVLoader(str(path), encoding="utf-8")
            documents = loader.load()

        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    content = json.dumps(item, indent=2) if isinstance(item, dict) else str(item)
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": str(path), "record_index": idx}
                        )
                    )
            else:
                documents.append(
                    Document(
                        page_content=json.dumps(data, indent=2),
                        metadata={"source": str(path)}
                    )
                )
        else:
            # Fallback text load
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            documents = [Document(page_content=content, metadata={"source": str(path)})]

        # Normalize metadata across all loaded documents
        for doc in documents:
            doc.metadata.setdefault("source", str(path.name))
            doc.metadata["filename"] = path.name
            doc.metadata["extension"] = ext
            doc.metadata.setdefault("category", "general_policy")
            doc.metadata.setdefault("version", "1.0")

        return documents

    def load_all(self) -> List[Document]:
        """Loads all supported files found in the raw_data_dir."""
        all_docs: List[Document] = []
        if not self.raw_data_dir.exists():
            return all_docs

        for file_path in self.raw_data_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                docs = self.load_file(file_path)
                all_docs.extend(docs)

        return all_docs
