import os
import re
import json
import time
import math
import hashlib
import mimetypes
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Iterable

import requests
# from tenacity import retry, stop_after_attempt, wait_exponential

# Text loaders
import pdfplumber
import docx
import pandas as pd
import frontmatter
# -------- Data Models --------
@dataclass
class DocumentMeta:
    doc_id: str
    source_path: str
    source_type: str
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    version: Optional[str] = None
    content_sha256: Optional[str] = None

@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    metadata: Dict
    # embedding is not stored here; pushed directly to vector DB


# -------- Utils --------
def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def norm_ws(text: str) -> str:
    return re.sub(r"\s+\n", "\n", re.sub(r"[ \t]+", " ", text)).strip()


# -------- Loaders (pluggable) --------
class BaseLoader:
    def load(self, path: str) -> Dict:
        raise NotImplementedError

class PdfLoader(BaseLoader):
    def load(self, path: str) -> Dict:
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append({"page": i, "text": norm_ws(text)})
        full_text = "\n\n".join([p["text"] for p in pages])
        return {
            "text": full_text,
            "pages": pages,
            "meta": {
                "page_count": len(pages),
            }
        }

class DocxLoader(BaseLoader):
    def load(self, path: str) -> Dict:
        d = docx.Document(path)
        paras = [norm_ws(p.text) for p in d.paragraphs if p.text and p.text.strip()]
        text = "\n\n".join(paras)
        return {"text": text, "pages": None, "meta": {"paragraphs": len(paras)}}

class MarkdownLoader(BaseLoader):
    def load(self, path: str) -> Dict:
        post = frontmatter.load(path)
        text = post.content
        meta = dict(post.metadata) if post.metadata else {}
        return {"text": norm_ws(text), "pages": None, "meta": {"frontmatter": meta}}

class ExcelLoader(BaseLoader):
    def load(self, path: str) -> Dict:
        # Simple approach: concatenate each sheet to a textual table
        xls = pd.read_excel(path, sheet_name=None, dtype=str)
        parts = []
        for sheet_name, df in xls.items():
            df = df.fillna("")
            # Limit extremely wide sheets for readability
            snippet = df.to_string(index=False)
            parts.append(f"# Sheet: {sheet_name}\n{snippet}")
        text = "\n\n".join(parts)
        return {"text": norm_ws(text), "pages": None, "meta": {"sheets": list(xls.keys())}}

def pick_loader(path: str) -> BaseLoader:
    ext = os.path.splitext(path.lower())[1]
    if ext in [".pdf"]:
        return PdfLoader()
    if ext in [".docx"]:
        return DocxLoader()
    if ext in [".md", ".markdown"]:
        return MarkdownLoader()
    if ext in [".xlsx", ".xls"]:
        return ExcelLoader()
    # Fallback: read as plain text
    return PlainTextLoader()

class PlainTextLoader(BaseLoader):
    def load(self, path: str) -> Dict:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return {"text": norm_ws(text), "pages": None, "meta": {}}
    
