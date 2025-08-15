from typing import List
import re

# Tokenization (fallback if tiktoken unavailable)
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(s: str) -> int:
        return len(_enc.encode(s))
except Exception:
    def count_tokens(s: str) -> int:
        # naive fallback
        return max(1, math.ceil(len(s) / 4))

# -------- Chunking --------
def sentence_split(text: str) -> List[str]:
    # Lightweight sentence split by punctuation
    parts = re.split(r"(?<=[。！？.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]

def chunk_text(
    text: str,
    max_tokens: int = 400,
    overlap: int = 50,
    min_sent_tokens: int = 5
) -> List[str]:
    sents = sentence_split(text)
    chunks = []
    buf = []
    buf_tokens = 0

    for s in sents:
        t = count_tokens(s)
        if t >= max_tokens:
            # Hard split overly long sentence
            for i in range(0, len(s), max_tokens * 4):
                sub = s[i:i + max_tokens * 4]
                chunks.append(sub)
            buf, buf_tokens = [], 0
            continue

        if buf_tokens + t <= max_tokens or (buf_tokens < min_sent_tokens):
            buf.append(s)
            buf_tokens += t
        else:
            chunks.append(" ".join(buf))
            # overlap
            if overlap > 0:
                tail = []
                tk = 0
                for ss in reversed(buf):
                    tk += count_tokens(ss)
                    tail.append(ss)
                    if tk >= overlap:
                        break
                buf = list(reversed(tail))
                buf_tokens = sum(count_tokens(x) for x in buf)
            else:
                buf, buf_tokens = [], 0
            buf.append(s)
            buf_tokens += t

    if buf:
        chunks.append(" ".join(buf))
    return [c.strip() for c in chunks if c.strip()]
