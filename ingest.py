"""
Ingestion + chunking pipeline for The Unofficial Guide.

  - Loads every PDF in documents/ with pdfplumber
  - Saves the raw extracted text to data/raw/ (consistent format, before cleaning)
  - Cleans each document and reconstructs it into records:
        * RateMyProfessors -> one record per review (+ a ratings-summary record)
        * Reddit            -> one record for the post + one per comment
  - Chunks records with a recursive splitter (~800 chars / ~100 overlap)
  - Attaches metadata (source file, title, URL, professor) to every chunk
  - Saves chunks to data/chunks.json and prints samples for inspection

Run:  python ingest.py
"""

import json
import os
import re

import pdfplumber

# --- paths -----------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(ROOT, "documents")
RAW_DIR = os.path.join(ROOT, "data", "raw")
CHUNKS_PATH = os.path.join(ROOT, "data", "chunks.json")

# --- chunking parameters (from planning.md: recursive, ~800 / ~100) --------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SCHOOL = "NYU Tandon, Computer Engineering"


# ---------------------------------------------------------------------------
# 1. LOAD: read each PDF into raw text and persist it before cleaning
# ---------------------------------------------------------------------------
def load_raw(pdf_path):
    """Extract all text from a PDF with pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages).strip()


def save_raw(name, text):
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(os.path.join(RAW_DIR, name + ".txt"), "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# 2. CLEAN + RECONSTRUCT RECORDS
#    pdfplumber inserts a newline at every visual line-wrap, which breaks
#    reviews/comments mid-sentence. We rejoin those soft wraps and rebuild
#    one record per review (RMP) or post/comment (Reddit).
# ---------------------------------------------------------------------------
def _pop_source(lines):
    """Return (url, remaining_lines) by extracting the 'Source: <url>' line."""
    url = ""
    rest = []
    for ln in lines:
        if ln.lower().startswith("source:"):
            url = ln.split(":", 1)[1].strip()
        else:
            rest.append(ln)
    return url, rest


def parse_ratemyprofessors(lines):
    """One record per review, plus a ratings-summary record. Each record is
    prefixed with the professor's name so the chunk is self-contained."""
    title = lines[0].strip()
    professor = title.split(" - ")[0].strip()

    # Split header (dept / rating / courses) from the reviews.
    try:
        rev_start = lines.index("Student Reviews") + 1
    except ValueError:
        rev_start = 1
    header = [ln.strip() for ln in lines[1:rev_start - 1] if ln.strip()]
    review_lines = lines[rev_start:]

    # Rebuild reviews: a new review begins on a line starting with a quote;
    # everything else is a soft-wrapped continuation of the current review.
    reviews, current = [], ""
    for ln in review_lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith('"') and current:
            reviews.append(current.strip())
            current = ln
        else:
            current = (current + " " + ln).strip() if current else ln
    if current:
        reviews.append(current.strip())

    summary = f"Professor {professor} ({SCHOOL}). " + " ".join(header)
    records = [summary]
    records += [f"Professor {professor} student review: {r}" for r in reviews]
    return professor, records


def parse_reddit(lines):
    """One record for the original post, one per comment. Soft wraps rejoined."""
    title = lines[0].strip()

    def section_index(label):
        for i, ln in enumerate(lines):
            if ln.strip() == label:
                return i
        return -1

    op_i, com_i = section_index("Original Post"), section_index("Comments")
    op_lines = lines[op_i + 1:com_i] if op_i >= 0 else []
    op_text = " ".join(ln.strip() for ln in op_lines if ln.strip())

    records = []
    if op_text:
        records.append(f"{title} - Original post: {op_text}")

    # Comments: an author line is short and ends with ':'. Following lines are
    # the comment body until the next author line.
    author_re = re.compile(r"^[A-Za-z0-9_][\w ,.()'/-]{0,60}:$")
    author, body = None, ""

    def flush():
        nonlocal author, body
        if author and body.strip():
            records.append(f"{title} - Comment by {author}: {body.strip()}")
        author, body = None, ""

    for ln in lines[com_i + 1:] if com_i >= 0 else []:
        s = ln.strip()
        if not s:
            continue
        if author_re.match(s):
            flush()
            author = s[:-1]
        elif author:
            body = (body + " " + s).strip()
    flush()
    return None, records


def to_records(name, text):
    """Dispatch on document type and return (professor, [record strings])."""
    lines = text.split("\n")
    _, lines = _pop_source([ln for ln in lines])
    url = ""
    for ln in text.split("\n"):
        if ln.lower().startswith("source:"):
            url = ln.split(":", 1)[1].strip()
            break
    title = lines[0].strip() if lines else name

    if "RateMyProfessors" in title:
        professor, records = parse_ratemyprofessors(lines)
        doc_type = "ratemyprofessors"
    else:
        professor, records = parse_reddit(lines)
        doc_type = "reddit"

    meta = {"source": name, "title": title, "url": url, "type": doc_type,
            "professor": professor}
    return meta, records


# ---------------------------------------------------------------------------
# 3. CHUNK: recursive splitter, respecting record boundaries first
#    Same algorithm as LangChain's RecursiveCharacterTextSplitter, implemented
#    directly so the pipeline has no heavy transformers/torch dependency:
#    try to split on the largest natural separator (blank line, then newline,
#    then sentence, then space); only fall back to a smaller one when a piece
#    still exceeds chunk_size. Adjacent small pieces are merged up to the size
#    limit, carrying chunk_overlap characters of context between chunks.
# ---------------------------------------------------------------------------
class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size, chunk_overlap, separators):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text):
        return self._split(text, self.separators)

    def _split(self, text, separators):
        # pick the first separator that occurs in the text
        sep, rest = separators[-1], []
        for i, s in enumerate(separators):
            if s == "":
                sep, rest = "", []
                break
            if s in text:
                sep, rest = s, separators[i + 1:]
                break

        splits = list(text) if sep == "" else text.split(sep)
        final, buffer = [], []
        for piece in splits:
            if len(piece) <= self.chunk_size:
                buffer.append(piece)
            else:
                final.extend(self._merge(buffer, sep))
                buffer = []
                # piece too big: recurse with finer separators
                final.extend(self._split(piece, rest) if rest else [piece])
        final.extend(self._merge(buffer, sep))
        return [c for c in final if c.strip()]

    def _merge(self, splits, sep):
        """Greedily join small splits into chunks <= chunk_size, with overlap."""
        sep_len = len(sep)
        chunks, current, total = [], [], 0
        for piece in splits:
            extra = sep_len if current else 0
            if total + len(piece) + extra > self.chunk_size and current:
                chunks.append(sep.join(current).strip())
                # drop from the front until we're under the overlap budget
                while total > self.chunk_overlap and current:
                    total -= len(current[0]) + (sep_len if len(current) > 1 else 0)
                    current.pop(0)
            current.append(piece)
            total += len(piece) + (sep_len if len(current) > 1 else 0)
        if current:
            chunks.append(sep.join(current).strip())
        return chunks


def build_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_records(meta, records, splitter):
    """Most records are short enough to stay whole; only long comments split."""
    chunks = []
    for record in records:
        for piece in splitter.split_text(record):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append({"text": piece, "metadata": dict(meta)})
    return chunks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    pdf_files = sorted(f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf"))
    print(f"Found {len(pdf_files)} PDFs in {DOCS_DIR}\n")

    splitter = build_splitter()
    all_chunks = []

    for fname in pdf_files:
        name = os.path.splitext(fname)[0]
        raw = load_raw(os.path.join(DOCS_DIR, fname))
        save_raw(name, raw)                       # persist before cleaning
        meta, records = to_records(name, raw)
        chunks = chunk_records(meta, records, splitter)
        all_chunks.extend(chunks)
        print(f"  {fname:48s} -> {len(records):3d} records, {len(chunks):3d} chunks")

    # attach a stable per-document chunk index
    counters = {}
    for c in all_chunks:
        src = c["metadata"]["source"]
        counters[src] = counters.get(src, 0)
        c["metadata"]["chunk_index"] = counters[src]
        counters[src] += 1

    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # --- inspection -------------------------------------------------------
    print(f"\nTotal chunks: {len(all_chunks)}  (saved to {os.path.relpath(CHUNKS_PATH, ROOT)})")
    lengths = [len(c["text"]) for c in all_chunks]
    print(f"Chunk length: min {min(lengths)}, max {max(lengths)}, "
          f"avg {sum(lengths) // len(lengths)} chars")

    print("\n--- 5 sample chunks ---")
    step = max(1, len(all_chunks) // 5)
    for c in all_chunks[::step][:5]:
        m = c["metadata"]
        print(f"\n[{m['source']} #{m['chunk_index']}]  ({len(c['text'])} chars)")
        print(c["text"])


if __name__ == "__main__":
    main()
