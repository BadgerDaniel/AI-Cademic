"""
Ingest the RAG papers with Baidu Unlimited-OCR-2 and push them to Pinecone.

Three resumable stages, each reading the previous stage's files:

  1. ocr    PDF -> page images -> OCR text per page   (ocr_output/<paper>/pages.json)
  2. chunk  pages -> chunks with file/page/section     (ocr_output/chunks.jsonl)
  3. embed  chunks -> text-embedding-3-small -> Pinecone namespace + dicembed_uocr.json

The live app (UI_CADEMIC_v4.4_Alpha.py) reads the default namespace and dicembed.json,
neither of which this script touches. Everything goes to a separate namespace
("uocr" by default) and a separate JSON file, so the two can be compared before switching.

Usage
-----
    # Stage 1 needs the GPU environment (see requirements-ocr.txt). On a 12 GB card,
    # keep --pages-per-batch small.
    python ingest_unlimited_ocr.py ocr --pages-per-batch 4

    # Baseline with no GPU: use the PDF's embedded text layer instead of the model.
    # Useful for testing the pipeline and as an evaluation comparison.
    python ingest_unlimited_ocr.py ocr --backend pdftext --out ocr_output_pdftext

    # SGLang server instead of in-process Transformers (Linux/WSL2 only).
    python ingest_unlimited_ocr.py ocr --backend sglang --server-url http://127.0.0.1:10000

    python ingest_unlimited_ocr.py chunk
    python ingest_unlimited_ocr.py embed --dry-run        # no API calls, prints a summary
    python ingest_unlimited_ocr.py embed --namespace uocr  # needs OPENAI_API_KEY, PINECONE_API_KEY

    python ingest_unlimited_ocr.py all  # all three stages in order

Model usage follows the model card: https://huggingface.co/baidu/Unlimited-OCR-2
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = REPO_ROOT / "Academic_Papers_for_RAG"
DEFAULT_OUT_DIR = REPO_ROOT / "ocr_output"
DEFAULT_DICT_PATH = REPO_ROOT / "dicembed_uocr.json"
MODEL_NAME = "baidu/Unlimited-OCR-2"
EMBED_MODEL = "text-embedding-3-small"  # must match the app's query embedding (1536 dims)
INDEX_NAME = "ai-cademic"
PAGE_MARKER = "<PAGE>"

# PDFs in the folder that are not source papers (image_desc2.pdf is generated output
# from the old image-description notebooks).
DEFAULT_SKIP = {"image_desc2.pdf"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def slugify(name: str) -> str:
    """ASCII-only, Pinecone-safe ID fragment."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return s or "doc"


def list_pdfs(pdf_dir: Path, only: list[str] | None, skip: set[str]) -> list[Path]:
    pdfs = sorted(p for p in pdf_dir.glob("*.pdf") if p.name not in skip)
    if only:
        wanted = {o.lower() for o in only}
        pdfs = [p for p in pdfs if p.name.lower() in wanted or p.stem.lower() in wanted]
    return pdfs


def render_pages(pdf_path: Path, img_dir: Path, dpi: int) -> list[Path]:
    """Render each page to PNG (cached). Mirrors the model card's pdf_to_images."""
    import fitz  # PyMuPDF

    img_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with fitz.open(pdf_path) as doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for i, page in enumerate(doc):
            p = img_dir / f"page_{i + 1:04d}.png"
            if not p.exists():
                page.get_pixmap(matrix=mat).save(str(p))
            paths.append(p)
    return paths


def split_pages(text: str, expected: int) -> list[str]:
    """
    Split multi-page output on <PAGE> markers. The model card says there is one
    marker per input page, in input order. If the count doesn't match, keep the
    whole batch on its first page and warn rather than misattribute pages.
    """
    parts = text.split(PAGE_MARKER)
    if parts and not parts[0].strip():
        parts = parts[1:]
    if len(parts) == expected:
        return [p.strip() for p in parts]
    log(f"    ! expected {expected} pages, found {len(parts)} <PAGE> segments; "
        f"keeping the batch as one block on its first page")
    return [text.replace(PAGE_MARKER, "\n\n").strip()] + [""] * (expected - 1)


# The model descends from DeepSeek-OCR, which can emit grounding tags such as
# <|ref|>Title<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>. That format is a guess for this
# model, not documented. Raw output is always saved, so this cleanup can be changed
# and the chunk stage re-run without re-running OCR.
_DET_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.S)
_TAG_RE = re.compile(r"<\|/?(?:ref|grounding)\|>")
_EOS_RE = re.compile(r"<\|end▁of▁sentence\|>|</s>")


def clean_ocr_text(text: str) -> str:
    text = _DET_RE.sub("", text)
    text = _TAG_RE.sub(" ", text)
    text = _EOS_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------

class TransformersBackend:
    """In-process inference, as on the model card. Works on native Windows."""

    def __init__(self, infer_kwargs: dict):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            sys.exit("CUDA is not available to torch. Install a CUDA build of torch "
                     "(see requirements-ocr.txt) or use --backend pdftext.")
        log(f"Loading {MODEL_NAME} on {torch.cuda.get_device_name(0)} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            MODEL_NAME, trust_remote_code=True, use_safetensors=True,
            torch_dtype=torch.bfloat16,
        ).eval().cuda()
        self.infer_kwargs = infer_kwargs
        self.torch = torch

    def run(self, images: list[Path], raw_dir: Path) -> str:
        raw_dir.mkdir(parents=True, exist_ok=True)
        files = [str(p) for p in images]
        with self.torch.inference_mode():
            if len(files) == 1:
                out = self.model.infer(self.tokenizer, image_file=files[0],
                                       output_path=str(raw_dir), save_results=True,
                                       **self.infer_kwargs)
                text = out[0] if isinstance(out, tuple) else out
                # Single-page output may lack a marker; add one so splitting is uniform.
                if PAGE_MARKER not in (text or ""):
                    text = PAGE_MARKER + (text or "")
            else:
                text, _num_tokens = self.model.infer_multi(
                    self.tokenizer, image_files=files, output_path=str(raw_dir),
                    save_results=True, **self.infer_kwargs)
        self.torch.cuda.empty_cache()
        return text or ""


class SGLangBackend:
    """Client for the SGLang server from the model card (Linux / WSL2)."""

    def __init__(self, server_url: str, crop_mode: bool, tokens_per_page: int):
        import requests

        self.url = server_url.rstrip("/") + "/v1/chat/completions"
        self.crop_mode = crop_mode
        self.tokens_per_page = tokens_per_page
        self.session = requests.Session()
        self.session.trust_env = False
        try:
            from sglang.srt.sampling.custom_logit_processor import (
                DeepseekOCRNoRepeatNGramLogitProcessor,
            )
            self.ngram = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
        except ImportError:
            self.ngram = None
            log("  ! sglang isn't importable here, so the no-repeat n-gram processor is "
                "off. Output may loop on repetitive pages.")

    @staticmethod
    def _encode(path: Path) -> dict:
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}

    def run(self, images: list[Path], raw_dir: Path) -> str:
        prompt = PAGE_MARKER + "<image>"
        content = [self._encode(p) for p in images]
        content.append({"type": "text", "text": prompt * len(images) + "Multi page parsing."})
        payload = {
            "model": "Unlimited-OCR-2",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.tokens_per_page * len(images),
            "temperature": 0,
            "skip_special_tokens": False,
            "images_config": {"crop_mode": self.crop_mode},
            "stream": False,
        }
        if self.ngram:
            payload["custom_logit_processor"] = self.ngram
            payload["custom_params"] = {
                "ngram_size": 35,
                "window_size": 1024 if len(images) > 1 else 128,
                "page_token_id": 128827,
                "eos_token_id": 1,
                "max_page_tokens": 20000,
            }
        r = self.session.post(self.url, json=payload, timeout=6000)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


class PdfTextBackend:
    """No model: PyMuPDF's embedded text layer. Baseline and pipeline smoke test."""

    def __init__(self, pdf_path: Path):
        import fitz

        with fitz.open(pdf_path) as doc:
            self.pages = [page.get_text("text") for page in doc]

    def run_pages(self, page_numbers: list[int]) -> str:
        return "".join(PAGE_MARKER + self.pages[n - 1] for n in page_numbers)


# ---------------------------------------------------------------------------
# Stage 1: OCR
# ---------------------------------------------------------------------------

def stage_ocr(args) -> None:
    out_dir: Path = args.out
    pdfs = list_pdfs(args.pdf_dir, args.only, DEFAULT_SKIP | set(args.skip or []))
    if not pdfs:
        sys.exit(f"No PDFs found in {args.pdf_dir}")
    log(f"OCR backend={args.backend}, {len(pdfs)} PDFs -> {out_dir}")

    backend = None
    if args.backend == "transformers":
        backend = TransformersBackend(json.loads(args.infer_kwargs))
    elif args.backend == "sglang":
        backend = SGLangBackend(args.server_url, not args.base_mode, args.tokens_per_page)

    for pdf in pdfs:
        doc_dir = out_dir / slugify(pdf.stem)
        pages_file = doc_dir / "pages.json"
        if pages_file.exists() and not args.force:
            log(f"- {pdf.name}: already done, skipping (use --force to redo)")
            continue
        doc_dir.mkdir(parents=True, exist_ok=True)

        if args.backend == "pdftext":
            import fitz
            with fitz.open(pdf) as d:
                n_pages = d.page_count
            images = None
            text_backend = PdfTextBackend(pdf)
        else:
            images = render_pages(pdf, doc_dir / "images", args.dpi)
            n_pages = len(images)
        log(f"- {pdf.name}: {n_pages} pages")

        # Per-batch results are cached so an interrupted run (e.g. CUDA OOM) resumes.
        pages: list[dict] = []
        bs = max(1, args.pages_per_batch)
        for start in range(0, n_pages, bs):
            nums = list(range(start + 1, min(start + bs, n_pages) + 1))
            raw_file = doc_dir / f"raw_p{nums[0]:04d}-{nums[-1]:04d}.txt"
            if raw_file.exists() and not args.force:
                raw = raw_file.read_text(encoding="utf-8")
            else:
                t0 = time.time()
                if args.backend == "pdftext":
                    raw = text_backend.run_pages(nums)
                else:
                    raw = backend.run([images[n - 1] for n in nums],
                                      doc_dir / "model_output" / raw_file.stem)
                raw_file.write_text(raw, encoding="utf-8")
                log(f"    pages {nums[0]}-{nums[-1]} in {time.time() - t0:.1f}s")
            for n, page_text in zip(nums, split_pages(raw, len(nums))):
                pages.append({"page": n, "text": page_text})

        pages_file.write_text(json.dumps({
            "source": pdf.name,
            "backend": args.backend,
            "pages_per_batch": bs,
            "dpi": args.dpi,
            "pages": pages,
        }, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Stage 2: chunk
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")


def chunk_page(text: str, max_chars: int, section: str) -> tuple[list[tuple[str, str]], str]:
    """
    Split one page into (section, chunk_text) pieces.

    Paragraphs (blank-line separated) are packed up to max_chars. Tables and other
    long paragraphs stay whole unless they exceed 3x max_chars. The last paragraph
    of each chunk is repeated at the start of the next for overlap. Returns the
    section heading that is current at the end of the page, so it carries over to
    the next page.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[tuple[str, str]] = []
    buf: list[str] = []
    buf_section = section

    def flush():
        nonlocal buf
        if buf:
            chunks.append((buf_section, "\n\n".join(buf)))
            buf = buf[-1:] if len(buf) > 1 and len(buf[-1]) < max_chars // 2 else []

    for para in paras:
        m = _HEADING_RE.match(para.splitlines()[0])
        if m:
            flush()
            buf = []
            section = m.group(2).strip()
            buf_section = section
        if len(para) > 3 * max_chars:  # hard split huge blocks
            flush()
            for i in range(0, len(para), max_chars):
                chunks.append((section, para[i:i + max_chars]))
            buf = []
            continue
        if buf and sum(len(b) for b in buf) + len(para) > max_chars:
            flush()
            buf_section = section
        buf.append(para)
    if buf and (not chunks or buf != [chunks[-1][1].split("\n\n")[-1]]):
        chunks.append((buf_section, "\n\n".join(buf)))
    return chunks, section


def stage_chunk(args) -> None:
    out_dir: Path = args.out
    page_files = sorted(out_dir.glob("*/pages.json"))
    if not page_files:
        sys.exit(f"No pages.json files under {out_dir}. Run the ocr stage first.")

    records = []
    for pf in page_files:
        doc = json.loads(pf.read_text(encoding="utf-8"))
        slug = pf.parent.name
        section = ""
        for page in doc["pages"]:
            text = clean_ocr_text(page["text"])
            if len(text) < args.min_chars:
                continue
            pieces, section = chunk_page(text, args.max_chars, section)
            for ci, (sec, chunk) in enumerate(pieces):
                if len(chunk) < args.min_chars:
                    continue
                records.append({
                    "id": f"uocr-{slug}-p{page['page']:04d}-c{ci:02d}",
                    "source": doc["source"],
                    "page": page["page"],
                    "section": sec,
                    "text": chunk,
                })
        log(f"- {doc['source']}: {sum(r['source'] == doc['source'] for r in records)} chunks")

    chunks_file = out_dir / "chunks.jsonl"
    with chunks_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens = [len(r["text"]) for r in records] or [0]
    log(f"{len(records)} chunks -> {chunks_file} "
        f"(avg {sum(lens) // len(lens)} chars, max {max(lens)})")


# ---------------------------------------------------------------------------
# Stage 3: embed + upsert
# ---------------------------------------------------------------------------

def context_string(r: dict) -> str:
    """What the app pastes into the prompt: text plus a citation header."""
    head = f"[{r['source']}, p.{r['page']}" + (f", {r['section']}" if r["section"] else "") + "]"
    return f"{head}\n{r['text']}"


def stage_embed(args) -> None:
    chunks_file = args.out / "chunks.jsonl"
    if not chunks_file.exists():
        sys.exit(f"{chunks_file} not found. Run the chunk stage first.")
    records = [json.loads(line) for line in chunks_file.read_text(encoding="utf-8").splitlines() if line]

    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        sys.exit("Duplicate chunk IDs. Re-run the chunk stage.")

    if args.dry_run:
        total = sum(len(r["text"]) for r in records)
        log(f"[dry run] {len(records)} chunks, about {total // 4:,} tokens to embed, "
            f"namespace '{args.namespace}' in index '{INDEX_NAME}', dict -> {args.dict_path}")
        for r in records[:3]:
            log("-" * 60 + "\n" + context_string(r)[:400])
        return

    from openai import OpenAI
    from pinecone import Pinecone

    openai_key = os.environ.get("OPENAI_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not openai_key or not pinecone_key:
        sys.exit("Set OPENAI_API_KEY and PINECONE_API_KEY in the environment.")
    client = OpenAI(api_key=openai_key)
    index = Pinecone(api_key=pinecone_key).Index(INDEX_NAME)

    if args.namespace in ("", "__default__") and not args.allow_default_namespace:
        sys.exit("Refusing to write to the default namespace the live app uses. "
                 "Pass --allow-default-namespace if you really mean it.")

    dic = {}
    bs = args.batch_size
    for i in range(0, len(records), bs):
        batch = records[i:i + bs]
        texts = [context_string(r) for r in batch]
        resp = client.embeddings.create(input=texts, model=EMBED_MODEL)
        vectors = []
        for r, text, d in zip(batch, texts, resp.data):
            vectors.append({
                "id": r["id"],
                "values": d.embedding,
                # Pinecone allows 40 KB of metadata per vector; chunks are far smaller.
                "metadata": {"source": r["source"], "page": r["page"],
                             "section": r["section"][:200], "text": text},
            })
            dic[r["id"]] = text
        index.upsert(vectors=vectors, namespace=args.namespace)
        log(f"  upserted {i + len(batch)}/{len(records)}")

    args.dict_path.write_text(json.dumps(dic, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Done. {len(dic)} vectors in namespace '{args.namespace}'; dict -> {args.dict_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["ocr", "chunk", "embed", "all"])
    ap.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="working directory for stage outputs")
    ap.add_argument("--only", nargs="*", help="process only these PDFs (file name or stem)")
    ap.add_argument("--skip", nargs="*", help="extra PDF file names to skip")
    ap.add_argument("--force", action="store_true", help="redo OCR even if cached")

    g = ap.add_argument_group("ocr")
    g.add_argument("--backend", choices=["transformers", "sglang", "pdftext"], default="transformers")
    g.add_argument("--pages-per-batch", type=int, default=4,
                   help="pages per model call; lower this if you hit CUDA out-of-memory (default 4)")
    g.add_argument("--dpi", type=int, default=300)
    g.add_argument("--infer-kwargs", default="{}",
                   help='extra JSON kwargs for model.infer / infer_multi, e.g. \'{"crop_mode": false}\'')
    g.add_argument("--server-url", default="http://127.0.0.1:10000")
    g.add_argument("--base-mode", action="store_true", help="sglang: crop_mode=False (base, not gundam)")
    g.add_argument("--tokens-per-page", type=int, default=8000)

    g = ap.add_argument_group("chunk")
    g.add_argument("--max-chars", type=int, default=1800)
    g.add_argument("--min-chars", type=int, default=40)

    g = ap.add_argument_group("embed")
    g.add_argument("--namespace", default="uocr")
    g.add_argument("--allow-default-namespace", action="store_true")
    g.add_argument("--dict-path", type=Path, default=DEFAULT_DICT_PATH)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    if args.stage in ("ocr", "all"):
        stage_ocr(args)
    if args.stage in ("chunk", "all"):
        stage_chunk(args)
    if args.stage in ("embed", "all"):
        stage_embed(args)


if __name__ == "__main__":
    main()
