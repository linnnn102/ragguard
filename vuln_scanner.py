"""
vuln_scanner.py — RAG-powered vulnerability scanner
=====================================================
Scans a Python file function-by-function using:
  - CWE/CVE knowledge base (Parquet + numpy from Colab)
  - nomic-embed-text for retrieval
  - Llama 3 via Ollama for analysis

Usage:
    python vuln_scanner.py <target.py> [options]

Options:
    --chunks-zip   PATH   Path to rag_chunks.zip from Colab  [default: ./rag_chunks.zip]
    --chunks-dir   PATH   Or point directly at an unzipped folder
    --output       PATH   JSON report output path            [default: ./vuln_report.json]
    --model        STR    Ollama model tag                   [default: llama3]
    --ollama-url   URL    Ollama base URL                    [default: http://localhost:11434]
    --min-severity STR    Skip findings below this level     [default: LOW]
    --top-k        INT    Chunks to retrieve per function    [default: 6]
    --no-color            Disable terminal colours
"""

import ast
import sys
import json
import time
import zipfile
import argparse
import textwrap
import datetime
import re
from pathlib import Path
from typing import Optional

import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── Terminal colours ───────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    ORANGE = "\033[38;5;208m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    GRAY   = "\033[90m"
    WHITE  = "\033[97m"

USE_COLOR = True

def col(text: str, *codes: str) -> str:
    if not USE_COLOR:
        return text
    return "".join(codes) + text + C.RESET

SEVERITY_COLOR = {
    "CRITICAL": C.RED,
    "HIGH":     C.ORANGE,
    "MEDIUM":   C.YELLOW,
    "LOW":      C.GREEN,
    "INFO":     C.CYAN,
}

SEVERITY_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}

# ── Knowledge base loader ──────────────────────────────────────────────────────

class KnowledgeBase:
    """Loads CWE/CVE chunks and embeddings from the Colab zip output."""

    def __init__(self, source: Path, embed_model: str = "nomic-ai/nomic-embed-text-v2-moe"):
        self.chunks_dir = self._resolve_dir(source)
        print(col(f"[KB] Loading knowledge base from {self.chunks_dir} ...", C.CYAN))

        self.df_cwe = pd.read_parquet(self.chunks_dir / "cwe_chunks.parquet")
        self.df_cve = pd.read_parquet(self.chunks_dir / "cve_chunks.parquet")
        self.emb_cwe = np.load(self.chunks_dir / "cwe_embeddings.npy")
        self.emb_cve = np.load(self.chunks_dir / "cve_embeddings.npy")

        manifest_path = self.chunks_dir / "manifest.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
            embed_model = self.manifest.get("embed_model", embed_model)

        print(col(f"[KB] CWE chunks: {len(self.df_cwe)} | CVE chunks: {len(self.df_cve)}", C.CYAN))
        print(col(f"[KB] Loading embedding model: {embed_model} ...", C.CYAN))
        self.embedder = SentenceTransformer(embed_model, trust_remote_code=True)
        print(col("[KB] Ready.", C.GREEN))

    def _resolve_dir(self, source: Path) -> Path:
        """Accept either a .zip file or an already-unzipped directory."""
        if source.suffix == ".zip":
            extract_to = source.parent / source.stem
            if not extract_to.exists():
                print(col(f"[KB] Unzipping {source} → {extract_to} ...", C.CYAN))
                with zipfile.ZipFile(source, "r") as zf:
                    zf.extractall(extract_to)
            # The zip may have an extra nesting level — find the actual chunks dir
            candidates = list(extract_to.rglob("cwe_chunks.parquet"))
            if not candidates:
                raise FileNotFoundError(
                    f"cwe_chunks.parquet not found inside {source}. "
                    "Make sure you downloaded the full rag_chunks.zip from Colab."
                )
            return candidates[0].parent
        return source

    def embed_query(self, text: str) -> np.ndarray:
        return self.embedder.encode(
            f"search_query: {text}",
            normalize_embeddings=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        min_severity: str = "LOW",
    ) -> list[dict]:
        """
        Retrieve top-k chunks relevant to query.
        """
        min_rank = SEVERITY_RANK.get(min_severity, 0)
        q_emb = self.embed_query(query)

        # Score CWE chunks
        cwe_scores = self.emb_cwe @ q_emb
        cwe_hits = sorted(
            [(float(s), "CWE", i) for i, s in enumerate(cwe_scores)],
            reverse=True,
        )[: top_k // 2 + 1]

        # Score CVE chunks with severity filter
        cve_scores = self.emb_cve @ q_emb
        cve_hits = []
        for i, s in enumerate(cve_scores):
            sev = self.df_cve.iloc[i].get("cvss_severity", "LOW")
            if SEVERITY_RANK.get(sev, 0) >= min_rank:
                cve_hits.append((float(s), "CVE", i))
        cve_hits = sorted(cve_hits, reverse=True)[: top_k // 2 + 1]

        # Interleave: definition first, then instance
        merged = []
        for cwe_h, cve_h in zip(cwe_hits, cve_hits):
            merged.append(cwe_h)
            merged.append(cve_h)
        merged = merged[:top_k]

        results = []
        for score, src, idx in merged:
            row = (self.df_cwe.iloc[idx] if src == "CWE" else self.df_cve.iloc[idx]).to_dict()
            row["_score"] = score
            row["_source"] = src
            results.append(row)
        return results

    def format_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            src = chunk["_source"]
            if src == "CWE":
                header = f"[Ref {i} | {chunk.get('chunk_id','')} | relevance {chunk['_score']:.2f}]"
            else:
                sev   = chunk.get("cvss_severity", "?")
                score = chunk.get("cvss_score", "?")
                header = (
                    f"[Ref {i} | {chunk.get('cve_id','')} | CVSS {score} {sev}"
                    f" | {chunk.get('cwe_id','')} | relevance {chunk['_score']:.2f}]"
                )
            parts.append(f"{header}\n{chunk['chunk_text']}")
        return "\n\n".join(parts)


# ── AST-based Python file parser ───────────────────────────────────────────────

class FunctionExtractor(ast.NodeVisitor):
    """Walks a Python AST and collects function definitions with metadata."""

    def __init__(self, source_lines: list[str]):
        self.source_lines = source_lines
        self.functions: list[dict] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._record(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._record(node)
        self.generic_visit(node)

    def _record(self, node):
        start = node.lineno - 1
        end   = node.end_lineno
        src   = "".join(self.source_lines[start:end])
        docstring = ast.get_docstring(node) or ""

        # Collect argument names for context
        args = [a.arg for a in node.args.args]

        self.functions.append({
            "name":      node.name,
            "lineno":    node.lineno,
            "end_lineno": node.end_lineno,
            "args":      args,
            "docstring": docstring,
            "source":    src,
        })


def extract_functions(file_path: Path) -> list[dict]:
    source = file_path.read_text(encoding="utf-8")
    lines  = source.splitlines(keepends=True)
    tree   = ast.parse(source, filename=str(file_path))
    extractor = FunctionExtractor(lines)
    extractor.visit(tree)
    return extractor.functions


# ── Ollama LLM client ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a professional application security analyst specializing in Python code review.
You identify software vulnerabilities by mapping code patterns to the MITRE CWE taxonomy
and real-world CVE examples provided in the context.

For each vulnerability you find, you respond ONLY with a valid JSON array.
Each element must have exactly these fields:
  - "cwe_id":      string, e.g. "CWE-89"
  - "cwe_name":    string, short name of the weakness
  - "severity":    string, one of CRITICAL / HIGH / MEDIUM / LOW / INFO
  - "confidence":  number between 0.0 and 1.0
  - "line_hint":   string, rough location in the function e.g. "line 3" or "argument handling"
  - "description": string, explain what the vulnerability is and why this code is vulnerable
  - "evidence":    string, the specific code snippet or pattern that is vulnerable
  - "solution":    string, concrete fix with corrected code example if possible
  - "references":  array of strings, relevant CVE IDs from the context if any

If the function has NO vulnerabilities, return an empty array: []
Do not include any text outside the JSON array.\
"""

def build_prompt(fn: dict, context: str) -> str:
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Analyze the following Python function for security vulnerabilities.\n"
        f"Use the CWE/CVE reference context below to ground your findings.\n\n"
        f"--- FUNCTION: {fn['name']} (line {fn['lineno']}) ---\n"
        f"```python\n{fn['source']}\n```\n\n"
        f"--- CWE/CVE REFERENCE CONTEXT ---\n{context}\n\n"
        f"Respond ONLY with a JSON array of findings (or [] if none)."
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self._check_connection()

    def _check_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model in m for m in models):
                print(col(
                    f"[WARN] Model '{self.model}' not found in Ollama. "
                    f"Available: {models}\n"
                    f"       Run: ollama pull {self.model}",
                    C.YELLOW,
                ))
            else:
                print(col(f"[Ollama] Connected — model: {self.model}", C.GREEN))
        except requests.ConnectionError:
            print(col(
                f"[ERROR] Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (`ollama serve`).",
                C.RED,
            ))
            sys.exit(1)

    def generate(self, prompt: str, max_retries: int = 2) -> str:
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature":        0.1,
                "repeat_penalty":     1.15,
                "num_predict":        1024,
                "stop": ["<|eot_id|>", "<|end_of_text|>"],
            },
        }
        for attempt in range(max_retries + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=120,
                )
                r.raise_for_status()
                return r.json().get("response", "")
            except requests.Timeout:
                if attempt < max_retries:
                    print(col(f"  [Ollama] Timeout, retrying ({attempt+1}/{max_retries})...", C.YELLOW))
                    time.sleep(2)
                else:
                    raise


# ── JSON extraction from LLM output ───────────────────────────────────────────

def extract_json_array(text: str) -> list[dict]:
    """
    Robustly extract a JSON array from LLM output.
    Handles markdown code fences, leading/trailing text, and partial JSON.
    """
    text = text.strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find the first [...] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Nothing parseable — return empty
    return []


def validate_finding(f: dict) -> dict:
    """Normalise and fill in missing fields in a finding dict."""
    severity = str(f.get("severity", "LOW")).upper()
    if severity not in SEVERITY_RANK:
        severity = "LOW"
    confidence = float(f.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    return {
        "cwe_id":      str(f.get("cwe_id", "CWE-Unknown")),
        "cwe_name":    str(f.get("cwe_name", "")),
        "severity":    severity,
        "confidence":  round(confidence, 2),
        "line_hint":   str(f.get("line_hint", "")),
        "description": str(f.get("description", "")),
        "evidence":    str(f.get("evidence", "")),
        "solution":    str(f.get("solution", "")),
        "references":  list(f.get("references", [])),
    }


# ── Report printer ─────────────────────────────────────────────────────────────

def severity_badge(severity: str) -> str:
    sev_col = SEVERITY_COLOR.get(severity, C.WHITE)
    return col(f" {severity} ", C.BOLD, sev_col)

def confidence_bar(conf: float) -> str:
    filled = int(conf * 10)
    bar = "█" * filled + "░" * (10 - filled)
    pct = f"{conf*100:.0f}%"
    if conf >= 0.8:
        return col(f"{bar} {pct}", C.GREEN)
    elif conf >= 0.5:
        return col(f"{bar} {pct}", C.YELLOW)
    else:
        return col(f"{bar} {pct}", C.GRAY)

def print_report(
    target_file: Path,
    results: list[dict],
    total_functions: int,
    elapsed: float,
):
    divider     = col("─" * 70, C.GRAY)
    big_divider = col("═" * 70, C.GRAY)

    print()
    print(big_divider)
    print(col("  VULNERABILITY SCAN REPORT", C.BOLD, C.WHITE))
    print(big_divider)
    print(f"  Target  : {col(str(target_file), C.CYAN)}")
    print(f"  Scanned : {total_functions} functions")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Date    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(big_divider)

    all_findings = [f for r in results for f in r["findings"]]
    if not all_findings:
        print(col("\n  ✓ No vulnerabilities detected.\n", C.GREEN, C.BOLD))
        return

    # Summary counts per severity
    from collections import Counter
    counts = Counter(f["severity"] for f in all_findings)
    print("\n  SUMMARY")
    print(divider)
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        if counts.get(sev, 0):
            print(f"  {severity_badge(sev)}  {counts[sev]} finding(s)")
    print()

    # Per-function findings
    for fn_result in results:
        if not fn_result["findings"]:
            continue

        fn = fn_result["function"]
        print(divider)
        print(
            col(f"  ƒ {fn['name']}", C.BOLD, C.WHITE)
            + col(f"  (line {fn['lineno']}–{fn['end_lineno']})", C.GRAY)
        )
        print()

        for i, finding in enumerate(fn_result["findings"], 1):
            sev  = finding["severity"]
            conf = finding["confidence"]

            print(f"  [{i}] {severity_badge(sev)}  "
                  f"{col(finding['cwe_id'], C.BOLD)}  {col(finding['cwe_name'], C.WHITE)}")
            print(f"       Confidence : {confidence_bar(conf)}")
            if finding["line_hint"]:
                print(f"       Location   : {col(finding['line_hint'], C.GRAY)}")
            print()

            # Description
            desc_lines = textwrap.wrap(finding["description"], width=64)
            print(col("  Description:", C.BOLD))
            for line in desc_lines:
                print(f"    {line}")
            print()

            # Evidence
            if finding["evidence"]:
                print(col("  Evidence:", C.BOLD))
                for line in finding["evidence"].splitlines():
                    print(col(f"    {line}", C.YELLOW))
                print()

            # Solution
            if finding["solution"]:
                print(col("  Solution:", C.BOLD))
                sol_lines = finding["solution"].splitlines()
                for line in sol_lines:
                    print(f"    {line}")
                print()

            # References
            if finding["references"]:
                refs = ", ".join(finding["references"][:5])
                print(col("  References:", C.BOLD) + f" {col(refs, C.GRAY)}")
                print()

    print(big_divider)
    total = len(all_findings)
    critical_high = counts.get("CRITICAL", 0) + counts.get("HIGH", 0)
    print(f"  {col(str(total), C.BOLD)} total finding(s) — "
          f"{col(str(critical_high), C.RED, C.BOLD)} critical/high priority")
    print(big_divider)
    print()


# ── JSON report writer ─────────────────────────────────────────────────────────

def save_json_report(
    target_file: Path,
    results: list[dict],
    output_path: Path,
    elapsed: float,
    model: str,
):
    all_findings = [f for r in results for f in r["findings"]]
    from collections import Counter
    counts = Counter(f["severity"] for f in all_findings)

    report = {
        "meta": {
            "tool":        "vuln_scanner",
            "version":     "1.0.0",
            "target":      str(target_file.resolve()),
            "model":       model,
            "scan_date":   datetime.datetime.utcnow().isoformat() + "Z",
            "duration_s":  round(elapsed, 2),
        },
        "summary": {
            "total_functions": sum(1 for r in results),
            "functions_with_findings": sum(1 for r in results if r["findings"]),
            "total_findings": len(all_findings),
            "by_severity": {
                sev: counts.get(sev, 0)
                for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
            },
        },
        "results": [
            {
                "function":  r["function"]["name"],
                "file":      str(target_file),
                "line_start": r["function"]["lineno"],
                "line_end":   r["function"]["end_lineno"],
                "findings":   r["findings"],
            }
            for r in results
        ],
    }

    output_path.write_text(json.dumps(report, indent=2))
    print(col(f"[Report] JSON saved → {output_path}", C.CYAN))
    return report


# ── Main scanner ───────────────────────────────────────────────────────────────

def scan_file(
    target: Path,
    kb: KnowledgeBase,
    llm: OllamaClient,
    top_k: int = 6,
    min_severity: str = "LOW",
) -> list[dict]:
    """
    Scan every function in target file.
    Returns list of {function, findings} dicts.
    """
    print(col(f"\n[Scanner] Extracting functions from {target.name} ...", C.CYAN))
    functions = extract_functions(target)

    if not functions:
        print(col("[Scanner] No functions found in file.", C.YELLOW))
        return []

    print(col(f"[Scanner] Found {len(functions)} function(s). Starting analysis...\n", C.CYAN))

    results = []
    for idx, fn in enumerate(functions, 1):
        fn_label = col(f"ƒ {fn['name']}", C.BOLD)
        fn_lineno = fn['lineno']
        print(f"  [{idx}/{len(functions)}] {fn_label} "
              f"{col(f'(line {fn_lineno})', C.GRAY)} ... ", end="", flush=True)

        # Build RAG query from function signature + docstring + source keywords
        rag_query = (
            f"Python function '{fn['name']}' vulnerability analysis. "
            f"Arguments: {', '.join(fn['args'])}. "
            f"{fn['docstring'][:200] if fn['docstring'] else ''} "
            f"{fn['source'][:400]}"
        )

        # Retrieve relevant CWE/CVE context
        chunks  = kb.retrieve(rag_query, top_k=top_k, min_severity=min_severity)
        context = kb.format_context(chunks)

        # Build prompt and query LLM
        prompt = build_prompt(fn, context)
        try:
            raw_response = llm.generate(prompt)
        except Exception as e:
            print(col(f"ERROR ({e})", C.RED))
            results.append({"function": fn, "findings": [], "error": str(e)})
            continue

        # Parse LLM response
        raw_findings = extract_json_array(raw_response)
        findings     = [validate_finding(f) for f in raw_findings
                        if isinstance(f, dict)]

        # Filter by minimum severity
        min_rank = SEVERITY_RANK.get(min_severity, 0)
        findings = [f for f in findings
                    if SEVERITY_RANK.get(f["severity"], 0) >= min_rank]

        # Sort findings within function by severity
        findings.sort(key=lambda f: SEVERITY_RANK.get(f["severity"], 0), reverse=True)

        count = len(findings)
        if count == 0:
            print(col("✓ clean", C.GREEN))
        else:
            sev_summary = ", ".join(
                f"{v}×{k}" for k, v in
                sorted(
                    {f["severity"]: sum(1 for x in findings if x["severity"] == f["severity"])
                     for f in findings}.items(),
                    key=lambda x: -SEVERITY_RANK.get(x[0], 0)
                ) #.items()
            )
            print(col(f"⚠ {count} finding(s) [{sev_summary}]",
                      SEVERITY_COLOR.get(findings[0]["severity"], C.WHITE)))

        results.append({"function": fn, "findings": findings})

    return results


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG-powered Python vulnerability scanner using Llama 3 + Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("target",
        type=Path, help="Python file to scan")
    parser.add_argument("--chunks-zip",
        type=Path, default=Path("./rag_chunks.zip"),
        help="Path to rag_chunks.zip from Colab (default: ./rag_chunks.zip)")
    parser.add_argument("--chunks-dir",
        type=Path, default=None,
        help="Or point directly at an unzipped chunks folder")
    parser.add_argument("--output",
        type=Path, default=Path("./vuln_report.json"),
        help="JSON report output path (default: ./vuln_report.json)")
    parser.add_argument("--model",
        default="llama3",
        help="Ollama model tag (default: llama3)")
    parser.add_argument("--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--min-severity",
        default="LOW", choices=["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
        help="Minimum severity to report (default: LOW)")
    parser.add_argument("--top-k",
        type=int, default=6,
        help="Chunks to retrieve per function (default: 6)")
    parser.add_argument("--no-color",
        action="store_true", help="Disable terminal colours")
    return parser.parse_args()


def main():
    global USE_COLOR
    args = parse_args()
    USE_COLOR = not args.no_color

    # Validate target file
    if not args.target.exists():
        print(col(f"[ERROR] Target file not found: {args.target}", C.RED))
        sys.exit(1)

    # Resolve knowledge base source
    kb_source = args.chunks_dir if args.chunks_dir else args.chunks_zip
    if not kb_source.exists():
        print(col(
            f"[ERROR] Knowledge base not found at {kb_source}\n"
            f"        Run the Colab notebook first and download rag_chunks.zip,\n"
            f"        then place it next to this script or pass --chunks-zip <path>.",
            C.RED,
        ))
        sys.exit(1)

    print(col("\n  RAG Vulnerability Scanner  ", C.BOLD, C.WHITE))
    print(col(f"  Target : {args.target}", C.CYAN))
    print(col(f"  Model  : {args.model} @ {args.ollama_url}\n", C.CYAN))

    # Load knowledge base
    kb = KnowledgeBase(kb_source)

    # Connect to Ollama
    llm = OllamaClient(base_url=args.ollama_url, model=args.model)

    # Run scan
    t0 = time.time()
    results = scan_file(
        target=args.target,
        kb=kb,
        llm=llm,
        top_k=args.top_k,
        min_severity=args.min_severity,
    )
    elapsed = time.time() - t0

    # Print terminal report
    print_report(
        target_file=args.target,
        results=results,
        total_functions=len(results),
        elapsed=elapsed,
    )

    # Save JSON report
    save_json_report(
        target_file=args.target,
        results=results,
        output_path=args.output,
        elapsed=elapsed,
        model=args.model,
    )


if __name__ == "__main__":
    main()
