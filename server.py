"""
mcp_vuln_server.py — MCP vulnerability analysis server
=======================================================
Exposes three tools via the MCP protocol (fastmcp):

  1. analyze_code        — RAG + Llama 3 static vulnerability analysis
  2. fuzz_target         — ffuf fuzzing guided by Tool 1 report
  3. suggest_mitigations — RAG + Llama 3 code-level fix suggestions

Usage:
    python mcp_vuln_server.py

Configuration (edit CONFIG block below or set env vars):
    CHUNKS_ZIP      path to rag_chunks.zip from Colab
    OLLAMA_URL      Ollama base URL
    OLLAMA_MODEL    Ollama model tag
    SECLISTS_PATH   root of SecLists installation
    FFUF_BIN        path to ffuf binary
    REPORT_PATH     where Tool 1 writes / Tool 2 reads vuln_report.json
"""

import os
import json
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import requests
import numpy as np
import pandas as pd
from fastmcp import FastMCP

# ── reuse scanner internals ────────────────────────────────────────────────────
# Import everything we already built in vuln_scanner.py.
# Place mcp_vuln_server.py in the same folder as vuln_scanner.py.
from vuln_scanner_bf_dot import (
    KnowledgeBase,
    OllamaClient,
    extract_functions,
    build_prompt,
    extract_json_array,
    validate_finding,
    save_json_report,
    SEVERITY_RANK,
    SYSTEM_PROMPT,
)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "chunks_zip"   : Path(os.getenv("CHUNKS_ZIP",    "./rag_chunks.zip")),
    "ollama_url"   : os.getenv("OLLAMA_URL",          "http://localhost:11434"),
    "ollama_model" : os.getenv("OLLAMA_MODEL",         "llama3"),
    "seclists_path": Path(os.getenv("SECLISTS_PATH",  Path(__file__).parent / "lib/SecLists")),
    "ffuf_bin"     : os.getenv("FFUF_BIN",             "ffuf"),
    "report_path"  : Path(os.getenv("REPORT_PATH",    "./vuln_report.json")),
    "top_k"        : int(os.getenv("TOP_K",            "6")),
}

# ── CWE → SecLists wordlist mapping ───────────────────────────────────────────
# Maps CWE IDs to the most relevant SecLists wordlist paths (relative to SECLISTS_PATH).
# Extend this dict as you add more CWEs to your knowledge base.
CWE_WORDLIST_MAP = {
    "CWE-89":  [
        "Fuzzing/Databases/SQLi/MySQL-SQLi-Login-Bypass.fuzzdb.txt",
        "Fuzzing/Databases/SQLi/Generic-SQLi.txt",
    ],
    "CWE-79":  [
        "Fuzzing/XSS/XSS-Jhaddix.txt",
        "Fuzzing/XSS/XSS-BruteLogic.txt",
    ],
    "CWE-78":  [
        "Fuzzing/command-injection-commix.txt",
    ],
    "CWE-22":  [
        "Fuzzing/LFI/LFI-gracefulsecurity-linux.txt",
        "Fuzzing/LFI/LFI-LFISuite-pathtotest-huge.txt",
    ],
    "CWE-287": [
        "Fuzzing/Authentication/Authentication.txt",
    ],
    "CWE-352": [
        "Fuzzing/CSRF/CSRF-token-wordlist.txt",
    ],
    "CWE-434": [
        "Fuzzing/Extensions.fuzz.txt",
    ],
    "CWE-502": [
        "Fuzzing/Polyglots/Polyglots.txt",
    ],
    "CWE-918": [
        "Fuzzing/SSRF/SSRF-targets.txt",
    ],
    "CWE-20":  [
        "Fuzzing/Polyglots/Polyglots.txt",
        "Fuzzing/special-chars.txt",
    ],
    # Fallback for unmapped CWEs
    "_default": [
        "Discovery/Web-Content/common.txt",
    ],
}

MITIGATION_SYSTEM_PROMPT = """\
You are am expert application security engineer specialising in Python secure code review.
You are given a vulnerability finding and the relevant CWE/CVE reference context.
Your task is to provide:
  1. A clear explanation of why the code is vulnerable
  2. A concrete, minimal code fix — show the corrected function in full
  3. Any additional hardening recommendations beyond the immediate fix
  4. The CWE and any relevant CVE IDs from the context

Respond ONLY with a valid JSON object with these fields:
  - "cwe_id":        string
  - "explanation":   string — why the code is vulnerable
  - "fixed_code":    string — the corrected function as a code string
  - "hardening":     array of strings — additional recommendations
  - "references":    array of strings — CVE IDs from context if relevant

Do not include any text outside the JSON object.\
"""

# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
_kb:  Optional[KnowledgeBase] = None
_llm: Optional[OllamaClient]  = None

def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase(CONFIG["chunks_zip"])
    return _kb

def get_llm() -> OllamaClient:
    global _llm
    if _llm is None:
        _llm = OllamaClient(
            base_url=CONFIG["ollama_url"],
            model=CONFIG["ollama_model"],
        )
    return _llm

# ── MCP server instance ────────────────────────────────────────────────────────
mcp = FastMCP(
    name="vuln-analysis-server",
    instructions=(
        "Vulnerability analysis server with three tools:\n"
        "1. analyze_code — static RAG analysis of a Python file\n"
        "2. fuzz_target  — ffuf fuzzing guided by the analysis report\n"
        "3. suggest_mitigations — RAG-grounded code fix suggestions\n"
        "Run tools in order: analyze_code → fuzz_target → suggest_mitigations."
    ),
)

# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — analyze_code
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def analyze_code(
    file_path: str,
    min_severity: str = "LOW",
    top_k: int = 6,
) -> dict:
    """
    Statically analyze a Python file for security vulnerabilities using
    RAG-augmented Llama 3. Extracts each function, retrieves relevant
    CWE/CVE context, and asks the LLM to identify vulnerabilities.

    Args:
        file_path:    Absolute or relative path to the Python file to scan.
        min_severity: Minimum severity to include — CRITICAL/HIGH/MEDIUM/LOW/INFO.
        top_k:        Number of RAG chunks to retrieve per function (default 6).

    Returns:
        dict with keys:
          - status:    "ok" or "error"
          - summary:   finding counts by severity
          - findings:  list of all findings across all functions
          - report_path: path where full JSON report was written
    """
    target = Path(file_path)
    if not target.exists():
        return {"status": "error", "message": f"File not found: {file_path}"}

    kb  = get_kb()
    llm = get_llm()

    try:
        functions = extract_functions(target)
    except SyntaxError as e:
        return {"status": "error", "message": f"Python syntax error in target: {e}"}

    if not functions:
        return {"status": "ok", "summary": {}, "findings": [],
                "message": "No functions found in file."}

    all_results = []
    t0 = time.time()

    for fn in functions:
        rag_query = (
            f"Python function '{fn['name']}' vulnerability. "
            f"Args: {', '.join(fn['args'])}. "
            f"{fn['source'][:400]}"
        )
        chunks  = kb.retrieve(rag_query, top_k=top_k, min_severity=min_severity)
        context = kb.format_context(chunks)
        prompt  = build_prompt(fn, context)

        try:
            raw = llm.generate(prompt)
        except Exception as e:
            all_results.append({"function": fn, "findings": [], "error": str(e)})
            continue

        raw_findings = extract_json_array(raw)
        findings = [validate_finding(f) for f in raw_findings if isinstance(f, dict)]
        min_rank = SEVERITY_RANK.get(min_severity, 0)
        findings = [f for f in findings if SEVERITY_RANK.get(f["severity"], 0) >= min_rank]
        findings.sort(key=lambda f: SEVERITY_RANK.get(f["severity"], 0), reverse=True)
        all_results.append({"function": fn, "findings": findings})

    elapsed = time.time() - t0

    # Persist report so Tool 2 can read it
    report = save_json_report(
        target_file=target,
        results=all_results,
        output_path=CONFIG["report_path"],
        elapsed=elapsed,
        model=CONFIG["ollama_model"],
    )

    all_findings = [f for r in all_results for f in r["findings"]]
    from collections import Counter
    counts = Counter(f["severity"] for f in all_findings)

    return {
        "status":      "ok",
        "summary":     dict(counts),
        "findings":    all_findings,
        "report_path": str(CONFIG["report_path"]),
        "functions_scanned": len(functions),
        "duration_s":  round(elapsed, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — fuzz_target
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_wordlists(cwe_id: str) -> list[Path]:
    """Map a CWE ID to its SecLists wordlist paths, filter to those that exist."""
    seclists = CONFIG["seclists_path"]
    candidates = CWE_WORDLIST_MAP.get(cwe_id, CWE_WORDLIST_MAP["_default"])
    resolved = []
    for rel in candidates:
        p = seclists / rel
        if p.exists():
            resolved.append(p)
    if not resolved:
        # last resort: use common.txt if it exists
        fallback = seclists / "Discovery/Web-Content/common.txt"
        if fallback.exists():
            resolved.append(fallback)
    return resolved


def _build_ffuf_cmd(
    target_url: str,
    wordlist: Path,
    cwe_id: str,
    extra_flags: list[str],
    match_codes: str = "200",
    threads: int = 20,
    timeout: int = 10,
    url_encode: bool = True,
) -> list[str]:
    """Build an ffuf command list for a given CWE and wordlist."""
    out_path = f"/tmp/ffuf_{cwe_id.replace('-','_')}_{wordlist.stem}.json"
    cmd = [
        CONFIG["ffuf_bin"],
        "-u",   target_url,
        "-w",   str(wordlist),
        "-mc",  match_codes,
        "-o",   out_path,
        "-of",  "json",
        "-t",   str(threads),
        "-timeout", str(timeout),
        "-v",                     # verbose — logs each request, helps debug zero hits
    ]
    if url_encode:
        cmd += ["-enc", "url"]    # URL-encode payloads, matches your working command
    cmd.extend(extra_flags)
    return cmd


def _run_ffuf(cmd: list[str]) -> dict:
    """Run an ffuf command and parse its JSON output."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        out_path = None
        for i, part in enumerate(cmd):
            if part == "-o" and i + 1 < len(cmd):
                out_path = Path(cmd[i + 1])
                break

        hits = []
        if out_path and out_path.exists():
            try:
                ffuf_out = json.loads(out_path.read_text())
                for result in ffuf_out.get("results", []):
                    hits.append({
                        "url":    result.get("url"),
                        "status": result.get("status"),
                        "length": result.get("length"),
                        "words":  result.get("words"),
                        "input":  result.get("input", {}).get("FUZZ", ""),
                    })
            except json.JSONDecodeError:
                pass

        return {
            "command":   " ".join(cmd),
            "returncode": proc.returncode,
            "hits":      hits,
            "hit_count": len(hits),
            "stderr":    proc.stderr[:500] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"command": " ".join(cmd), "error": "ffuf timed out after 300s"}
    except FileNotFoundError:
        return {"command": " ".join(cmd),
                "error": f"ffuf binary not found at '{CONFIG['ffuf_bin']}'"}


@mcp.tool()
def fuzz_target(
    base_url: str = "http://127.0.0.1:5055/user/FUZZ",
    report_path: Optional[str] = None,
    match_codes: str = "200",
    extra_ffuf_flags: Optional[list[str]] = None,
    max_wordlists_per_cwe: int = 1,
    url_encode: bool = True,
) -> dict:
    """
    Run ffuf fuzzing against a localhost target, guided by the vulnerability
    report produced by analyze_code. Selects SecLists wordlists based on
    the CWE IDs found in the report.

    Args:
        base_url:              Target URL with FUZZ keyword marking the injection
                               point. Examples:
                                 "http://127.0.0.1:5055/user/FUZZ"       (path)
                                 "http://127.0.0.1:5055/search?q=FUZZ"   (query param)
                                 "http://127.0.0.1:5055/FUZZ"            (path discovery)
        report_path:           Path to vuln_report.json. Defaults to the path
                               written by analyze_code.
        match_codes:           HTTP status codes to treat as hits, comma-separated.
                               Default "200". Use "200,302,403" to widen scope.
        extra_ffuf_flags:      Additional ffuf CLI flags as a list,
                               e.g. ["-H", "Cookie: session=abc", "-X", "POST"].
        max_wordlists_per_cwe: How many wordlists to use per CWE (default 1).
        url_encode:            URL-encode payloads before sending (default True,
                               matches -enc url in manual ffuf runs).

    Returns:
        dict with keys:
          - status:     "ok" or "error"
          - cwe_jobs:   list of {cwe_id, wordlist, command, hits} per job
          - total_hits: total number of positive ffuf responses
          - summary:    plain-text summary of what was run
    """
    rpath = Path(report_path) if report_path else CONFIG["report_path"]
    if not rpath.exists():
        return {
            "status": "error",
            "message": (
                f"Report not found at {rpath}. "
                "Run analyze_code first to generate the report."
            ),
        }

    report = json.loads(rpath.read_text())
    all_findings = [
        f
        for result in report.get("results", [])
        for f in result.get("findings", [])
    ]

    if not all_findings:
        return {
            "status":  "ok",
            "message": "No findings in report — nothing to fuzz.",
            "cwe_jobs": [],
            "total_hits": 0,
        }

    # Deduplicate CWE IDs, keep highest-severity representative per CWE
    cwe_seen: dict[str, dict] = {}
    for f in sorted(all_findings,
                    key=lambda x: SEVERITY_RANK.get(x["severity"], 0),
                    reverse=True):
        cid = f["cwe_id"]
        if cid not in cwe_seen:
            cwe_seen[cid] = f

    extra_flags = extra_ffuf_flags or []

    if "FUZZ" not in base_url:
        return {
            "status": "error",
            "message": (
                "base_url must contain FUZZ at the injection point. "
                "Examples: http://127.0.0.1:5055/user/FUZZ  or  http://127.0.0.1:5055/search?q=FUZZ"
            ),
        }
    target_url = base_url

    print(f"[ffuf] Target  : {target_url}")
    print(f"[ffuf] Codes   : {match_codes}")
    print(f"[ffuf] Encode  : {url_encode}")

    cwe_jobs   = []
    total_hits = 0

    for cwe_id, finding in cwe_seen.items():
        wordlists = _resolve_wordlists(cwe_id)[:max_wordlists_per_cwe]

        if not wordlists:
            cwe_jobs.append({
                "cwe_id":   cwe_id,
                "severity": finding["severity"],
                "wordlist": None,
                "hits":     [],
                "hit_count": 0,
                "error":    f"No SecLists wordlist found for {cwe_id} under {CONFIG['seclists_path']}",
            })
            continue

        for wl in wordlists:
            print(f"[ffuf] {cwe_id} wordlist: {wl.name}")
            cmd    = _build_ffuf_cmd(target_url, wl, cwe_id, extra_flags,
                                     match_codes=match_codes, url_encode=url_encode)
            print(f"[ffuf] cmd: {' '.join(cmd)}")
            result = _run_ffuf(cmd)

            job = {
                "cwe_id":   cwe_id,
                "cwe_name": finding.get("cwe_name", ""),
                "severity": finding["severity"],
                "wordlist": str(wl),
                "command":  result.get("command"),
                "hits":     result.get("hits", []),
                "hit_count": result.get("hit_count", 0),
            }
            if "error" in result:
                job["error"] = result["error"]

            total_hits += job["hit_count"]
            cwe_jobs.append(job)

    summary_lines = [
        f"Fuzzed {len(cwe_seen)} unique CWE(s) against {target_url}",
        f"Total ffuf hits: {total_hits}",
    ]
    for job in cwe_jobs:
        line = f"  {job.get('cwe_id','?')} ({job.get('severity','?')}): {job.get('hit_count', 0)} hit(s)"
        if job.get("error"):
            line += f" [ERROR: {job['error']}]"
        summary_lines.append(line)

    return {
        "status":     "ok",
        "cwe_jobs":   cwe_jobs,
        "total_hits": total_hits,
        "target_url": target_url,
        "summary":    "\n".join(summary_lines),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — suggest_mitigations
# ══════════════════════════════════════════════════════════════════════════════

def _mitigation_prompt(finding: dict, fn_source: str, context: str) -> str:
    """Build a Llama 3 prompt for mitigation suggestions."""
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{MITIGATION_SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Vulnerability finding:\n"
        f"  CWE: {finding['cwe_id']} — {finding.get('cwe_name','')}\n"
        f"  Severity: {finding['severity']}\n"
        f"  Description: {finding['description']}\n"
        f"  Evidence: {finding.get('evidence','')}\n\n"
        f"Vulnerable function source:\n"
        f"```python\n{fn_source}\n```\n\n"
        f"CWE/CVE reference context:\n{context}\n\n"
        f"Respond ONLY with a JSON object containing: "
        f"explanation, fixed_code, hardening (array), references (array)."
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


@mcp.tool()
def suggest_mitigations(
    report_path: Optional[str] = None,
    min_severity: str = "MEDIUM",
    top_k: int = 6,
) -> dict:
    """
    For each vulnerability in the analyze_code report, retrieve RAG context
    and ask Llama 3 to produce a concrete code fix and hardening advice.

    Args:
        report_path:  Path to vuln_report.json. Defaults to Tool 1 output path.
        min_severity: Only suggest mitigations for findings at or above this
                      severity (default MEDIUM — skips INFO/LOW noise).
        top_k:        RAG chunks to retrieve per finding (default 6).

    Returns:
        dict with keys:
          - status:       "ok" or "error"
          - mitigations:  list of {function, cwe_id, explanation,
                          fixed_code, hardening, references}
          - total:        number of mitigations generated
    """
    rpath = Path(report_path) if report_path else CONFIG["report_path"]
    if not rpath.exists():
        return {
            "status": "error",
            "message": (
                f"Report not found at {rpath}. "
                "Run analyze_code first."
            ),
        }

    report = json.loads(rpath.read_text())
    kb  = get_kb()
    llm = get_llm()
    min_rank = SEVERITY_RANK.get(min_severity, 0)

    mitigations = []

    for result in report.get("results", []):
        fn_name   = result.get("function", "unknown")
        fn_source = result.get("source", "")

        # Try to recover source from the report if not stored
        # (save_json_report doesn't include raw source — fall back gracefully)
        if not fn_source:
            target = Path(report.get("meta", {}).get("target", ""))
            if target.exists():
                try:
                    fns = extract_functions(target)
                    match = next((f for f in fns if f["name"] == fn_name), None)
                    if match:
                        fn_source = match["source"]
                except Exception:
                    fn_source = f"# source unavailable for {fn_name}"

        for finding in result.get("findings", []):
            if SEVERITY_RANK.get(finding["severity"], 0) < min_rank:
                continue

            # RAG query: combine CWE name + evidence for targeted retrieval
            rag_query = (
                f"{finding['cwe_id']} {finding.get('cwe_name','')} "
                f"mitigation fix secure code. "
                f"{finding.get('evidence','')[:200]}"
            )
            chunks  = kb.retrieve(rag_query, top_k=top_k)
            context = kb.format_context(chunks)
            prompt  = _mitigation_prompt(finding, fn_source, context)

            try:
                raw = llm.generate(prompt, max_retries=2)
            except Exception as e:
                mitigations.append({
                    "function": fn_name,
                    "cwe_id":   finding["cwe_id"],
                    "error":    str(e),
                })
                continue

            # Parse JSON — LLM returns a single object not an array here
            text = raw.strip()
            import re
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
            text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
            text = text.strip()

            parsed = None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                    except json.JSONDecodeError:
                        pass

            if parsed and isinstance(parsed, dict):
                mitigations.append({
                    "function":    fn_name,
                    "cwe_id":      finding["cwe_id"],
                    "severity":    finding["severity"],
                    "explanation": parsed.get("explanation", ""),
                    "fixed_code":  parsed.get("fixed_code", ""),
                    "hardening":   parsed.get("hardening", []),
                    "references":  parsed.get("references", []),
                })
            else:
                mitigations.append({
                    "function":  fn_name,
                    "cwe_id":    finding["cwe_id"],
                    "severity":  finding["severity"],
                    "raw_response": raw[:1000],
                    "error":     "Could not parse LLM response as JSON",
                })

    return {
        "status":      "ok",
        "mitigations": mitigations,
        "total":       len(mitigations),
        "min_severity_filter": min_severity,
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting MCP vulnerability server...")
    print(f"  Chunks zip  : {CONFIG['chunks_zip']}")
    print(f"  Ollama URL  : {CONFIG['ollama_url']}")
    print(f"  Ollama model: {CONFIG['ollama_model']}")
    print(f"  SecLists    : {CONFIG['seclists_path']}")
    print(f"  Report path : {CONFIG['report_path']}")
    print()
    mcp.run()
