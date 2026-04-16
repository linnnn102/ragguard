"""
Usage:
    python run_pipeline.py <target.py> [options]

Options:
    --target-url    URL   ffuf target URL with FUZZ keyword
                          [default: http://127.0.0.1:5055/user/FUZZ]
    --min-severity  STR   Minimum severity to report  [default: LOW]
    --mit-severity  STR   Minimum severity to mitigate [default: MEDIUM]
    --skip-fuzz           Skip the fuzzing step
    --skip-mitigate       Skip the mitigation step
    --output        PATH  Final combined report  [default: ./full_report.json]
    --debug               Print raw LLM output for every mitigation attempt
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ── Import tools and internals directly from server.py ────────────────────────
try:
    from server import analyze_code, fuzz_target
    from server import get_kb, get_llm, CONFIG, SEVERITY_RANK, _mitigation_prompt
    from vuln_scanner import extract_functions
except ImportError as e:
    print(f"[ERROR] Could not import server.py: {e}")
    print("        Make sure server.py and vuln_scanner.py are in the same directory.")
    sys.exit(1)


# ── Triple-quote fixer ─────────────────────────────────────────────────────────
# The LLM sometimes writes `"fixed_code": """..."""` which is invalid JSON.
# This converts every """...""" block into a properly escaped JSON string.
def _fix_triple_quotes(s: str) -> str:
    def replacer(m):
        inner = m.group(1)
        inner = inner.replace("\\", "\\\\")
        inner = inner.replace('"',  '\\"')
        inner = inner.replace("\n", "\\n")
        inner = inner.replace("\r", "\\r")
        inner = inner.replace("\t", "\\t")
        return f'"{inner}"'
    return re.sub(r'"""(.*?)"""', replacer, s, flags=re.DOTALL)


# ── Mitigation runner ──────────────────────────────────────────────────────────
def _suggest_mitigations_fixed(min_severity: str = "MEDIUM", top_k: int = 6) -> dict:
    rpath = CONFIG["report_path"]
    if not rpath.exists():
        return {"status": "error", "message": f"Report not found at {rpath}. Run analyze_code first."}

    report   = json.loads(rpath.read_text())
    kb       = get_kb()
    llm      = get_llm()
    min_rank = SEVERITY_RANK.get(min_severity, 0)
    mitigations = []

    for result in report.get("results", []):
        fn_name   = result.get("function", "unknown")
        fn_source = result.get("source", "")

        if not fn_source:
            target = Path(report.get("meta", {}).get("target", ""))
            if target.exists():
                try:
                    fns   = extract_functions(target)
                    match = next((f for f in fns if f["name"] == fn_name), None)
                    if match:
                        fn_source = match["source"]
                except Exception:
                    fn_source = f"# source unavailable for {fn_name}"

        for finding in result.get("findings", []):
            if SEVERITY_RANK.get(finding["severity"], 0) < min_rank:
                continue

            rag_query = (
                f"{finding['cwe_id']} {finding.get('cwe_name', '')} "
                f"mitigation fix secure code. "
                f"{finding.get('evidence', '')[:200]}"
            )
            chunks  = kb.retrieve(rag_query, top_k=top_k)
            context = kb.format_context(chunks)
            prompt  = _mitigation_prompt(finding, fn_source, context)

            try:
                raw = llm.generate(prompt, max_retries=2)
            except Exception as e:
                mitigations.append({"function": fn_name, "cwe_id": finding["cwe_id"], "error": str(e)})
                continue

            # Strip markdown fences
            text = raw.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
            text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
            text = text.strip()

            # # Fix triple-quoted strings before any parse attempt
            text = _fix_triple_quotes(text)

            # PRIMARY: extract fixed_code before JSON parsing
            # The fixed_code value often contains special characters 
            # for exampleSQL strings with unescaped double-quotes (e.g. query = "CREATE TABLE...") 
            # it will break json.loads.
            # pull it out first, replace it with a safe placeholder, parse the rest cleanly, then inject it back.
            fixed_code_raw = None
            text_for_parse = text
 
            fc_match = re.search(
                r'"fixed_code"\s*:\s*"(.*?)"(?=\s*,\s*"|\s*})',
                text, re.DOTALL
            )
            if fc_match:
                fixed_code_raw = fc_match.group(1)
                # Decode any \n \t the LLM did escape correctly
                fixed_code_raw = fixed_code_raw.replace("\\n", "\n") \
                                               .replace("\\t", "\t") \
                                               .replace('\\"', '"')
                # Replace the field value with a safe placeholder
                text_for_parse = text[:fc_match.start(1)] \
                                 + "__FIXED_CODE__" \
                                 + text[fc_match.end(1):]
            
            def _try_parse(s: str) -> dict | None:
                """Try json.loads directly, then on the first { } block."""
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    m = re.search(r"\{.*\}", s, re.DOTALL)
                    if m:
                        try:
                            return json.loads(m.group())
                        except json.JSONDecodeError:
                            pass
                return None

            # Primary: parse with placeholder, inject fixed_code back
            parsed = _try_parse(text_for_parse)
            if parsed is not None:
                parsed["fixed_code"] = fixed_code_raw or ""

            # Backup: blank out fixed_code entirely, parse the rest
            if parsed is None:
                backup_text = re.sub(
                    r'"fixed_code"\s*:\s*".*?"(?=\s*,\s*"|\s*})',
                    '"fixed_code": ""',
                    text, flags=re.DOTALL
                )
                parsed = _try_parse(backup_text)
                if parsed is not None:
                    parsed["fixed_code"] = fixed_code_raw or "[could not parse fixed_code]"

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
                    "function":     fn_name,
                    "cwe_id":       finding["cwe_id"],
                    "severity":     finding["severity"],
                    "raw_response": text[:2000],
                    "error":        "Could not parse LLM response as JSON",
                })

    return {"status": "ok", "mitigations": mitigations, "total": len(mitigations)}


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(args):
    full_report = {
        "target":      args.target,
        "analysis":    None,
        "fuzzing":     None,
        "mitigations": None,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — analyze_code
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 1 — Static vulnerability analysis")
    print("=" * 60)

    analysis = analyze_code(
        file_path=args.target,
        min_severity=args.min_severity,
        top_k=6,
    )
    full_report["analysis"] = analysis

    if analysis.get("status") == "error":
        print(f"[ERROR] Analysis failed: {analysis.get('message')}")
        sys.exit(1)

    print(f"Functions scanned : {analysis.get('functions_scanned', '?')}")
    print(f"Duration          : {analysis.get('duration_s', '?')}s")
    print(f"Findings summary  : {analysis.get('summary', {})}")
    print(f"Report written to : {analysis.get('report_path')}")

    findings = analysis.get("findings", [])
    if not findings:
        print("\nNo vulnerabilities found — stopping pipeline.")
        _save(full_report, args.output)
        return

    print()
    for f in findings:
        print(f"  [{f.get('severity','?')}] {f.get('cwe_id','?')} "
              f"{f.get('cwe_name','')} "
              f"(confidence {f.get('confidence', 0):.0%})")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — fuzz_target
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_fuzz:
        print("\n" + "=" * 60)
        print("STEP 2 — Guided fuzzing with ffuf + SecLists")
        print("=" * 60)

        fuzzing = fuzz_target(base_url=args.target_url)
        full_report["fuzzing"] = fuzzing

        if fuzzing.get("status") == "error":
            print(f"[WARN] Fuzzing error: {fuzzing.get('message')}")
        else:
            print(fuzzing.get("summary", ""))
            for job in fuzzing.get("cwe_jobs", []):
                if job.get("hit_count", 0) > 0:
                    print(f"\n  Hits for {job['cwe_id']} "
                          f"({Path(job.get('wordlist','')).name}):")
                    for hit in job.get("hits", [])[:5]:
                        print(f"    [{hit['status']}] {hit['url']}"
                              f"  payload: {str(hit['input'])[:60]}")
                if job.get("error"):
                    print(f"  [WARN] {job['cwe_id']}: {job['error']}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — suggest_mitigations
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_mitigate:
        print("\n" + "=" * 60)
        print("STEP 3 — RAG-grounded mitigation suggestions")
        print("=" * 60)

        mitigations = _suggest_mitigations_fixed(
            min_severity=args.mit_severity,
        )
        full_report["mitigations"] = mitigations

        if mitigations.get("status") == "error":
            print(f"[WARN] Mitigation error: {mitigations.get('message')}")
        else:
            total = mitigations.get("total", 0)
            print(f"Generated {total} mitigation(s):\n")
            for m in mitigations.get("mitigations", []):
                if m.get("error"):
                    print(f"  [{m['cwe_id']}] ERROR: {m['error']}")
                    if m.get("raw_response"):
                        print(f"  Raw: {m['raw_response']}")
                    continue
                print(f"  Function : {m['function']}")
                print(f"  CWE      : {m['cwe_id']} ({m.get('severity','')})")
                print(f"  Why      : {m.get('explanation','')}")
                if m.get("hardening"):
                    print("  Hardening:")
                    for tip in m["hardening"][:3]:
                        print(f"    • {tip}")
                if m.get("fixed_code"):
                    print("  Fixed code:")
                    for line in m["fixed_code"].splitlines():
                        print(f"    {line}")
                print()

    _save(full_report, args.output)


def _save(report, path):
    Path(path).write_text(json.dumps(report, indent=2))
    print(f"\n[Runner] Full report saved → {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Direct pipeline runner — no MCP needed")
    p.add_argument("target", help="Python file to scan")
    p.add_argument("--target-url",    default="http://127.0.0.1:5055/user/FUZZ")
    p.add_argument("--min-severity",  default="LOW",
                   choices=["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"])
    p.add_argument("--mit-severity",  default="MEDIUM",
                   choices=["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
                   help="Minimum severity for mitigations (default: MEDIUM)")
    p.add_argument("--skip-fuzz",     action="store_true")
    p.add_argument("--skip-mitigate", action="store_true")
    p.add_argument("--output",        default="./full_report.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)