"""
MCP client pipeline runner
=============================================
Calls the three MCP tools in server.py over stdio transport:
  1. analyze_code        — static RAG analysis
  2. fuzz_target         — ffuf fuzzing guided by the report
  3. suggest_mitigations — RAG-grounded code fix suggestions

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
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ── JSON parsing helpers ───────────────────────────────────────────────────────

def _fix_triple_quotes(s: str) -> str:
    """Convert LLM triple-quoted strings e.g. \"\"\"...\"\"\" into valid JSON strings."""
    def replacer(m):
        inner = m.group(1)
        inner = inner.replace("\\", "\\\\")
        inner = inner.replace('"',  '\\"')
        inner = inner.replace("\n", "\\n")
        inner = inner.replace("\r", "\\r")
        inner = inner.replace("\t", "\\t")
        return f'"{inner}"'
    return re.sub(r'"""(.*?)"""', replacer, s, flags=re.DOTALL)


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


def _parse_mitigation_response(raw: str) -> dict | None:
    """
    Robustly parse a single mitigation JSON object from LLM output.

    The LLM frequently breaks JSON in fixed_code by embedding SQL or Python
    strings with unescaped double-quotes. Strategy:
      1. Strip markdown fences and fix triple-quotes
      2. PRIMARY: extract fixed_code value before parsing, replace with
         placeholder, parse the rest cleanly, inject fixed_code back
      3. BACKUP: blank out fixed_code entirely, parse everything else,
         keep fixed_code as a raw string — loses formatting but saves
         explanation, hardening, and references
    """
    # Strip markdown fences
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
    text = text.strip()

    # Fix triple-quoted strings
    text = _fix_triple_quotes(text)

    # PRIMARY: extract fixed_code before parsing
    fixed_code_raw = None
    text_for_parse = text

    fc_match = re.search(
        r'"fixed_code"\s*:\s*"(.*?)"(?=\s*,\s*"|\s*})',
        text, re.DOTALL
    )
    if fc_match:
        fixed_code_raw = fc_match.group(1)
        fixed_code_raw = (fixed_code_raw
                          .replace("\\n", "\n")
                          .replace("\\t", "\t")
                          .replace('\\"', '"'))
        text_for_parse = (text[:fc_match.start(1)]
                          + "__FIXED_CODE__"
                          + text[fc_match.end(1):])

    parsed = _try_parse(text_for_parse)
    if parsed is not None:
        parsed["fixed_code"] = fixed_code_raw or ""
        return parsed

    # BACKUP: blank fixed_code, parse everything else
    backup_text = re.sub(
        r'"fixed_code"\s*:\s*".*?"(?=\s*,\s*"|\s*})',
        '"fixed_code": ""',
        text, flags=re.DOTALL
    )
    parsed = _try_parse(backup_text)
    if parsed is not None:
        parsed["fixed_code"] = fixed_code_raw or "[could not parse fixed_code]"
        return parsed

    return None


# ── Pipeline ───────────────────────────────────────────────────────────────────

async def run_pipeline(args):
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).parent / "server.py")],
        env={**os.environ},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"[MCP] Connected. Tools: {[t.name for t in tools.tools]}\n")

            full_report = {
                "target":      args.target,
                "analysis":    None,
                "fuzzing":     None,
                "mitigations": None,
            }

            # ══════════════════════════════════════════════════════════════
            # STEP 1 — analyze_code
            # ══════════════════════════════════════════════════════════════
            print("=" * 60)
            print("STEP 1 — Static vulnerability analysis")
            print("=" * 60)

            result1 = await session.call_tool(
                "analyze_code",
                arguments={
                    "file_path":    args.target,
                    "min_severity": args.min_severity,
                    "top_k":        6,
                },
            )
            analysis = json.loads(result1.content[0].text)
            full_report["analysis"] = analysis

            if analysis.get("status") == "error":
                print(f"[ERROR] Analysis failed: {analysis.get('message')}")
                _save(full_report, args.output)
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

            # ══════════════════════════════════════════════════════════════
            # STEP 2 — fuzz_target
            # ══════════════════════════════════════════════════════════════
            if not args.skip_fuzz:
                print("\n" + "=" * 60)
                print("STEP 2 — Guided fuzzing with ffuf + SecLists")
                print("=" * 60)

                result2 = await session.call_tool(
                    "fuzz_target",
                    arguments={"base_url": args.target_url},
                )
                fuzzing = json.loads(result2.content[0].text)
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

            # ══════════════════════════════════════════════════════════════
            # STEP 3 — suggest_mitigations
            # ══════════════════════════════════════════════════════════════
            if not args.skip_mitigate:
                print("\n" + "=" * 60)
                print("STEP 3 — RAG-grounded mitigation suggestions")
                print("=" * 60)

                result3 = await session.call_tool(
                    "suggest_mitigations",
                    arguments={"min_severity": args.mit_severity},
                )

                # The server returns a mitigations list. Each entry's fixed_code
                # may have broken JSON inside, so we re-parse any error entries
                # through _parse_mitigation_response on the client side.
                raw_text = result3.content[0].text
                try:
                    raw_result = json.loads(raw_text)
                except json.JSONDecodeError:
                    raw_result = {"status": "ok", "mitigations": []}
                mitigations_out = []

                for m in raw_result.get("mitigations", []):
                    if not m.get("error"):
                        mitigations_out.append(m)
                        continue

                    raw_text = m.get("raw_response", "")
                    if raw_text:
                        parsed = _parse_mitigation_response(raw_text)
                        if parsed:
                            mitigations_out.append({
                                "function":    m.get("function", "unknown"),
                                "cwe_id":      m.get("cwe_id", ""),
                                "severity":    m.get("severity", ""),
                                "explanation": parsed.get("explanation", ""),
                                "fixed_code":  parsed.get("fixed_code", ""),
                                "hardening":   parsed.get("hardening", []),
                                "references":  parsed.get("references", []),
                            })
                            continue

                    mitigations_out.append(m)

                mitigations = {
                    "status":      raw_result.get("status", "ok"),
                    "total":       len(mitigations_out),
                    "mitigations": mitigations_out,
                }
                full_report["mitigations"] = mitigations

                if mitigations.get("status") == "error":
                    print(f"[WARN] Mitigation error: {mitigations.get('message')}")
                else:
                    print(f"Generated {mitigations['total']} mitigation(s):\n")
                    for m in mitigations["mitigations"]:
                        if m.get("error"):
                            print(f"  [{m['cwe_id']}] ERROR: {m['error']}")
                            if m.get("raw_response"):
                                print(f"  Raw (first 300 chars): {m['raw_response'][:300]}")
                            continue
                        print(f"  Function : {m['function']}")
                        print(f"  CWE      : {m['cwe_id']} ({m.get('severity','')})")
                        print(f"  Why      : {m.get('explanation','')}")
                        if m.get("hardening"):
                            print("  Hardening:")
                            for tip in m["hardening"][:3]:
                                print(f"    • {tip}")
                        if m.get("fixed_code"):
                            print("  Fixed code (first 20 lines):")
                            for line in m["fixed_code"].splitlines()[:20]:
                                print(f"    {line}")
                        print()

            _save(full_report, args.output)


def _save(report, path):
    Path(path).write_text(json.dumps(report, indent=2))
    print(f"\n[Client] Full report saved → {path}")


def parse_args():
    p = argparse.ArgumentParser(description="MCP client pipeline runner")
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
    asyncio.run(run_pipeline(args))