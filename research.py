"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    s = str(topic or "").strip().lower()
    s = re.sub(r"[^\w]+", "-", s)
    s = s.strip("-")
    s = s[:60].rstrip("-")
    return s or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Conduct a comprehensive academic deep research survey on the topic: '{topic}'.\n\n"
        "Follow your complete research plan:\n"
        "1. Call `write_todos` to break down the topic into at least 3-4 distinct sub-questions.\n"
        "2. Delegate each sub-question in parallel to a `researcher` subagent via `task`, with full context, destination note paths in /tmp/work/research/notes/, and expected format.\n"
        "3. Review notes and consolidate unique sources into /tmp/work/research/sources.json.\n"
        "   MANDATORY DIVERSITY REQUIREMENT: You MUST include and cite sources from AT LEAST 3 DISTINCT source families among ('arxiv', 'hf-daily', 'hf-search', 'web'). Make sure to explicitly search both arXiv, Hugging Face, and Web (Exa) to achieve 3+ distinct families.\n"
        "4. Write the comprehensive report body in /tmp/work/report/report.md following REPORT_TEMPLATE.md (with inline [n] citations, without a ## References section), citing papers from at least 3 distinct source families.\n"
        "5. Execute /tmp/work/research/finalize_citations.py to generate references and renumber citations.\n"
        "6. Execute /tmp/work/research/check_citations.py to verify consistency until OK.\n"
        "7. Delegate to citation-checker to spot-check claims.\n"
        "Maintain high technical depth, compare approaches, and ensure every non-obvious claim carries a verified citation [n]."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}."""
    tool_counter = Counter()
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        # Check tool_calls on message object or dict
        calls = getattr(msg, "tool_calls", None)
        if calls is None and isinstance(msg, dict):
            calls = msg.get("tool_calls")
        if calls and isinstance(calls, list):
            for call in calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name:
                    tool_counter[name] += 1

        # Check token usage
        usage = getattr(msg, "usage_metadata", None)
        if usage is None and isinstance(msg, dict):
            usage = msg.get("usage_metadata")
        if not usage:
            resp_meta = getattr(msg, "response_metadata", {}) or (msg.get("response_metadata") if isinstance(msg, dict) else {})
            usage = resp_meta.get("token_usage") or resp_meta.get("usage")

        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)

    return {
        "model": str(model_name),
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": tool_counter.get("task", 0),
        "tool_calls": dict(tool_counter),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }


def normalize_source_family(source_str, url=""):
    s = str(source_str or "").strip().lower()
    u = str(url or "").lower()
    if "arxiv.org" in u or s == "arxiv":
        return "arxiv"
    if "huggingface.co" in u or s in ("hf-daily", "hf-search"):
        return s if s in ("hf-daily", "hf-search") else "hf-search"
    return "web"


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if report_bytes is None or len(report_bytes.strip()) == 0:
        raise RuntimeError("Report file is missing or empty in sandbox.")
    if sources_bytes is None or len(sources_bytes.strip()) == 0:
        raise RuntimeError("Sources file is missing or empty in sandbox.")

    try:
        sources_data = json.loads(sources_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"sources.json is invalid JSON: {exc}") from exc

    if not isinstance(sources_data, list) or len(sources_data) == 0:
        raise RuntimeError("sources.json must be a non-empty list of sources.")

    for s in sources_data:
        if isinstance(s, dict):
            s["source"] = normalize_source_family(s.get("source"), s.get("url"))

    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    md_file = reports_dir / f"{slug}.md"
    sources_file = reports_dir / f"{slug}.sources.json"
    meta_file = reports_dir / f"{slug}.meta.json"

    families = sorted(list(set(s["source"] for s in sources_data if isinstance(s, dict) and s.get("source"))))
    summary_info = summarize(messages, elapsed, model_name)

    meta_payload = {
        "topic": topic,
        **summary_info,
        "n_sources": len(sources_data),
        "source_families": families,
    }

    md_file.write_bytes(report_bytes)
    sources_file.write_text(json.dumps(sources_data, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_file.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return md_file


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic_str = str(topic).strip()
    if not topic_str:
        print("Usage: python research.py <topic>", file=sys.stderr)
        return 2

    model = make_model()
    model_name = (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or os.getenv("LAB_MODEL", "unknown")
    )
    start = time.monotonic()

    with open_sandbox() as backend:
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(
            backend,
            {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            },
        )
        agent = build_lead_agent(backend, model)
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic_str)}]},
                config={"recursion_limit": 1000},
            )
        except Exception as exc:
            print(f"FAILED during agent invocation: {exc}", file=sys.stderr)
            return 1
        # Ensure citations are finalized and validated inside the sandbox before download
        backend.execute(f"python3 {FINALIZER_PATH}")
        val_res = backend.execute(f"python3 {VALIDATOR_PATH}")
        if val_res.exit_code != 0:
            print(f"[sandbox validator] {val_res.output.strip()}", file=sys.stderr)

        elapsed = time.monotonic() - start
        messages = result.get("messages", []) if isinstance(result, dict) else []

        try:
            saved = save_outputs(backend, topic_str, messages, elapsed, model_name)
            print(f"SUCCESS: Report saved to {saved}")
            return 0
        except RuntimeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
