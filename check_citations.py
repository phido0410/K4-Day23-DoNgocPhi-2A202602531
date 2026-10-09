"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


import re


_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REF_LINE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")


def _extract_urls(text):
    urls = []
    for raw in re.findall(r"https?://\S+", text):
        while raw and raw[-1] in ".,;!?]":
            raw = raw[:-1]
        if raw.endswith(")") and raw.count(")") > raw.count("("):
            raw = raw[:-1]
        if raw:
            urls.append(raw)
    return urls


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []

    # 1. Validate sources list
    if not isinstance(sources, list) or len(sources) == 0:
        return ["no sources in sources.json"]

    seen_urls = set()
    sources_by_n = {}
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"invalid source entry (not a dict): {entry!r}")
            continue
        n = entry.get("n")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"source n must be an int, got {n!r}")
        url = str(entry.get("url") or "")
        if not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] url must start with http:// or https://, got {url!r}")
        if url in seen_urls:
            problems.append(f"duplicate url in sources.json: {url}")
        seen_urls.add(url)
        if isinstance(n, int) and not isinstance(n, bool):
            if n in sources_by_n:
                problems.append(f"duplicate source number n={n} in sources.json")
            sources_by_n[n] = entry

    source_numbers = set(sources_by_n.keys())

    # 2. Split report at "## References"
    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing '## References' heading")
        body = report_text
        ref_section = ""
    else:
        body = report_text[:matches[-1].start()]
        ref_section = report_text[matches[-1].end():]

    # 3. Find cited numbers in the BODY only
    cited = set()
    segments = _CODE.split(body)
    for i, segment in enumerate(segments):
        if i % 2 == 1:
            continue  # inside code block / span
        for match in _GROUP.finditer(segment):
            for num in _group_numbers(match.group(1)):
                cited.add(num)

    # 4. Cross-check cited with sources
    for n in sorted(cited):
        if n not in source_numbers:
            problems.append(f"[{n}] cited but missing from sources.json")

    for n in sorted(source_numbers):
        if n not in cited:
            problems.append(f"source [{n}] never cited")

    # 5 & 6. Validate References section
    if matches:
        ref_lines_by_n = {}
        for line in ref_section.strip().splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            line_match = _REF_LINE.match(line_str)
            if not line_match:
                continue
            ref_n = int(line_match.group(1))
            if ref_n in ref_lines_by_n:
                problems.append(f"duplicate reference line for [{ref_n}]")
            else:
                ref_lines_by_n[ref_n] = line_str

            if ref_n not in source_numbers:
                problems.append(f"reference [{ref_n}] is not in sources.json")

        for n, source_entry in sources_by_n.items():
            if n not in ref_lines_by_n:
                problems.append(f"missing reference line for source [{n}]")
            else:
                ref_line = ref_lines_by_n[n]
                clean_urls = _extract_urls(ref_line)
                if len(clean_urls) != 1:
                    problems.append(
                        f"reference line [{n}] must contain exactly one URL (found {len(clean_urls)})"
                    )
                else:
                    expected_url = str(source_entry.get("url") or "").strip()
                    if clean_urls[0] != expected_url:
                        problems.append(
                            f"reference line [{n}] URL mismatch: expected {expected_url} but found {clean_urls[0]}"
                        )

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
