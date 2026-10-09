"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree

from dotenv import load_dotenv
import httpx
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again."""
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as e:
            if attempt == attempts - 1:
                raise
            if e.retry_after is not None and e.retry_after > 0:
                delay = float(e.retry_after)
            else:
                delay = base * (2 ** attempt) + random.uniform(0.1, 1.0)
            delay = min(delay, cap)
            time.sleep(delay)


def _http_get(url, params=None, headers=None, timeout=20.0):
    """Helper to perform HTTP GET with retryable classification."""
    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=timeout)
    except httpx.TransportError as te:
        raise RetryableError(f"Transport error: {te}") from te

    if resp.status_code == 429 or (500 <= resp.status_code <= 504):
        retry_after = None
        ra = resp.headers.get("Retry-After")
        if ra and ra.isdigit():
            retry_after = float(ra)
        raise RetryableError(f"HTTP {resp.status_code}", retry_after=retry_after)
    resp.raise_for_status()
    return resp


# ---- TODO 2: arXiv ----
_LAST_ARXIV_CALL = 0.0


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    try:
        terms = re.findall(r"[\w\-]+", query)
        if not terms:
            return "NO RESULTS"

        global _LAST_ARXIV_CALL
        elapsed = time.time() - _LAST_ARXIV_CALL
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        _LAST_ARXIV_CALL = time.time()

        search_query = " AND ".join(f"all:{t}" for t in terms)
        clamped_max = max(1, min(max_results, 30))
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": clamped_max,
        }

        resp = with_retry(lambda: _http_get(ARXIV_URL, params=params), attempts=6, base=3.0, cap=60.0)

        root = xml.etree.ElementTree.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "NO RESULTS"

        records = []
        for entry in entries:
            id_elem = entry.find("atom:id", ns)
            pub_elem = entry.find("atom:published", ns)
            title_elem = entry.find("atom:title", ns)
            sum_elem = entry.find("atom:summary", ns)

            raw_id = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
            paper_id = raw_id.split("/abs/")[-1]
            paper_id = re.sub(r"v\d+$", "", paper_id)

            url = f"https://arxiv.org/abs/{paper_id}"
            published = pub_elem.text.strip()[:10] if pub_elem is not None and pub_elem.text else ""

            title = re.sub(r"\s+", " ", title_elem.text).strip() if title_elem is not None and title_elem.text else "Untitled"
            summary = re.sub(r"\s+", " ", sum_elem.text).strip() if sum_elem is not None and sum_elem.text else ""
            if len(summary) > 600:
                summary = summary[:600] + "..."

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary
            })

        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        clamped_limit = max(1, min(limit, 100))
        params = {"limit": clamped_limit}
        if date and date.strip():
            params["date"] = date.strip()

        resp = with_retry(lambda: _http_get(HF_DAILY_URL, params=params), attempts=5, base=1.5, cap=30.0)
        items = resp.json()
        if not isinstance(items, list) or not items:
            return "NO RESULTS"

        records = []
        kw = keyword.strip().lower() if keyword else ""
        for item in items:
            paper = item.get("paper") if isinstance(item, dict) else None
            if not paper or not paper.get("id"):
                continue
            p_id = paper["id"]
            title = re.sub(r"\s+", " ", str(paper.get("title") or item.get("title") or "")).strip()
            summary = re.sub(r"\s+", " ", str(paper.get("summary") or item.get("summary") or "")).strip()
            if kw and (kw not in title.lower() and kw not in summary.lower()):
                continue
            if len(summary) > 600:
                summary = summary[:600] + "..."

            pub = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            upvotes = int(paper.get("upvotes") or item.get("upvotes") or 0)
            github = paper.get("githubRepo") or ""
            stars = int(paper.get("githubStars") or 0)
            records.append({
                "id": p_id,
                "url": f"https://huggingface.co/papers/{p_id}",
                "published": pub,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars
            })

        records.sort(key=lambda r: r.get("upvotes", 0), reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        clamped_limit = max(1, min(limit, 50))
        params = {"q": query, "limit": clamped_limit}
        resp = with_retry(lambda: _http_get(HF_SEARCH_URL, params=params), attempts=5, base=1.5, cap=30.0)
        items = resp.json()
        if not isinstance(items, list) or not items:
            return "NO RESULTS"

        records = []
        for item in items:
            paper = item.get("paper") if isinstance(item, dict) else item
            if not isinstance(paper, dict) or not paper.get("id"):
                continue
            p_id = paper["id"]
            title = re.sub(r"\s+", " ", str(paper.get("title") or item.get("title") or "")).strip()
            raw_summary = paper.get("ai_summary") or paper.get("summary") or item.get("summary") or ""
            summary = re.sub(r"\s+", " ", str(raw_summary)).strip()
            if len(summary) > 600:
                summary = summary[:600] + "..."

            pub = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            upvotes = int(paper.get("upvotes") or item.get("upvotes") or 0)
            github = paper.get("githubRepo") or ""
            stars = int(paper.get("githubStars") or 0)
            records.append({
                "id": p_id,
                "url": f"https://huggingface.co/papers/{p_id}",
                "published": pub,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars
            })

        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _redact_key(msg: str) -> str:
    key = os.getenv("EXA_API_KEY", "").strip()
    if key and key in msg:
        msg = msg.replace(key, "[REDACTED]")
    return msg


def _call_exa(tool_name: str, arguments: dict) -> str:
    key = os.getenv("EXA_API_KEY", "").strip()
    endpoint = f"{EXA_URL}?exaApiKey={key}" if key else EXA_URL
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments}
    }

    def _do_post():
        try:
            resp = httpx.post(endpoint, headers=headers, json=payload, timeout=30.0)
        except httpx.TransportError as te:
            raise RetryableError(f"Exa transport error: {te}") from te

        if resp.status_code == 429 or (500 <= resp.status_code <= 504):
            ra = resp.headers.get("Retry-After")
            retry_after = float(ra) if ra and ra.isdigit() else 10.0
            raise RetryableError(f"Exa HTTP {resp.status_code}", retry_after=retry_after)

        resp.raise_for_status()
        text = resp.text

        data_obj = None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                raw_json = line[5:].strip()
                try:
                    data_obj = json.loads(raw_json)
                    break
                except json.JSONDecodeError:
                    continue
        if data_obj is None:
            try:
                data_obj = resp.json()
            except Exception as exc:
                raise RuntimeError(f"Exa invalid response format: {text[:200]}") from exc

        if "error" in data_obj:
            err_msg = str(data_obj["error"])
            if "rate" in err_msg.lower() or "limit" in err_msg.lower():
                raise RetryableError(f"Exa rate limit: {err_msg}", retry_after=15.0)
            raise RuntimeError(f"Exa error: {err_msg}")

        result = data_obj.get("result", {})
        content = result.get("content", [])
        texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        full_text = "\n\n".join(texts).strip()

        meta_str = str(result.get("_meta", "")).lower()
        text_lower = full_text.lower()
        if (
            "rate limit" in text_lower
            or "too many requests" in text_lower
            or "rate_limit" in meta_str
            or "exceeded" in meta_str
        ):
            raise RetryableError("Exa rate limit signal detected in result", retry_after=20.0)

        return full_text

    return with_retry(_do_post, attempts=5, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        obj = objective.strip() or f"find academic research and survey papers about {query}"
        num = max(1, min(num_results, 10))
        res = _call_exa("web_search_exa", {"query": query, "objective": obj, "numResults": num})
        return res if res else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact_key(str(exc))}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        res = _call_exa("web_fetch_exa", {"urls": [url]})
        if not res:
            return "NO RESULTS"
        if len(res) > 12000:
            res = res[:12000] + "\n\n...[TRUNCATED]"
        return res
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_redact_key(str(exc))}"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
