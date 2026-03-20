#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量抓取网页 + 文本清洗。

示例：
    python fetch_and_clean.py --input urls.txt --output-dir data
    python fetch_and_clean.py --input urls.txt --output-dir data --retries 3 --export-markdown --export-jsonl
    python fetch_and_clean.py --resume-failures-from data/summary.json --output-dir rerun_data

特性：
- robots.txt 礼貌检查
- 请求失败自动重试 + 退避等待
- 运行日志输出到终端和文件
- 失败任务重跑
- 导出 clean_text、Markdown、JSONL、metadata、summary
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

requests_spec = importlib.util.find_spec("requests")
requests = importlib.import_module("requests") if requests_spec else None
bs4_spec = importlib.util.find_spec("bs4")
BeautifulSoup = getattr(importlib.import_module("bs4"), "BeautifulSoup") if bs4_spec else None

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}
DEFAULT_LOG_FILE = "fetch.log"
DEFAULT_SUMMARY_FILE = "summary.json"


@dataclass
class PageResult:
    url: str
    status: str
    title: str
    fetched_at: str
    raw_html_path: Optional[str]
    clean_text_path: Optional[str]
    metadata_path: Optional[str]
    markdown_path: Optional[str] = None
    attempts: int = 0
    error: Optional[str] = None


@dataclass
class SimpleResponse:
    text: str
    status_code: int
    headers: Dict[str, str]


class FallbackHTMLExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.in_title = False
        self.skip_stack: List[str] = []
        self.blocks: List[str] = []
        self.current_parts: List[str] = []
        self.current_tag: Optional[str] = None

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = dict(attrs)
        classes = attrs_dict.get("class", "")
        skip_tags = {
            "script", "style", "noscript", "svg", "canvas", "header",
            "footer", "nav", "aside", "form", "button",
        }
        if tag in skip_tags:
            self.skip_stack.append(tag)
            return
        if tag == "title":
            self.in_title = True
            return
        if self.skip_stack:
            return
        if tag in {"h1", "h2", "h3", "h4", "p", "li"}:
            self.current_tag = tag
            self.current_parts = []
        if tag == "table":
            self.blocks.append("\n[TABLE]")
        if tag in {"tr", "br"}:
            self.blocks.append("")
        if tag in {"td", "th"}:
            self.current_parts.append(" | ")
        noise_classes = [
            "advertisement", "ads", "promo", "newsletter", "social",
            "related-posts", "breadcrumb", "breadcrumbs", "cookie",
            "banner", "sidebar", "comments",
        ]
        if any(noise in classes for noise in noise_classes):
            self.skip_stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if self.skip_stack and self.skip_stack[-1] == tag:
            self.skip_stack.pop()
            return
        if tag == "title":
            self.in_title = False
            return
        if self.skip_stack:
            return
        if tag in {"h1", "h2", "h3", "h4", "p", "li"} and self.current_tag == tag:
            text = normalize_whitespace("".join(self.current_parts))
            if text:
                if tag.startswith("h"):
                    self.blocks.append(f"\n## {text}\n")
                elif tag == "li":
                    self.blocks.append(f"- {text}")
                else:
                    self.blocks.append(text)
            self.current_parts = []
            self.current_tag = None
        if tag == "table":
            self.blocks.append("[/TABLE]\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data
            return
        if self.skip_stack:
            return
        if self.current_tag:
            self.current_parts.append(data)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_urls(input_file: str) -> List[str]:
    urls: List[str] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def load_failed_urls(summary_file: str) -> List[str]:
    with open(summary_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results", [])
    urls = [item["url"] for item in results if item.get("status") in {"error", "skipped"}]
    return dedupe_keep_order(urls)


def dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def robots_allowed(url: str, user_agent: str = "*") -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


def fetch_html(url: str, timeout: int = 20) -> SimpleResponse:
    if requests is not None:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return SimpleResponse(resp.text, resp.status_code, dict(resp.headers))

    req = Request(url, headers=DEFAULT_HEADERS)
    with urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
        return SimpleResponse(
            text=text,
            status_code=getattr(resp, "status", 200),
            headers=dict(resp.headers.items()),
        )


def fetch_with_retry(
    url: str,
    timeout: int,
    retries: int,
    retry_backoff: float,
    logger: logging.Logger,
) -> tuple[SimpleResponse, int]:
    last_error: Optional[Exception] = None
    max_attempts = max(1, retries + 1)

    for attempt in range(1, max_attempts + 1):
        try:
            response = fetch_html(url, timeout=timeout)
            return response, attempt
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep_sec = retry_backoff * attempt
            logger.warning(
                "Fetch failed for %s on attempt %s/%s: %s. Retrying in %.1fs",
                url,
                attempt,
                max_attempts,
                exc,
                sleep_sec,
            )
            time.sleep(sleep_sec)

    assert last_error is not None
    raise last_error


def extract_main_content(soup):
    selectors = ["article", "main", '[role="main"]', "body"]
    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            return node
    return soup


def remove_noise_tags(node) -> None:
    noise_selectors = [
        "script", "style", "noscript", "svg", "canvas", "header", "footer", "nav", "aside",
        "form", "button", "figure .share", ".advertisement", ".ads", ".promo", ".newsletter",
        ".social", ".related-posts", ".breadcrumb", ".breadcrumbs", ".cookie", ".banner",
        ".sidebar", ".comments",
    ]
    for sel in noise_selectors:
        for tag in node.select(sel):
            tag.decompose()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text_from_html(html: str, url: str) -> Dict[str, Any]:
    if BeautifulSoup is None:
        parser = FallbackHTMLExtractor()
        parser.feed(html)
        clean_text = post_clean_text("\n".join(parser.blocks))
        return {
            "url": url,
            "title": normalize_whitespace(parser.title),
            "clean_text": clean_text,
            "markdown_text": build_markdown_document(normalize_whitespace(parser.title), url, clean_text),
        }

    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    main_node = extract_main_content(soup)
    remove_noise_tags(main_node)
    blocks: List[str] = []

    for elem in main_node.descendants:
        if not getattr(elem, "name", None):
            continue
        if elem.name in {"h1", "h2", "h3", "h4"}:
            txt = elem.get_text(" ", strip=True)
            if txt:
                blocks.append(f"\n## {txt}\n")
        elif elem.name == "li":
            txt = elem.get_text(" ", strip=True)
            if txt:
                blocks.append(f"- {txt}")
        elif elem.name == "p":
            txt = elem.get_text(" ", strip=True)
            if txt:
                blocks.append(txt)
        elif elem.name == "table":
            table_text = extract_table_text(elem)
            if table_text:
                blocks.append(table_text)

    clean_text = post_clean_text("\n".join(blocks))
    return {
        "url": url,
        "title": title,
        "clean_text": clean_text,
        "markdown_text": build_markdown_document(title, url, clean_text),
    }


def build_markdown_document(title: str, url: str, clean_text: str) -> str:
    lines = [f"# {title or 'Untitled'}", "", f"Source: {url}", "", clean_text.strip()]
    return "\n".join(lines).strip() + "\n"


def extract_table_text(table_tag) -> str:
    rows_out: List[str] = []
    rows = table_tag.find_all("tr")
    for tr in rows:
        cells = tr.find_all(["th", "td"])
        vals = [c.get_text(" ", strip=True) for c in cells]
        vals = [v for v in vals if v]
        if vals:
            rows_out.append(" | ".join(vals))
    if rows_out:
        return "\n[TABLE]\n" + "\n".join(rows_out) + "\n[/TABLE]\n"
    return ""


def post_clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: List[str] = []
    seen_short = set()
    for line in lines:
        if not line:
            cleaned_lines.append("")
            continue
        if re.fullmatch(r"[\W_]+", line):
            continue
        key = line.lower()
        if len(line) <= 40:
            if key in seen_short:
                continue
            seen_short.add(key)
        if is_noise_line(line):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = normalize_whitespace(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise_line(line: str) -> bool:
    lower = line.lower()
    noise_patterns = [
        "cookie", "privacy policy", "terms of use", "subscribe", "sign up", "follow us",
        "advertisement", "share this", "back to top", "menu", "search", "facebook",
        "instagram", "twitter", "linkedin",
    ]
    if any(pattern in lower for pattern in noise_patterns):
        return True
    return len(line) <= 2


def safe_filename(url: str) -> str:
    digest = sha1_text(url)[:12]
    parsed = urlparse(url)
    host = parsed.netloc.replace(":", "_")
    path = parsed.path.strip("/").replace("/", "_") or "root"
    base = f"{host}__{path}__{digest}"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", base)[:180]


def save_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_paths(output_dir: str, url: str) -> Dict[str, str]:
    base = safe_filename(url)
    return {
        "raw_html": os.path.join(output_dir, "raw_html", base + ".html"),
        "clean_text": os.path.join(output_dir, "clean_text", base + ".txt"),
        "metadata": os.path.join(output_dir, "metadata", base + ".json"),
        "markdown": os.path.join(output_dir, "markdown", base + ".md"),
    }


def process_url(
    url: str,
    output_dir: str,
    delay_sec: float,
    check_robots: bool,
    timeout: int,
    retries: int,
    retry_backoff: float,
    skip_existing: bool,
    export_markdown: bool,
    jsonl_path: Optional[str],
    logger: logging.Logger,
) -> PageResult:
    fetched_at = datetime.now(timezone.utc).isoformat()

    for subdir in ["raw_html", "clean_text", "metadata", "markdown"]:
        ensure_dir(os.path.join(output_dir, subdir))

    paths = build_paths(output_dir, url)

    if skip_existing and os.path.exists(paths["metadata"]):
        logger.info("Skipping %s because metadata already exists", url)
        return PageResult(
            url=url,
            status="skipped_existing",
            title="",
            fetched_at=fetched_at,
            raw_html_path=paths["raw_html"] if os.path.exists(paths["raw_html"]) else None,
            clean_text_path=paths["clean_text"] if os.path.exists(paths["clean_text"]) else None,
            metadata_path=paths["metadata"],
            markdown_path=paths["markdown"] if os.path.exists(paths["markdown"]) else None,
            attempts=0,
            error=None,
        )

    try:
        if check_robots and not robots_allowed(url, user_agent="*"):
            logger.warning("Blocked by robots.txt: %s", url)
            return PageResult(url, "skipped", "", fetched_at, None, None, None, None, 0, "Blocked by robots.txt")

        resp, attempts = fetch_with_retry(url, timeout, retries, retry_backoff, logger)
        html = resp.text
        save_text(paths["raw_html"], html)

        parsed = clean_text_from_html(html, url)
        save_text(paths["clean_text"], parsed["clean_text"])

        markdown_path: Optional[str] = None
        if export_markdown:
            save_text(paths["markdown"], parsed["markdown_text"])
            markdown_path = paths["markdown"]

        metadata = {
            "url": url,
            "title": parsed["title"],
            "status_code": resp.status_code,
            "content_type": resp.headers.get("Content-Type", ""),
            "fetched_at": fetched_at,
            "attempts": attempts,
            "raw_html_path": paths["raw_html"],
            "clean_text_path": paths["clean_text"],
            "markdown_path": markdown_path,
            "content_length": len(html),
        }
        save_json(paths["metadata"], metadata)

        if jsonl_path:
            append_jsonl(
                jsonl_path,
                {
                    "url": url,
                    "title": parsed["title"],
                    "fetched_at": fetched_at,
                    "attempts": attempts,
                    "clean_text": parsed["clean_text"],
                    "markdown_path": markdown_path,
                },
            )

        if delay_sec > 0:
            time.sleep(delay_sec)

        return PageResult(
            url=url,
            status="ok",
            title=parsed["title"],
            fetched_at=fetched_at,
            raw_html_path=paths["raw_html"],
            clean_text_path=paths["clean_text"],
            metadata_path=paths["metadata"],
            markdown_path=markdown_path,
            attempts=attempts,
            error=None,
        )
    except Exception as exc:
        logger.error("Failed to process %s: %s", url, exc)
        return PageResult(url, "error", "", fetched_at, None, None, None, None, retries + 1, str(exc))


def setup_logger(output_dir: str, log_file: str, verbose: bool) -> logging.Logger:
    ensure_dir(output_dir)
    logger = logging.getLogger("fetch_and_clean")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(os.path.join(output_dir, log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def write_summary(output_dir: str, results: List[PageResult], source_mode: str) -> str:
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_mode": source_mode,
        "total": len(results),
        "ok": sum(1 for r in results if r.status == "ok"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "skipped_existing": sum(1 for r in results if r.status == "skipped_existing"),
        "error": sum(1 for r in results if r.status == "error"),
        "results": [asdict(r) for r in results],
    }
    summary_path = os.path.join(output_dir, DEFAULT_SUMMARY_FILE)
    save_json(summary_path, summary)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量抓取网页并清洗文本")
    parser.add_argument("--input", help="URL 列表文件，每行一个 URL")
    parser.add_argument("--output-dir", default="data", help="输出目录")
    parser.add_argument("--delay-sec", type=float, default=2.0, help="请求间隔秒数")
    parser.add_argument("--timeout", type=int, default=20, help="单次请求超时秒数")
    parser.add_argument("--retries", type=int, default=2, help="失败后的重试次数")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="重试退避基数秒数")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE, help="日志文件名，写入 output-dir 下")
    parser.add_argument("--resume-failures-from", help="从既有 summary.json 中读取 error/skipped URL 重新跑")
    parser.add_argument("--skip-existing", action="store_true", help="如果 metadata 已存在，则跳过该 URL")
    parser.add_argument("--export-markdown", action="store_true", help="额外导出 Markdown 版本")
    parser.add_argument("--export-jsonl", action="store_true", help="额外导出聚合 JSONL 文件")
    parser.add_argument("--no-robots-check", action="store_true", help="不检查 robots.txt（不推荐）")
    parser.add_argument("--verbose", action="store_true", help="终端打印更详细日志")
    args = parser.parse_args()

    if not args.input and not args.resume_failures_from:
        parser.error("必须提供 --input 或 --resume-failures-from 其中之一")
    return args


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    logger = setup_logger(args.output_dir, args.log_file, args.verbose)

    if args.resume_failures_from:
        urls = load_failed_urls(args.resume_failures_from)
        source_mode = f"resume_failures_from:{args.resume_failures_from}"
    else:
        urls = read_urls(args.input)
        source_mode = f"input:{args.input}"

    urls = dedupe_keep_order(urls)
    jsonl_path = os.path.join(args.output_dir, "records.jsonl") if args.export_jsonl else None
    if jsonl_path:
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
        save_text(jsonl_path, "")

    logger.info("Loaded %s URLs", len(urls))
    logger.info("Source mode: %s", source_mode)
    logger.info("Retries=%s timeout=%ss delay=%.1fs", args.retries, args.timeout, args.delay_sec)
    if requests is None:
        logger.info("requests 未安装，已切换到 urllib 标准库抓取")
    if BeautifulSoup is None:
        logger.info("bs4 未安装，已切换到内置 HTMLParser 清洗")

    results: List[PageResult] = []
    for idx, url in enumerate(urls, start=1):
        logger.info("(%s/%s) Processing: %s", idx, len(urls), url)
        result = process_url(
            url=url,
            output_dir=args.output_dir,
            delay_sec=args.delay_sec,
            check_robots=not args.no_robots_check,
            timeout=args.timeout,
            retries=args.retries,
            retry_backoff=args.retry_backoff,
            skip_existing=args.skip_existing,
            export_markdown=args.export_markdown,
            jsonl_path=jsonl_path,
            logger=logger,
        )
        results.append(result)
        logger.info("Result status=%s attempts=%s url=%s", result.status, result.attempts, url)

    summary_path = write_summary(args.output_dir, results, source_mode)
    logger.info("Summary saved to: %s", summary_path)
    if jsonl_path:
        logger.info("JSONL saved to: %s", jsonl_path)


if __name__ == "__main__":
    main()
