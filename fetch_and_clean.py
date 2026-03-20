#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
抓取网页 + 文本清洗
用法示例：
    python fetch_and_clean.py --input urls.txt --output-dir data

urls.txt 每行一个 URL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

try:
    import requests  # type: ignore
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:
    BeautifulSoup = None

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}


@dataclass
class PageResult:
    url: str
    status: str
    title: str
    fetched_at: str
    raw_html_path: Optional[str]
    clean_text_path: Optional[str]
    metadata_path: Optional[str]
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
        skip_tags = {"script", "style", "noscript", "svg", "canvas", "header", "footer", "nav", "aside", "form", "button"}
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
            "advertisement", "ads", "promo", "newsletter", "social", "related-posts",
            "breadcrumb", "breadcrumbs", "cookie", "banner", "sidebar", "comments",
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
        return SimpleResponse(
            text=resp.text,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    req = Request(url, headers=DEFAULT_HEADERS)
    with urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
        return SimpleResponse(
            text=text,
            status_code=getattr(resp, "status", 200),
            headers=dict(resp.headers.items()),
        )


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
        text = post_clean_text("\n".join(parser.blocks))
        return {
            "url": url,
            "title": normalize_whitespace(parser.title),
            "clean_text": text,
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

    text = post_clean_text("\n".join(blocks))
    return {"url": url, "title": title, "clean_text": text}


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


def process_url(url: str, output_dir: str, delay_sec: float = 2.0, check_robots: bool = True) -> PageResult:
    from datetime import datetime, timezone

    fetched_at = datetime.now(timezone.utc).isoformat()
    raw_dir = os.path.join(output_dir, "raw_html")
    clean_dir = os.path.join(output_dir, "clean_text")
    meta_dir = os.path.join(output_dir, "metadata")
    ensure_dir(raw_dir)
    ensure_dir(clean_dir)
    ensure_dir(meta_dir)

    base = safe_filename(url)
    raw_html_path = os.path.join(raw_dir, base + ".html")
    clean_text_path = os.path.join(clean_dir, base + ".txt")
    metadata_path = os.path.join(meta_dir, base + ".json")

    try:
        if check_robots and not robots_allowed(url, user_agent="*"):
            return PageResult(url, "skipped", "", fetched_at, None, None, None, "Blocked by robots.txt")

        resp = fetch_html(url)
        html = resp.text
        save_text(raw_html_path, html)

        parsed = clean_text_from_html(html, url)
        save_text(clean_text_path, parsed["clean_text"])

        metadata = {
            "url": url,
            "title": parsed["title"],
            "status_code": resp.status_code,
            "content_type": resp.headers.get("Content-Type", ""),
            "fetched_at": fetched_at,
            "raw_html_path": raw_html_path,
            "clean_text_path": clean_text_path,
            "content_length": len(html),
        }
        save_json(metadata_path, metadata)
        time.sleep(delay_sec)
        return PageResult(url, "ok", parsed["title"], fetched_at, raw_html_path, clean_text_path, metadata_path)
    except Exception as e:
        return PageResult(url, "error", "", fetched_at, None, None, None, str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取网页并清洗文本")
    parser.add_argument("--input", required=True, help="URL 列表文件，每行一个 URL")
    parser.add_argument("--output-dir", default="data", help="输出目录")
    parser.add_argument("--delay-sec", type=float, default=2.0, help="请求间隔秒数")
    parser.add_argument("--no-robots-check", action="store_true", help="不检查 robots.txt（不推荐）")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    urls = read_urls(args.input)
    results: List[PageResult] = []

    print(f"[INFO] Loaded {len(urls)} URLs")
    if requests is None:
        print("[INFO] requests 未安装，已切换到 urllib 标准库抓取")
    if BeautifulSoup is None:
        print("[INFO] bs4 未安装，已切换到内置 HTMLParser 清洗")

    for idx, url in enumerate(urls, start=1):
        print(f"[INFO] ({idx}/{len(urls)}) Processing: {url}")
        result = process_url(url, args.output_dir, args.delay_sec, not args.no_robots_check)
        results.append(result)
        if result.status == "ok":
            print(f"[OK] {result.title}")
        else:
            print(f"[WARN] {result.status}: {result.error}")

    summary = {
        "total": len(results),
        "ok": sum(1 for r in results if r.status == "ok"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "error": sum(1 for r in results if r.status == "error"),
        "results": [asdict(r) for r in results],
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    save_json(summary_path, summary)
    print(f"[DONE] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
