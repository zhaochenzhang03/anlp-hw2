import argparse
import json
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString
from pypdf import PdfReader


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


# Below this many characters, the main-content element is treated as empty.
MIN_MAIN_CHARS = 300


def fetch_html(url: str) -> str:
    """
    Download the HTML content of a webpage.
    """
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20,
    )

    response.raise_for_status()

    return response.text


def clean_text(text: str) -> str:
    """
    Normalize whitespace while keeping paragraph boundaries.
    """
    text = text.replace("\xa0", " ")

    # Remove repeated spaces/tabs inside a line.
    text = re.sub(r"[ \t]+", " ", text)

    # Strip whitespace at line edges.
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Remove excessive blank lines.
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# Wikipedia elements that are never article content.
WIKIPEDIA_BOILERPLATE = [
    "sup.reference",
    ".mw-editsection",
    ".mw-references-wrap",
    ".reflist",
    ".references",
    ".navbox",
    ".navbox-styles",
    ".navbar",
    ".vertical-navbox",
    ".sidebar",
    "#toc",
    ".toc",
    ".hatnote",
    ".ambox",
    ".metadata",
    ".noprint",
    ".catlinks",
    ".sistersitebox",
    ".side-box",
    ".portalbox",
    ".shortdescription",
    ".mw-empty-elt",
    "link",
]

# Trailing Wikipedia sections that hold citations or links, not content.
WIKIPEDIA_SKIP_SECTIONS = {
    "see also",
    "notes",
    "references",
    "citations",
    "sources",
    "footnotes",
    "bibliography",
    "works cited",
    "further reading",
    "external links",
}

BLOCK_TAGS = [
    "p", "div", "section", "blockquote", "pre", "figure", "figcaption",
    "ul", "ol", "li", "dl", "dt", "dd",
    "table", "caption",
    "h1", "h2", "h3", "h4", "h5", "h6",
]


def _h2_block(tag):
    """
    Return the element that represents an h2 heading at sibling level,
    or None. Newer Wikipedia HTML wraps headings in div.mw-heading.
    """
    if tag.name == "h2":
        return tag

    if tag.name == "div" and "mw-heading2" in (tag.get("class") or []):
        return tag

    return None


def extract_wikipedia_text(body) -> str:
    """
    Extract article text from a Wikipedia .mw-parser-output element,
    keeping sentences intact and paragraphs/sections separated.
    """
    for tag in body.select(", ".join(WIKIPEDIA_BOILERPLATE)):
        tag.decompose()

    # Drop trailing sections such as References and External links,
    # from the heading up to the next h2 heading.
    for heading in body.find_all("h2"):
        if heading.get_text(strip=True).lower() not in WIKIPEDIA_SKIP_SECTIONS:
            continue

        block = heading
        if heading.parent is not None and _h2_block(heading.parent):
            block = heading.parent

        node = block.next_sibling
        while node is not None and not (
            getattr(node, "name", None) and _h2_block(node)
        ):
            following = node.next_sibling
            node.extract()
            node = following

        block.decompose()

    # Source-HTML newlines are not meaningful; collapse them so inline
    # links do not break sentences.
    for string in body.find_all(string=True):
        if type(string) is NavigableString and string.find_parent("pre") is None:
            string.replace_with(re.sub(r"\s+", " ", string))

    for br in body.find_all("br"):
        br.replace_with("\n")

    # Keep table rows (e.g. infoboxes) on one line each.
    for row in body.find_all("tr"):
        cells = [
            re.sub(r"\s+", " ", cell.get_text()).strip()
            for cell in row.find_all(["th", "td"], recursive=False)
        ]
        row.replace_with("\n" + " | ".join(c for c in cells if c) + "\n")

    for tag in body.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    return clean_text(body.get_text())


def extract_page(html: str) -> tuple[str, str]:
    """
    Extract the page title and readable text from HTML.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove content that is almost never useful for retrieval.
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "iframe",
        ]
    ):
        tag.decompose()

    # Extract title.
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    else:
        title = ""

    # Wikipedia pages get dedicated article-body extraction.
    wiki_body = soup.select_one("#mw-content-text .mw-parser-output")

    if wiki_body is not None:
        return title, extract_wikipedia_text(wiki_body)

    description = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    description = (description.get("content") or "").strip() if description else ""

    # Prefer the main content of the webpage when available; fall back to
    # the whole body when <main>/<article> is (nearly) empty, as on some
    # JavaScript-heavy sites.
    text = ""
    for main in (soup.find("main"), soup.find("article"), soup.body, soup):
        if main is None:
            continue

        # Keep some paragraph structure instead of joining everything together.
        text = clean_text(main.get_text(separator="\n", strip=True))

        if len(text) >= MIN_MAIN_CHARS:
            break

    # The meta description often states what the page is about in one line,
    # and is all that JavaScript-rendered pages offer without a browser.
    if description and description not in text:
        text = f"{description}\n\n{text}".strip()

    return title, text


def extract_pdf(content: bytes) -> tuple[str, str]:
    """
    Extract the title (from metadata, if any) and text of a PDF, one
    paragraph-separated block per page.
    """
    reader = PdfReader(BytesIO(content))

    title = ""
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip()

    pages = [page.extract_text() or "" for page in reader.pages]

    return title, clean_text("\n\n".join(pages))


def crawl_page(url: str, category: str, fetch_url: str | None = None, title: str | None = None) -> dict:
    """
    Download and parse one webpage or PDF into a structured document.

    fetch_url, if given, is where the content is actually downloaded from
    (e.g. a Wayback Machine copy of an official page that blocks crawlers);
    the document still records the official url, plus archived_url.
    """
    response = requests.get(fetch_url or url, headers=HEADERS, timeout=60)
    response.raise_for_status()

    is_pdf = "pdf" in response.headers.get("content-type", "") or response.content[:5] == b"%PDF-"
    if is_pdf:
        page_title, text = extract_pdf(response.content)
    else:
        page_title, text = extract_page(response.text)

    document = {
        "url": url,
        "title": title or page_title,
        "category": category,
        "text": text,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }

    if fetch_url:
        document["archived_url"] = fetch_url

    return document


def save_document(document: dict, output_path: Path) -> None:
    """
    Save one crawled document as JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            document,
            f,
            ensure_ascii=False,
            indent=2,
        )


def sitemap_urls(sitemap_url: str) -> list[str]:
    """
    Return the page URLs listed in one sitemap (urlset) file.
    """
    xml = fetch_html(sitemap_url)

    return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)


def expand_sources(sources: list[dict], failures: list[dict]) -> list[dict]:
    """
    Turn sitemap entries into one entry per matching page.

    A sitemap entry has "sitemap" (one URL or a list) instead of "url",
    plus an optional "include" regex. Its pages are saved under a
    subdirectory named after the entry.
    """
    expanded = []

    for source in sources:
        if "sitemap" not in source:
            expanded.append(source)
            continue

        sitemaps = source["sitemap"]
        if isinstance(sitemaps, str):
            sitemaps = [sitemaps]

        include = re.compile(source.get("include", ""))
        seen = set()

        for sitemap in sitemaps:
            try:
                urls = sitemap_urls(sitemap)
            except Exception as e:
                print(f"Sitemap FAILED: {sitemap}: {e}")
                failures.append({"url": sitemap, "error": str(e)})
                continue

            for url in urls:
                if url in seen or not include.search(url):
                    continue

                seen.add(url)
                slug = re.sub(r"[^A-Za-z0-9_-]+", "_", url.rstrip("/").rsplit("/", 1)[-1])
                expanded.append(
                    {
                        "name": f"{source['name']}/{slug}",
                        "url": url,
                        "category": source.get("category", "general"),
                    }
                )

        print(f"Sitemap source {source['name']}: {len(seen)} pages")

    return expanded


def batch_crawl(
    sources_path: Path,
    output_dir: Path,
    failures_path: Path,
    delay: float,
) -> None:
    """
    Crawl every source listed in a JSON file, skipping failures.
    """
    with sources_path.open("r", encoding="utf-8") as f:
        sources = json.load(f)

    failures = []
    sources = expand_sources(sources, failures)

    for i, source in enumerate(sources):
        if i > 0:
            time.sleep(delay)

        url = source["url"]
        print(f"[{i + 1}/{len(sources)}] Crawling: {url}")

        try:
            document = crawl_page(
                url=url,
                category=source.get("category", "general"),
                fetch_url=source.get("archived_url"),
                title=source.get("title"),
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            failures.append({"url": url, "error": str(e)})
            continue

        output_path = output_dir / f"{source['name']}.json"
        save_document(
            document=document,
            output_path=output_path,
        )

        print(f"  Title: {document['title']}")
        print(f"  Characters extracted: {len(document['text'])}")
        print(f"  Saved to: {output_path}")

    print(f"\nPages attempted: {len(sources)}, failures (pages + sitemaps): {len(failures)}")

    failures_path.parent.mkdir(parents=True, exist_ok=True)

    with failures_path.open("w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    print(f"Failures written to: {failures_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download one webpage (--url) or a list of webpages (--sources) "
            "and save their text as JSON."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--url",
        help="URL of the webpage to crawl.",
    )

    mode.add_argument(
        "--sources",
        help="JSON file listing sources (name, url, category) to batch crawl.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory for per-page JSON files in batch mode.",
    )

    parser.add_argument(
        "--failures",
        default="outputs/crawl_failures.json",
        help="Where to record failed URLs and errors in batch mode.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to sleep between requests in batch mode.",
    )

    parser.add_argument(
        "--category",
        default="general",
        help="Category of the webpage, e.g. cmu, sports, museum.",
    )

    parser.add_argument(
        "--output",
        help="Output JSON file (single-URL mode).",
    )

    args = parser.parse_args()

    if args.sources:
        batch_crawl(
            sources_path=Path(args.sources),
            output_dir=Path(args.output_dir),
            failures_path=Path(args.failures),
            delay=args.delay,
        )
        return

    if not args.output:
        parser.error("--output is required with --url")

    print(f"Crawling: {args.url}")

    document = crawl_page(
        url=args.url,
        category=args.category,
    )

    print(f"Title: {document['title']}")
    print(f"Characters extracted: {len(document['text'])}")

    output_path = Path(args.output)

    save_document(
        document=document,
        output_path=output_path,
    )

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()