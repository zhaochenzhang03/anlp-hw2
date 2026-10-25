import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path


# Chunk sizes are measured in approximate tokens: words and punctuation marks.
TOKEN_RE = re.compile(r"\w+|[^\w\s]")

MIN_TOKENS = 300
MAX_TOKENS = 500
OVERLAP_TOKENS = 75

# Room left in each chunk for a "Section: ..." line.
SECTION_RESERVE_TOKENS = 12

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")

# Repeated UI labels on event pages; they carry no content.
EVENT_UI_LINES = {
    "Save",
    "Map",
    "Open in Google Maps",
    "Buy Tickets",
    "Details",
    "Full View",
    "Website",
}

# Event-page labels whose value is on the next line.
EVENT_LABELS = {"Phone", "Admission"}

EVENT_DATE_RE = re.compile(
    r"^(January|February|March|April|May|June|July|August|September|"
    r"October|November|December) \d{1,2}, \d{4}"
)
EVENT_TIME_RE = re.compile(r"^(\d{1,2}):\d\d ([AP]M)$")
EVENT_END_TIME_RE = re.compile(r"^–(\d{1,2}):\d\d ([AP]M)$")

# Footer, newsletter and navigation wording on generic pages.
FOOTER_RE = re.compile(
    r"privacy|cookie|terms of (use|service)|©|copyright|all rights reserved|"
    r"subscribe|newsletter|mailing list|email list|sign up|follow (us|on)|"
    r"facebook|instagram|twitter|youtube|tiktok|linkedin|^share$|^tweet$|"
    r"site credits|advertising|final score|^winner:|^details$|played moment|"
    r"^(home|away) team$|^(stadium|location|attendance|duration):$|"
    r"^score by period$|^(team|period|f)$|captcha|^submit$|^required$|"
    r"indicates required|^email( address)?$|^phone number$|^contact us$",
    re.IGNORECASE,
)

# A line repeated in this many documents is site navigation or footer.
SHARED_LINE_MIN_DOCS = 3

# A trailing chunk is dropped when this share of its tokens is footer or
# shared navigation.
BOILERPLATE_SHARE = 0.7

# Repeated UI labels on athletics schedule pages.
SCHEDULE_UI_LINES = {
    "Game Center",
    "Box Score",
    "Recap",
    "Photos",
    "Book Your Tailgate!",
    "History",
    "Game Info",
    "Listen",
    "Live Stats",
    "Tickets",
    "Watch",
    "Add To Calendar",
    "Text Only",
    "Skip Ad",
    "Sponsored by:",
    "Days",
    "Hours",
    "Minutes",
    "Seconds",
    "All Games",
    "Home Games",
    "Away Games",
    "Conference Games",
    "Non-conference Games",
    "View",
    "Type:",
    "List View",
    "Table View",
    "not selected",
}

# Lines that are only punctuation left over from table cells.
PUNCT_LINE_RE = re.compile(r"^[\s|/,;:()•º·\-–]*$")

# Season/year dropdowns, e.g. "2026", "2025-26".
YEAR_LINE_RE = re.compile(r"^(19|20)\d\d(-\d\d)?$")

MILLAGE_RE = re.compile(r"^\d+\.\d{3,4}$")
SCHEDULE_DATE_RE = re.compile(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}$")
GAME_DATE_RE = re.compile(r"^[A-Z][a-z]{2} \d{1,2} \([A-Z][a-z]{2}\)$")

# Pages whose table cells were extracted one per line. Each rule says how
# to rebuild a table row so it lands on a single line (and in one chunk):
#   begin_after: table starts after this line
#   end:         a row ends at a line matching this regex
#   start:       start(lines, i) says whether a row starts at line i
#   row_check:   fixed-width rows; check(lines) decides if a row starts here
#   header:      column names, prepended to every chunk of the page
OPPONENT_MARKERS = ("vs", "at")

# Pitt lists each game as "vs/at, opponent, ..., date, time".
OPPONENT_FIRST_SCHEDULE = {
    "start": lambda lines, i: lines[i] in OPPONENT_MARKERS,
}

# Duquesne and RMU list each game as "date, ..., vs/at, opponent, ...";
# RMU repeats the date right after the opponent, which is not a new game.
DATE_FIRST_SCHEDULE = {
    "start": lambda lines, i: (
        GAME_DATE_RE.match(lines[i]) is not None
        and not (i >= 2 and lines[i - 2] in OPPONENT_MARKERS)
    ),
}

TABLE_RULES = {
    "allegheny_municipal_millage": {
        "begin_after": "Land",
        "end": MILLAGE_RE,
        "header": "Municipality | Tax collector | Phone | Millage (mills)",
    },
    "allegheny_school_millage": {
        "begin_after": "Land",
        "end": MILLAGE_RE,
        "header": "School district | Municipalities | Millage (mills)",
    },
    "penguins_schedule_2026_27_cbs": {
        "start": lambda lines, i: SCHEDULE_DATE_RE.match(lines[i]) is not None,
    },
    "steelers_roster": {
        "row_width": 8,
        "row_check": lambda row: (
            re.fullmatch(r"\d{1,2}", row[1]) is not None
            and re.fullmatch(r"[A-Z/]{1,5}", row[2]) is not None
        ),
        "header": "Player | # | Pos | HT | WT | Age | Exp | College",
    },
}

# A row never spans more lines than this; the rest is ordinary text.
MAX_ROW_LINES = 25

# Schedule page widgets that follow the table (related news and videos,
# an empty box-score template); they are dropped.
TABLE_END_RE = re.compile(r"^(Related (Videos|Headlines|News)|Score By Period)$")


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_documents(raw_dir: Path) -> tuple[list[dict], list[str]]:
    """
    Load every crawled document under raw_dir, skipping files that are
    not valid documents (failure logs, malformed JSON, missing fields).
    """
    documents = []
    skipped = []

    for path in sorted(raw_dir.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            skipped.append(str(path))
            continue

        if not (
            isinstance(data, dict)
            and isinstance(data.get("text"), str)
            and data["text"].strip()
            and isinstance(data.get("url"), str)
        ):
            skipped.append(str(path))
            continue

        relative = path.relative_to(raw_dir)
        doc_id = relative.with_suffix("").as_posix()

        documents.append(
            {
                "doc_id": doc_id,
                "source": relative.parts[0].removesuffix(".json"),
                "source_file": relative.as_posix(),
                "url": data["url"],
                "title": (data.get("title") or "").strip(),
                "category": data.get("category", "general"),
                "retrieved_at": data.get("retrieved_at", ""),
                "text": data["text"],
            }
        )

    return documents, skipped


def deduplicate(documents: list[dict]) -> tuple[list[dict], list[tuple[str, str]]]:
    """
    Keep the first document (in sorted path order) for each normalized text.
    Returns the kept documents and (removed doc_id, kept doc_id) pairs.
    """
    seen = {}
    kept = []
    removed = []

    for doc in documents:
        key = hashlib.sha1(normalize_for_dedup(doc["text"]).encode("utf-8")).hexdigest()

        if key in seen:
            removed.append((doc["doc_id"], seen[key]))
            continue

        seen[key] = doc["doc_id"]
        kept.append(doc)

    return kept, removed


def doc_type(doc: dict) -> str:
    if doc["source"] == "cmoa":
        return "museum_record"

    if doc["source"] in ("visitpittsburgh_events", "arcade_comedy_events"):
        return "event"

    if table_rule(doc) is not None:
        return "table"

    return "web"


def table_rule(doc: dict) -> dict | None:
    if doc["source"] in TABLE_RULES:
        return TABLE_RULES[doc["source"]]

    if re.search(r"_schedule_\d{4}", doc["source"]):
        if doc["source"].startswith("pitt_"):
            return OPPONENT_FIRST_SCHEDULE
        return DATE_FIRST_SCHEDULE

    return None


def clean_event_text(text: str) -> str:
    """
    Strip repeated UI labels and placeholder times from an event page.
    """
    lines = text.split("\n")

    while lines and lines[0] in ("Home", "< Back"):
        lines = lines[1:]

    remove_placeholder_times(lines)

    cleaned = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if line in EVENT_UI_LINES:
            i += 1
            continue

        if line in EVENT_LABELS and i + 1 < len(lines) and lines[i + 1].strip():
            cleaned.append(f"{line}: {lines[i + 1]}")
            i += 2
            continue

        cleaned.append(line)
        i += 1

    return "\n".join(cleaned)


def remove_placeholder_times(lines: list[str]) -> None:
    """
    Remove the times an event page shows when no real time is set, in place.
    Only the header times right after the date are touched:
    - a "12:00 AM" start (the site's default; e.g. football games listed at
      12:00 AM-3:00 AM), together with its end time;
    - an end of "12:00 PM" after a PM start, which would be before the start.
    Unknown times are left out, never replaced.
    """
    for i in range(min(len(lines), 3)):
        if EVENT_DATE_RE.match(lines[i]):
            break
    else:
        return

    start = i + 1
    if start >= len(lines):
        return

    start_time = EVENT_TIME_RE.match(lines[start])
    if start_time is None:
        return

    has_end = start + 1 < len(lines) and EVENT_END_TIME_RE.match(lines[start + 1])

    if lines[start] == "12:00 AM":
        del lines[start:start + (2 if has_end else 1)]
    elif (
        has_end
        and lines[start + 1] == "–12:00 PM"
        and start_time.group(2) == "PM"
        and start_time.group(1) != "12"
    ):
        del lines[start + 1]


def drop_noise_lines(lines: list[str], ui_lines: set[str]) -> list[str]:
    """
    Remove UI labels, punctuation-only lines, year dropdowns (runs of five
    or more bare years), and immediately repeated lines.
    """
    lines = [
        line for line in lines
        if line not in ui_lines and not (line and PUNCT_LINE_RE.match(line))
    ]

    kept = []
    i = 0
    while i < len(lines):
        j = i
        while j < len(lines) and YEAR_LINE_RE.match(lines[j]):
            j += 1

        if j - i >= 5:
            i = j
            continue

        for line in lines[i:max(j, i + 1)]:
            if not kept or line != kept[-1] or not line:
                kept.append(line)

        i = max(j, i + 1)

    return kept


def join_row(cells: list[str]) -> str:
    return " | ".join(cells)


def rebuild_table_rows(text: str, rule: dict) -> str:
    """
    Join the one-cell-per-line rows of a table page into one line per row.
    Lines outside the table are kept as they are.
    """
    lines = drop_noise_lines(text.split("\n"), SCHEDULE_UI_LINES)
    out = []

    begin = 0
    if "begin_after" in rule and rule["begin_after"] in lines:
        begin = lines.index(rule["begin_after"]) + 1
        out.extend(lines[:begin])

    i = begin

    if "row_width" in rule:
        width = rule["row_width"]
        while i < len(lines):
            row = lines[i:i + width]
            if len(row) == width and rule["row_check"](row):
                out.append(join_row(row))
                i += width
            else:
                out.append(lines[i])
                i += 1

        return "\n".join(out)

    if "end" in rule:
        row = []
        while i < len(lines):
            line = lines[i]
            row.append(line)
            i += 1

            if rule["end"].match(line):
                # A second number right after is the land millage column.
                if i < len(lines) and rule["end"].match(lines[i]):
                    row.append(lines[i])
                    i += 1

                out.append(join_row(row))
                row = []
            elif len(row) >= MAX_ROW_LINES:
                out.extend(row)
                row = []

        out.extend(row)

        return "\n".join(out)

    # Rows that begin at a start line and run to the next one.
    starts = [k for k in range(i, len(lines)) if rule["start"](lines, k)]
    if not starts:
        return "\n".join(out + lines[i:])

    out.extend(lines[i:starts[0]])

    table_end = next(
        (k for k in range(starts[0], len(lines)) if TABLE_END_RE.match(lines[k])),
        len(lines),
    )
    starts = [k for k in starts if k < table_end]

    for n, begin in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else table_end

        if end - begin <= MAX_ROW_LINES:
            out.append(join_row(lines[begin:end]))
        elif n + 1 == len(starts):
            # The last row runs into the page footer.
            out.append(join_row(lines[begin:begin + MAX_ROW_LINES]))
            out.extend(lines[begin + MAX_ROW_LINES:end])
        else:
            # Too long to be one row (e.g. a standings box); keep as is.
            out.extend(lines[begin:end])

    return "\n".join(out)


def is_heading(block: str) -> bool:
    """
    A paragraph-separated block that looks like a section heading, e.g.
    "Geography" or "19th century": one short line, without the
    punctuation of table rows and image captions.
    """
    return (
        "\n" not in block
        and len(block.split()) <= 5
        and not re.search(r"[.!?:|,;(%]", block)
        and re.match(r"[A-Za-z]|\d+(st|nd|rd|th|s)\b", block) is not None
    )


def split_long_text(text: str, limit: int) -> list[str]:
    """
    Split text that is over the token limit: by sentences first, then by
    words for sentences that are still too long.
    """
    pieces = []

    for sentence in SENTENCE_SPLIT_RE.split(text):
        if count_tokens(sentence) <= limit:
            pieces.append(sentence)
            continue

        words = sentence.split()
        current = []
        for word in words:
            if current and count_tokens(" ".join(current + [word])) > limit:
                pieces.append(" ".join(current))
                current = []
            current.append(word)

        if current:
            pieces.append(" ".join(current))

    return pieces


def build_units(text: str, limit: int) -> list[dict]:
    """
    Turn text into units for chunking. Each unit is a line (or a sentence of
    a line that is too long), with the strength of the boundary before it:
    2 = paragraph break, 1 = line break, 0 = sentence break. Units also
    record the section heading they fall under, when the text has headings.
    """
    units = []
    has_paragraphs = "\n\n" in text
    heading = ""

    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue

        if has_paragraphs and is_heading(block):
            heading = block

        for line_number, line in enumerate(block.split("\n")):
            line = line.strip()
            if not line:
                continue

            boundary = 2 if line_number == 0 and has_paragraphs else 1

            if count_tokens(line) <= limit:
                pieces = [line]
            else:
                pieces = split_long_text(line, limit)

            for piece_number, piece in enumerate(pieces):
                units.append(
                    {
                        "text": piece,
                        "tokens": count_tokens(piece),
                        "boundary": boundary if piece_number == 0 else 0,
                        "heading": heading,
                    }
                )

    return units


def join_units(units: list[dict]) -> str:
    text = ""
    for unit in units:
        if not text:
            text = unit["text"]
        elif unit["boundary"] == 2:
            text += "\n\n" + unit["text"]
        elif unit["boundary"] == 1:
            text += "\n" + unit["text"]
        else:
            text += " " + unit["text"]

    return text


def chunk_units(units: list[dict], max_tokens: int, min_tokens: int, overlap: int) -> list[list[dict]]:
    """
    Group units into chunks of about min_tokens..max_tokens, cutting at the
    strongest boundary available, with about `overlap` tokens repeated
    from the end of the previous chunk.
    """
    chunks = []
    start = 0

    while start < len(units):
        total = 0
        end = start
        while end < len(units) and total + units[end]["tokens"] <= max_tokens:
            total += units[end]["tokens"]
            end += 1

        end = max(end, start + 1)

        if end >= len(units):
            chunks.append(units[start:])
            break

        # Cut before the strongest boundary once the chunk is big enough;
        # among equally strong boundaries, take the latest.
        best = end
        best_strength = -1
        size = 0
        for k in range(start, end):
            size += units[k]["tokens"]
            cut = k + 1
            if size >= min_tokens and units[cut]["boundary"] >= best_strength:
                best = cut
                best_strength = units[cut]["boundary"]

        chunks.append(units[start:best])

        # Overlap: the largest suffix within the budget, preferring one that
        # starts at a line or paragraph boundary.
        next_start = best
        size = 0
        candidates = []
        for k in range(best - 1, start, -1):
            size += units[k]["tokens"]
            if size > overlap:
                break
            candidates.append(k)

        line_starts = [k for k in candidates if units[k]["boundary"] >= 1]
        if line_starts:
            next_start = line_starts[-1]
        elif candidates:
            next_start = candidates[-1]

        start = next_start

    return chunks


def compact_museum_text(text: str) -> str:
    """
    Fit a museum record into MAX_TOKENS for retrieval. The structured
    fields come first and are kept whole; only the trailing provenance
    is shortened.
    """
    if count_tokens(text) <= MAX_TOKENS:
        return text

    fields, _, provenance = text.partition("\nProvenance: ")
    marker = " … (provenance shortened)"
    budget = MAX_TOKENS - count_tokens(fields) - count_tokens("Provenance:" + marker)

    words = []
    for word in provenance.split():
        if count_tokens(" ".join(words + [word])) > budget:
            break
        words.append(word)

    return f"{fields}\nProvenance: {' '.join(words)}{marker}"


def chunk_document(doc: dict) -> list[dict]:
    """
    Return the chunks for one document as {"text": ...} dicts. A museum
    record whose text had to be shortened also keeps its "full_text".
    """
    kind = doc_type(doc)
    text = doc["text"]
    title = doc["title"]

    if kind == "museum_record":
        # One chunk per artwork: title, artist, accession number, credit
        # line, department, dates and location stay together.
        compact = compact_museum_text(text)
        if compact == text:
            return [{"text": text}]
        return [{"text": compact, "full_text": text}]

    header = ""

    if kind == "event":
        text = clean_event_text(text)
    elif kind == "table":
        rule = table_rule(doc)
        text = rebuild_table_rows(text, rule)
        header = rule.get("header", "")

    # Pages that already open with their title do not need it repeated.
    prefix_lines = [title] if title and not text.startswith(title) else []
    if header:
        prefix_lines.append(header)

    prefix = "\n".join(prefix_lines)
    prefix_tokens = count_tokens(prefix)

    if kind == "event" and count_tokens(text) + prefix_tokens <= MAX_TOKENS:
        return [{"text": f"{prefix}\n{text}" if prefix else text}]

    budget = MAX_TOKENS - prefix_tokens - SECTION_RESERVE_TOKENS
    units = build_units(text, budget)

    chunks = []
    for group in chunk_units(units, budget, MIN_TOKENS - prefix_tokens, OVERLAP_TOKENS):
        lines = list(prefix_lines)

        heading = group[0]["heading"]
        if heading and group[0]["text"] != heading:
            lines.append(f"Section: {heading}")

        lines.append(join_units(group))
        chunks.append({"text": "\n".join(lines)})

    return chunks


def shared_lines(documents: list[dict]) -> set[str]:
    """
    Lines that appear in several generic pages: navigation and footers.
    """
    counts = Counter()
    for doc in documents:
        if doc_type(doc) in ("web", "table"):
            counts.update({line.strip() for line in doc["text"].split("\n") if line.strip()})

    return {line for line, n in counts.items() if n >= SHARED_LINE_MIN_DOCS}


def is_boilerplate(text: str, title: str, shared: set[str]) -> bool:
    """
    Whether a chunk is mostly footer, newsletter or navigation text.
    """
    lines = [
        line.strip() for line in text.split("\n")
        if line.strip() and line.strip() != title and not line.startswith("Section: ")
    ]
    if not lines:
        return True

    total = sum(count_tokens(line) for line in lines)
    noise = sum(
        count_tokens(line) for line in lines
        if line in shared or FOOTER_RE.search(line) or PUNCT_LINE_RE.match(line)
    )

    return total == 0 or noise / total >= BOILERPLATE_SHARE


def build_chunks(documents: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Chunk every document. Trailing chunks of generic pages that are mostly
    boilerplate are dropped; their chunk IDs are returned.
    """
    shared = shared_lines(documents)
    records = []
    dropped = []

    for doc in documents:
        chunks = chunk_document(doc)

        if doc_type(doc) in ("web", "table"):
            while len(chunks) > 1 and is_boilerplate(chunks[-1]["text"], doc["title"], shared):
                dropped.append(f"{doc['doc_id']}#{len(chunks) - 1}")
                chunks.pop()

        for index, chunk in enumerate(chunks):
            text = chunk["text"]
            record = {
                "chunk_id": f"{doc['doc_id']}#{index}",
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "source_file": doc["source_file"],
                "url": doc["url"],
                "title": doc["title"],
                "category": doc["category"],
                "doc_type": doc_type(doc),
                "retrieved_at": doc["retrieved_at"],
                "chunk_index": index,
                "n_tokens": count_tokens(text),
                "text": text,
            }
            if "full_text" in chunk:
                record["full_text"] = chunk["full_text"]

            records.append(record)

    return records, dropped


def print_diagnostics(documents, skipped, removed, records, dropped) -> None:
    sizes = [r["n_tokens"] for r in records]

    print(f"Source documents loaded: {len(documents)}")
    print(f"Files skipped (not documents): {len(skipped)}")
    print(f"Duplicates removed: {len(removed)}")
    for doc_id, kept in removed:
        print(f"  {doc_id} (same text as {kept})")

    print(f"Boilerplate trailing chunks dropped: {len(dropped)}")
    for chunk_id in dropped:
        print(f"  {chunk_id}")

    print(f"Total chunks: {len(records)}")
    print(f"Museum records shortened for retrieval: {sum('full_text' in r for r in records)}")

    print("Chunks by category:")
    for category, n in sorted(Counter(r["category"] for r in records).items()):
        print(f"  {category}: {n}")

    print("Chunks by document type:")
    for kind, n in sorted(Counter(r["doc_type"] for r in records).items()):
        print(f"  {kind}: {n}")

    print(
        "Chunk tokens: "
        f"mean {statistics.mean(sizes):.0f}, "
        f"median {statistics.median(sizes):.0f}, "
        f"min {min(sizes)}, max {max(sizes)}, "
        f"over 512: {sum(n > 512 for n in sizes)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Split crawled documents into chunks for retrieval."
    )

    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory of crawled JSON documents (searched recursively).",
    )

    parser.add_argument(
        "--output",
        default="data/processed/chunks.jsonl",
        help="Output JSONL file, one chunk per line.",
    )

    args = parser.parse_args()

    loaded, skipped = load_documents(Path(args.raw_dir))
    documents, removed = deduplicate(loaded)
    records, dropped = build_chunks(documents)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print_diagnostics(loaded, skipped, removed, records, dropped)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
