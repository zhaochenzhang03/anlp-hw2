import argparse
import csv
import io
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

from crawl import HEADERS, clean_text, save_document


# Carnegie Museum of Art's CC0 collection data release (2015). The original
# cmoa/collection GitHub repository is gone; this is a public mirror of it.
DEFAULT_CSV_URL = "https://raw.githubusercontent.com/roll/collection/master/cmoa.csv"

# CSV columns rendered into the document text, with their labels.
OBJECT_FIELDS = [
    ("creation_date", "Date"),
    ("medium", "Medium"),
    ("classification", "Classification"),
    ("accession_number", "Accession number"),
    ("credit_line", "Credit line"),
    ("department", "Department"),
    ("date_acquired", "Date acquired"),
    ("physical_location", "Location (2015 data release)"),
]


def load_rows(csv_source: str) -> list[dict]:
    """
    Read the collection CSV from a local path or a URL.
    """
    if csv_source.startswith("http"):
        response = requests.get(csv_source, headers=HEADERS, timeout=60)
        response.raise_for_status()
        text = response.content.decode("utf-8")
    else:
        text = Path(csv_source).read_text(encoding="utf-8")

    return list(csv.DictReader(io.StringIO(text)))


def format_creator(row: dict) -> str:
    """
    Describe one creator, e.g. "Jean-Léon Gérôme (French, 1824–1904)".
    """
    name = row["full_name"].strip()
    details = [row["nationality"].strip()]

    years = "–".join(
        d[:4] for d in (row["birth_date"], row["death_date"]) if d.strip()
    )
    details.append(years)

    details = ", ".join(d for d in details if d)

    return f"{name} ({details})" if details else name


def build_document(rows: list[dict], retrieved_at: str) -> dict:
    """
    Build one document from all CSV rows of a single object.
    Objects with several creators appear as several rows.
    """
    first = rows[0]
    title = first["title"].strip()

    creators = []
    for row in rows:
        if row["full_name"].strip():
            creator = format_creator(row)
            if creator not in creators:
                creators.append(creator)

    lines = [
        f"{title}",
        "Carnegie Museum of Art collection record",
        f"Title: {title}",
    ]

    if creators:
        lines.append(f"Artist: {'; '.join(creators)}")

    for field, label in OBJECT_FIELDS:
        value = first[field].strip()
        if value:
            lines.append(f"{label}: {value}")

    provenance = first["provenance_text"].strip()
    if provenance:
        lines.append(f"Provenance: {provenance}")

    return {
        "url": first["web_url"].strip() or DEFAULT_CSV_URL,
        "title": f"{title} | Carnegie Museum of Art",
        "category": "museums",
        "text": clean_text("\n".join(lines)),
        "retrieved_at": retrieved_at,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Carnegie Museum of Art collection data into one JSON "
            "document per object."
        )
    )

    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV_URL,
        help="Path or URL of cmoa.csv.",
    )

    parser.add_argument(
        "--classification",
        default="paintings",
        help='Only keep objects with this classification, or "all".',
    )

    parser.add_argument(
        "--output-dir",
        default="data/raw/cmoa",
        help="Directory for the per-object JSON files.",
    )

    args = parser.parse_args()

    print(f"Loading: {args.csv}")
    rows = load_rows(args.csv)

    objects = {}
    for row in rows:
        if args.classification != "all" and row["classification"] != args.classification:
            continue

        objects.setdefault(row["id"], []).append(row)

    retrieved_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)

    for object_rows in objects.values():
        document = build_document(object_rows, retrieved_at)

        accession = object_rows[0]["accession_number"].strip() or object_rows[0]["id"]
        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", accession)

        save_document(
            document=document,
            output_path=output_dir / f"{filename}.json",
        )

    print(f"Saved {len(objects)} objects to: {output_dir}")


if __name__ == "__main__":
    main()
