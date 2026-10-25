import argparse
from datetime import datetime, timezone
from pathlib import Path

import requests

from crawl import HEADERS, save_document


# MLB's official Stats API: the Pirates (team 134) 40-man roster with each
# player's biographical fields.
DEFAULT_URL = "https://statsapi.mlb.com/api/v1/teams/134/roster/40Man?hydrate=person"

# Public player page, recorded as each document's URL.
PLAYER_PAGE = "https://www.mlb.com/player/{slug}"


def player_text(entry: dict) -> str:
    """
    Render one roster entry as a short factual record.
    """
    person = entry["person"]
    name = person["fullName"]
    position = entry["position"]["name"]

    birthplace = ", ".join(
        part for part in (
            person.get("birthCity"),
            person.get("birthStateProvince"),
            person.get("birthCountry"),
        ) if part
    )

    lines = [
        f"{name} – Pittsburgh Pirates roster",
        f"{name} is a {position.lower()} on the Pittsburgh Pirates 40-man roster.",
        f"Name: {name}",
        f"Full name: {person.get('fullFMLName', name)}",
        f"Team: Pittsburgh Pirates",
        f"Position: {position}",
        f"Jersey number: {entry.get('jerseyNumber') or person.get('primaryNumber', '')}",
        f"Roster status: {entry['status']['description']}",
    ]

    if person.get("birthDate"):
        lines.append(f"Born: {person['birthDate']} (birth year {person['birthDate'][:4]})")
    if birthplace:
        lines.append(f"Birthplace: {birthplace}")
        lines.append(f"{name} was born in {birthplace}.")
    if person.get("mlbDebutDate"):
        lines.append(f"MLB debut: {person['mlbDebutDate']}")
        lines.append(f"{name} made his MLB debut on {person['mlbDebutDate']}.")
    if person.get("height"):
        lines.append(f"Height: {person['height']}")
    if person.get("weight"):
        lines.append(f"Weight: {person['weight']} lb")
    if person.get("batSide"):
        lines.append(f"Bats: {person['batSide']['description']}")
    if person.get("pitchHand"):
        lines.append(f"Throws: {person['pitchHand']['description']}")

    return "\n".join(line for line in lines if not line.endswith(": "))


def main():
    parser = argparse.ArgumentParser(description="Build one document per Pittsburgh Pirates 40-man roster player.")
    parser.add_argument("--url", default=DEFAULT_URL, help="MLB Stats API roster URL.")
    parser.add_argument("--output-dir", default="data/raw/pirates_roster")
    args = parser.parse_args()

    response = requests.get(args.url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    roster = response.json()["roster"]

    retrieved_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)

    for entry in roster:
        person = entry["person"]
        document = {
            "url": PLAYER_PAGE.format(slug=person["nameSlug"]),
            "title": f"{person['fullName']} | Pittsburgh Pirates",
            "category": "sports",
            "text": player_text(entry),
            "retrieved_at": retrieved_at,
            "api_url": args.url,
        }
        save_document(document, output_dir / f"{person['nameSlug']}.json")

    print(f"Saved {len(roster)} player documents to {output_dir}")


if __name__ == "__main__":
    main()
