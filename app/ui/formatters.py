"""Formatting helpers for the Streamlit UI."""

from __future__ import annotations


def format_source_reference(source: str) -> str:
    """Format a raw source reference for human-readable display."""
    if " | row:" in source:
        return source
    return source.replace(" -- ", " - ")


def build_visible_source_references(sources_used: list[str]) -> list[str]:
    """Build the compact source references shown in visible citation chips."""
    visible_sources: list[str] = []
    seen: set[str] = set()

    for source in sources_used:
        visible_source = format_source_reference(source)
        if " | row:" in source:
            file_name, _, rest = source.partition(" | row: ")
            row_name = rest.split(" | ", maxsplit=1)[0].strip()
            visible_source = f"{file_name} - {row_name} row"
        elif " | sorted by:" in source:
            file_name, _, rest = source.partition(" | sorted by: ")
            row_name = ""
            if " | row: " in rest:
                row_name = rest.split(" | row: ", maxsplit=1)[1].strip()
            visible_source = f"{file_name} - {row_name} row" if row_name else file_name
        elif " | filter:" in source:
            file_name = source.split(" | filter:", maxsplit=1)[0].strip()
            visible_source = f"{file_name} - filtered rows"
        elif " | exists check:" in source:
            file_name = source.split(" | exists check:", maxsplit=1)[0].strip()
            visible_source = f"{file_name} - existence check"
        elif " | rows counted" in source:
            file_name = source.split(" | rows counted", maxsplit=1)[0].strip()
            visible_source = f"{file_name} - row count"
        elif " | listed rows" in source:
            file_name = source.split(" | listed rows", maxsplit=1)[0].strip()
            visible_source = f"{file_name} - listed rows"

        if visible_source not in seen:
            seen.add(visible_source)
            visible_sources.append(visible_source)

    return visible_sources
