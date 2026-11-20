"""Shared offline HTML presentation. Author: Athul Prem | _AP_pnc."""

from html import escape
import os
from pathlib import Path
from urllib.parse import quote


REPORT_CSS = """
:root { color-scheme: light; --ink: #17243a; --muted: #43536a; --line: #d5dfeb;
  --accent: #116461; --paper: #fff; --canvas: #112037; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.55 system-ui, sans-serif; }
main { width: min(80rem, calc(100% - 2rem)); margin: clamp(1.25rem, 4vw, 3rem) auto; }
h1 { font-size: clamp(1.65rem, 4vw, 2.5rem); line-height: 1.15; margin: .25rem 0 .75rem; }
h2 { font-size: 1.25rem; line-height: 1.3; margin: 1.5rem 0 .5rem; }
p { max-width: 85ch; margin: .75rem 0; }
a { color: var(--accent); text-decoration-thickness: .09em; text-underline-offset: .18em; }
code, .sources { overflow-wrap: anywhere; }
.author, .scope, .availability { color: var(--muted); }
.author { font-size: .875rem; font-weight: 650; margin: .75rem 0 .25rem; }
.report-navigation { font-size: .9rem; }
.table-scroll { max-width: 100%; overflow-x: auto; margin: .75rem 0 1rem; }
table { width: 100%; border-collapse: collapse; font-size: .875rem; font-variant-numeric: tabular-nums; }
th, td { padding: .55rem .7rem; border-bottom: 1px solid var(--line); text-align: left; }
th { background: #e9f1f7; white-space: nowrap; }
tbody tr:nth-child(even) { background: #f7fafc; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
"""


def report_link(label: str, href: str, *, download: bool = False) -> str:
    attribute = " download" if download else ""
    return f'<a href="{escape(href, quote=True)}"{attribute}>{escape(label)}</a>'


def available_link(directory: Path, target: Path, label: str, *, unavailable: str = "Unavailable",
                   download: bool = False) -> str:
    """Render a local file link or an explicit availability message."""
    target = target.resolve()
    if not target.is_file():
        return f'<span class="availability">{escape(label)} ({escape(unavailable)})</span>'
    try:
        href = quote(Path(os.path.relpath(target, directory.resolve())).as_posix(), safe="/")
    except ValueError:  # Different Windows drives.
        href = target.as_uri()
    return report_link(label, href, download=download)


def scroll_table(table: str, label: str) -> str:
    return (f'<div class="table-scroll" role="region" tabindex="0" aria-label="{escape(label, quote=True)}">'
            f'{table}</div>')


def report_header(title: str, identifier: str, *, navigation: str = "", scope: str = "") -> str:
    nav = f'<nav class="report-navigation" aria-label="Report navigation">{navigation}</nav>' if navigation else ""
    description = f'<p class="scope">{escape(scope)}</p>' if scope else ""
    return (f'<header>{nav}<p class="author">Athul Prem | athul_simulation_pnc_{escape(identifier)}</p>'
            f'<h1>{escape(title)}</h1>{description}</header>')


def report_page(title: str, identifier: str, body: str, *, navigation: str = "", scope: str = "") -> str:
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{escape(title)} | Athul Prem</title>\n<style>{REPORT_CSS}</style></head>\n'
            f'<body><main>{report_header(title, identifier, navigation=navigation, scope=scope)}'
            f'{body}</main></body></html>\n')
