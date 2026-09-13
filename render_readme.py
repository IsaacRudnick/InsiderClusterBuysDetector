"""Render README.md to README.html with inline CSS so it opens offline."""

import os
from pathlib import Path

import markdown


ROOT = Path(__file__).parent
SRC = ROOT / "README.md"
DST = ROOT / "README.html"

CSS = """
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; background: #f5f7fa; color: #1f2937; line-height: 1.55; }
  main { max-width: 820px; margin: 0 auto; padding: 32px 28px 60px;
         background: #fff; border-left: 1px solid #e5e7eb;
         border-right: 1px solid #e5e7eb; min-height: 100vh; }
  h1, h2, h3 { color: #111827; margin-top: 1.8em; margin-bottom: 0.6em; }
  h1 { font-size: 28px; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px;
       margin-top: 0; }
  h2 { font-size: 20px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }
  h3 { font-size: 16px; }
  hr { border: 0; border-top: 1px solid #e5e7eb; margin: 28px 0; }
  a { color: #2563eb; text-decoration: none; }
  a:hover { text-decoration: underline; }
  code { background: #f3f4f6; padding: 1px 5px; border-radius: 3px;
         font-size: 0.92em; font-family: "SFMono-Regular", Consolas, monospace; }
  pre { background: #111827; color: #f9fafb; padding: 14px 16px;
        border-radius: 6px; overflow-x: auto; line-height: 1.45; }
  pre code { background: transparent; padding: 0; color: inherit; font-size: 0.88em; }
  blockquote { border-left: 4px solid #fbbf24; background: #fffbeb; margin: 12px 0;
               padding: 10px 16px; color: #78350f; border-radius: 0 4px 4px 0; }
  ul, ol { padding-left: 1.6em; }
  li { margin: 4px 0; }
  table { border-collapse: collapse; margin: 14px 0; }
  th, td { border: 1px solid #e5e7eb; padding: 6px 10px; font-size: 0.95em; }
  th { background: #f3f4f6; }
"""


def main() -> int:
    if not SRC.exists():
        print(f"Not found: {SRC}")
        return 1
    body = markdown.markdown(
        SRC.read_text(encoding="utf-8"),
        extensions=["fenced_code", "tables", "sane_lists"],
    )
    html_doc = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8" />\n'
        '<title>Insider Cluster-Buy Detector - README</title>\n'
        f"<style>{CSS}</style>\n"
        '</head>\n<body>\n<main>\n'
        f"{body}\n"
        "</main>\n</body>\n</html>\n"
    )
    DST.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {DST.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
