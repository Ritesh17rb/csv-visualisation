from __future__ import annotations

import html
import json
from importlib.resources import files


PACKAGE_NAME = "csv_visualisation"


def _read_resource(*parts: str) -> str:
    return files(PACKAGE_NAME).joinpath(*parts).read_text(encoding="utf-8")


def _inline_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_html(payload: dict) -> str:
    template = _read_resource("templates", "index.html")
    replacements = {
        "__TITLE__": html.escape(str(payload["meta"].get("displayName") or "CSV Data Explorer")),
        "__APP_CSS__": _read_resource("static", "app.css"),
        "__D3_JS__": _read_resource("vendor", "d3.v7.min.js"),
        "__APP_DATA__": _inline_json(payload),
        "__APP_JS__": _read_resource("static", "app.js"),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template
