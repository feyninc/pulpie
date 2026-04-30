"""Reconstruct main-content HTML by removing 'other' blocks from the original."""

from __future__ import annotations

from lxml import etree, html


def extract_main_html(map_html: str, labels: dict[str, str]) -> str:
    """Remove boilerplate blocks from HTML, keeping the original document structure.

    Keeps the full HTML document intact and only removes elements whose _item_id
    is labeled "other". This preserves proper nesting for downstream conversion
    (e.g. html2text).

    Args:
        map_html: Original HTML with _item_id attributes on block elements.
        labels: Dict mapping item_id -> "main" or "other".

    Returns:
        HTML string with boilerplate blocks removed.
    """
    if not labels or not map_html:
        return ""

    other_ids = {k for k, v in labels.items() if v == "other"}
    if not other_ids:
        # Everything is main — just strip _item_id attributes and return
        return _strip_item_ids(map_html)

    parser = html.HTMLParser(remove_comments=True)
    try:
        doc = html.document_fromstring(map_html, parser=parser)
    except Exception:
        return map_html

    # Remove elements labeled as "other"
    elements_to_remove = []
    for el in doc.iter():
        if not isinstance(el, html.HtmlElement):
            continue
        item_id = el.get("_item_id")
        if item_id and item_id in other_ids:
            elements_to_remove.append(el)

    for el in elements_to_remove:
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)

    # Clean up _item_id attributes from remaining elements
    for el in doc.iter():
        if isinstance(el, html.HtmlElement) and "_item_id" in el.attrib:
            del el.attrib["_item_id"]

    return etree.tostring(doc, method="html", encoding="unicode")


def _strip_item_ids(html_str: str) -> str:
    """Remove all _item_id attributes from HTML string."""
    import re

    return re.sub(r'\s*_item_id="[^"]*"', "", html_str)
