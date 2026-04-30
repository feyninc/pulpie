"""HTML simplification for block-level classification.

Converts raw HTML into a simplified representation where each content block
has a unique _item_id. Matches the MinerU-HTML simplification format that
the model was trained on.
"""

from __future__ import annotations

from lxml import etree, html

TAGS_TO_REMOVE = frozenset(
    {
        "script",
        "style",
        "link",
        "meta",
        "head",
        "title",
        "iframe",
        "frame",
        "noscript",
        "svg",
        "math",
    }
)

INLINE_TAGS = frozenset(
    {
        "a",
        "abbr",
        "acronym",
        "b",
        "bdo",
        "big",
        "br",
        "button",
        "cite",
        "code",
        "dfn",
        "em",
        "font",
        "i",
        "img",
        "input",
        "kbd",
        "label",
        "map",
        "mark",
        "nobr",
        "object",
        "option",
        "optgroup",
        "q",
        "s",
        "samp",
        "select",
        "small",
        "span",
        "strike",
        "strong",
        "sub",
        "sup",
        "textarea",
        "time",
        "u",
        "var",
    }
)

BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "center",
        "details",
        "dialog",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "select",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)

ATTRS_TO_REMOVE = frozenset(
    {
        "href",
        "src",
        "action",
        "onclick",
        "onload",
        "onerror",
        "data-src",
        "data-href",
        "srcset",
        "loading",
        "decoding",
        "fetchpriority",
        "sizes",
        "media",
        "crossorigin",
        "integrity",
        "referrerpolicy",
        "target",
        "rel",
        "download",
        "hreflang",
        "type",
        "value",
        "name",
        "method",
        "enctype",
        "novalidate",
        "autocomplete",
        "autofocus",
        "placeholder",
        "required",
        "disabled",
        "readonly",
        "checked",
        "selected",
        "multiple",
        "min",
        "max",
        "step",
        "pattern",
        "maxlength",
        "minlength",
        "tabindex",
        "accesskey",
        "contenteditable",
        "draggable",
        "hidden",
        "spellcheck",
        "translate",
        "dir",
        "lang",
        "xmlns",
        "xml:lang",
        "role",
        "aria-label",
        "aria-labelledby",
        "aria-describedby",
        "aria-hidden",
        "aria-expanded",
        "aria-controls",
        "aria-haspopup",
        "aria-current",
        "aria-selected",
        "aria-live",
        "aria-atomic",
        "aria-busy",
    }
)

# Keep: class, id, alt, colspan, rowspan, _item_id (added by us)

MAX_LIST_ITEMS = 3


def simplify(raw_html: str, cutoff_length: int = 500) -> tuple[str, str]:
    """Simplify raw HTML for block-level classification.

    Args:
        raw_html: Raw HTML string.
        cutoff_length: Maximum character length for text in each block.

    Returns:
        Tuple of (simplified_html, map_html).
        - simplified_html: Cleaned HTML with _item_id on each block.
        - map_html: Original HTML structure with _item_id markers for reconstruction.
    """
    parser = html.HTMLParser(remove_comments=True, remove_pis=True)
    try:
        doc = html.document_fromstring(raw_html, parser=parser)
    except Exception:
        doc = html.document_fromstring("<html><body></body></html>", parser=parser)

    _remove_tags(doc)

    blocks = _extract_blocks(doc)

    simplified_parts = ['<html><head><meta charset="utf-8"></head><body>']
    for item_id, element in enumerate(blocks, start=1):
        element.set("_item_id", str(item_id))
        cleaned = _clean_element(element, cutoff_length)
        simplified_parts.append(cleaned)
    simplified_parts.append("</body></html>")

    simplified = "".join(simplified_parts)
    map_html = etree.tostring(doc, method="html", encoding="unicode")

    return simplified, map_html


def _remove_tags(doc: html.HtmlElement) -> None:
    """Remove script, style, and other non-content tags."""
    to_remove = [el for el in doc.iter() if el.tag in TAGS_TO_REMOVE]
    for el in to_remove:
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)


def _extract_blocks(doc: html.HtmlElement) -> list[html.HtmlElement]:
    """Walk the DOM and extract block-level elements as flat list."""
    blocks: list[html.HtmlElement] = []
    body = doc.body if doc.body is not None else doc
    _walk(body, blocks)
    return blocks


def _walk(element: html.HtmlElement, blocks: list[html.HtmlElement]) -> None:
    """Recursively walk DOM, collecting leaf block elements."""
    if element.tag in INLINE_TAGS:
        return

    # Lists: treat the whole list as one block (matches MinerU behavior)
    if element.tag in ("ul", "ol", "dl"):
        if _has_text_content(element):
            blocks.append(element)
        return

    # Select: treat as one block
    if element.tag == "select":
        blocks.append(element)
        return

    has_block_children = any(
        child.tag in BLOCK_TAGS for child in element if isinstance(child, html.HtmlElement)
    )

    if not has_block_children and element.tag in BLOCK_TAGS:
        if _has_text_content(element) or element.tag == "table":
            blocks.append(element)
        return

    if not has_block_children and element.tag not in BLOCK_TAGS:
        if _has_text_content(element):
            blocks.append(element)
        return

    for child in element:
        if not isinstance(child, html.HtmlElement):
            continue
        if child.tag in BLOCK_TAGS or _has_block_descendants(child):
            _walk(child, blocks)
        elif _has_text_content(child):
            blocks.append(child)


def _has_block_descendants(element: html.HtmlElement) -> bool:
    """Check if element has any block-level descendants."""
    for desc in element.iterdescendants():
        if isinstance(desc, html.HtmlElement) and desc.tag in BLOCK_TAGS:
            return True
    return False


def _has_text_content(element: html.HtmlElement) -> bool:
    """Check if element has meaningful text content."""
    text = element.text_content()
    return bool(text and text.strip())


def _clean_element(element: html.HtmlElement, cutoff_length: int) -> str:
    """Clean an element for simplified output: strip URLs, truncate text, simplify lists."""
    import copy

    el = copy.deepcopy(element)

    # Simplify lists: keep first N items + "..."
    if el.tag in ("ul", "ol"):
        items = el.findall("li")
        if len(items) > MAX_LIST_ITEMS:
            for item in items[MAX_LIST_ITEMS:]:
                el.remove(item)
            ellipsis = etree.SubElement(el, "span")
            ellipsis.text = "..."

    # Clean attributes on all elements
    for sub in el.iter():
        if not isinstance(sub, html.HtmlElement):
            continue
        for attr in list(sub.attrib):
            if attr in ATTRS_TO_REMOVE:
                del sub.attrib[attr]

    # Truncate text content
    _truncate_text(el, cutoff_length)

    return etree.tostring(el, method="html", encoding="unicode")


def _sanitize_text(text: str) -> str:
    """Remove control characters that lxml rejects."""
    import re

    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def _truncate_text(element: html.HtmlElement, max_length: int) -> None:
    """Truncate text content within element to max_length chars."""
    total = [0]

    def _truncate_node(node):
        if node.text:
            text = _sanitize_text(node.text)
            remaining = max_length - total[0]
            if remaining <= 0:
                node.text = ""
            elif len(text) > remaining:
                node.text = text[:remaining] + "..."
                total[0] = max_length
            else:
                node.text = text
                total[0] += len(text)

        for child in node:
            if not isinstance(child, html.HtmlElement):
                continue
            _truncate_node(child)
            if child.tail:
                tail = _sanitize_text(child.tail)
                remaining = max_length - total[0]
                if remaining <= 0:
                    child.tail = ""
                elif len(tail) > remaining:
                    child.tail = tail[:remaining] + "..."
                    total[0] = max_length
                else:
                    child.tail = tail
                    total[0] += len(tail)

    _truncate_node(element)
