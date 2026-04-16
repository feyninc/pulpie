//! Stage 5: Markdown cleanup.
//!
//! Simple regex-based post-processing of assembled markdown.

use regex::Regex;
use std::sync::LazyLock;

static RE_MULTI_BLANK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n{3,}").unwrap());
static RE_TRAILING_WS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[ \t]+\n").unwrap());
static RE_ORPHAN_BOLD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\*{2,4}(?:\s*\*{2,4})").unwrap());
static RE_EMPTY_LINK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[]\([^)]*\)").unwrap());

pub fn postprocess(md: &str) -> String {
    let s = RE_MULTI_BLANK.replace_all(md, "\n\n");
    let s = RE_TRAILING_WS.replace_all(&s, "\n");
    let s = RE_ORPHAN_BOLD.replace_all(&s, "");
    let s = RE_EMPTY_LINK.replace_all(&s, "");
    s.trim().to_string()
}
