//! Stage 4: HTML to Markdown conversion.
//!
//! Ported from konbu/src/markdown.rs. Iterative stack-based converter
//! with smart spacing handling.

use scraper::{ElementRef, Node, Selector};
use std::sync::LazyLock;

static SEL_PRE: LazyLock<Selector> = LazyLock::new(|| Selector::parse("pre").unwrap());
static SEL_TH_TD: LazyLock<Selector> = LazyLock::new(|| Selector::parse("th, td").unwrap());

/// Convert an element to Markdown.
///
/// Processes the element itself (including its tag), not just its children.
/// This ensures heading, blockquote, list etc. formatting is preserved when
/// the element is the root of a segmented block.
pub fn md_from(element: ElementRef<'_>) -> String {
    let options = MarkdownOptions::default();
    let mut converter = Converter::new(&options);
    converter.convert_element(&element);
    converter.finish()
}

/// Options for markdown conversion.
#[derive(Debug, Clone)]
pub struct MarkdownOptions {
    pub include_links: bool,
    pub include_images: bool,
    pub include_tables: bool,
}

impl Default for MarkdownOptions {
    fn default() -> Self {
        Self {
            include_links: false,
            include_images: false,
            include_tables: true,
        }
    }
}

// ============================================================================
// Markdown Builder
// ============================================================================

struct MarkdownBuilder {
    buf: String,
}

impl MarkdownBuilder {
    fn new() -> Self {
        Self { buf: String::with_capacity(4096) }
    }

    fn last_char(&self) -> Option<char> {
        self.buf.chars().last()
    }

    fn second_last_char(&self) -> Option<char> {
        if self.buf.len() < 2 { return None; }
        self.buf.chars().nth_back(1)
    }

    fn push_raw(&mut self, s: &str) {
        self.buf.push_str(s);
    }

    fn push_char(&mut self, c: char) {
        self.buf.push(c);
    }

    fn push_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let first = text.chars().next().unwrap();
        if self.needs_space_before(first) {
            self.buf.push(' ');
        }
        self.buf.push_str(text);
    }

    fn push_block(&mut self) {
        self.buf.push_str("\n\n");
    }

    fn push_marker(&mut self, marker: &str) {
        let first = marker.chars().next().unwrap_or(' ');
        if self.needs_space_before(first) {
            self.buf.push(' ');
        }
        self.buf.push_str(marker);
    }

    fn needs_space_before(&self, next_char: char) -> bool {
        let Some(last) = self.last_char() else {
            return false;
        };

        if last.is_whitespace() || matches!(last, '(' | '[' | '"' | '\'' | '\n') {
            return false;
        }

        if matches!(next_char, '.' | ',' | '!' | '?' | ':' | ';' | ')' | ']' | '"' | '\'') {
            return false;
        }

        if matches!(last, '*' | '_' | '`') {
            if let Some(prev) = self.second_last_char() {
                if prev.is_alphanumeric() && next_char.is_alphanumeric() {
                    return true;
                }
            }
            return false;
        }

        last.is_alphanumeric() || matches!(last, ')' | ']')
    }

    fn finish(self) -> String {
        clean_output(self.buf)
    }
}

// ============================================================================
// Conversion Context
// ============================================================================

#[derive(Clone, Copy)]
enum ListType {
    Unordered,
    Ordered(usize),
}

struct Context<'a> {
    options: &'a MarkdownOptions,
    list_stack: Vec<ListType>,
    in_pre: bool,
    in_strong: bool,
    in_em: bool,
}

impl<'a> Context<'a> {
    fn new(options: &'a MarkdownOptions) -> Self {
        Self {
            options,
            list_stack: Vec::new(),
            in_pre: false,
            in_strong: false,
            in_em: false,
        }
    }

    fn list_depth(&self) -> usize {
        self.list_stack.len()
    }

    fn current_list(&self) -> Option<ListType> {
        self.list_stack.last().copied()
    }

    fn push_list(&mut self, list_type: ListType) {
        self.list_stack.push(list_type);
    }

    fn pop_list(&mut self) {
        self.list_stack.pop();
    }
}

// ============================================================================
// Iterative Stack Types
// ============================================================================

enum StackAction<'a> {
    Children(ElementRef<'a>),
    Text(String),
    Suffix(String, Option<StateRestore>),
}

enum StateRestore {
    EndPre,
    EndStrong,
    EndEm,
    PopList,
}

fn push_children_to_stack<'a>(stack: &mut Vec<StackAction<'a>>, element: &ElementRef<'a>) {
    let children: Vec<_> = element.children().collect();
    for child in children.into_iter().rev() {
        match child.value() {
            Node::Text(t) => {
                stack.push(StackAction::Text(t.text.to_string()));
            }
            Node::Element(_) => {
                if let Some(el) = ElementRef::wrap(child) {
                    stack.push(StackAction::Children(el));
                }
            }
            _ => {}
        }
    }
}

// ============================================================================
// Main Converter
// ============================================================================

struct Converter<'a> {
    out: MarkdownBuilder,
    ctx: Context<'a>,
}

impl<'a> Converter<'a> {
    fn new(options: &'a MarkdownOptions) -> Self {
        Self {
            out: MarkdownBuilder::new(),
            ctx: Context::new(options),
        }
    }

    fn finish(self) -> String {
        self.out.finish()
    }

    /// Convert the element itself (not just its children) to markdown.
    fn convert_element(&mut self, element: &ElementRef<'_>) {
        let mut stack: Vec<StackAction<'_>> = Vec::new();
        stack.push(StackAction::Children(*element));

        while let Some(action) = stack.pop() {
            match action {
                StackAction::Text(ref text) => {
                    self.handle_text(text);
                }
                StackAction::Suffix(ref suffix, ref restore) => {
                    if !suffix.is_empty() {
                        self.out.push_raw(suffix);
                    }
                    if let Some(r) = restore {
                        match r {
                            StateRestore::EndPre => self.ctx.in_pre = false,
                            StateRestore::EndStrong => self.ctx.in_strong = false,
                            StateRestore::EndEm => self.ctx.in_em = false,
                            StateRestore::PopList => { self.ctx.pop_list(); self.out.push_char('\n'); }
                        }
                    }
                }
                StackAction::Children(el) => {
                    self.process_element(&mut stack, el);
                }
            }
        }
    }

    fn convert_children(&mut self, element: &ElementRef<'_>) {
        let mut stack: Vec<StackAction<'_>> = Vec::new();
        push_children_to_stack(&mut stack, element);

        while let Some(action) = stack.pop() {
            match action {
                StackAction::Text(ref text) => {
                    self.handle_text(text);
                }
                StackAction::Suffix(ref suffix, ref restore) => {
                    if !suffix.is_empty() {
                        self.out.push_raw(suffix);
                    }
                    if let Some(r) = restore {
                        match r {
                            StateRestore::EndPre => self.ctx.in_pre = false,
                            StateRestore::EndStrong => self.ctx.in_strong = false,
                            StateRestore::EndEm => self.ctx.in_em = false,
                            StateRestore::PopList => { self.ctx.pop_list(); self.out.push_char('\n'); }
                        }
                    }
                }
                StackAction::Children(el) => {
                    self.process_element(&mut stack, el);
                }
            }
        }
    }

    fn process_element<'b>(&mut self, stack: &mut Vec<StackAction<'b>>, el: ElementRef<'b>) {
        let tag = el.value().name();

        match tag {
            "script" | "style" | "noscript" | "iframe" | "svg" => {}

            tag @ ("h1" | "h2" | "h3" | "h4" | "h5" | "h6") => {
                let level = tag.chars().nth(1).unwrap().to_digit(10).unwrap() as usize;
                self.out.push_block();
                self.out.push_raw(&"#".repeat(level));
                self.out.push_char(' ');
                stack.push(StackAction::Suffix("\n\n".to_string(), None));
                push_children_to_stack(stack, &el);
            }

            "p" => {
                self.out.push_block();
                stack.push(StackAction::Suffix("\n\n".to_string(), None));
                push_children_to_stack(stack, &el);
            }

            "div" | "section" | "article" | "main" | "span"
            | "thead" | "tbody" | "tfoot" => {
                push_children_to_stack(stack, &el);
            }

            "br" => self.out.push_raw("  \n"),
            "hr" => self.out.push_raw("\n\n---\n\n"),

            "strong" | "b" => {
                if self.ctx.in_strong {
                    push_children_to_stack(stack, &el);
                } else {
                    self.ctx.in_strong = true;
                    self.out.push_marker("**");
                    stack.push(StackAction::Suffix("**".to_string(), Some(StateRestore::EndStrong)));
                    push_children_to_stack(stack, &el);
                }
            }
            "em" | "i" => {
                if self.ctx.in_em {
                    push_children_to_stack(stack, &el);
                } else {
                    self.ctx.in_em = true;
                    self.out.push_marker("*");
                    stack.push(StackAction::Suffix("*".to_string(), Some(StateRestore::EndEm)));
                    push_children_to_stack(stack, &el);
                }
            }
            "u" => {
                self.out.push_marker("__");
                stack.push(StackAction::Suffix("__".to_string(), None));
                push_children_to_stack(stack, &el);
            }
            "s" | "strike" | "del" => {
                self.out.push_marker("~~");
                stack.push(StackAction::Suffix("~~".to_string(), None));
                push_children_to_stack(stack, &el);
            }
            "code" => {
                if self.ctx.in_pre {
                    push_children_to_stack(stack, &el);
                } else if el.select(&SEL_PRE).next().is_some() {
                    if let Some(lang) = extract_language(&el) {
                        self.out.push_raw("\n\n```");
                        self.out.push_raw(&lang);
                        self.out.push_char('\n');
                        self.ctx.in_pre = true;
                        stack.push(StackAction::Suffix("\n```\n\n".to_string(), Some(StateRestore::EndPre)));
                        push_children_to_stack(stack, &el);
                    } else {
                        push_children_to_stack(stack, &el);
                    }
                } else {
                    self.out.push_marker("`");
                    stack.push(StackAction::Suffix("`".to_string(), None));
                    push_children_to_stack(stack, &el);
                }
            }

            "a" => {
                if !self.ctx.options.include_links {
                    push_children_to_stack(stack, &el);
                } else {
                    let href = el.value().attr("href").unwrap_or("").to_string();
                    self.out.push_marker("[");
                    stack.push(StackAction::Suffix(format!("]({})", href), None));
                    push_children_to_stack(stack, &el);
                }
            }
            "img" => {
                if self.ctx.options.include_images {
                    let src = el.value().attr("src").unwrap_or("");
                    let alt = el.value().attr("alt").unwrap_or("");
                    self.out.push_raw(&format!("![{alt}]({src})"));
                }
            }

            "pre" => {
                if self.ctx.in_pre {
                    push_children_to_stack(stack, &el);
                } else {
                    let lang = extract_language(&el).unwrap_or_default();
                    self.out.push_raw("\n\n```");
                    self.out.push_raw(&lang);
                    self.out.push_char('\n');
                    self.ctx.in_pre = true;
                    stack.push(StackAction::Suffix("\n```\n\n".to_string(), Some(StateRestore::EndPre)));
                    push_children_to_stack(stack, &el);
                }
            }

            "blockquote" => {
                self.out.push_block();
                let mut inner = Converter::new(self.ctx.options);
                inner.ctx.in_pre = self.ctx.in_pre;
                inner.convert_children(&el);
                let text = inner.out.buf;
                let trimmed = text.trim();
                for line in trimmed.lines() {
                    let line = line.trim();
                    if !line.is_empty() {
                        self.out.push_raw("> ");
                        self.out.push_raw(line);
                        self.out.push_char('\n');
                    }
                }
                self.out.push_char('\n');
            }

            "ul" => {
                self.out.push_block();
                self.ctx.push_list(ListType::Unordered);
                stack.push(StackAction::Suffix(String::new(), Some(StateRestore::PopList)));
                push_children_to_stack(stack, &el);
            }
            "ol" => {
                self.out.push_block();
                self.ctx.push_list(ListType::Ordered(1));
                stack.push(StackAction::Suffix(String::new(), Some(StateRestore::PopList)));
                push_children_to_stack(stack, &el);
            }
            "li" => {
                let depth = self.ctx.list_depth();
                let indent = "  ".repeat(depth.saturating_sub(1));
                self.out.push_raw(&indent);
                match self.ctx.current_list() {
                    Some(ListType::Ordered(n)) => {
                        self.out.push_raw(&format!("{}. ", n));
                        if let Some(ListType::Ordered(num)) = self.ctx.list_stack.last_mut() {
                            *num += 1;
                        }
                    }
                    Some(ListType::Unordered) | None => {
                        self.out.push_raw("* ");
                    }
                }
                stack.push(StackAction::Suffix("\n".to_string(), None));
                push_children_to_stack(stack, &el);
            }

            "table" => {
                if self.ctx.options.include_tables {
                    self.out.push_block();
                    stack.push(StackAction::Suffix("\n".to_string(), None));
                    push_children_to_stack(stack, &el);
                }
            }
            "tr" => {
                self.out.push_char('|');
                let is_in_thead = el
                    .parent()
                    .and_then(|p| p.value().as_element())
                    .map(|e| e.name() == "thead")
                    .unwrap_or(false);
                let suffix = if is_in_thead {
                    let cols = el.select(&SEL_TH_TD).count();
                    format!("\n|{}\n", " --- |".repeat(cols))
                } else {
                    "\n".to_string()
                };
                stack.push(StackAction::Suffix(suffix, None));
                push_children_to_stack(stack, &el);
            }
            "th" | "td" => {
                self.out.push_char(' ');
                stack.push(StackAction::Suffix(" |".to_string(), None));
                push_children_to_stack(stack, &el);
            }

            // Skip form controls — no useful text content
            "input" | "button" | "select" | "textarea" => {}

            _ => {
                push_children_to_stack(stack, &el);
            }
        }
    }

    fn handle_text(&mut self, text: &str) {
        if self.ctx.in_pre {
            if !text.is_empty() {
                self.out.push_raw(text);
            }
            return;
        }

        let has_leading = text.starts_with(char::is_whitespace);
        let has_trailing = text.ends_with(char::is_whitespace);
        let words: Vec<&str> = text.split_whitespace().collect();

        if words.is_empty() {
            if has_leading || has_trailing {
                self.out.push_text(" ");
            }
            return;
        }

        let normalized = words.join(" ");

        if has_leading {
            if let Some(last) = self.out.last_char() {
                if !last.is_whitespace() {
                    self.out.push_char(' ');
                }
            }
        }

        self.out.push_text(&normalized);
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn extract_language(el: &ElementRef<'_>) -> Option<String> {
    el.value()
        .attr("class")
        .and_then(|c| c.split_whitespace().find(|s| s.starts_with("language-")))
        .map(|s| s[9..].to_string())
}

fn clean_output(md: String) -> String {
    let mut result = String::with_capacity(md.len());
    let mut blank = false;
    let mut in_code = false;

    for line in md.lines() {
        if line.trim().starts_with("```") {
            in_code = !in_code;
        }

        if in_code {
            result.push_str(line);
            result.push('\n');
            blank = false;
        } else if line.trim().is_empty() {
            if !blank {
                result.push('\n');
                blank = true;
            }
        } else {
            let trimmed = line.trim_start();
            let indent = &line[..line.len() - trimmed.len()];

            let is_list = trimmed.starts_with("- ") || trimmed.starts_with("* ") ||
                          trimmed.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false);
            let is_blockquote = trimmed.starts_with("> ");

            if is_list || is_blockquote {
                result.push_str(indent);
                result.push_str(&normalize_spaces(trimmed.trim_end()));
            } else {
                result.push_str(&normalize_spaces(line.trim()));
            }
            result.push('\n');
            blank = false;
        }
    }

    fix_emphasis(result.trim())
}

fn normalize_spaces(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut prev_space = false;
    let mut in_backticks = false;

    for c in s.chars() {
        if c == '`' {
            in_backticks = !in_backticks;
            result.push(c);
            prev_space = false;
        } else if in_backticks {
            result.push(c);
            prev_space = false;
        } else if c == ' ' {
            if !prev_space {
                result.push(c);
            }
            prev_space = true;
        } else {
            result.push(c);
            prev_space = false;
        }
    }

    result
}

fn fix_emphasis(md: &str) -> String {
    let chars: Vec<char> = md.chars().collect();
    let len = chars.len();
    let mut result = String::with_capacity(md.len() + 100);
    let mut i = 0;
    let mut in_bold = false;
    let mut in_italic = false;
    let mut in_code_block = false;

    while i < len {
        if i + 2 < len && chars[i] == '`' && chars[i + 1] == '`' && chars[i + 2] == '`' {
            in_code_block = !in_code_block;
            result.push_str("```");
            i += 3;
            continue;
        }

        if in_code_block {
            result.push(chars[i]);
            i += 1;
            continue;
        }

        if chars[i] == '*' {
            let marker_len = if i + 1 < len && chars[i + 1] == '*' { 2 } else { 1 };
            let is_bold = marker_len == 2;
            let in_emphasis = if is_bold { in_bold } else { in_italic };

            let prev = if i > 0 { Some(chars[i - 1]) } else { None };
            let next_idx = i + marker_len;
            let _next = if next_idx < len { Some(chars[next_idx]) } else { None };

            if !in_emphasis {
                if needs_space_before_opening(prev) {
                    result.push(' ');
                }
                result.push_str(if is_bold { "**" } else { "*" });
                i += marker_len;
                while i < len && chars[i] == ' ' {
                    i += 1;
                }
                if is_bold { in_bold = true; } else { in_italic = true; }
            } else {
                while result.ends_with(' ') {
                    result.pop();
                }
                result.push_str(if is_bold { "**" } else { "*" });
                i += marker_len;
                let next = if i < len { Some(chars[i]) } else { None };
                if needs_space_after_closing(next) {
                    result.push(' ');
                }
                if is_bold { in_bold = false; } else { in_italic = false; }
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

fn needs_space_before_opening(prev: Option<char>) -> bool {
    prev.map(|c| c.is_alphanumeric() || matches!(c, ',' | ':' | ';')).unwrap_or(false)
}

fn needs_space_after_closing(next: Option<char>) -> bool {
    next.map(|c| c.is_alphanumeric()).unwrap_or(false)
}
