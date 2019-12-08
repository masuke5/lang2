use crate::span::Span;

pub fn escape_string(raw: &str) -> String {
    let mut s = String::new();
    for ch in raw.chars() {
        let n = ch as u32;
        match ch {
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            '\\' => s.push_str("\\\\"),
            '\0' => s.push_str("\\0"),
            '"' => s.push_str("\\\""),
            _ if n <= 0x1f || n == 0x7f => s.push_str(&format!("\\x{:02x}", n)),
            ch => s.push(ch),
        }
    }

    s
}

pub fn span_to_string(span: &Span) -> String {
    format!("\x1b[33m{}:{}-{}:{}\x1b[0m", span.start_line, span.start_col, span.end_line, span.end_col)
}

pub fn align(x: usize, n: usize) -> usize {
    (x + (n - 1)) & !(n - 1)
}
