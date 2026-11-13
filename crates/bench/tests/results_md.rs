//! Requirements on bench/RESULTS.md, the hub's record of measured numbers.
//!
//! These read the committed results; they do not measure anything.

use std::fs;
use std::path::PathBuf;

const HOSTS: [&str; 3] = ["macmini", "ubuntu", "windows"];
const F32_BYTES_PER_SAMPLE: f64 = 4.0 * 4096.0;
const BF16_BYTES_PER_SAMPLE: f64 = 2.0 * 4096.0;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn results() -> String {
    fs::read_to_string(repo().join("bench/RESULTS.md")).expect("bench/RESULTS.md")
}

struct Table {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    fn col(&self, name: &str) -> Option<usize> {
        self.header.iter().position(|cell| cell == name)
    }
}

fn cells(line: &str) -> Vec<String> {
    let inner = line.trim().trim_start_matches('|').trim_end_matches('|');
    inner.split('|').map(|cell| cell.trim().trim_matches('`').to_string()).collect()
}

/// Every markdown table in `text`, in order.
fn tables(text: &str) -> Vec<Table> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut index = 0;
    while index + 1 < lines.len() {
        let line = lines[index].trim();
        let next = lines[index + 1].trim();
        if line.starts_with('|') && next.starts_with("| ---") {
            let header = cells(line);
            let mut rows = Vec::new();
            index += 2;
            while index < lines.len() && lines[index].trim().starts_with('|') {
                rows.push(cells(lines[index]));
                index += 1;
            }
            out.push(Table { header, rows });
        } else {
            index += 1;
        }
    }
    out
}

/// Text from the `## name` heading up to the next `## ` heading.
fn section<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let start = text.find(&format!("\n## {name}\n"))? + 1;
    let body = &text[start + 3 + name.len()..];
    let end = body.find("\n## ").map_or(body.len(), |at| at + 1);
    Some(&body[..end])
}

fn number(cell: &str) -> Option<f64> {
    cell.trim().parse::<f64>().ok().filter(|value| value.is_finite())
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-6 * right.abs().max(1.0)
}

#[test]
fn link_matrix_measures_every_pair_both_ways_for_30_seconds() {
    let text = results();
    let body = section(&text, "Link matrix").expect("a `## Link matrix` section");
    let table = tables(body)
        .into_iter()
        .find(|table| table.col("from").is_some() && table.col("bytes/s").is_some())
        .expect("a link table with from and bytes/s columns");
    let col = |name: &str| table.col(name).unwrap_or_else(|| panic!("link table column {name}"));
    let (from, to, seconds, rate, f32c, bf16c, status) = (
        col("from"),
        col("to"),
        col("seconds"),
        col("bytes/s"),
        col("f32 ceiling samples/s"),
        col("bf16 ceiling samples/s"),
        col("status"),
    );
    let (from_medium, to_medium) = (col("from medium"), col("to medium"));
    for a in HOSTS {
        for b in HOSTS {
            if a == b {
                continue;
            }
            let row = table
                .rows
                .iter()
                .find(|row| row[from] == a && row[to] == b)
                .unwrap_or_else(|| panic!("no link row {a} -> {b}"));
            assert_eq!(row[status], "ok", "{a} -> {b} status");
            assert_eq!(number(&row[seconds]), Some(30.0), "{a} -> {b} runs 30 s");
            assert!(
                !row[from_medium].is_empty() && !row[to_medium].is_empty(),
                "{a} -> {b} medium"
            );
            let bytes = number(&row[rate]).expect("bytes/s is a number");
            assert!(bytes > 0.0);
            let f32_ceiling = number(&row[f32c]).expect("f32 ceiling is a number");
            let bf16_ceiling = number(&row[bf16c]).expect("bf16 ceiling is a number");
            assert!(close(f32_ceiling, bytes / F32_BYTES_PER_SAMPLE), "{a} -> {b} f32 ceiling");
            assert!(close(bf16_ceiling, bytes / BF16_BYTES_PER_SAMPLE), "{a} -> {b} bf16 ceiling");
        }
    }
}

#[test]
fn subset_headings_print_the_port_they_tried() {
    let text = results();
    let Some(start) = text.find("\n### Subset groups\n") else {
        return;
    };
    let body = &text[start + 1..];
    let body = &body[..body[4..].find("\n## ").map_or(body.len(), |at| at + 5)];
    let lines: Vec<&str> = body.lines().collect();
    let starts: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].starts_with("- `")).collect();
    assert!(!starts.is_empty(), "subset section has no headings");
    for (n, &at) in starts.iter().enumerate() {
        let heading = lines[at];
        let ports: Vec<&str> = heading
            .split("port `")
            .nth(1)
            .and_then(|rest| rest.split('`').next())
            .unwrap_or_else(|| panic!("heading without a port: {heading}"))
            .split(", ")
            .collect();
        for port in &ports {
            assert!(port.parse::<u16>().is_ok(), "heading port is not a port: {heading}");
        }
        let end = starts.get(n + 1).copied().unwrap_or(lines.len());
        for line in &lines[at + 1..end] {
            if let Some(rest) = line.trim().strip_prefix("\"port\": ") {
                let tried = rest.trim_end_matches(',');
                assert!(ports.contains(&tried), "{heading} omits tried port {tried}");
            }
        }
    }
}

/// Top-level `samples_per_sec` of a pretty-printed, key-sorted run JSON.
fn top_level_samples(json: &str) -> Option<f64> {
    let key = "\n  \"samples_per_sec\": ";
    let at = json.find(key)? + key.len();
    let value: String = json[at..]
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '-' | '+'))
        .collect();
    value.parse().ok()
}

fn ok_samples(text: &str) -> Vec<f64> {
    let mut out = Vec::new();
    for table in tables(text) {
        let (Some(samples), Some(status)) = (table.col("samples/sec"), table.col("status")) else {
            continue;
        };
        if table
            .col("precision")
            .is_some_and(|precision| table.rows.iter().any(|row| row[precision] != "f32"))
        {
            panic!("a samples/sec table mixes in a non-f32 headline");
        }
        for row in &table.rows {
            if row[status] == "ok" {
                out.push(number(&row[samples]).expect("ok row has a number"));
            }
        }
    }
    out
}

#[test]
fn baseline_b_states_number_configuration_and_json() {
    let text = results();
    let line =
        text.lines().find(|line| line.starts_with("Baseline B: ")).expect("a `Baseline B: ` line");
    let value = number(line["Baseline B: ".len()..].split_whitespace().next().unwrap_or(""))
        .expect("B is a number");
    let file = line
        .split('`')
        .find(|part| part.starts_with("bench/runs/") && part.ends_with(".json"))
        .expect("B names its JSON file");
    let json = fs::read_to_string(repo().join(file)).expect("B's JSON file exists");
    assert_eq!(top_level_samples(&json), Some(value), "B equals its JSON samples_per_sec");
    assert!(line.contains(" on "), "B names its configuration: {line}");
    let measured = ok_samples(&text);
    assert!(!measured.is_empty());
    let best = measured.iter().copied().fold(f64::MIN, f64::max);
    assert!(close(value, best), "B {value} is the best measured configuration {best}");
}

fn rows_with<'a>(tables: &'a [Table], needle: &str) -> Vec<&'a Vec<String>> {
    tables
        .iter()
        .flat_map(|table| table.rows.iter())
        .filter(|row| row.iter().any(|cell| cell.contains(needle)))
        .collect()
}

#[test]
fn wsl2_single_host_sits_next_to_native() {
    let text = results();
    let body = section(&text, "Single host").expect("single host section");
    let table = tables(body).into_iter().next().expect("single host table");
    let ssh = table.col("ssh").expect("ssh column");
    let runtime = table.col("runtime").expect("runtime column");
    let windows: Vec<usize> =
        (0..table.rows.len()).filter(|&index| table.rows[index][ssh] == "windows").collect();
    assert_eq!(windows.len(), 2, "windows has a native row and a WSL2 row");
    assert_eq!(windows[1], windows[0] + 1, "the two windows rows are adjacent");
    let kinds: Vec<&str> =
        windows.iter().map(|&index| table.rows[index][runtime].as_str()).collect();
    assert!(kinds.contains(&"native"), "{kinds:?}");
    assert!(kinds.iter().any(|kind| kind.starts_with("wsl2")), "{kinds:?}");
}

#[test]
fn wsl2_pipeline_times_both_splits_or_records_the_error() {
    let text = results();
    let body = section(&text, "Same-OS pipeline (WSL2)").expect("a WSL2 pipeline section");
    let all = tables(body);
    let rows = rows_with(&all, "windows[wsl2]");
    let table = all.iter().find(|table| table.col("policy").is_some()).expect("pipeline table");
    let policy = table.col("policy").unwrap();
    let status = table.col("status").unwrap();
    let split = table.col("split").unwrap();
    for wanted in ["equal", "proportional"] {
        let row = rows
            .iter()
            .find(|row| row[policy] == wanted)
            .unwrap_or_else(|| panic!("no {wanted} row for windows[wsl2]"));
        if row[status] == "ok" {
            assert!(row[split].starts_with('['), "{wanted} split recorded");
        } else {
            assert!(
                body.contains(&format!("Exact error for `{wanted}`")),
                "{wanted} error recorded"
            );
        }
    }
}

#[test]
fn macmini_joins_a_group_only_if_gloo_forms() {
    let text = results();
    let all = tables(&text);
    let joined = all.iter().any(|table| {
        let (Some(hosts), Some(status)) = (table.col("hosts"), table.col("status")) else {
            return false;
        };
        table.rows.iter().any(|row| row[hosts].contains("macmini") && row[status] == "ok")
    });
    if joined {
        return;
    }
    let line = text
        .lines()
        .find(|line| line.contains("macmini") && line.contains("not formable"))
        .expect("macmini is marked `not formable`");
    let after = &text[text.find(line).unwrap() + line.len()..];
    assert!(after.trim_start().starts_with("```"), "the gloo error follows: {line}");
}
