//! Why workload W does not run on linalg-pipeline today, and the shared metric.

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use linalg_core::{Error, Tensor, Transport};
use linalg_cpu::{CpuBackend, CpuStore};
use linalg_pipeline::{Placement, TrainOutput};

pub const LAYERS: usize = 12;
pub const WIDTH: usize = 4096;
pub const MICROBATCH_ROWS: usize = 256;
pub const MICROBATCHES: usize = 8;
pub const SAMPLES_PER_STEP: usize = MICROBATCH_ROWS * MICROBATCHES;
pub const WARMUP_STEPS: usize = 5;
pub const MEASURED_STEPS: usize = 50;

struct IdleTransport;

impl Transport for IdleTransport {
    fn rank(&self) -> usize {
        0
    }

    fn world(&self) -> usize {
        1
    }

    fn send(&self, _dest: usize, _bytes: &[u8]) -> Result<(), Error> {
        Err(Error::Transport)
    }

    fn recv_timeout(&self, _src: usize, _timeout: Duration) -> Result<Vec<u8>, Error> {
        Err(Error::Transport)
    }
}

/// The pipeline entry point, monomorphized on the CPU backend.
pub(crate) fn train_entry(
    backend: &CpuBackend,
    transport: &mut IdleTransport,
    placement: &Placement,
    microbatches: usize,
    weights: &[Tensor<CpuStore>],
    inputs: &[Tensor<CpuStore>],
    dy: &[Tensor<CpuStore>],
) -> Result<TrainOutput<CpuStore>, Error> {
    linalg_pipeline::train(backend, transport, placement, microbatches, weights, inputs, dy)
}

pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let mid = ordered.len() / 2;
    if ordered.len() % 2 == 1 {
        Some(ordered[mid])
    } else {
        Some(0.5 * (ordered[mid - 1] + ordered[mid]))
    }
}

pub fn samples_per_sec(median_step_s: f64) -> Option<f64> {
    if !median_step_s.is_finite() || median_step_s <= 0.0 {
        return None;
    }
    Some(SAMPLES_PER_STEP as f64 / median_step_s)
}

pub struct Report {
    pub status: &'static str,
    pub reason: String,
    pub hosts_json: String,
    pub devices_json: String,
    pub split_json: &'static str,
    pub versions_json: String,
    pub commit: String,
    pub command: &'static str,
    pub per_step_times_s: Vec<f64>,
    pub samples_per_sec: Option<f64>,
    pub bytes_sent_per_boundary_json: &'static str,
}

fn repo_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn git_text(args: &[&str]) -> String {
    let output = Command::new("git").arg("-C").arg(repo_root()).args(args).output();
    let Ok(output) = output else {
        return String::new();
    };
    if !output.status.success() {
        return String::new();
    }
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn transport_sentence() -> String {
    match linalg_transport::localhost_mesh(2) {
        Ok(endpoints) => {
            let count = endpoints.len();
            drop(endpoints);
            format!(
                "the only multi-rank transport constructor is linalg_transport::localhost_mesh, \
                 which built {count} endpoints on 127.0.0.1"
            )
        }
        Err(err) => format!("linalg_transport::localhost_mesh(2) returned {err:?}"),
    }
}

pub fn blocked_reason() -> String {
    format!(
        "not run. Workload W is gelu(x @ W + b), mean-squared error, and SGD at 1e-3. \
         linalg_pipeline::train applies gelu(x @ W) and has no bias add. \
         It requires dy before the forward that produces y, so the mean-squared-error \
         gradient of those predictions cannot be formed inside the step. \
         {}",
        transport_sentence()
    )
}

fn retain_pipeline_entry(run: bool) {
    if run {
        let backend = CpuBackend::new();
        let mut transport = IdleTransport;
        let placement = linalg_pipeline::contiguous_placement(1, 1).expect("one stage");
        let _ = train_entry(&backend, &mut transport, &placement, 1, &[], &[], &[]);
    }
}

pub fn report() -> Report {
    retain_pipeline_entry(false);
    let host = git_text(&["rev-parse", "--abbrev-ref", "HEAD"]);
    let hostname = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string());
    let commit = git_text(&["rev-parse", "HEAD"]);
    let rustc = Command::new("rustc").arg("--version").output().ok().and_then(|output| {
        output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
    });
    let porcelain = git_text(&["status", "--porcelain"]);
    let versions = format!(
        "{{\n    \"linalg-pipeline\": \"0.1.0\",\n    \"rustc\": {},\n    \"branch\": {},\n    \
         \"git_status\": {}\n  }}",
        json_string(rustc.as_deref().unwrap_or("")),
        json_string(&host),
        json_string(&porcelain)
    );
    Report {
        status: "not run",
        reason: blocked_reason(),
        hosts_json: format!(
            "[{{\n    \"hostname\": {},\n    \"role\": \"local bench process\"\n  }}]",
            json_string(&hostname)
        ),
        devices_json: "[\"cpu\"]".to_string(),
        split_json: "null",
        versions_json: versions,
        commit,
        command: "cargo run -p linalg-bench -- bench/runs/linalg-pipeline.json",
        per_step_times_s: Vec::new(),
        samples_per_sec: None,
        bytes_sent_per_boundary_json: "[]",
    }
}

pub fn write_report(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let report = report();
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, report.to_json())?;
    Ok(())
}

impl Report {
    pub fn to_json(&self) -> String {
        format!(
            "{{\n  \"workload\": \"W\",\n  \"stack\": \"linalg-pipeline\",\n  \
             \"status\": {},\n  \"reason\": {},\n  \"hosts\": {},\n  \"devices\": {},\n  \
             \"split\": {},\n  \"versions\": {},\n  \"commit\": {},\n  \"command\": {},\n  \
             \"warmup_steps\": {WARMUP_STEPS},\n  \"measured_steps\": {MEASURED_STEPS},\n  \
             \"per_step_times_s\": {},\n  \"median_step_s\": null,\n  \"samples_per_sec\": {},\n  \
             \"bytes_sent_per_boundary\": {},\n  \"layers\": {LAYERS},\n  \"width\": {WIDTH},\n  \
             \"microbatch_rows\": {MICROBATCH_ROWS},\n  \"microbatches\": {MICROBATCHES}\n}}\n",
            json_string(self.status),
            json_string(&self.reason),
            self.hosts_json,
            self.devices_json,
            self.split_json,
            self.versions_json,
            json_string(&self.commit),
            json_string(self.command),
            json_f64_list(&self.per_step_times_s),
            json_option_f64(self.samples_per_sec),
            self.bytes_sent_per_boundary_json,
        )
    }
}

fn json_option_f64(value: Option<f64>) -> String {
    match value {
        Some(number) if number.is_finite() => format!("{number}"),
        _ => "null".to_string(),
    }
}

fn json_f64_list(values: &[f64]) -> String {
    if values.is_empty() {
        return "[]".to_string();
    }
    let body = values.iter().map(|value| format!("{value}")).collect::<Vec<_>>().join(", ");
    format!("[{body}]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_is_2048_over_the_median() {
        assert_eq!(SAMPLES_PER_STEP, 2048);
        assert_eq!(median(&[1.0, 3.0, 2.0, 4.0]), Some(2.5));
        assert_eq!(median(&[4.0, 1.0, 3.0]), Some(3.0));
        assert_eq!(median(&[]), None);
        assert_eq!(samples_per_sec(2.0), Some(1024.0));
        assert_eq!(samples_per_sec(0.0), None);
    }

    #[test]
    fn workload_w_is_not_run() {
        let report = report();
        assert_eq!(report.status, "not run");
        assert!(report.samples_per_sec.is_none());
        assert!(report.per_step_times_s.is_empty());
        assert!(report.reason.contains("bias"));
        assert!(report.reason.contains("mean-squared"));
        assert!(report.reason.contains("127.0.0.1"));
        let json = report.to_json();
        for key in [
            "hosts",
            "devices",
            "split",
            "versions",
            "commit",
            "per_step_times_s",
            "samples_per_sec",
            "bytes_sent_per_boundary",
        ] {
            assert!(json.contains(&format!("\"{key}\"")), "{key} missing");
        }
        assert!(json.contains("\"samples_per_sec\": null"));
        assert!(json.contains("\"per_step_times_s\": []"));
        assert!(json.contains("\"bytes_sent_per_boundary\": []"));
    }

    #[test]
    fn train_rejects_an_empty_stage() {
        let backend = CpuBackend::new();
        let mut transport = IdleTransport;
        let placement = linalg_pipeline::contiguous_placement(1, 1).unwrap();
        let result = train_entry(&backend, &mut transport, &placement, MICROBATCHES, &[], &[], &[]);
        assert!(matches!(result, Err(Error::Schedule)));
    }
}
