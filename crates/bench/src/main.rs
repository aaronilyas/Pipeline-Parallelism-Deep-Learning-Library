fn main() {
    let path =
        std::env::args().nth(1).unwrap_or_else(|| "bench/runs/linalg-pipeline.json".to_string());
    if let Err(err) = linalg_bench::write_report(&path) {
        eprintln!("linalg-bench: {err}");
        std::process::exit(1);
    }
    println!("{path}");
}
