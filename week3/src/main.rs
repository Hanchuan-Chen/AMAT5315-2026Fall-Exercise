fn main() {
    if let Err(message) = ising::cli::run() {
        eprintln!("error: {message}");
        std::process::exit(2);
    }
}
