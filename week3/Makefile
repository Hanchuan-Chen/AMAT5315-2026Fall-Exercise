.PHONY: reproduce snapshots sweep plot test

reproduce: snapshots sweep plot

snapshots:
	cargo run --release -- snapshots

sweep:
	cargo run --release -- sweep

plot:
	cargo run --release -- plot artifacts

test:
	cargo test --release

