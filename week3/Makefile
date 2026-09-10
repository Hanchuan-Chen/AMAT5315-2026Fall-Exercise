.PHONY: reproduce snapshots sweep plot analyze test

reproduce: snapshots sweep plot analyze

snapshots:
	cargo run --release -- snapshots

sweep:
	cargo run --release -- sweep

plot:
	cargo run --release -- plot artifacts

analyze:
	cargo run --release -- analyze artifacts --blocks 50

test:
	cargo test --release
