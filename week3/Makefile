.PHONY: reproduce snapshots sweep plot analyze wolff compare test

reproduce: snapshots sweep plot analyze

snapshots:
	cargo run --release -- snapshots

sweep:
	cargo run --release -- sweep

plot:
	cargo run --release -- plot artifacts

analyze:
	cargo run --release -- analyze artifacts --blocks 50

wolff:
	cargo run --release -- sweep --wolff
	cargo run --release -- analyze artifacts-wolff --blocks 50

compare:
	cargo run --release -- compare artifacts artifacts-wolff --output tau-compare.png

test:
	cargo test --release
