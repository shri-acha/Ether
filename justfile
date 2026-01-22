build:
	cargo +nightly build --release
fmt:
	cargo +nightly fmt
run:
	cargo +nightly run --release
test:
	cargo +nightly test -- --show-output
test-one TEST:
	cargo +nightly test {{TEST}} -- --show-output 
stdlib:
  rustc stdlib/stdlib.rs --crate-type staticlib -o ./stdlib/ethstdlib.a
