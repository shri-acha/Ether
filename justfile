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
llvm-file-compile FILE:
  llc src/tests/llvm-ir-files/{{FILE}}.ll -o {{FILE}}.o --filetype=obj --relocation-model=pic
link-stdlib FILE:
  gcc {{FILE}}.o -L. -l:./stdlib/ethstdlib.a -o {{FILE}}.out
