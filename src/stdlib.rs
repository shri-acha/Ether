#[unsafe(no_mangle)]
extern "C" fn __Eth_print(val:*const i8){
    print!("{:?}",val);
}
