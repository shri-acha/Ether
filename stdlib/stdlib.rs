use std::ffi::{CStr, CString};
use std::io::{self, Write};
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn __Eth_print(val: *const c_char) {
    unsafe {
        if val.is_null() {
            eprintln!("Error: null pointer passed to __Eth_print");
            return;
        }
        
        let c_str = CStr::from_ptr(val);
        match c_str.to_str() {
            Ok(s) => println!("{}", s),
            Err(e) => {
                eprintln!("Error decoding string: {:?}", e);
                eprintln!("Raw bytes: {:?}", c_str.to_bytes());
            }
        }
    }
}

static mut READ_BUFFER: [u8; 4096] = [0; 4096];

#[no_mangle]
pub extern "C" fn __Eth_read() -> *const c_char {
    let mut input = String::new();
    
    match io::stdin().read_line(&mut input) {
        Ok(_) => {
            // Remove trailing newline/carriage return
            let trimmed = input.trim_end();
            
            unsafe {
                // IMPORTANT: Zero out the entire buffer first!
                READ_BUFFER = [0; 4096];
                
                let bytes = trimmed.as_bytes();
                let len = bytes.len().min(4095);
                
                if len > 0 {
                    READ_BUFFER[..len].copy_from_slice(&bytes[..len]);
                }
                READ_BUFFER[len] = 0; // Null terminator
                READ_BUFFER.as_ptr() as *const c_char
            }
        }
        Err(e) => {
            eprintln!("Error reading: {}", e);
            std::ptr::null()
        }
    }
}
