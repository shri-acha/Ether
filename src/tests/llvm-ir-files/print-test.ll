; ModuleID = 'my_module'
source_filename = "my_module"

@str = private unnamed_addr constant [18 x i8] c"hello from Ether!\00", align 1

declare void @__Eth_print_str(ptr)

declare void @__Eth_print_i64(i64)

declare ptr @__Eth_read()

define i64 @main() {
entry:
  call void @__Eth_print_str(ptr @str)
  %x = alloca {}, align 8
  store {} zeroinitializer, ptr %x, align 1
  ret i64 0
}
