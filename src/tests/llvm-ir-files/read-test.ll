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
  %read_call = call ptr @__Eth_read()
  %input_value = alloca ptr, align 8
  store ptr %read_call, ptr %input_value, align 8
  %input_value1 = load ptr, ptr %input_value, align 8
  call void @__Eth_print_str(ptr %input_value1)
  %y = alloca {}, align 8
  store {} zeroinitializer, ptr %y, align 1
  ret i64 0
}
