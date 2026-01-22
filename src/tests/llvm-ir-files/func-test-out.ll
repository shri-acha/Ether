; ModuleID = 'my_module'
source_filename = "my_module"

declare void @__Eth_print(ptr)

declare ptr @__Eth_read()

define i64 @add(i64 %0, i64 %1) {
entry:
  %a = alloca i64, align 8
  store i64 %0, ptr %a, align 4
  %b = alloca i64, align 8
  store i64 %1, ptr %b, align 4
  %a1 = load i64, ptr %a, align 4
  %b2 = load i64, ptr %b, align 4
  %add = add i64 %a1, %b2
  ret i64 %add
}

define i64 @main() {
entry:
  %x = alloca i64, align 8
  store i64 10, ptr %x, align 4
  %y = alloca i64, align 8
  store i64 20, ptr %y, align 4
  %x1 = load i64, ptr %x, align 4
  %y2 = load i64, ptr %y, align 4
  %call = call i64 @add(i64 %x1, i64 %y2)
  %result = alloca i64, align 8
  store i64 %call, ptr %result, align 4
  %result3 = load i64, ptr %result, align 4
  ret i64 %result3
}
