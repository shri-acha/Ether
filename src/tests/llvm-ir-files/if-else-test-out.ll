; ModuleID = 'my_module'
source_filename = "my_module"

declare void @__Eth_print(ptr)

declare ptr @__Eth_read()

define i64 @fibonacci(i64 %0) {
entry:
  %n = alloca i64, align 8
  store i64 %0, ptr %n, align 4
  %n1 = load i64, ptr %n, align 4
  %le = icmp sle i64 %n1, 1
  br i1 %le, label %then, label %else

then:                                             ; preds = %entry
  %n2 = load i64, ptr %n, align 4
  ret i64 %n2

else:                                             ; preds = %entry
  %n3 = load i64, ptr %n, align 4
  %sub = sub i64 %n3, 1
  %call = call i64 @fibonacci(i64 %sub)
  %n4 = load i64, ptr %n, align 4
  %sub5 = sub i64 %n4, 2
  %call6 = call i64 @fibonacci(i64 %sub5)
  %add = add i64 %call, %call6
  ret i64 %add

ifcont:                                           ; No predecessors!
  ret i64 0
}

define i64 @main() {
entry:
  %call = call i64 @fibonacci(i64 7)
  %result = alloca i64, align 8
  store i64 %call, ptr %result, align 4
  %result1 = load i64, ptr %result, align 4
  ret i64 %result1
}
