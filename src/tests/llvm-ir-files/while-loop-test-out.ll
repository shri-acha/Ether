; ModuleID = 'my_module'
source_filename = "my_module"

declare void @__Eth_print_str(ptr)

declare void @__Eth_print_i64(i64)

declare ptr @__Eth_read()

define internal i64 @sum_to_n(i64 %0) {
entry:
  %n = alloca i64, align 8
  store i64 %0, ptr %n, align 4
  %sum = alloca i64, align 8
  store i64 0, ptr %sum, align 4
  %i = alloca i64, align 8
  store i64 0, ptr %i, align 4
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %i1 = load i64, ptr %i, align 4
  %n2 = load i64, ptr %n, align 4
  %le = icmp sle i64 %i1, %n2
  br i1 %le, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %sum3 = load i64, ptr %sum, align 4
  %i4 = load i64, ptr %i, align 4
  %add = add i64 %sum3, %i4
  store i64 %add, ptr %sum, align 4
  %i5 = load i64, ptr %i, align 4
  %add6 = add i64 %i5, 1
  store i64 %add6, ptr %i, align 4
  br label %while.cond

while.end:                                        ; preds = %while.cond
  %sum7 = load i64, ptr %sum, align 4
  ret i64 %sum7
}

define i64 @main() {
entry:
  %call = call i64 @sum_to_n(i64 10)
  ret i64 %call
}
