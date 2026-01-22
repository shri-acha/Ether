; ModuleID = 'my_module'
source_filename = "my_module"

declare void @__Eth_print(ptr)

declare ptr @__Eth_read()

define i64 @main() {
entry:
  %end_value = alloca i64, align 8
  store i64 6, ptr %end_value, align 4
  %result = alloca i64, align 8
  store i64 0, ptr %result, align 4
  br label %for.init

for.init:                                         ; preds = %entry
  %for.counter = alloca i64, align 8
  store i64 0, ptr %for.counter, align 4
  %end_value1 = load i64, ptr %end_value, align 4
  %for.start = alloca i64, align 8
  store i64 2, ptr %for.start, align 4
  %for.end2 = alloca i64, align 8
  store i64 %end_value1, ptr %for.end2, align 4
  store i64 2, ptr %for.counter, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %for.init
  %counter.val = load i64, ptr %for.counter, align 4
  %end.val = load i64, ptr %for.end2, align 4
  %for.cond3 = icmp slt i64 %counter.val, %end.val
  br i1 %for.cond3, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %i = alloca i64, align 8
  %counter.load = load i64, ptr %for.counter, align 4
  store i64 %counter.load, ptr %i, align 4
  %result4 = load i64, ptr %result, align 4
  %i5 = load i64, ptr %i, align 4
  %add = add i64 %result4, %i5
  store i64 %add, ptr %result, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %counter.inc.load = load i64, ptr %for.counter, align 4
  %counter.inc = add i64 %counter.inc.load, 1
  store i64 %counter.inc, ptr %for.counter, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %result6 = load i64, ptr %result, align 4
  ret i64 %result6
}
