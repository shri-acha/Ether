# Match Statement in Ether

The Ether programming language now supports `match` statements, similar to Rust's match expression.

## Syntax

```ether
match (expression) {
    pattern1 => {
        // code block
    }
    pattern2 => {
        // code block
    }
    _ => {
        // default case (wildcard)
    }
}
```

## Pattern Types

### 1. Literal Patterns
Match against specific literal values (integers, floats, booleans, strings, chars):

```ether
fn classify_number(n: int): int {
    match (n) {
        0 => {
            return 10;
        }
        1 => {
            return 20;
        }
        2 => {
            return 30;
        }
        _ => {
            return 99;
        }
    }
}
```

### 2. Enum Variant Patterns
Match against specific enum variants:

```ether
enum Status {
    Success,
    Error,
    Pending
}

fn check_status(s: Status): int {
    match (s) {
        Status::Success => {
            return 1;
        }
        Status::Error => {
            return 0;
        }
        Status::Pending => {
            return 2;
        }
    }
}
```

### 3. Wildcard Pattern
The underscore `_` pattern matches any value and is typically used as a default case:

```ether
match (value) {
    1 => {
        // handle 1
    }
    2 => {
        // handle 2
    }
    _ => {
        // handle all other values
    }
}
```

You can also use named identifiers as wildcards:

```ether
match (value) {
    0 => {
        return 0;
    }
    other => {
        // 'other' acts as a wildcard, matching any value
        return 1;
    }
}
```

## Implementation Details

- Match statements are compiled to efficient if-else chains in LLVM IR
- Enum variant matching is done by comparing discriminant values
- All match arms must have compatible return types (enforced by type checker)
- Pattern matching does not currently support variable capture (bindings)

## Examples

See `examples/match_example.etr` for a complete working example.
