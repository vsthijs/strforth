# Builtin words

- `true` and `false` - push the associated values to the stack.
- `dup` - duplicate the top element.
- `swap` - swap the 2 top elements.
- `+`, `-`, `*` - do arithmatic on the 2 top elements.
- `divmod` - divides the 2 top elements. first pushes the quotient, and then
  the modulus
- `=`, `>`, `<`, `!=`, `>=`, `<=` - compare the 2 top elements.
- `neg` - negate the top element.
- `not` - speaks for itself.
- `syscall<n>` - execute a syscall with the specified amount of arguments. The
  linux exit syscall is nr 60, and accepts one argument. This would be
  `0 60 syscall1`.
