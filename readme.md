# Strforth

A forth compiler by Thijs van STRaaten. There is no intermediate mode.

Interaction with the OS is done using syscalls. Forth binaries do not link with anything. No Libc. No GNU.

## Syntax

Forth is a stack-oriented programming language. This means that expressions are written in postfix notation. For example, `34 35 +`, adds 34 and 35. Here `+` is a word. There are builtin words for comparison, arithmatic, control-flow and low-level things. If a token is not a number or a string, it is automatically a word.

- Comments are written in `(` and `)`. These parenthesis have to be surrounded with whitespace, just like every other token. Nested comments are supported.
- Conditionals are done with the `[` and `]`. At `[`, if the top of the stack is equal to false, the body is skipped.
- Loops are done with the `{` and `}`. At `}`, if the top of the stack is not equal to false, there is a jump back to the opening `{`.

### User-defined words

Example:

```forth
: my_word
  34 35 +
;
```

newlines are ignored and handled as whitespace, so it can be written on one line as:

```forth
: my_word 34 35 + ;
``` 

## Hello World

```forth
include "std.sf"
"Hello, World!" print
```

## Internals

A program is compiled using nasm. To call a function in assembly, a stack is
needed without data on it. To work around this, 2 stacks are needed. a
datastack and a callstack. When calling and returning from a function, the
stacks are swapped.

A string pushes 2 elements on the stack: the length, and the pointer to the string

With an exception for strings, the source code is literally split by whitespace. This is why the `(`, `)`, `[`, `]`, `{`, `}`, etc. all have to be surrounded by whitespace.
