# Strforth

A forth implementation by Thijs van STRaaten, without an intermediate mode.

## Syntax

Like regular forth, values are pushed on the stack as they are in the source file. Words are also executing like regular.
Words are defined by `: word body ;`

## Internals

A program is compiled using nasm. To call a function in assembly, a stack is
needed without data on it. To work around this, 2 stacks are needed. a
datastack and a callstack. When calling and returning from a function, the
stacks are swapped.

A string pushes 2 elements on the stack: the length, and the pointer to the string