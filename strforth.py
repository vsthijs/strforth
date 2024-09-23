#!/usr/bin/env python3

# TODO: implement syscall words
# TODO: implement function definitions
# TODO: implement function calls
# FIXME: comments raise empty error

"""
linux x86_64 syscall table

| syscall | rax | rdi       | rsi | rdx   | 
|---------|-----|-----------|-----|-------|
| write   | 1   | fd        | buf | count |
| exit    | 60  | exit code |
"""

from dataclasses import dataclass
import subprocess
from typing import Generator, Any, Literal, TypeAlias
import sys


LONGSTRINGCNT = 8
SHORTSTRINGCNT = 2
MICROSTRINGCNT = 1
STRINGCNT = SHORTSTRINGCNT


def fatal_err(
    *values,
    sep: str = " ",
    end: str = "\n",
    flush: bool = False,
    code: int = 1,
):
    global program_name
    print(
        program_name + ": err: ",
        *values,
        sep=sep,
        end=end,
        file=sys.stderr,
        flush=flush,
    )
    exit(code)


@dataclass
class Token:
    kind: Literal["word", "string", "integer"]
    data: str | int
    file: str
    line: int

    def __eq__(self, value: object) -> bool:
        return self.data == value


class AstNode:
    file: str
    line: int

    def __init__(self, file: str, line: int) -> None:
        self.file, self.line = file, line


class Word(AstNode):
    name: str

    def __init__(self, tok: Token) -> None:
        super().__init__(tok.file, tok.line)
        assert isinstance(tok.data, str)
        self.name = tok.data

    def __str__(self) -> str:
        return self.name


class LiteralValue(AstNode):
    value: str | int

    def __init__(self, tok: Token) -> None:
        super().__init__(tok.file, tok.line)
        self.value = tok.data

    def __str__(self) -> str:
        if isinstance(self.value, int):
            return str(self.value)
        else:
            return f"'{self.value}'"


class Block(AstNode):
    kind: Literal["loop", "cond"]
    inner: list[AstNode]

    def __init__(
        self, kind: Literal["loop", "cond"], inner: list[AstNode], file: str, line: int
    ) -> None:
        super().__init__(file, line)
        self.kind, self.inner = kind, inner

    def __str__(self) -> str:
        return " ".join(
            [
                ("[" if self.kind == "cond" else "{"),
                *[str(i) for i in self.inner],
                ("]" if self.kind == "cond" else "}"),
            ]
        )


class FunctionDef(AstNode):
    name: str | None
    inner: list[AstNode]

    def __init__(
        self, name: str | None, inner: list[AstNode], file: str, line: int
    ) -> None:
        super().__init__(file, line)
        self.name, self.inner = name, inner

    def __str__(self) -> str:
        return " ".join(
            [
                ":",
                (self.name if self.name else ""),
                *[str(i) for i in self.inner],
                ";",
            ]
        )


def countlines(s: str) -> int:
    return s.count("\n")


def lex(source: str, file: str):
    ip = 0
    while ip < len(source):
        ch = source[ip]
        if ch in "\"'":
            ip += 1
            data = ""
            while ip < len(source) and source[ip] not in "\"'":
                data += source[ip]
                ip += 1
            ip += 1
            yield Token("string", data, file, countlines(source[:ip]))
        elif ch.isspace():
            while ip < len(source) and source[ip].isspace():
                ip += 1
        else:
            data = ""
            while ip < len(source) and not source[ip].isspace():
                data += source[ip]
                ip += 1
            try:
                data = int(data)
            except ValueError:
                pass
            yield Token(
                "integer" if isinstance(data, int) else "word",
                data,
                file,
                countlines(source[:ip]),
            )


def parse(lexer: Generator[Token, Any, None]):
    tokens = list(lexer)

    def inner_parse(ip: int) -> tuple[AstNode, int]:
        if tokens[ip] == ":":  # function definition
            # : <name> <definition> ;
            ip += 2
            name = tokens[ip - 1]
            assert isinstance(name.data, str)
            body = []
            while ip < len(tokens) and tokens[ip] != ";":
                el, ip = inner_parse(ip)
                body.append(el)
            ip += 1
            return (
                FunctionDef(
                    name.data,
                    body,
                    tokens[ip - 1].file,
                    tokens[ip - 1].line,
                ),
                ip,
            )
        elif tokens[ip] == "(":  # comment
            # ( comment )
            level = 1
            ip += 1
            while ip < len(tokens) and level > 0:
                if tokens[ip] == "(":
                    level += 1
                elif tokens[ip] == ")":
                    level -= 1
                ip += 1
            return inner_parse(ip)
        elif tokens[ip] == "[" or tokens[ip] == "{":  # conditional or loop block
            start = tokens[ip].data
            assert isinstance(start, str)
            end = {"[": "]", "{": "}"}[start]
            ip += 1
            body = []
            while ip < len(tokens) and tokens[ip] != end:
                el, ip = inner_parse(ip)
                body.append(el)
            ip += 1
            return (
                Block(
                    "cond" if end == "]" else "loop",
                    body,
                    tokens[ip - 1].file,
                    tokens[ip - 1].line,
                ),
                ip,
            )
        elif tokens[ip].kind == "word":
            ip += 1
            return Word(tokens[ip - 1]), ip
        else:
            ip += 1
            return LiteralValue(tokens[ip - 1]), ip

    ip = 0
    while ip < len(tokens):
        el, ip = inner_parse(ip)
        yield el


# def evaluate(expr: AstNode, stack: list[int | str | bool]):
# if isinstance(expr, LiteralValue):
#     stack.append(expr.value)
# elif isinstance(expr, Word):
#     if expr.name == "dbg":
#         print(stack.pop())
#     elif expr.name == "dup":
#         stack.append(stack[-1])
#     elif expr.name == "dec":
#         assert isinstance((val := stack.pop()), int)
#         stack.append(val - 1)
#     else:
#         assert False
# elif isinstance(expr, FunctionDef):
#     assert False
# elif isinstance(expr, Block):
#     if expr.kind == "cond":
#         if stack.pop() != 0:
#             for inex in expr.inner:
#                 evaluate(inex, stack)
#     elif expr.kind == "loop":
#         while True:
#             for inex in expr.inner:
#                 evaluate(inex, stack)
#             if stack.pop() == 0:
#                 break
# else:
#     assert False


DataType = Literal["int", "bool", "str"]
builtin_words = ["dbg"]


class NasmAmd64Linux:
    def __init__(self, prog: list[AstNode]) -> None:
        self.asts = prog
        self.checker: list[DataType] = []
        self.curr_func = "_start"
        self.funcs: dict[str, list[str]] = {"_start": []}
        self.block_bumps: dict[str, int] = {"_start": 0}
        self.curr_scope = []
        self.literals: list[tuple[str, Any]] = []

    def _ret(self):
        self.curr_func = self.curr_scope.pop()

    def stexpect(self, *types: DataType) -> bool:
        """Expect one of the given data types."""
        if len(self.checker) <= 0:
            return False
        return self.checker.pop() in types

    def comp_word(self, word: str):
        if word in ["true", "false"]:
            self._push_bool({"true": True, "false": False}[word])
        elif word == "dup":
            self._inst("pop rax")
            self._inst("push rax")
            self._inst("push rax")
        elif word == "dec":
            self._inst("pop rax")
            self._inst("dec rax")
            self._inst("push rax")
        elif word == "exit":
            assert self.stexpect("int")
            self._inst("mov rax, 60")
            self._inst("pop rdi")
            self._inst("syscall")
        elif word == "print":
            # fd  | buf | count
            # rdi | rsi | rdx
            assert self.stexpect("str")
            self._inst("pop rax")
            asmtype = {1: "BYTE", 2: "WORD", 4: "DWORD", 8: "QWORD"}[STRINGCNT]
            self._inst("xor rdx, rdx")
            self._inst(f"mov dx, {asmtype} [rax]")
            self._inst("mov rsi, rax")
            self._inst(f"add rsi, {STRINGCNT}")
            self._inst("mov rdi, 1")
            self._inst("mov rax, 1")
            self._inst("syscall")
        else:  # ?
            assert False

    def _switch_stack(self):
        assert False

    def _inst(self, line: str):
        self.funcs[self.curr_func].append(line)

    def _push_bool(self, val: bool):
        self.checker.append("bool")
        self._inst(f"push {int(val)}")

    def _push_int(self, val: int):
        self.checker.append("int")
        self._inst(f"push {val}")

    def _new_literal_id(self) -> str:
        return f"_sf_lit_{len(self.literals)}"

    def _new_block_id(self) -> str:
        self.block_bumps[self.curr_func] += 1
        return f".{self.block_bumps[self.curr_func]-1}"

    def _push_str(self, val: str):
        self.checker.append("str")
        literal = self._new_literal_id()
        self.literals.append((literal, val))
        self._inst(f"push {literal}")

    def _get_asm(self) -> str:
        asm: list[str] = ["section .data"]
        for name, value in self.literals:
            if isinstance(value, str):
                definst = {1: "db", 2: "dw", 4: "dd", 8: "dq"}[STRINGCNT]
                asm.append(f"{name}: {definst} {len(value.encode())+1}")
                asm.append(f'{len(name)*" "}  db "{value}", 10, 0')

        asm.append("section .text")
        asm.append("global _start")
        for func in self.funcs:
            asm.append(f"{func}:")
            if func != "_start":
                asm.append(f"  jmp .end")  # dont execute, but only define
            asm.append(f".start:")
            for inst in self.funcs[func]:
                indent = 0 if inst.endswith(":") else 2
                asm.append(" " * indent + inst)
            asm.append(f".end:")
        asm.append("ret")
        return "\n".join(asm) + "\n"

    def _conditional(self, inner: list[AstNode]):
        begin = self._new_block_id()
        end = self._new_block_id()
        self._inst("pop rax")
        self._inst("cmp rax, 0")
        self._inst(f"je {end}")  # jump to the end if 0
        self._inst(begin + ":")
        for ii in inner:
            self._process_node(ii)
        self._inst(end + ":")

    def _loop(self, inner: list[AstNode]):
        begin = self._new_block_id()
        end = self._new_block_id()
        self._inst(begin + ":")
        for ii in inner:
            self._process_node(ii)
        self._inst("pop rax")
        self._inst("cmp rax, 0")
        self._inst(f"jne {begin}")  # jump to the end if 0
        self._inst(end + ":")

    def func_def(self, name: str | None, inner: list[AstNode]):
        assert False

    def _process_node(self, node: AstNode):
        if isinstance(node, LiteralValue):
            if isinstance(node.value, str):
                self._push_str(node.value)
            elif isinstance(node.value, int):
                self._push_int(node.value)
            else:
                assert False
        elif isinstance(node, Word):
            self.comp_word(node.name)
        elif isinstance(node, FunctionDef):
            self.func_def(node.name, node.inner)
        elif isinstance(node, Block):
            if node.kind == "cond":
                self._conditional(node.inner)
            elif node.kind == "loop":
                self._loop(node.inner)
        else:
            assert False

    def process_ast(self):
        while len(self.asts) > 0:
            ii = self.asts[0]
            self._process_node(ii)
            self.asts.pop(0)


def exec_cmd(cmd: list[str], check: bool = True):
    print(">", *cmd)
    if check and (result := subprocess.run(cmd, capture_output=True)).returncode != 0:
        raise Exception(
            f"err: nasm ({result.returncode}): {result.stderr.decode()}".strip()
        )


def compile_file(file: str, of: str):
    with open(file) as f:
        compiler = NasmAmd64Linux(list(parse(lex(f.read(), file))))
    compiler.process_ast()
    binfile = file.removesuffix(".sf")
    asmfile = binfile + ".asm"
    ofile = binfile + ".o"
    with open(asmfile, "w") as f:
        f.write(compiler._get_asm())
    exec_cmd(["nasm", "-felf64", asmfile])
    exec_cmd(["ld", ofile, "-o", of])


def shift(xs):
    return xs[0], xs[1:]


def show_usage():
    global program_name
    print(
        """Usage:
  PROG <file>               compiles the given source file.
  PLEN        -r | --run    runs the resulting executable if compilation was succesfull.
  PLEN        -o <output>   saves the resulting executable as specified.
""".replace(
            "PROG", program_name
        ).replace(
            "PLEN", " " * len(program_name)
        )
    )


def main():
    global program_name
    program_name, argv = shift(sys.argv)
    inpfile: str | None = None
    outfile: str | None = None
    runargs: list[str] = []

    # simple flags with their flag names and default values
    flags: dict[str, bool] = {
        "run": False,
    }

    if len(argv) <= 0:
        show_usage()

    while len(argv):
        arg, argv = shift(argv)
        if arg.startswith("-"):
            if arg in ["-h", "--help"]:
                show_usage()
                exit(0)
            elif arg.startswith("--"):
                if arg.startswith("--no-"):
                    if (flag := arg.removeprefix("--no-")) in flags:
                        flags[flag] = False
                    else:
                        fatal_err(f"unknown flag: {arg}")
                elif (flag := arg.removeprefix("--")) in flags:
                    flags[flag] = True
                elif arg == "--":
                    runargs = argv  # pass all the arguments to the run command
                    argv = []  # ignore those commands
                    flags["run"] = True
                else:
                    fatal_err(f"unknown flag: {arg}")
            elif arg.startswith("-o"):
                if arg == "-o":
                    outfile, argv = shift(argv)  # -o a.out
                else:
                    outfile = arg.removeprefix("-o")  # -oa.out
        else:
            if inpfile:
                fatal_err("can't compile multiple source files")
            inpfile = arg

    if inpfile is None:
        fatal_err("no source file given")
    assert isinstance(inpfile, str), "unreachable: type checker"

    if outfile is None:
        outfile = inpfile.removesuffix(".sf")

    try:
        compile_file(inpfile, outfile)
    except Exception as e:
        fatal_err(str(e))

    if flags["run"]:
        exec_cmd([outfile, *runargs], False)


if __name__ == "__main__":
    main()
