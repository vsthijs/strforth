#!/usr/bin/env python3

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
import os
import traceback


LONGSTRINGCNT = 8
SHORTSTRINGCNT = 2
MICROSTRINGCNT = 1
STRINGCNT = SHORTSTRINGCNT


def escape_string(s: str) -> str:
    for old, new in {
        "\\t": "\t",
        "\\n": "\n",
        "\\0": "\0",
        "\\\\": "\\",
    }.items():
        s = s.replace(old, new)
    return s


def nasm_string(s: str) -> str:
    """converts 'Hello World!\n' to 'Hello World!', 10 for nasm"""
    special_chars = {"\t": 9, "\n": 10}
    line = ""
    instring = False
    for ii in s:
        if ii in special_chars:
            if instring:
                line += '", '
                instring = False
            line += f"{special_chars[ii]}, "
        else:
            if not instring:
                line += '"'
                instring = True
            line += ii
    if instring:
        line += '"'
    return line.removeprefix(", ")


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
    name: str
    inner: list[AstNode]

    def __init__(self, name: str, inner: list[AstNode], file: str, line: int) -> None:
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
            yield Token("string", escape_string(data), file, countlines(source[:ip]))
        elif ch.isspace():
            while ip < len(source) and source[ip].isspace():
                ip += 1
        elif ch == "#":
            while ip < len(source) and source[ip] != "\n":
                ip += 1
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

    def inner_parse(ip: int) -> tuple[AstNode | None, int]:
        assert ip < len(tokens)
        if tokens[ip] == ":":  # function definition
            # : <name> <definition> ;
            ip += 2
            name = tokens[ip - 1]
            assert isinstance(name.data, str)
            body = []
            while ip < len(tokens) and tokens[ip] != ";":
                el, ip = inner_parse(ip)
                if el:
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
            return None, ip
        elif tokens[ip] == "[" or tokens[ip] == "{":  # conditional or loop block
            start = tokens[ip].data
            assert isinstance(start, str)
            end = {"[": "]", "{": "}"}[start]
            ip += 1
            body = []
            while ip < len(tokens) and tokens[ip] != end:
                el, ip = inner_parse(ip)
                if el:
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
        if el:
            yield el


DataType = Literal["int", "bool", "str"]
builtin_words = ["dbg"]


class NasmAmd64Linux:
    # TODO: add .bss section for the datastack

    def __init__(self, prog: list[AstNode]) -> None:
        self.asts = prog
        self.checker: list[DataType] = []

        self.funcs: dict[str, list[str]] = {}
        self.curr_func: str | None = None
        self.toplevel: list[str] = []
        self.scope: list[str | None] = []

        self.block_bump = 0

        self.literals: list[tuple[str, Any]] = []

        self._switch_stack()

    def _ret(self):
        self._switch_stack()
        self._inst("ret")

    def _finalize_fn(self):
        self._ret()
        self.curr_func = self.scope.pop()

    def _call(self, name: str):
        self._switch_stack()
        self._inst(f"call {name}")
        self._switch_stack()

    def comp_word(self, word: str):
        self._inst(f"; {word}")
        if word in ["true", "false"]:
            self._push_bool({"true": True, "false": False}[word])
        elif word == "dup":
            self._inst("pop rax")
            self._inst("push rax")
            self._inst("push rax")
        elif word == "drop":
            self._inst("pop rax")
        elif word == "swap":
            self._inst("pop rbx")
            self._inst("pop rax")
            self._inst("push rbx")
            self._inst("push rax")
        elif word in ["+", "-", "*"]:
            self._inst("pop rbx")
            self._inst("pop rax")
            self._inst({"+": "add", "-": "sub", "*": "mul"}[word] + " rax, rbx")
            self._inst("push rax")
        elif word == "divmod":
            self._inst("pop rax")
            self._inst("pop rbx")
            self._inst("xor rdx, rdx")
            self._inst("div rbx")  # now rax contains the quotient and rdx the modulus.
            self._inst("push rax")
            self._inst("push rdx")
        elif word in ["=", ">", "<", "!=", ">=", "<="]:
            self._inst("pop rcx")
            self._inst("pop rbx")
            self._inst("cmp rcx, rbx")
            self._inst(
                {
                    "=": "setz",
                    ">": "setl",
                    "<": "setg",
                    "!=": "setnz",
                    ">=": "setle",
                    "<=": "setge",
                }[word]
                + " al"
            )
            self._inst("movzx rax, al")
            self._inst("push rax")
        elif word == "-":
            self._inst("pop rax")
            self._inst("neg rax")
            self._inst("push rax")
        elif word == "not":
            self._inst("pop rax")
            self._inst("test rax, rax")
            self._inst("sete al")
            self._inst("movzx rax, al")
            self._inst("push rax")
        elif word.startswith("syscall") and len(word) == 8:
            argn = int(word[-1])
            self._inst("pop rax")
            if argn >= 1:
                self._inst("pop rdi")
            if argn >= 2:  # etc.
                self._inst("pop rsi")
            if argn >= 3:  # syscall3
                self._inst("pop rdx")
            if argn >= 4:  # syscall4
                self._inst("pop r10")
            if argn >= 5:  # syscall5
                self._inst("pop r8")
            if argn >= 6:  # syscall6
                self._inst("pop r9")
            self._inst("syscall")
            self._inst("push rax")
        elif word in self.funcs:
            self._call(word)
        else:  # ?
            assert False, f"unknown word: {word}"

    def _switch_stack(self):
        # print(__file__ + ":" + str(sys._getframe().f_back.f_lineno) + ": " + str(sys._getframe().f_back.f_code.co_name) + ": " + str(self.curr_func))  # type: ignore
        for ii in [
            "; === Switch stacks ===",
            "mov rax, rsp",
            "mov rsp, [__strforth_altstack]",
            "mov [__strforth_altstack], rax",
        ]:
            self._inst(ii)

    def _inst(self, line: str):
        if self.curr_func:
            self.funcs[self.curr_func].append(line)
        else:
            self.toplevel.append(line)

    def _push_bool(self, val: bool):
        self._inst(f"; push {str(val).lower()}")
        self._inst(f"push {int(val)}")

    def _push_int(self, val: int):
        self._inst(f"; push {val}")
        self._inst(f"push {val}")

    def _new_literal_id(self) -> str:
        return f"_sf_lit_{len(self.literals)}"

    def _new_block_id(self) -> str:
        self.block_bump += 1
        return f".{self.block_bump-1}"

    def _push_str(self, val: str):
        comment_string = val.replace("\n", "\\n")
        self._inst(f"; push '{comment_string}'")
        literal = self._new_literal_id()
        self.literals.append((literal, val))
        self._push_int(len(val.encode()))
        self._inst(f"push {literal}")

    def _get_asm(self) -> str:
        asm: list[str] = [
            "section .bss",
            "__strforth_datastack: resb 4096",
            "section .data",
            "__strforth_altstack: dq __strforth_datastack + 4096",
        ]
        for name, value in self.literals:
            if isinstance(value, str):
                asm.append(f"{name}: db {nasm_string(value)}")

        asm.append("section .text")
        asm.append("global _start")
        asm.append("_start:")
        for func in self.funcs:
            asm.append(f"  jmp {func}.end")  # dont execute, but only define
            asm.append(f"{func}:")
            for inst in self.funcs[func]:
                indent = 0 if inst.endswith(":") else 2
                asm.append(" " * indent + inst)
            asm.append(f"{func}.end:")
        for inst in self.toplevel:
            indent = 0 if inst.endswith(":") else 2
            asm.append(" " * indent + inst)
        return "\n".join(asm) + "\n"

    def _conditional(self, inner: list[AstNode]):
        self._inst(f"; conditional")
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
        self._inst(f"; loop")
        begin = self._new_block_id()
        end = self._new_block_id()
        self._inst(begin + ":")
        for ii in inner:
            self._process_node(ii)
        self._inst("pop rax")
        self._inst("cmp rax, 0")
        self._inst(f"jne {begin}")  # jump to the end if 0
        self._inst(end + ":")

    def func_def(self, name: str, inner: list[AstNode]):
        self.scope.append(self.curr_func)
        self.curr_func = name
        self.funcs[name] = []

        if len(inner) < 1:
            print(f"warning: {name} has empty definition")
            self._inst("ret")
            self.curr_func = self.scope.pop()
        else:
            self._switch_stack()
            for ii in inner:
                self._process_node(ii)
            self._finalize_fn()

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
            assert False, type(node)

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


def echo_cmd(cmd: list[str]):
    print(">", *cmd)
    return subprocess.call(cmd)


def compile_file(file: str, of: str, keep_artifacts: bool):
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

    if not keep_artifacts:
        os.remove(asmfile)
        os.remove(ofile)


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


def test_file(ifile: str) -> tuple[bool, str]:
    print(f"=== Compiling {ifile} ===")
    try:
        compile_file(ifile, (ofile := ifile.removesuffix(".sf")), False)
    except Exception as e:
        traceback.print_exc()
        return False, (str(e))
    print(f"=== Running {ofile} ===")
    if echo_cmd(["./" + ofile]) != 0:
        return False, "runtime error"
    return True, "Ok"


def main():
    global program_name
    program_name, argv = shift(sys.argv)
    inpfile: str | None = None
    outfile: str | None = None
    runargs: list[str] = []

    # simple flags with their flag names and default values
    flags: dict[str, bool] = {"run": False, "artifacts": False, "test-all": False}

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
            elif arg == "-r":
                flags["run"] = True
            else:
                fatal_err(f"unknown commandline flag: {arg}")
        else:
            if inpfile:
                fatal_err("can't compile multiple source files")
            inpfile = arg

    if flags["test-all"]:
        if inpfile is None:
            fatal_err("no directory path given")
        assert isinstance(inpfile, str), "unreachable: type checker"
        if not os.path.exists(inpfile):
            fatal_err("directory does not exist")
        if not os.path.isdir(inpfile):
            fatal_err("given path is not a directory")
        all_files = list(
            filter(
                lambda x: x.endswith(".sf"),
                (os.path.join(inpfile, ii) for ii in os.listdir(inpfile)),
            )
        )
        if len(all_files) == 0:
            fatal_err("no files to test")

        failed_tests: list[str] = []
        for ii in all_files:
            status, msg = test_file(ii)
            if not status:
                failed_tests.append(ii)
        print("=== Test result ===")
        print(f"ran {len(all_files)} tests, of which {len(failed_tests)} failed:")
        print(", ".join(failed_tests))
    else:
        if inpfile is None:
            fatal_err("no source file given")
        assert isinstance(inpfile, str), "unreachable: type checker"

        if outfile is None:
            outfile = "./" + inpfile.removesuffix(".sf")

        try:
            compile_file(inpfile, outfile, flags["artifacts"])
        except Exception as e:
            # fatal_err(str(e))
            raise e

        if flags["run"]:
            exit(echo_cmd([outfile, *runargs]))


if __name__ == "__main__":
    main()
