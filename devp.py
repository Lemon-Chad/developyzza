import string
import os
import math
import random
from _thread import start_new_thread

# CONSTANTS

NUMBERS = '0123456789'
NUMDOT = '0123456789.'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + NUMBERS
KEYWORDS = (
    'async',
    'import',
    'bin',
    'function',
    'attr',
    '&',
    '|',
    'bake',
    'var',
    'for',
    'while',
    'null',
    'if',
    'elif',
    'else',
    'return',
    'continue',
    'break',
    'method',
    'ingredients',
    'recipe'
)
BREAK = ';'
IGNORE = ' \t\n'
LIBRARIES = {}


# FUNCTIONS


def string_with_arrows(text, pos_start, pos_end):
    result = ''

    # Calculate indices
    idx_start = max(text.rfind('\n', 0, pos_start.idx), 0)
    idx_end = text.find('\n', idx_start + 1)
    if idx_end < 0: idx_end = len(text)

    # Generate each line
    line_count = pos_end.ln - pos_start.ln + 1
    for i in range(line_count):
        # Calculate line columns
        line = text[idx_start:idx_end]
        col_start = pos_start.col if i == 0 else 0
        col_end = pos_end.col if i == line_count - 1 else len(line) - 1

        # Append to result
        result += line + '\n'
        result += ' ' * col_start + '^' * (col_end - col_start)

        # Re-calculate indices
        idx_start = idx_end
        idx_end = text.find('\n', idx_start + 1)
        if idx_end < 0: idx_end = len(text)

    return result.replace('\t', '')


# POSITION


class Position:
    def __init__(self, idx: int, ln: int, col: int, fn: str, ftext: str) -> None:
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftext = ftext

    def advance(self, current_char: str = None):
        self.idx += 1
        self.col += 1

        if current_char == BREAK:
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftext)


# ERRORS


class Error:
    def __init__(self, pos_start: Position, pos_end: Position, error_name, details) -> None:
        self.error_name = error_name
        self.details = details
        self.pos_start = pos_start
        self.pos_end = pos_end

    def as_string(self) -> str:
        return f'{self.error_name}: {self.details}\nFile {self.pos_start.fn}, line {self.pos_start.ln + 1}\n\n' \
               f'{string_with_arrows(self.pos_start.ftext, self.pos_start, self.pos_end)}'


class IllegalCharError(Error):
    def __init__(self, start_pos: Position, end_pos: Position, details: str):
        super().__init__(start_pos, end_pos, 'Illegal Character', details)


class ExpectedCharError(Error):
    def __init__(self, start_pos: Position, end_pos: Position, details: str):
        super().__init__(start_pos, end_pos, 'Expected Character', details)


class RTError(Error):
    def __init__(self, start_pos: Position, end_pos: Position, details: str, context):
        super().__init__(start_pos, end_pos, 'Runtime Error', details)
        self.context = context

    def as_string(self) -> str:
        return f'{self.generate_traceback()}' \
               f'{self.error_name}: {self.details}\nFile {self.pos_start.fn}, line {self.pos_start.ln + 1}\n\n' \
               f'{string_with_arrows(self.pos_start.ftext, self.pos_start, self.pos_end)}'

    def generate_traceback(self) -> str:
        result = ''
        pos = self.pos_start
        ctx = self.context

        while ctx:
            result = f'  File {pos.fn}, line {pos.ln + 1}, in {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent

        return f'Traceback (most recent call last):\n{result}'


class InvalidSyntax(Error):
    def __init__(self, start_pos: Position, end_pos: Position, details: str):
        super().__init__(start_pos, end_pos, 'Invalid Syntax', details)


# TOKENS

TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_STRING = 'STRING'
TT_BOOL = 'BOOL'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_EOF = 'EoF'
TT_NEWLINE = 'NEWLINE'
TT_POWER = 'POWER'
TT_IDENTIFIER = 'IDENTIFIER'
TT_KEYWORD = 'KEYWORD'
TT_EQ = 'EQ'
TT_EE = 'EE'
TT_NE = 'NE'
TT_LT = 'LT'
TT_GT = 'GT'
TT_LTE = 'LTE'
TT_GTE = 'GTE'
TT_AND = 'AND'
TT_OR = 'OR'
TT_NOT = 'NOT'
TT_CLACCESS = '::'
TT_MOD = 'MOD'
TT_QUERY = 'QUERY'
TT_BITE = 'BITE'
TT_DEFAULTQUE = 'DEFAULTQUE'
TT_QUEBACK = 'QUEBACK'
TT_LAMBDA = 'LAMBDA'
TT_STEP = 'STEP'
TT_COMMA = 'COMMA'
TT_OPENSIGN = "OPENSIGN"
TT_CLOSESIGN = "CLOSESIGN"
TT_LSQUARE = 'LSQUARE'
TT_RSQUARE = 'RSQUARE'
TT_OPEN = 'TT_OPEN'
TT_CLOSE = 'TT_CLOSE'
TT_PLE = 'TT_PLUSEQUALS'
TT_MIE = 'TT_MINUSEQUALS'
TT_MUE = 'TT_MULTIPLYEQUALS'
TT_DIE = 'TT_DIVIDEEQUALS'
TT_POE = 'TT_POWEREQUALS'
TT_INCR = 'TT_INCREMENT'
TT_DECR = 'TT_DECREMENT'
TT_DICT = 'TT_DICTIONARY'
TT_DOT = 'TT_DOT'

TOKEY = {
    '[': TT_LSQUARE,
    '::': TT_CLACCESS,
    '%': TT_MOD,
    ']': TT_RSQUARE,
    ',': TT_COMMA,
    '+': TT_PLUS,
    '++': TT_INCR,
    '--': TT_DECR,
    '>>': TT_STEP,
    ':': TT_BITE,
    '$': TT_QUEBACK,
    '$_': TT_DEFAULTQUE,
    '?': TT_QUERY,
    '-': TT_MINUS,
    '*': TT_MUL,
    '/': TT_DIV,
    '(': TT_LPAREN,
    ')': TT_RPAREN,
    '^': TT_POWER,
    '=>': TT_EQ,
    '&': TT_AND,
    '|': TT_OR,
    '->': TT_LAMBDA,
    ';': TT_NEWLINE,
    '{': TT_OPEN,
    '}': TT_CLOSE,
    '^=': TT_POE,
    '*=': TT_MUE,
    '/=': TT_DIE,
    '+=': TT_PLE,
    '-=': TT_MIE,
    '.': TT_DOT
}


class Token:
    def __init__(self, type_: str, value=None, pos_start: Position = None, pos_end: Position = None) -> None:
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_end.copy() if pos_end else pos_start.copy().advance()

    def __repr__(self) -> str:
        return f'{self.type}:{self.value}' if self.value else f'{self.type}'

    def matches(self, type_, value):
        return self.type == type_ and self.value == value


# NODES


class ImportNode:
    def __init__(self, file_name_tok):
        self.file_name_tok = file_name_tok
        self.pos_start = file_name_tok.pos_start
        self.pos_end = file_name_tok.pos_end


class FuncDefNode:
    def __init__(self, var_name_tok, arg_name_toks, body_node, autoreturn, asynchronous):
        self.var_name_tok = var_name_tok
        self.asynchronous = asynchronous
        self.arg_name_toks = arg_name_toks
        self.body_node = body_node
        self.autoreturn = autoreturn

        self.pos_start = self.var_name_tok.pos_start if self.var_name_tok else \
            self.arg_name_toks[0].pos_start if self.arg_name_toks else self.body_node.pos_start

        self.pos_end = self.body_node.pos_end


class MethDefNode:
    def __init__(self, var_name_tok, arg_name_toks, body_node, autoreturn, bin_, asynchronous):
        self.var_name_tok = var_name_tok
        self.asynchronous = asynchronous
        self.arg_name_toks = arg_name_toks
        self.body_node = body_node
        self.autoreturn = autoreturn
        self.bin = bin_

        self.pos_start = self.var_name_tok.pos_start

        self.pos_end = self.body_node.pos_end


class ClassDefNode:
    def __init__(self, class_name_tok, attribute_name_toks, arg_name_toks, make_node, methods, pos_end):
        self.class_name_tok = class_name_tok
        self.make_node = make_node
        self.attribute_name_toks = attribute_name_toks
        self.arg_name_toks = arg_name_toks
        self.methods = methods
        self.pos_start = self.class_name_tok.pos_start
        self.pos_end = pos_end


class ClassNode:
    def __init__(self, context, methods):
        self.context = context
        self.methods = methods


class CallNode:
    def __init__(self, node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes

        self.pos_start = self.node_to_call.pos_start
        self.pos_end = self.arg_nodes[-1].pos_end if self.arg_nodes else self.node_to_call.pos_end


class ReturnNode:
    def __init__(self, node_to_return, pos_start, pos_end):
        self.node_to_return = node_to_return

        self.pos_start = pos_start
        self.pos_end = pos_end


class BreakNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end


class ContinueNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end


class ListNode:
    def __init__(self, element_nodes, pos_start, pos_end):
        self.elements = element_nodes
        self.pos_start = pos_start
        self.pos_end = pos_end


class DictNode:
    def __init__(self, dict_, pos_start, pos_end):
        self.dict = dict_

        self.pos_start = pos_start
        self.pos_end = pos_end

    def get(self, key):
        return self.dict[key]

    def delete(self, key):
        del self.dict[key]
        return self

    def set(self, key, value):
        self.dict[key] = value
        return self


class ValueNode:
    def __init__(self, tok: Token) -> None:
        self.tok = tok

        self.pos_start = tok.pos_start
        self.pos_end = tok.pos_end

    def __repr__(self) -> str:
        return f'{self.tok}'


class NullNode(ValueNode):
    pass


class NumberNode(ValueNode):
    pass


class StringNode(ValueNode):
    pass


class BooleanNode(ValueNode):
    def __repr__(self) -> str:
        return 'true' if self.tok.value else 'false'


class QueryNode:
    def __init__(self, cases, else_case):
        self.else_case = else_case
        self.cases = cases

        self.pos_start = cases[0][0].pos_start
        self.pos_end = (else_case if else_case else cases[-1])[0].pos_end


class ForNode:
    def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node, retnull):
        self.var_name_tok = var_name_tok
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.step_value_node = step_value_node
        self.body_node = body_node
        self.retnull = retnull

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.body_node.pos_end


class WhileNode:
    def __init__(self, condition_node, body_node, retnull):
        self.condition_node = condition_node
        self.body_node = body_node
        self.retnull = retnull

        self.pos_start = self.condition_node.pos_start
        self.pos_end = self.body_node.pos_end


class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end


class VarAssignNode:
    def __init__(self, var_name_tok, value_node, locked=False):
        self.var_name_tok = var_name_tok
        self.value_node = value_node
        self.locked = locked

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end


class VarNode:
    def __init__(self, value_node, locked=False):
        self.locked = locked
        self.value_node = value_node


class AttrAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end


class AttrDeclareNode:
    def __init__(self, attr_name_tok):
        self.attr_name_tok = attr_name_tok

        self.pos_start = self.attr_name_tok.pos_start
        self.pos_end = self.attr_name_tok.pos_end


class AttrNode:
    def __init__(self, value_node):
        self.value_node = value_node


class AttrAssignNode:
    def __init__(self, var_name_tok, value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end


class ClaccessNode:
    def __init__(self, cls, atr):
        self.class_tok = cls
        self.attr_name_tok = atr

        self.pos_start = cls.pos_start
        self.pos_end = atr.pos_end


class BinOpNode:
    def __init__(self, left_node, op_tok: Token, right_node) -> None:
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = left_node.pos_start
        self.pos_end = right_node.pos_end

    def __repr__(self) -> str:
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'


class UnaryOpNode:
    def __init__(self, op_tok: Token, node) -> None:
        self.op_tok = op_tok
        self.node = node

        self.pos_start = op_tok.pos_start
        self.pos_end = node.pos_end

    def __repr__(self) -> str:
        return f'({self.op_tok}, {self.node})'


# LEXER


class Lexer:
    def __init__(self, fn: str, text: str) -> None:
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self) -> None:
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def next(self, i: int = 1) -> str:
        return self.text[self.pos.idx + i] if self.pos.idx + i < len(self.text) else None

    def skip_comment(self):
        self.advance()

        while self.current_char and self.current_char not in ';\n':
            self.advance()

        self.advance()

    def make_tokens(self) -> tuple:
        tokens = []
        while self.current_char:
            if self.current_char in IGNORE:
                self.advance()
            elif self.next() and self.current_char + self.next() == "<>":
                self.skip_comment()
            elif self.current_char in ('"', "'"):
                tokens.append(self.make_string(self.current_char))
            elif self.current_char == '}':
                tokens.append(Token(TT_CLOSE, pos_start=self.pos))
                self.advance()
            elif self.next() and self.current_char + self.next() in TOKEY:
                tokens.append(Token(TOKEY[self.current_char + self.next()], pos_start=self.pos,
                                    pos_end=self.pos.copy().advance().advance()))
                self.advance(); self.advance()
            elif self.current_char in TOKEY:
                tokens.append(Token(TOKEY[self.current_char], pos_start=self.pos))
                self.advance()
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char in NUMBERS:
                tokens.append(self.make_number())
            elif self.current_char in ('!', '<', '>', '='):
                tok, error = self.make_equals_expr()
                if error: return [], error
                tokens.append(tok)
            else:
                char, pos_start = self.current_char, self.pos.copy()
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, f"'{char}'")
        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_string(self, q='"'):
        string_ = ''
        pos_start = self.pos.copy()
        escape_character = False
        self.advance()

        escape_characters = {
            'n': '\n',
            't': '\t'
        }

        while self.current_char and (self.current_char != q or escape_character):
            if escape_character:
                string_ += escape_characters.get(self.current_char, self.current_char)
                escape_character = False
            elif self.current_char == '\\':
                escape_character = True
            else:
                string_ += self.current_char
            self.advance()

        self.advance()
        return Token(TT_STRING, string_, pos_start, self.pos)

    def make_equals_expr(self):
        pos_start = self.pos.copy()
        char = self.current_char
        self.advance()

        if self.current_char == '=':
            self.advance()
            return Token({
                             '!': TT_NE,
                             '>': TT_GTE,
                             '<': TT_LTE,
                             '=': TT_EE
                         }[char], pos_start=pos_start), None
        elif char == '=':
            self.advance()
            return None, ExpectedCharError(pos_start, self.pos,
                                           f"'=' (after '{char}')")
        else:
            return Token({
                             '>': TT_GT,
                             '<': TT_LT,
                             '!': TT_NOT
                         }[char], pos_start=pos_start), None

    def make_identifier(self) -> Token:
        id_str = str()
        pos_start = self.pos.copy()

        while self.current_char and self.current_char in LETTERS_DIGITS + '_':
            id_str += self.current_char
            self.advance()

        if id_str in ('true', 'false'):
            return Token(TT_BOOL, True if id_str == 'true' else False, pos_start, self.pos)

        tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER

        return Token(tok_type, id_str, pos_start, self.pos)

    def make_number(self) -> Token:
        num = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char and self.current_char in NUMDOT:
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num += '.'
            else:
                num += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num), pos_start, self.pos)
        return Token(TT_FLOAT, float(num), pos_start, self.pos)


# PARSE RESULT


class ParseResult:
    node = None
    error: Error = None
    advance_count: int = 0
    to_reverse_count: int = 0

    def register_advancement(self):
        self.advance_count += 1

    def try_register(self, res):
        if res.error:
            self.to_reverse_count = res.advance_count
            return None
        return self.register(res)

    def register(self, res):
        self.advance_count += res.advance_count
        if res.error: self.error = res.error
        return res.node

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or not self.advance_count:
            self.error = error
        return self


# PARSER


# noinspection DuplicatedCode
class Parser:
    current_tok: Token

    def __init__(self, tokens) -> None:
        self.tokens = tokens
        self.tok_idx = -1
        self.tokount = len(tokens)
        self.advance()

    def advance(self) -> Token:
        self.tok_idx += 1
        self.update_tok()
        return self.current_tok

    def reverse(self, amount=1) -> Token:
        self.tok_idx -= amount
        self.update_tok()
        return self.current_tok

    def update_tok(self) -> None:
        if 0 <= self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]

    def parse(self):
        res = self.statements()
        if not res.error and self.current_tok.type != TT_EOF:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '+', '-', '*', '^', or '/'"
            ))
        return res

    def if_expr(self):
        res = ParseResult()
        all_cases = res.register(self.if_expr_cases('if'))
        if res.error: return res
        cases, else_case = all_cases
        return res.success(QueryNode(cases, else_case))

    def elif_expr(self):
        return self.if_expr_cases('elif')

    def else_expr(self):
        res = ParseResult()
        else_case = None

        if self.current_tok.matches(TT_KEYWORD, 'else'):
            res.register_advancement()
            self.advance()

            if not self.current_tok.type == TT_OPEN:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '{'"
                ))
            res.register_advancement()
            self.advance()

            statements = res.register(self.statements())
            if res.error: return res
            else_case = (statements, True)

            if not self.current_tok.type == TT_CLOSE:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '}'"
                ))

            res.register_advancement()
            self.advance()

        return res.success(else_case)

    def elif_else(self):
        res = ParseResult()
        cases, else_case = [], None

        if self.current_tok.matches(TT_KEYWORD, 'elif'):
            all_cases = res.register(self.elif_expr())
            if res.error: return res
            cases, else_case = all_cases
        else:
            else_case = res.register(self.else_expr())
            if res.error: return res

        return res.success((cases, else_case))

    def if_expr_cases(self, case_keyword):
        res = ParseResult()
        cases = []

        if not self.current_tok.matches(TT_KEYWORD, case_keyword):
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{case_keyword}'"
            ))

        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.type == TT_OPEN:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{'"
            ))

        res.register_advancement()
        self.advance()

        statements = res.register(self.statements())
        if res.error: return res
        cases.append((condition, statements, True))

        if self.current_tok.type != TT_CLOSE:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '}'"
            ))
        res.register_advancement()
        self.advance()
        all_cases = res.register(self.elif_else())
        if res.error: return res
        new_cases, else_case = all_cases
        cases.extend(new_cases)

        return res.success((cases, else_case))

    def for_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(TT_KEYWORD, 'for'):
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'for'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                'Expected identifier'
            ))

        var_name = self.current_tok
        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_LAMBDA:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected weak assignment ('->')"
            ))

        res.register_advancement()
        self.advance()

        start_value = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.type == TT_BITE:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected ':'"
            ))

        res.register_advancement()
        self.advance()

        end_value = res.register(self.expr())

        if self.current_tok.type == TT_STEP:
            res.register_advancement()
            self.advance()

            step_value = res.register(self.expr())
        else:
            step_value = None

        if self.current_tok.type == TT_OPEN:
            res.register_advancement()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res

            if not self.current_tok.type == TT_CLOSE:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '}'"
                ))

            res.register_advancement()
            self.advance()

            return res.success(ForNode(var_name, start_value, end_value, step_value, body, True))

        elif self.current_tok.type == TT_EQ:
            res.register_advancement()
            self.advance()

            body = res.register(self.statement())
            if res.error: return res

            return res.success(ForNode(var_name, start_value, end_value, step_value, body, False))

        else:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{' or '=>'"
            ))

    def while_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(TT_KEYWORD, 'while'):
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'while'"
            ))

        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if self.current_tok.type == TT_EQ:
            res.register_advancement()
            self.advance()

            body = res.register(self.statement())
            if res.error: return res

            return res.success(WhileNode(condition, body, False))
        elif self.current_tok.type == TT_OPEN:
            res.register_advancement()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res

            if not self.current_tok.type == TT_CLOSE:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '}'"
                ))

            res.register_advancement()
            self.advance()

            return res.success(WhileNode(condition, body, True))
        else:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{' or '=>'"
            ))

    def class_def(self):
        res = ParseResult()

        if not self.current_tok.matches(TT_KEYWORD, 'recipe'):
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'recipe'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected identifier"
            ))

        class_name_tok = self.current_tok
        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_OPEN:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{'"
            ))
        self.advance()
        res.register_advancement()

        attribute_declarations = []
        if self.current_tok.type == TT_IDENTIFIER:
            attribute_declarations.append(self.current_tok)
            self.advance()
            res.register_advancement()
            while self.current_tok.type == TT_COMMA:
                res.register_advancement()
                self.advance()
                if self.current_tok.type != TT_IDENTIFIER:
                    return res.failure(InvalidSyntax(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected identifier"
                    ))
                attribute_declarations.append(self.current_tok)
                self.advance()
                res.register_advancement()
            if self.current_tok.type != TT_NEWLINE:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ;"
                ))
            self.advance()
            res.register_advancement()
        arg_name_toks = []
        ingredient_node = ListNode(
            [],
            class_name_tok.pos_start,
            class_name_tok.pos_end
        )
        methods = []
        while self.current_tok.type == TT_KEYWORD and self.current_tok.value in ('method', 'ingredients'):
            if self.current_tok.matches(TT_KEYWORD, "ingredients"):
                self.advance()
                res.register_advancement()
                arg_name_toks = []
                if self.current_tok.type == TT_LT:
                    self.advance()
                    res.register_advancement()
                    if self.current_tok.type == TT_IDENTIFIER:
                        arg_name_toks.append(self.current_tok)
                        res.register_advancement()
                        self.advance()

                        while self.current_tok.type == TT_COMMA:
                            res.register_advancement()
                            self.advance()

                            if self.current_tok.type != TT_IDENTIFIER:
                                return res.failure(InvalidSyntax(
                                    self.current_tok.pos_start, self.current_tok.pos_end,
                                    "Expected identifier"
                                ))

                            arg_name_toks.append(self.current_tok)
                            res.register_advancement()
                            self.advance()

                    if self.current_tok.type != TT_GT:
                        return res.failure(InvalidSyntax(
                            self.current_tok.pos_start, self.current_tok.pos_end,
                            "Expected ',' or '>'"
                        ))
                    self.advance()
                    res.register_advancement()
                if self.current_tok.type != TT_OPEN:
                    return res.failure(InvalidSyntax(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected '{'"
                    ))
                self.advance()
                res.register_advancement()
                ingredient_node = res.register(self.statements())
                if res.error: return res

                if self.current_tok.type != TT_CLOSE:
                    return res.failure(InvalidSyntax(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected '}'"
                    ))

                self.advance()
                res.register_advancement()
            elif self.current_tok.matches(TT_KEYWORD, 'method'):
                res.register_advancement()
                self.advance()

                bin_ = False
                asynchronous = False
                while self.current_tok.value in ('bin', 'async') and self.current_tok.type == TT_KEYWORD:
                    if self.current_tok.matches(TT_KEYWORD, 'bin'):
                        bin_ = True
                        res.register_advancement()
                        self.advance()
                    elif self.current_tok.matches(TT_KEYWORD, 'async'):
                        asynchronous = True
                        res.register_advancement()
                        self.advance()

                if self.current_tok.type != TT_IDENTIFIER:
                    return res.failure(InvalidSyntax(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected identifier"
                    ))
                var_name_tok = self.current_tok
                res.register_advancement()
                self.advance()
                args = []
                if self.current_tok.type == TT_LT:
                    self.advance()
                    res.register_advancement()
                    if self.current_tok.type == TT_IDENTIFIER:
                        args.append(self.current_tok)
                        res.register_advancement()
                        self.advance()

                        while self.current_tok.type == TT_COMMA:
                            res.register_advancement()
                            self.advance()

                            if self.current_tok.type != TT_IDENTIFIER:
                                return res.failure(InvalidSyntax(
                                    self.current_tok.pos_start, self.current_tok.pos_end,
                                    "Expected identifier"
                                ))

                            args.append(self.current_tok)
                            res.register_advancement()
                            self.advance()

                    if self.current_tok.type != TT_GT:
                        return res.failure(InvalidSyntax(
                            self.current_tok.pos_start, self.current_tok.pos_end,
                            "Expected ',' or '>'"
                        ))
                    self.advance()
                    res.register_advancement()

                if self.current_tok.type == TT_LAMBDA:
                    res.register_advancement()
                    self.advance()
                    node_to_return = res.register(self.expr())
                    if res.error: return res

                    methods.append(MethDefNode(
                        var_name_tok,
                        args,
                        node_to_return,
                        True,
                        bin_,
                        asynchronous
                    ))

                elif self.current_tok.type == TT_OPEN:
                    res.register_advancement()
                    self.advance()
                    node_to_return = res.register(self.statements())
                    if res.error: return res

                    if self.current_tok.type != TT_CLOSE:
                        return res.failure(InvalidSyntax(
                            self.current_tok.pos_start, self.current_tok.pos_end,
                            "Expected '}'"
                        ))

                    self.advance()
                    res.register_advancement()

                    methods.append(MethDefNode(
                        var_name_tok,
                        args,
                        node_to_return,
                        False,
                        bin_,
                        asynchronous
                    ))
        if self.current_tok.type != TT_CLOSE:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected }"
            ))
        self.advance()
        res.register_advancement()
        return res.success(ClassDefNode(
            class_name_tok,
            attribute_declarations,
            arg_name_toks,
            ingredient_node,
            methods,
            self.current_tok.pos_end.copy()
        ))

    def func_def(self):
        res = ParseResult()

        if not self.current_tok.matches(TT_KEYWORD, 'function'):
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'function'"
            ))

        res.register_advancement()
        self.advance()

        asynchronous = False
        if self.current_tok.matches(TT_KEYWORD, 'async'):
            asynchronous = True
            self.advance()
            res.register_advancement()

        if self.current_tok.type == TT_IDENTIFIER:
            var_name_tok = self.current_tok
            res.register_advancement()
            self.advance()
        else:
            var_name_tok = None
        arg_name_toks = []
        if self.current_tok.type == TT_LT:
            self.advance()
            res.register_advancement()
            if self.current_tok.type == TT_IDENTIFIER:
                arg_name_toks.append(self.current_tok)
                res.register_advancement()
                self.advance()

                while self.current_tok.type == TT_COMMA:
                    res.register_advancement()
                    self.advance()

                    if self.current_tok.type != TT_IDENTIFIER:
                        return res.failure(InvalidSyntax(
                            self.current_tok.pos_start, self.current_tok.pos_end,
                            "Expected identifier"
                        ))

                    arg_name_toks.append(self.current_tok)
                    res.register_advancement()
                    self.advance()

            if self.current_tok.type != TT_GT:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ',' or '>'"
                ))
            self.advance()
            res.register_advancement()

        if self.current_tok.type == TT_LAMBDA:
            res.register_advancement()
            self.advance()
            node_to_return = res.register(self.expr())
            if res.error: return res

            return res.success(FuncDefNode(
                var_name_tok,
                arg_name_toks,
                node_to_return,
                True,
                asynchronous
            ))

        elif self.current_tok.type == TT_OPEN:
            res.register_advancement()
            self.advance()
            node_to_return = res.register(self.statements())
            if res.error: return res

            if self.current_tok.type != TT_CLOSE:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '}'"
                ))

            self.advance()
            res.register_advancement()

            return res.success(FuncDefNode(
                var_name_tok,
                arg_name_toks,
                node_to_return,
                False,
                asynchronous
            ))

        else:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '->' or '{'"
            ))

    def atom(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.matches(TT_KEYWORD, "recipe"):
            class_def = res.register(self.class_def())
            if res.error: return res
            return res.success(class_def)

        elif tok.matches(TT_KEYWORD, "import"):
            self.advance()
            res.register_advancement()
            file_name_tok = self.current_tok
            if file_name_tok.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntax(
                    file_name_tok.pos_start, file_name_tok.pos_end,
                    "Expected module name"
                ))
            self.advance()
            res.register_advancement()
            return res.success(ImportNode(file_name_tok))

        elif tok.matches(TT_KEYWORD, "attr"):
            self.advance()
            res.register_advancement()
            var_name_tok = self.current_tok
            if self.current_tok.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected identifier"
                ))
            self.advance()
            res.register_advancement()
            return res.success(AttrAccessNode(var_name_tok))

        elif tok.type in (TT_INT, TT_FLOAT):
            res.register_advancement()
            self.advance()
            return res.success(NumberNode(tok))

        elif tok.type == TT_STRING:
            res.register_advancement()
            self.advance()
            return res.success(StringNode(tok))

        elif tok.type == TT_IDENTIFIER:
            res.register_advancement()
            self.advance()
            if self.current_tok.type == TT_CLACCESS:
                res.register_advancement()
                self.advance()
                under = self.current_tok
                if under.type != TT_IDENTIFIER:
                    return res.failure(InvalidSyntax(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected identifier"
                    ))
                self.advance()
                res.register_advancement()
                return res.success(ClaccessNode(
                    VarAccessNode(tok),
                    under
                ))
            return res.success(VarAccessNode(tok))

        elif tok.type == TT_BOOL:
            res.register_advancement()
            self.advance()
            return res.success(BooleanNode(tok))

        elif tok.matches(TT_KEYWORD, 'if'):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)

        elif tok.type == TT_QUERY:
            query_expr = res.register(self.query_expr())
            if res.error: return res
            return res.success(query_expr)

        elif tok.matches(TT_KEYWORD, 'null'):
            res.register_advancement()
            self.advance()
            return res.success(NullNode(tok))

        elif tok.matches(TT_KEYWORD, 'for'):
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)

        elif tok.matches(TT_KEYWORD, 'function'):
            func_def = res.register(self.func_def())
            if res.error: return res
            return res.success(func_def)

        elif tok.matches(TT_KEYWORD, 'while'):
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)

        elif tok.type == TT_LSQUARE:
            list_expr = res.register(self.list_expr())
            if res.error: return res
            return res.success(list_expr)

        elif tok.type == TT_OPEN:
            dict_expr = res.register(self.dict_expr())
            if res.error: return res
            return res.success(dict_expr)

        elif tok.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ')'"
                ))

        return res.failure(InvalidSyntax(
            tok.pos_start, tok.pos_end,
            "Expected int, float, identifier, '+', '-', or '('"
        ))

    def query_expr(self):
        res = ParseResult()
        cases = []
        else_case = None

        def get_statement():
            res.register_advancement()
            self.advance()

            condition = res.register(self.expr())
            if res.error: return res

            if not self.current_tok.type == TT_BITE:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected ':'"
                ))

            res.register_advancement()
            self.advance()

            expr_ = res.register(self.expr())
            if res.error: return res
            cases.append((condition, expr_, False))

        if not self.current_tok.type == TT_QUERY:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '?'"
            ))

        r = get_statement()
        if r: return r

        while self.current_tok.type == TT_QUEBACK:
            r = get_statement()
            if r: return r

        if self.current_tok.type == TT_DEFAULTQUE:
            res.register_advancement()
            self.advance()

            if not self.current_tok.type == TT_BITE:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected ':'"
                ))

            res.register_advancement()
            self.advance()

            expr = res.register(self.statement())
            if res.error: return res
            else_case = (expr, False)

        return res.success(QueryNode(cases, else_case))

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS, TT_INCR, TT_DECR):
            res.register_advancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))

        return self.pow()

    def list_expr(self):
        res = ParseResult()
        element_nodes = []
        pos_start = self.current_tok.pos_start.copy()

        if self.current_tok.type != TT_LSQUARE:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '['"
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type == TT_RSQUARE:
            res.register_advancement()
            self.advance()
        else:
            element_nodes.append(res.register(self.expr()))
            if res.error: return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected item"
            ))

            while self.current_tok.type == TT_COMMA:
                res.register_advancement()
                self.advance()

                element_nodes.append(res.register(self.expr()))
                if res.error: return res

            if self.current_tok.type != TT_RSQUARE: return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected ',' or ']'"
            ))

            res.register_advancement()
            self.advance()

        return res.success(ListNode(
            element_nodes,
            pos_start,
            self.current_tok.pos_end
        ))

    def dict_expr(self):
        res = ParseResult()
        dic = {}
        pos_start = self.current_tok.pos_start.copy()

        if self.current_tok.type != TT_OPEN:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{'"
            ))
        res.register_advancement()
        self.advance()

        def kv():
            key = res.register(self.expr())
            if res.error: return res

            if self.current_tok.type != TT_BITE:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ':'"
                ))
            self.advance()
            res.register_advancement()

            value = res.register(self.expr())
            if res.error: return res
            dic[key] = value

        if self.current_tok.type != TT_CLOSE:
            x = kv()
            if x: return x

        while self.current_tok and self.current_tok.type == TT_COMMA:
            self.advance()
            res.register_advancement()
            x = kv()
            if x: return x

        if self.current_tok.type != TT_CLOSE:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '}'"
            ))
        res.register_advancement()
        self.advance()

        return res.success(DictNode(dic, pos_start, self.current_tok.pos_end.copy()))

    def statements(self):
        res = ParseResult()
        statements = []
        pos_start = self.current_tok.pos_start.copy()

        while self.current_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

        statement = res.register(self.statement())
        if res.error: return res
        statements.append(statement)

        more_statements = True

        while True:
            newline_count = 0
            while self.current_tok.type == TT_NEWLINE:
                res.register_advancement()
                self.advance()
                newline_count += 1
            if not newline_count:
                more_statements = False

            if not more_statements: break
            statement = res.try_register(self.statement())
            if not statement:
                self.reverse(res.to_reverse_count)
                more_statements = False
                continue
            statements.append(statement)

        self.reverse()
        if self.current_tok.type != TT_NEWLINE:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Missing semicolon, found {self.current_tok.value}"
            ))
        self.advance()

        return res.success(ListNode(
            statements,
            pos_start,
            self.current_tok.pos_end.copy()
        ))

    def statement(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()

        if self.current_tok.matches(TT_KEYWORD, 'return'):
            res.register_advancement()
            self.advance()
            expr = res.try_register(self.expr())
            if not expr:
                self.reverse(res.to_reverse_count)
            return res.success(ReturnNode(expr, pos_start, self.current_tok.pos_end.copy()))

        if self.current_tok.matches(TT_KEYWORD, 'continue'):
            res.register_advancement()
            self.advance()
            return res.success(ContinueNode(pos_start, self.current_tok.pos_end.copy()))

        if self.current_tok.matches(TT_KEYWORD, 'break'):
            res.register_advancement()
            self.advance()
            return res.success(BreakNode(pos_start, self.current_tok.pos_end.copy()))

        expr = res.register(self.expr())
        if res.error: return res
        return res.success(expr)

    def call(self):
        res = ParseResult()
        atom = res.register(self.atom())
        if res.error: return res

        if self.current_tok.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            arg_nodes = []
            if self.current_tok.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
            else:
                arg_nodes.append(res.register(self.expr()))
                if res.error: return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected argument"
                ))

                while self.current_tok.type == TT_COMMA:
                    res.register_advancement()
                    self.advance()

                    arg_nodes.append(res.register(self.expr()))
                    if res.error: return res

                if self.current_tok.type != TT_RPAREN:
                    return res.failure(InvalidSyntax(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected ',' or ')'"
                    ))

                res.register_advancement()
                self.advance()
            return res.success(CallNode(
                atom,
                arg_nodes
            ))
        return res.success(atom)

    def pow(self):
        return self.bin_op(self.call, (TT_POWER, TT_MOD), self.factor)

    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV))

    def arith_expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def comp_expr(self):
        res = ParseResult()

        if self.current_tok.type == TT_NOT:
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()

            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(op_tok, node))

        node = res.register(self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))

        if res.error:
            return res.failure(InvalidSyntax(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected int, float, bool, identifier, '+', '-', '!', '^', '?', 'function', 'for', 'while' or '('"
            ))

        return res.success(node)

    def get_expr(self):
        res = ParseResult()
        node = res.register(self.bin_op(self.comp_expr, (TT_AND, TT_OR)))
        if res.error: return res
        return res.success(node)

    def expr(self):
        res = ParseResult()
        if self.current_tok.matches(TT_KEYWORD, 'attr'):
            res.register_advancement()
            self.advance()

            if self.current_tok.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected identifier"
                ))

            var_name = self.current_tok
            res.register_advancement()
            self.advance()

            if self.current_tok.type != TT_EQ:
                return res.success(AttrAccessNode(var_name))

            res.register_advancement()
            self.advance()

            expr = res.register(self.expr())
            if res.error: return res
            return res.success(AttrAssignNode(var_name, expr))

        elif self.current_tok.type == TT_KEYWORD and self.current_tok.value in ('var', 'bake'):
            locked = self.current_tok.value == 'bake'
            res.register_advancement()
            self.advance()

            if self.current_tok.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected identifier"
                ))

            var_name = self.current_tok
            res.register_advancement()
            self.advance()

            if self.current_tok.type != TT_EQ:
                return res.failure(InvalidSyntax(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '=>'"
                ))

            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr, locked))
        if self.current_tok.type == TT_IDENTIFIER:
            var_tok = self.current_tok
            self.advance()
            res.register_advancement()
            if self.current_tok.type in (TT_POE, TT_PLE, TT_MUE, TT_DIE, TT_MIE):
                op_tok = self.current_tok
                self.advance()
                res.register_advancement()
                value = res.register(self.expr())
                if res.error: return res
                return res.success(VarAssignNode(var_tok, BinOpNode(
                    VarAccessNode(var_tok),
                    Token({
                        TT_POE: TT_POWER,
                        TT_MUE: TT_MUL,
                        TT_DIE: TT_DIV,
                        TT_PLE: TT_PLUS,
                        TT_MIE: TT_MINUS
                    }[op_tok.type], op_tok.pos_start, op_tok.pos_end),
                    value
                ), False))
            elif self.current_tok.type in (TT_INCR, TT_DECR):
                op_tok = self.current_tok
                res.register_advancement()
                self.advance()
                return res.success(VarAssignNode(var_tok, UnaryOpNode(
                    op_tok,
                    VarAccessNode(var_tok)
                ), False))
            self.reverse()
        node = res.register(self.bin_op(self.get_expr, (TT_DOT,)))

        if res.error: return res.failure(InvalidSyntax(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected keyword, int, bool, float, identifier, '+', '-', '*', '/', '^', '!', '?', 'for', 'while' or '('"
        ))

        return res.success(node)

    def bin_op(self, left: callable, ops, right_func: callable = None):
        res = ParseResult()
        if right_func is None:
            right_func = left
        left = res.register(left())

        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()
            right = res.register(right_func())
            if res.error: return res
            left = BinOpNode(left, op_tok, right)
        return res.success(left)


# RUNTIME RESULT


class RTResult:
    value = None
    error = None
    func_return = None
    continue_loop = None
    break_loop = None

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = None
        self.error = None
        self.func_return = None
        self.continue_loop = False
        self.break_loop = False

    def register(self, res):
        if res.error: self.error = res.error
        self.func_return = res.func_return
        self.continue_loop = res.continue_loop
        self.break_loop = res.break_loop
        return res.value

    def success(self, value):
        self.reset()
        self.value = value
        return self

    def sreturn(self, value):
        self.reset()
        self.func_return = value
        return self

    def scontinue(self):
        self.reset()
        self.continue_loop = True
        return self

    def sbreak(self):
        self.reset()
        self.break_loop = True
        return self

    def failure(self, error):
        self.reset()
        self.error = error
        return self

    def should_return(self):
        return (
                self.error or
                self.func_return or
                self.break_loop or
                self.continue_loop
        )


# VALUES


class Value:
    value = None
    pos_start: Position
    pos_end: Position
    context = None

    def delete(self, other):
        return self.dictionary().delete(other)

    def mod(self, other):
        return self.number().mod(other)

    def get(self, other):
        return self.dictionary().get(other), None

    def dictionary(self):
        return self

    def __init__(self, value=None) -> None:
        self.value = value
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start: Position = None, pos_end: Position = None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def function(self): return self

    def execute(self, args):
        return self.function().execute(args)

    def set_context(self, context=None):
        self.context = context
        return self

    def copy(self):
        return type(self)(self.value).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def boolean(self):
        return self

    def number(self):
        return self

    def also(self, other):
        return self.boolean().also(other)

    def including(self, other):
        return self.boolean().including(other), None

    def invert(self):
        return self.boolean().invert(), None

    def add(self, other):
        return self.number().add(other)

    def sub(self, other):
        return self.number().sub(other)

    def mul(self, other):
        return self.number().mul(other)

    def div(self, other):
        return self.number().div(other)

    def fastpow(self, other):
        return self.number().fastpow(other)

    def eq(self, other):
        return Boolean(self.value == other.value), None

    def ne(self, other):
        return Boolean(self.value != other.value), None

    def lt(self, other):
        return self.number().lt(other)

    def lte(self, other):
        return self.number().lte(other)

    def string(self):
        return self

    def list(self):
        return self

    def append(self, other):
        return self.list().append(other)

    def extend(self, other):
        return self.list().extend(other)

    def pop(self, other):
        return self.list().pop(other)

    def remove(self, other):
        return self.list().remove(other)

    def __repr__(self) -> str:
        return self.string().value


class Boolean(Value):
    def invert(self):
        return Boolean(not self.value), None

    def type(self):
        return String("bool").set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def dictionary(self):
        return Dict({self.value: self.value}).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def list(self):
        return List([self.value]).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def string(self):
        return String('true' if self.value else 'false').set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def function(self):
        return Function(None, BooleanNode(Token(TT_BOOL, self.value, self.pos_start, self.pos_end)), None) \
            .set_context(self.context).set_pos(self.pos_end, self.pos_start)

    def also(self, other):
        other = other.boolean()
        return Boolean(self.value or other.value), None

    def including(self, other):
        other = other.boolean()
        return Boolean(self.value and other.value), None

    def __repr__(self) -> str:
        return 'true' if self.value else 'false'

    def boolean(self):
        return self

    def number(self):
        return Number(1 if self.value else 0).set_context(self.context)


class List(Value):
    def type(self):
        return String("list").set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def append(self, other):
        return List(self.value + [other.value]).set_context(self.context), None

    def extend(self, other):
        other = other.list()
        return List(self.value + other.value).set_context(self.context), None

    def add(self, other):
        return self.extend(other)

    def mod(self, other):
        return self.append(other)

    def contains(self, other):
        return Boolean(len([o for o in self.value if o.eq(other)[0].value]) > 0)

    def dictionary(self):
        return Dict({x: x for x in self.value}).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def len(self):
        return self.number()

    def dot(self, other):
        other = other.number()
        if other.value + 1 > len(self.value):
            return None, RTError(
                self.pos_start, self.pos_end,
                "List index out of range",
                self.context
            )
        if type(other.value) is float:
            return None, RTError(
                self.pos_start, self.pos_end,
                "List index must be int, not float",
                self.context
            )
        return self.value[other.value], None

    def div(self, other):
        return self.remove(other)

    def sub(self, other):
        return self.dot(other)

    def bite(self, other):
        return self.pop(other)

    def mul(self, other):
        other = other.number()
        return List(self.value * other.value).set_context(self.context), None

    def pop(self, other):
        other = other.number()
        if other.value + 1 > len(self.value):
            return None, RTError(
                self.pos_start, other.pos_end,
                'List index out of range',
                self.context
            )
        return List(self.value[:other.value] + self.value[(other.value + 1):]).set_context(self.context), None

    def remove(self, other):
        return List([value for value in self.value if value.value != other.value]).set_context(self.context), None

    def copy(self):
        return List(self.value).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def number(self):
        return Number(len(self.value)).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def boolean(self):
        return Boolean(len(self.value) > 0).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def function(self):
        return Function(None, ListNode(self.value, self.pos_start, self.pos_end), None).set_context(self.context) \
            .set_pos(self.pos_start, self.pos_end)

    def string(self):
        return String(str(self.value)).set_context(self.context).set_pos(self.pos_start, self.pos_end)


class Dict(Value):
    def type(self):
        return String("dict").set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def string(self):
        return String(str(self.value)).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def number(self):
        return Number(len(self.value)).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def boolean(self):
        return Boolean(len(self.value) > 0).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def list(self):
        return List(list(self.value)).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def function(self):
        return Function(
            None,
            DictNode(self.value, self.pos_start, self.pos_end),
            None
        ).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def add(self, other):
        other = other.dictionary()
        return Dict({**self.value, **other.value}).set_context(self.context), None

    def get(self, other):
        lx = [val for key, val in self.value.items() if key.eq(other)[0].value]
        return lx[0] if lx else Null(), None

    def delete(self, other):
        if self.contains(other):
            key = [ke for ke, va in self.value.items() if ke.eq(other)[0].value][0]
            del self.value[key]
        return Null()

    def contains(self, other):
        return len([value for key, value in self.value.items() if key.eq(other)[0].value]) > 0

    def set(self, other_a, other_b):
        self.value[other_a.set_pos(None, None).set_context(None)] = other_b.set_pos(None, None).set_context(None)
        return Null()


class BaseFunction(Value):
    def __init__(self, name):
        super().__init__()
        self.name = name or "<anonymous>"

    def new_context(self):
        new_context = Context(self.name, self.context, self.pos_start)
        new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
        return new_context

    def check_args(self, arg_names, args):
        res = RTResult()

        if len(args) > len(arg_names):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                f"{len(args) - len(arg_names)} too many args passed into '{self.name}'",
                self.context
            ))
        if len(args) < len(arg_names):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                f"{len(arg_names) - len(args)} too few args passed into '{self.name}'",
                self.context
            ))

        return res.success(None)

    # noinspection PyMethodMayBeStatic
    def populate_args(self, arg_names, args, exec_ctx):
        for i in range(len(args)):
            arg_name = arg_names[i]
            arg_value = args[i]
            arg_value.set_context(exec_ctx)
            exec_ctx.symbol_table.set(arg_name, arg_value)

    def check_pop_args(self, arg_names, args, exec_ctx):
        res = RTResult()
        res.register(self.check_args(arg_names, args))
        if res.should_return(): return res
        self.populate_args(arg_names, args, exec_ctx)
        if res.should_return(): return res
        return res.success(None)


class Function(BaseFunction):
    def __init__(self, name, body_node, arg_names, asynchronous=False, autoreturn=True):
        super().__init__(name)
        self.body_node = body_node
        self.autoreturn = autoreturn
        self.asynchronous = asynchronous
        self.arg_names = arg_names or []

    def type(self):
        return String("function").set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def list(self):
        return List(self.arg_names).set_pos(self.pos_start, self.pos_end).set_context(self.context) \
            .set_pos(self.pos_start, self.pos_end)

    def string(self):
        return String(self.name).set_pos(self.pos_start, self.pos_end).set_context(self.context) \
            .set_pos(self.pos_start, self.pos_end)

    def execute(self, args):
        res = RTResult()
        interpreter = Interpreter()
        exec_ctx = self.new_context()

        res.register(self.check_pop_args(self.arg_names, args, exec_ctx))
        if res.should_return():
            if self.asynchronous and res.error:
                print(f"Async function {self.name}:\n{res.error.as_string()}")
            return res

        value = res.register(interpreter.visit(self.body_node, exec_ctx))
        if res.should_return() and res.func_return is None:
            if self.asynchronous and res.error:
                print(f"Async function {self.name}:\n{res.error.as_string()}")
            return res

        ret_value = (value if self.autoreturn else None) or res.func_return or Null()
        return res.success(ret_value)

    def dictionary(self):
        return Dict({x: x for x in self.arg_names}).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def copy(self):
        return Function(self.name, self.body_node, self.arg_names, self.asynchronous, self.autoreturn)\
            .set_context(self.context)\
            .set_pos(self.pos_start, self.pos_end)

    def number(self):
        return Number(len(self.arg_names)).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def boolean(self):
        return Boolean(True).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def __repr__(self):
        return f"<function {self.name}>"


class CMethod(Function):
    def __init__(self, name, name_tok, context, body_node, arg_names, bin_, asynchronous=False, autoreturn=True):
        super().__init__(name, body_node, arg_names, asynchronous, autoreturn)
        self.bin = bin_
        self.name_tok = name_tok
        self.context = Context(self.name, context, self.pos_start)
        self.context.symbol_table = SymbolTable(context.symbol_table)

    def type(self):
        return String("method").set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def execute(self, args):
        x = super().execute(args)
        return x

    def set_context(self, context=None):
        self.context = context
        return self

    def copy(self):
        return CMethod(self.name, self.name_tok, self.context, self.body_node, self.arg_names, self.bin,
                       self.asynchronous, self.autoreturn).set_pos(self.pos_start, self.pos_end)\
            .set_context(self.context)

    def __repr__(self):
        return f"<classfunction-{self.name}>"


class Class(Value):
    pos_start: Position
    pos_end: Position
    context = None

    # noinspection PyMissingConstructor
    def __init__(self, name, attributes, make, methods):
        self.name = name
        self.make: CMethod = make
        self.methods: list[CMethod] = methods
        self.attributes = attributes
        self.asynchronous = False
        self.set_pos()
        self.set_context()

    def number(self):
        return Number(0).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def string(self):
        return String(self.name).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def dict(self):
        return Dict({}).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def list(self):
        return List([]).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def boolean(self):
        return Boolean(True).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def type(self):
        return self.string()

    def execute(self, args):
        res = RTResult()
        class_context = Context(self.name, self.context, self.pos_start)
        class_context.symbol_table = SymbolTable(self.context.symbol_table)
        for attribute in self.attributes:
            class_context.symbol_table.declareattr(attribute, class_context)
        make = self.make.copy()
        make.set_context(class_context)
        res.register(make.execute(args))
        if res.error: return res
        for m in self.methods:
            method = m.copy()
            if method.bin:
                class_context.symbol_table.setbin(method.name, method)
            else:
                class_context.symbol_table.declareattr(method.name_tok, class_context)
                class_context.symbol_table.setattr(method.name_tok.value, method)
            method.set_context(class_context)
        return res.success(ClassInstance(class_context))

    def copy(self):
        return Class(
            self.name,
            self.attributes.copy(),
            self.make.copy(),
            self.methods.copy()
        ).set_context(self.context).set_pos(self.pos_start, self.pos_end)


class ClassInstance:
    pos_start: Position
    pos_end: Position
    context = None

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def __init__(self, value):
        self.value: Context = value
        self.value.symbol_table.set("this", self.value.display_name)
        self.set_pos()
        self.set_context()

    def __repr__(self):
        return self.string().value

    def set_context(self, context=None):
        self.context = context
        return self

    def access(self, other):
        c = self.value.symbol_table.get(other)
        x = self.value.symbol_table.getattr(other)
        return x.set_context(self.value) if x else c.set_context(self.value) if c else Null()

    def copy(self):
        return ClassInstance(
            self.value
        ).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    # SUPERS

    def __getattr__(self, name):
        def method(*args):
            func = self.value.symbol_table.getbin(name)
            if func is None:
                return getattr(Value, name)(self, *args)
            return func.execute(args).value, func.execute(args).error
        return method

    def dictionary(self):
        res = RTResult()
        func = self.value.symbol_table.getbin("dictionary")
        if func is None:
            return Dict({}).set_context(self.context).set_pos(self.pos_start, self.pos_end)
        x = res.register(func.execute([]))
        if res.error: return Dict({}).set_context(self.context).set_pos(self.pos_start, self.pos_end)
        return x

    def function(self):
        res = RTResult()
        func = self.value.symbol_table.getbin("function")
        if func is None:
            return Null().set_context(self.context).set_pos(self.pos_start, self.pos_end)
        x = res.register(func.execute([]))
        if res.error: return Null().set_context(self.context).set_pos(self.pos_start, self.pos_end)
        return x

    def boolean(self):
        res = RTResult()
        func = self.value.symbol_table.getbin("boolean")
        if func is None:
            return Boolean(True).set_context(self.context).set_pos(self.pos_start, self.pos_end)
        x = res.register(func.execute([]))
        if res.error: return Boolean(True).set_context(self.context).set_pos(self.pos_start, self.pos_end)
        return x

    def number(self):
        res = RTResult()
        func = self.value.symbol_table.getbin("number")
        if func is None:
            return Number(0).set_context(self.context).set_pos(self.pos_start, self.pos_end)
        x = res.register(func.execute([]))
        if res.error: return Number(0).set_context(self.context).set_pos(self.pos_start, self.pos_end)
        return x

    def string(self):
        res = RTResult()
        func = self.value.symbol_table.getbin("string")
        if func is None:
            return String(self.value.display_name).set_context(self.context).set_pos(self.pos_start, self.pos_end)
        x = res.register(func.execute([]))
        if res.error:
            return String(self.value.display_name).set_context(self.context).set_pos(self.pos_start, self.pos_end)
        return x

    def type(self):
        res = RTResult()
        func = self.value.symbol_table.getbin("type")
        if func is None:
            return String(self.value.display_name).set_context(self.context).set_pos(self.pos_start, self.pos_end)
        x = res.register(func.execute([]))
        if res.error: return String(self.value.display_name).set_context(self.context)\
            .set_pos(self.pos_start, self.pos_end)
        return x

    def list(self):
        res = RTResult()
        func = self.value.symbol_table.getbin("list")
        if func is None:
            return List([]).set_context(self.context).set_pos(self.pos_start, self.pos_end)
        x = res.register(func.execute([]))
        if res.error:
            return List([]).set_context(self.context).set_pos(self.pos_start, self.pos_end)
        return x


class Null(Value):
    def __init__(self):
        super().__init__()

    def type(self):
        return String("null").set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def dictionary(self):
        return Dict({}).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def copy(self):
        return Null().set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def number(self):
        return Number(0).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def list(self):
        return List([]).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def boolean(self):
        return Boolean(False).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def function(self):
        return Function(None, NullNode(Token(
            TT_KEYWORD, "null", self.pos_start, self.pos_end
        )), None).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def string(self):
        return String('').set_context(self.context).set_pos(self.pos_start, self.pos_end)


class Library(BaseFunction):
    def __init__(self, name):
        super().__init__(name)
        self.asynchronous = False

    def type(self):
        return String("builtin").set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def execute(self, args):
        res = RTResult()
        exec_ctx = self.new_context()

        method_name = f'execute_{self.name}'
        method = getattr(self, method_name, self.no_visit_method)

        res.register(self.check_pop_args(method.arg_names, args, exec_ctx))
        if res.should_return(): return res

        return_value = res.register(method(exec_ctx))
        if res.should_return(): return res
        return res.success(return_value)

    def copy(self):
        return type(self)(self.name).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def __repr__(self):
        return f"<built-in function {self.name}>"

    def list(self):
        method_name = f'execute_{self.name}'
        method = getattr(self, method_name, self.no_visit_method)
        return List(
            list(method.arg_names),
        ).set_context(self).set_pos(self.pos_start, self.pos_end)

    def number(self):
        method_name = f'execute_{self.name}'
        method = getattr(self, method_name, self.no_visit_method)
        return Number(
            len(method.arg_names),
        ).set_context(self).set_pos(self.pos_start, self.pos_end)

    def boolean(self):
        return Boolean(True).set_context(self).set_pos(self.pos_start, self.pos_end)

    def string(self):
        return String(self.name).set_context(self).set_pos(self.pos_start, self.pos_end)

    def dictionary(self):
        return Dict({self.name: self.name}).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def no_visit_method(self, node, context):
        raise Exception(f'No execute_{self.name} method defined')


class BuiltInFunction(Library):
    def execute_type(self, exec_ctx):
        return RTResult().success(exec_ctx.symbol_table.get("value").type())
    execute_type.arg_names = ["value"]

    def execute_insert(self, exec_ctx):
        index = exec_ctx.symbol_table.get("index").number()
        list_ = exec_ctx.symbol_table.get("list").list()
        item = exec_ctx.symbol_table.get("item")
        if index.value >= len(list_.value):
            return RTResult().failure(RTError(
                index.pos_start, index.pos_end,
                "Index out of bounds!",
                exec_ctx
            ))
        list_.value.insert(index.value + 1, item)
        return RTResult().success(Null())
    execute_insert.arg_names = ['list', 'item', 'index']

    def execute_floating(self, exec_ctx):
        return RTResult().success(Boolean(type(
            exec_ctx.symbol_table.get("num").number().value
        ) is float))
    execute_floating.arg_names = ['num']

    def execute_random(self, _exec_ctx):
        return RTResult().success(Number(random.random()))
    execute_random.arg_names = []

    def execute_round(self, exec_ctx):
        return RTResult().success(Number(round(exec_ctx.symbol_table.get('num').number().value)))
    execute_round.arg_names = ['num']

    def execute_foreach(self, exec_ctx):
        res = RTResult()
        list_ = exec_ctx.symbol_table.get('list').list()
        func = exec_ctx.symbol_table.get('func').function()
        new = []
        for item in list_.value:
            item = res.register(func.execute([item]))
            if res.error: return res
            new.append(item.value)
        return res.success(
            List(new)
        )
    execute_foreach.arg_names = ['list', 'func']

    def execute_floor(self, exec_ctx):
        return RTResult().success(Number(math.floor(exec_ctx.symbol_table.get('num').number().value)))
    execute_floor.arg_names = ['num']

    def execute_ceil(self, exec_ctx):
        return RTResult().success(Number(math.ceil(exec_ctx.symbol_table.get('num').number().value)))
    execute_ceil.arg_names = ['num']

    def execute_abs(self, exec_ctx):
        return RTResult().success(Number(abs(exec_ctx.symbol_table.get('num').number().value)))
    execute_abs.arg_names = ['num']

    def execute_run(self, exec_ctx):
        fn = exec_ctx.symbol_table.get('fn')

        if type(fn) is not String:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be a string",
                exec_ctx
            ))

        fn = fn.value

        try:
            with open(fn, "r") as fil:
                script = fil.read()
        except Exception as e:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"Failed to load script \"{fn}\"\n" + str(e),
                exec_ctx
            ))

        _, error = run(fn, script)

        if error:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"Failed to finish executing script \"{fn}\"\n" + error.as_string(),
                exec_ctx
            ))

        return RTResult().success(Null())
    execute_run.arg_names = ['fn']

    def execute_size(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get('list').list()
        if type(list_) is not List:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be a list",
                exec_ctx
            ))
        return RTResult().success(Number(len(list_.value)))
    execute_size.arg_names = ['list']

    def execute(self, args):
        res = RTResult()
        exec_ctx = self.new_context()

        method_name = f'execute_{self.name}'
        method = getattr(self, method_name, self.no_visit_method)

        res.register(self.check_pop_args(method.arg_names, args, exec_ctx))
        if res.should_return(): return res

        return_value = res.register(method(exec_ctx))
        if res.should_return(): return res
        return res.success(return_value)

    def copy(self):
        return BuiltInFunction(self.name).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def execute_print(self, exec_ctx):
        print(exec_ctx.symbol_table.get('value').string(), end='')
        return RTResult().success(Null())
    execute_print.arg_names = ['value']

    def execute_println(self, exec_ctx):
        print(exec_ctx.symbol_table.get('value').string())
        return RTResult().success(Null())
    execute_println.arg_names = ['value']

    def execute_str(self, exec_ctx):
        return RTResult().success(exec_ctx.symbol_table.get("value").string())
    execute_str.arg_names = ['value']

    def execute_list(self, exec_ctx):
        return RTResult().success(exec_ctx.symbol_table.get("value").list())
    execute_list.arg_names = ['value']

    def execute_bool(self, exec_ctx):
        return RTResult().success(exec_ctx.symbol_table.get("value").boolean())
    execute_bool.arg_names = ['value']

    def execute_num(self, exec_ctx):
        return RTResult().success(exec_ctx.symbol_table.get("value").number())
    execute_num.arg_names = ['value']

    def execute_dict(self, exec_ctx):
        return RTResult().success(exec_ctx.symbol_table.get("value").dictionary())
    execute_dict.arg_names = ['value']

    def execute_split(self, exec_ctx):
        return RTResult().success(
            List([String(x) for x in exec_ctx.symbol_table.get("string").string().value.split(
                    exec_ctx.symbol_table.get("splitter").string().value
                )])
        )
    execute_split.arg_names = ['string', 'splitter']

    def execute_func(self, exec_ctx):
        return RTResult().success(exec_ctx.symbol_table.get("value").function())
    execute_func.arg_names = ['value']

    def execute_contains(self, exec_ctx):
        return RTResult().success(
            exec_ctx.symbol_table.get("list").list().contains(
                exec_ctx.symbol_table.get("value")
            )
        )
    execute_contains.arg_names = ['list', 'value']

    def execute_clear(self, _exec_ctx):
        os.system('cls' if os.name == 'nt' else 'clear')
        return RTResult().success(Null())
    execute_clear.arg_names = []

    def execute_is_number(self, exec_ctx):
        return RTResult().success(
            Boolean(True) if type(exec_ctx.symbol_table.get("value")) is Number else Boolean(False)
        )
    execute_is_number.arg_names = ['value']

    def execute_is_list(self, exec_ctx):
        return RTResult().success(
            Boolean(True) if type(exec_ctx.symbol_table.get("value")) is List else Boolean(False)
        )
    execute_is_list.arg_names = ['value']

    def execute_is_string(self, exec_ctx):
        return RTResult().success(
            Boolean(True) if type(exec_ctx.symbol_table.get("value")) is String else Boolean(False)
        )
    execute_is_string.arg_names = ['value']

    def execute_is_boolean(self, exec_ctx):
        return RTResult().success(
            Boolean(True) if type(exec_ctx.symbol_table.get("value")) is Boolean else Boolean(False)
        )
    execute_is_boolean.arg_names = ['value']

    def execute_is_function(self, exec_ctx):
        return RTResult().success(
            Boolean(True) if type(exec_ctx.symbol_table.get("value")) is BaseFunction else Boolean(False)
        )
    execute_is_function.arg_names = ['value']

    def execute_is_null(self, exec_ctx):
        return RTResult().success(
            Boolean(True) if type(exec_ctx.symbol_table.get("value")) is Null else Boolean(False)
        )
    execute_is_null.arg_names = ['value']

    def execute_printback(self, exec_ctx):
        print(exec_ctx.symbol_table.get('value').string())
        return RTResult().success(exec_ctx.symbol_table.get('value').string())
    execute_printback.arg_names = ['value']

    def execute_field(self, exec_ctx):
        text = input(exec_ctx.symbol_table.get('value').string())
        return RTResult().success(String(text))
    execute_field.arg_names = ['value']

    def execute_nfield(self, exec_ctx):
        while True:
            text = input(exec_ctx.symbol_table.get('value').string())
            try:
                number = int(text); break
            except ValueError:
                print(f"Please enter an integer.")
        return RTResult().success(Number(number))
    execute_nfield.arg_names = ['value']

    def execute_choose(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get("list").list()
        if not list_.boolean().value:
            return RTResult().success(Null())
        return RTResult().success(random.choice(list_.value))
    execute_choose.arg_names = ['list']

    def execute_randint(self, exec_ctx):
        minv = exec_ctx.symbol_table.get("min").number()
        maxv = exec_ctx.symbol_table.get("max").number()
        return RTResult().success(Number(random.randint(minv.value, maxv.value)))
    execute_randint.arg_names = ['min', 'max']

    def execute_append(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get("list").list()
        value = exec_ctx.symbol_table.get("value")

        if type(list_) is not List:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "First argument must be a list",
                exec_ctx
            ))

        list_.value.append(value)
        return RTResult().success(Null())
    execute_append.arg_names = ['list', 'value']

    def execute_get(self, exec_ctx):
        dict_ = exec_ctx.symbol_table.get('dict').dictionary()
        key = exec_ctx.symbol_table.get('key')

        if not dict_.contains(key):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"'{key}' not in dict",
                exec_ctx
            ))
        return RTResult().success(dict_.get(key)[0])
    execute_get.arg_names = ['dict', 'key']

    def execute_set(self, exec_ctx):
        dict_ = exec_ctx.symbol_table.get('dict').dictionary()
        key = exec_ctx.symbol_table.get('key')
        value = exec_ctx.symbol_table.get('value')
        return RTResult().success(dict_.set(key, value))
    execute_set.arg_names = ['dict', 'key', 'value']

    def execute_delete(self, exec_ctx):
        dict_ = exec_ctx.symbol_table.get('dict').dictionary()
        key = exec_ctx.symbol_table.get('key').set_context(None).set_pos(None, None)
        if not dict_.contains(key):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"'{key}' not in dict",
                exec_ctx
            ))
        return RTResult().success(dict_.delete(key))
    execute_delete.arg_names = ['dict', 'key']

    def execute_pop(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get("list").list()
        index = exec_ctx.symbol_table.get("index").number()

        if type(list_) is not List:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "First argument must be a list",
                exec_ctx
            ))
        if type(index) is not Number:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Second argument must be a number",
                exec_ctx
            ))

        try:
            element = list_.value.pop(index.value)
        except:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                'Index is out of bounds!',
                exec_ctx
            ))
        return RTResult().success(element)
    execute_pop.arg_names = ['list', 'index']

    def execute_extend(self, exec_ctx):
        listA = exec_ctx.symbol_table.get("listA").list()
        listB = exec_ctx.symbol_table.get("listB").list()

        if (type(listA), type(listB)) != (List, List):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Both arguments must be lists",
                exec_ctx
            ))

        listA.value.extend(listB.value)
        return RTResult().success(Null())
    execute_extend.arg_names = ['listA', 'listB']


class String(Value):
    def function(self):
        return Function(None, StringNode(Token(TT_STRING, self.value, self.pos_start, self.pos_end)), None) \
            .set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def type(self):
        return String("string").set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def dictionary(self):
        return Dict({self.value: self.value}).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def number(self):
        return Number(len(self.value)).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def list(self):
        return List([String(x) for x in list(self.value)]).set_context(self.context)\
            .set_pos(self.pos_start, self.pos_end)

    def __repr__(self):
        return f'{self.value}'

    def mul(self, other):
        other = other.number()
        return String(self.value * other.value).set_context(self.context), None

    def add(self, other):
        other = other.string()
        return String(self.value + other.value).set_context(self.context), None

    def boolean(self):
        return Boolean(len(self.value) > 0).set_context(self.context).set_pos(self.pos_start, self.pos_end)


class Number(Value):
    def add(self, other):
        other = other.number()
        return Number(other.value + self.value).set_context(self.context), None

    def type(self):
        return String("number").set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def dictionary(self):
        return Dict({self.value: self.value}).set_pos(self.pos_start, self.pos_end).set_context(self.context)

    def mod(self, other):
        other = other.number()
        return Number(self.value % other.value).set_context(self.context), None

    def string(self):
        return String(str(self.value)).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def list(self):
        return List([self.value]).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def function(self):
        return Function(None, NumberNode(Token(TT_FLOAT if type(self.value) is float else TT_INT,
                                               self.value, self.pos_start, self.pos_end)), None) \
            .set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def number(self):
        return self

    def sub(self, other):
        other = other.number()
        return Number(self.value - other.value).set_context(self.context), None

    def mul(self, other):
        other = other.number()
        return Number(other.value * self.value).set_context(self.context), None

    def div(self, other):
        other = other.number()
        if other.value == 0:
            return None, RTError(
                other.pos_start, other.pos_end,
                'Division by zero',
                self.context
            )
        return Number(self.value / other.value).set_context(self.context), None

    def boolean(self):
        return Boolean(self.value > 0).set_context(self.context).set_pos(self.pos_start, self.pos_end)

    def fastpow(self, other):
        other = other.number()
        return Number(self.value ** other.value).set_context(self.context), None

    def eq(self, other):
        return Boolean(self.value == other.value).set_context(self.context), None

    def ne(self, other):
        return Boolean(self.value != other.value).set_context(self.context), None

    def lt(self, other):
        return Boolean(self.value < other.value).set_context(self.context), None

    def lte(self, other):
        return Boolean(self.value <= other.value).set_context(self.context), None

    def invert(self):
        return Number(self.value * -1).set_context(self.context), None


# CONTEXT


class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None) -> None:
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None


# SYMBOL TABLE


class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.attributes = {}
        self.bins = {}
        self.parent = parent

    def get(self, name):
        value = self.symbols.get(name, None)
        if value is None and self.parent:
            return self.parent.get(name)
        return value.value_node if value else None

    def locked(self, name):
        value = self.symbols.get(name, None)
        if value is None and self.parent:
            return self.parent.get(name)
        return value.locked if value else None

    def remove(self, name):
        value = self.symbols.get(name, None)
        if value is None:
            if self.parent:
                self.parent.remove(name)
            return
        del self.symbols[name]

    def set(self, name, value, locked=False):
        if name in self.symbols and self.symbols[name].locked:
            return 'Baked variable already defined'
        self.symbols[name] = VarNode(value, locked)
        return None

    def setbin(self, name, value):
        self.bins[name] = value

    def getbin(self, name):
        if name in self.bins:
            return self.bins[name]
        return self.parent.getbin(name) if self.parent else None

    def declareattr(self, name_tok, context):
        self.attributes[name_tok.value] = AttrNode(Null().set_pos(name_tok.pos_start).set_context(context))

    def getattr(self, name):
        if name in self.attributes: return self.attributes[name].value_node
        return self.parent.getattr(name) if self.parent else None

    def setattr(self, name, value):
        if name in self.attributes:
            self.attributes[name] = AttrNode(value)
        elif self.parent:
            self.parent.setattr(name, value)


# INTERPRETER

# noinspection PyMethodMayBeStatic
class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def visit_StringNode(self, node, context):
        return RTResult().success(
            String(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_ImportNode(self, node, context):
        fn = node.file_name_tok.value
        file_name = fn + ".devp"
        path = os.getcwd()
        if not os.path.exists("C:\\DP\\modules\\"):
            os.mkdir("C:\\DP\\modules\\")
        os.chdir("C:\DP\\modules\\")
        if fn in LIBRARIES:
            imp = ClassInstance(LIBRARIES[fn]).set_pos(node.pos_start, node.pos_end).set_context(context)
        elif os.path.exists(f"{fn}\\{file_name}"):
            imp, error = imprt(fn, open(f"{fn}\\{file_name}", "r").read(), context, node.pos_start)
            imp.set_pos(node.pos_start, node.pos_end).set_context(context)
            if error: return RTResult().failure(error)
        else:
            os.chdir(path)
            if os.path.exists(file_name):
                imp, error = imprt(fn, open(file_name, "r").read(), context, node.pos_start)
                imp.set_pos(node.pos_start, node.pos_end).set_context(context)
                if error: return RTResult().failure(error)
            else:
                return RTResult().failure(RTError(
                    node.pos_start, node.pos_end,
                    "Module does not exist!",
                    context
                ))
        os.chdir(path)
        context.symbol_table.set(fn, imp)
        return RTResult().success(Null())

    def visit_NullNode(self, node, context):
        return RTResult().success(
            Null().set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_ListNode(self, node, context):
        res = RTResult()
        elements = []

        for element in node.elements:
            elements.append(res.register(self.visit(element, context)))
            if res.should_return(): return res

        return res.success(
            List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_DictNode(self, node: DictNode, context):
        res = RTResult()
        dic = Dict({})

        for key, value in node.dict.items():
            ke = res.register(self.visit(key, context))
            if res.should_return(): return res
            va = res.register(self.visit(value, context))
            if res.should_return(): return res
            dic.set(ke, va)

        return res.success(
            dic.set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_ForNode(self, node, context):
        res = RTResult()
        elements = []

        start_value = res.register(self.visit(node.start_value_node, context))
        if res.should_return(): return res

        end_value = res.register(self.visit(node.end_value_node, context))
        if res.should_return(): return res

        if node.step_value_node:
            step_value = res.register(self.visit(node.step_value_node, context))
            if res.should_return(): return res
        else:
            step_value = Number(1)

        i = start_value.value
        if step_value.value >= 0:
            condition = lambda: i < end_value.value
        else:
            condition = lambda: i > end_value.value

        while condition():
            context.symbol_table.set(node.var_name_tok.value, Number(i))
            i += step_value.value

            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and not res.continue_loop and not res.break_loop: return res

            if res.continue_loop: continue
            if res.break_loop: break

            elements.append(value)

        context.symbol_table.remove(node.var_name_tok.value)

        return res.success(
            Null() if node.retnull else
            List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_WhileNode(self, node, context):
        res = RTResult()
        elements = []

        while True:
            condition = res.register(self.visit(node.condition_node, context))
            if res.should_return(): return res

            if not condition.boolean().value: break

            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and not res.continue_loop and not res.break_loop: return res

            if res.continue_loop: continue
            if res.break_loop: break

            elements.append(value)

        return res.success(
            Null() if node.retnull else
            List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_MethDefNode(self, node: MethDefNode, context) -> RTResult:
        res = RTResult()

        func_name = node.var_name_tok.value
        name_tok = node.var_name_tok
        body_node = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        meth_value = CMethod(func_name, name_tok, context, body_node, arg_names, node.bin, node.asynchronous,
                             node.autoreturn)\
            .set_pos(node.pos_start, node.pos_end)

        context.symbol_table.set(func_name, meth_value)

        return res.success(meth_value)

    def visit_ClassDefNode(self, node: ClassDefNode, context):
        res = RTResult()

        name = node.class_name_tok.value
        class_context = Context(f"<{name}-context>", context, node.pos_start)
        class_context.symbol_table = SymbolTable(context.symbol_table)
        attributes = [attribute for attribute in node.attribute_name_toks]
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        make = CMethod("<make>", None, class_context, node.make_node, arg_names, False, False, False)\
            .set_pos(node.pos_start, node.pos_end)
        methods = [res.register(self.visit_MethDefNode(method, class_context)) for method in node.methods]

        class_value = Class(name, attributes, make, methods).set_context(class_context)\
            .set_pos(node.pos_start, node.pos_end)
        context.symbol_table.set(name, class_value)

        return res.success(class_value)

    def visit_FuncDefNode(self, node, context):
        res = RTResult()

        func_name = node.var_name_tok.value if node.var_name_tok else None
        body_node = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        func_value = Function(func_name, body_node, arg_names, node.asynchronous, node.autoreturn).set_context(context)\
            .set_pos(node.pos_start, node.pos_end)

        if node.var_name_tok:
            context.symbol_table.set(func_name, func_value)

        return res.success(func_value)

    def visit_ReturnNode(self, node, context):
        res = RTResult()

        if node.node_to_return:
            value = res.register(self.visit(node.node_to_return, context))
            if res.should_return(): return res
        else:
            value = Null()

        return res.sreturn(
            value
        )

    def visit_ContinueNode(self, _node, _context):
        return RTResult().scontinue()

    def visit_BreakNode(self, _node, _context):
        return RTResult().sbreak()

    def visit_CallNode(self, node, context):
        res = RTResult()
        args = []
        value_to_call = res.register(self.visit(node.node_to_call, context))
        if res.should_return(): return res
        if type(value_to_call) is not CMethod:
            value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end).set_context(context)

        for arg_node in node.arg_nodes:
            args.append(res.register(self.visit(arg_node, context)))
            if res.should_return(): return res

        if hasattr(value_to_call, 'asynchronous') and value_to_call.asynchronous:
            start_new_thread(value_to_call.execute, (args,))
            return res.success(Null().set_pos(node.pos_start, node.pos_end).set_context(context))
        return_value = res.register(value_to_call.execute(args))
        if res.should_return(): return res
        return res.success(return_value.copy().set_pos(node.pos_start, node.pos_end).set_context(context))

    def visit_QueryNode(self, node, context):
        res = RTResult()

        for condition, expr, retnull in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.should_return(): return res

            if condition_value.boolean().value:
                expr_value = res.register(self.visit(expr, context))
                if res.should_return(): return res
                return res.success(Null() if retnull else expr_value)

        if node.else_case:
            else_value = res.register(self.visit(node.else_case[0], context))
            if res.should_return(): return res
            return res.success(Null() if node.else_case[1] else else_value)

        return res.success(Null())

    def visit_BooleanNode(self, node, context):
        return RTResult().success(
            Boolean(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_ClaccessNode(self, node: ClaccessNode, context):
        res = RTResult()
        var = res.register(self.visit(
            node.class_tok,
            context
        ))

        if res.error: return res
        if type(var) is not ClassInstance:
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                "Expected class instance",
                context
            ))
        val = var.access(node.attr_name_tok.value)
        return res.success(val)

    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)

        if value is None:
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                f"'{var_name}' is not defined",
                context
            ))
        if type(value) is str:
            while context.display_name != value:
                if context.parent is None:
                    return res.failure(RTError(
                        node.pos_start, node.pos_end,
                        "Invalid 'this'",
                        context
                    ))
                context = context.parent
            return res.success(ClassInstance(context).copy().set_pos(node.pos_start, node.pos_end).set_context(context))

        value = value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
        return res.success(value)

    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, context))
        if res.should_return(): return res

        error = context.symbol_table.set(var_name, value, node.locked)
        if error:
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                error,
                context
            ))
        return res.success(value)

    def visit_AttrAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, context))
        if res.should_return(): return res

        error = context.symbol_table.setattr(var_name, value)
        if error:
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                error,
                context
            ))
        return res.success(value)

    def visit_AttrAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.symbol_table.getattr(var_name)

        if value is None:
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                f"'{var_name}' is not defined",
                context
            ))

        value = value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
        return res.success(value)

    def visit_BinOpNode(self, node: BinOpNode, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.should_return(): return res
        right = res.register(self.visit(node.right_node, context))
        if res.should_return(): return res

        result, error = {
            TT_PLUS: lambda x, y: x.add(y),
            TT_MINUS: lambda x, y: x.sub(y),
            TT_MUL: lambda x, y: x.mul(y),
            TT_DIV: lambda x, y: x.div(y),
            TT_POWER: lambda x, y: x.fastpow(y),
            TT_EE: lambda x, y: x.eq(y),
            TT_NE: lambda x, y: x.ne(y),
            TT_LT: lambda x, y: x.lt(y),
            TT_GT: lambda x, y: y.lt(x),
            TT_LTE: lambda x, y: x.lte(y),
            TT_GTE: lambda x, y: y.lte(x),
            TT_AND: lambda x, y: x.including(y),
            TT_OR: lambda x, y: x.also(y),
            TT_MOD: lambda x, y: x.mod(y),
            TT_DOT: lambda x, y: x.get(y)
        }[node.op_tok.type](left, right)
        if error:
            return res.failure(error)
        return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnaryOpNode(self, node: UnaryOpNode, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.should_return(): return res
        if node.op_tok.type == TT_MINUS:
            number, error = number.mul(Number(-1))
        elif node.op_tok.type in (TT_INCR, TT_DECR):
            number, error = number.add(Number(1 if node.op_tok.type == TT_INCR else -1))
        elif node.op_tok.type == TT_NOT:
            number, error = number.invert()
        else:
            number, error = number, None
        if error:
            return res.failure(error)
        return res.success(number.set_pos(node.pos_start, node.pos_end))


# RUN


global_symbol_table = SymbolTable()

BUILTINS = (
    "println",
    "pause",
    "insert",
    "floating",
    "contains",
    "split",
    "str",
    "bool",
    "list",
    "dict",
    "num",
    "func",
    "print",
    "printback",
    "field",
    "nfield",
    "clear",
    "is_number",
    "is_string",
    "is_bool",
    "is_null",
    "is_list",
    "is_function",
    "append",
    "pop",
    "extend",
    "run",
    "size",
    "abs",
    "random",
    "floor",
    "ceil",
    "round",
    "foreach",
    "get",
    "set",
    "delete",
    "choose",
    "randint",
    "type"
)

for builtin in BUILTINS:
    setattr(BuiltInFunction, builtin, BuiltInFunction(builtin))
    f = getattr(BuiltInFunction, builtin)
    global_symbol_table.set(builtin, f)

CONSTANTS = {
    'pi_': Number(math.pi),
    'e_': Number(math.e)
}

for k, v in CONSTANTS.items():
    global_symbol_table.set(k, v, True)


def run(fn, text):
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error: return None, error

    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    interpreter = Interpreter()
    context = Context(fn)
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)

    return result.value, result.error


def imprt(fn, text, parent, entry_pos):
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error: return None, error

    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    interpreter = Interpreter()
    context = Context(fn, parent, entry_pos)
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)

    return ClassInstance(context), result.error
