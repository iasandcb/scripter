"""Spoken Korean math -> AsciiMath, the same rules as mark-vector's math
dictation (frontend/static/js/kor-math.js there) so one vocabulary file
works in both.

The vocabulary is a CSV of "word, symbol" rules - mark-vector's "공용 수식
말" (spoken-math.csv). The word is everything before the first comma and the
symbol everything after it, so a symbol may itself be a comma ("과, ,");
"\\n" is a line break inside the formula and "$$" opens or closes a math
block ("수식시작, $$"). Lines starting with "#" are comments; a later line
wins over an earlier one for the same word. Spaces inside a word don't
matter - matching ignores spaces between Korean syllables, since
recognizers space Korean inconsistently.
"""
import re

HANGUL_RUN = re.compile(r'[가-힣]+|[^가-힣]+')
LONE_CAPITALS = re.compile(r'(?<![A-Za-z])[A-Z]{1,3}(?![A-Za-z])')


def parse_csv(content):
    rules = {}
    for line in content.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        word, comma, symbol = line.partition(',')
        if not comma:
            continue
        word = re.sub(r'\s+', '', word)
        symbol = symbol.strip()
        if symbol == '\\n':
            symbol = '\n'
        if word and symbol:
            rules[word] = symbol
    return rules


def _drop_sentence_punctuation(text):
    """Punctuation a recognizer adds on its own ("엑스 더하기 와이.") means
    nothing in a formula; a "." or "," between digits stays (3.14, 1,000)."""
    def keep(match):
        at, mark = match.start(), match.group()
        between_digits = (mark in '.,' and at > 0 and text[at - 1].isdigit()
                          and at + 1 < len(text) and text[at + 1].isdigit())
        return mark if between_digits else ' '
    return re.sub(r'[.,?!。、，]', keep, text)


class SpokenMath:
    def __init__(self, rules):
        self.rules = dict(rules)
        self.longest = max((len(word) for word in self.rules), default=0)
        # Words whose symbol is $$ open/close a math block wherever they're
        # heard; longest first, spaces allowed between syllables.
        commands = sorted((w for w, s in self.rules.items() if s == '$$'), key=len, reverse=True)
        self.block_command = (re.compile('|'.join(r'\s*'.join(map(re.escape, w)) for w in commands))
                              if commands else None)
        # Capitals are spoken with a prefix (라지 + 에프 -> F): whatever the
        # vocabulary puts in front of a lowercase letter's word to get its
        # capital.
        prefixes = set()
        for word, symbol in self.rules.items():
            if len(symbol) == 1 and 'A' <= symbol <= 'Z':
                for lower, lower_symbol in self.rules.items():
                    if lower_symbol == symbol.lower() and len(word) > len(lower) and word.endswith(lower):
                        prefixes.add(word[:-len(lower)])
        self.capital_prefixes = sorted(prefixes, key=len, reverse=True)

    @classmethod
    def from_csv(cls, content):
        return cls(parse_csv(content))

    def block_commands(self, text):
        """(start, end) spans of every block command in `text`."""
        if not self.block_command:
            return []
        return [match.span() for match in self.block_command.finditer(text)]

    def _convert_run(self, run, out):
        """Longest match first over Korean syllables (spaces already removed);
        a syllable no word starts with is kept as said (a particle, say)."""
        i, unknown = 0, ''
        while i < len(run):
            for size in range(min(self.longest, len(run) - i), 0, -1):
                symbol = self.rules.get(run[i:i + size])
                if symbol is not None and symbol != '$$':
                    break
            else:
                unknown += run[i]
                i += 1
                continue
            if unknown:
                out.append(unknown)
                unknown = ''
            out.append(symbol)
            i += size
        if unknown:
            out.append(unknown)

    def to_asciimath(self, text):
        """"엑스 승 2 더하기 라지 와이 는 열고 에이 과 비 닫고" ->
        "x ^ 2 + Y = ( a , b )". Anything that isn't Korean passes through
        as its own token, except that capitals standing alone - a recognizer
        spelling out "에프" as "F" - are read as lowercase variables unless
        they follow the capital prefix ("라지 F")."""
        out, run = [], ''
        for token in _drop_sentence_punctuation(text).split():
            for piece in HANGUL_RUN.findall(token):
                if '가' <= piece[0] <= '힣':
                    run += piece
                    continue
                prefix = next((p for p in self.capital_prefixes if run.endswith(p)), None) \
                    if len(piece) == 1 and piece.isascii() and piece.isalpha() else None
                if prefix:
                    run = run[:-len(prefix)]
                if run:
                    self._convert_run(run, out)
                run = ''
                out.append(piece.upper() if prefix else LONE_CAPITALS.sub(lambda m: m.group().lower(), piece))
        if run:
            self._convert_run(run, out)
        return re.sub(r' *\n *', '\n', ' '.join(out))
