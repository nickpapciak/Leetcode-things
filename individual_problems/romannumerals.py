import functools
# ^ you can make it faster at expense of memory

NUMERAL_MAP = {"M":1000, "CM":900, "D":500, "CD":400, "C":100, "XC":90, "L":50, "XL":40, "X":10, "IX":9, "V":5, "IV":4, "I":1}

@functools.cache
def int_to_roman(i: int) -> str: 
    for roman, traditional in NUMERAL_MAP.items(): 
        # how many times a certain numeral can 'fit' into i
        quo, rem = divmod(i, traditional)
        if quo > 0: 
            return (quo * roman) + int_to_roman(rem)
    # base case
    return ""

@functools.cache
def roman_to_int(r: str) -> int:
    # base case
    if r == '': 
        return 0
    # if there is a special subtraction pair
    if r[:2] in NUMERAL_MAP.keys(): 
        return NUMERAL_MAP[r[:2]] + roman_to_int(r[2:])
    # otherwise single numeral
    else: 
        return NUMERAL_MAP[r[0]] + roman_to_int(r[1:])

# tests all traditional roman numerals
assert any([x == roman_to_int(int_to_roman(x)) for x in range(1, 4000)]), "numeral failed"
