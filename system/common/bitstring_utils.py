"""
Utility functions for handling bitstrings (necessary for error corecting and serializing signature data)
"""

def pad_bitstring(b, n):
    #if b has number of bits that is not multiple of n, pad end with 0
    if len(b) % n != 0 :
        num_pad_bits = n - (len(b) % n)
        for i in range(num_pad_bits):
            b += "0"
    return b

def bitstring_to_ascii(b):
    """
    b : str containing only 0s and 1s, where...
    """
    b = pad_bitstring(b, 8)

    i = 0
    out = ""
    while i + 8 < len(b) + 8:
        out += chr(int(b[i:i+8], 2))
        i += 8
    return out


def ascii_to_bitstring(ascii):
    """
    ascii : 
    """
    out = ""
    for i in ascii:
        bits = bin(ord(i))[2:]
        if len(bits) < 8:
            for j in range(8 - len(bits)):
                bits = "0" + bits
        out += bits
    return out


def bitstring_to_bytes(bitstring):
    bitstring = pad_bitstring(bitstring, 8)
    return int(bitstring, 2).to_bytes((len(bitstring) + 7) // 8, 'big')

def bytes_to_bitstring(bytes_str):
    out = ""
    for c in bytes_str:
        bits = bin(c)[2:]
        if len(bits) < 8:
            for j in range(8 - len(bits)):
                bits = "0" + bits
        out += bits
    return out
