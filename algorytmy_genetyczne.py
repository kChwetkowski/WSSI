import random

min_range = 0
max_range = 10
def encode_binary(x, num_bits):
    """Encode a decimal value into a binary string representation."""
    max_value = 2 ** num_bits - 1
    scaled_value = round(x * (max_value)/(max_range - min_range))
    binary_string = bin(scaled_value)[2:].zfill(num_bits)
    return binary_string

def decode_binary(binary_string, num_bits):
    """Decode a binary string representation into a decimal value."""
    max_value = 2 ** num_bits - 1
    scaled_value = int(binary_string, 2)
    x =  round((scaled_value / max_value) * 10,7)
    return x

# Przykładowe użycie
num_bits = 8  # Number of bits to represent x
x = 6.5  # Decimal value of x

# Encoding
binary_string = encode_binary(x, num_bits)
print(f"Binary representation x: {binary_string}")

# Decoding
decoded_x = decode_binary(binary_string, num_bits)
print(f"Decoded value of x: {decoded_x}")
