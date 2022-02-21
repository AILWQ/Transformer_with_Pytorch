import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_positional_encoding(max_seq_len, d_model):
    # max_seq_len: 最大序列长度
    # d_model: 位置嵌入的维度 = 字嵌入的维度，以便直接相加
    positional_encoding = np.array([[pos / np.power(10000, 2 * i // d_model) for i in range(d_model)]
                                    if pos != 0 else np.zeros(d_model) for pos in range(max_seq_len)])

    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2]) # dim=2i
    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim=2i+1
    return positional_encoding

positional_encoding = get_positional_encoding(max_seq_len=10, d_model=20)
plt.figure(figsize=(10,10))
sns.heatmap(positional_encoding)
plt.title("Sinusoidal Function")
plt.xlabel("embedding dimension")
plt.ylabel("max sequence length")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(positional_encoding[1:, 1], label="dimension 1")
plt.plot(positional_encoding[1:, 2], label="dimension 2")
plt.plot(positional_encoding[1:, 3], label="dimension 3")
plt.legend()
plt.xlabel("Sequence length")
plt.ylabel("Period of Positional Encoding")
plt.show()

