import os
import matplotlib.pyplot as plt


def save_line_plot(values, title: str, ylabel: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_binary_plot(values, title: str, ylabel: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 3))
    plt.step(range(len(values)), values, where="mid")
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel(ylabel)
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()