from matplotlib import pyplot as plt
import csv
import numpy as np
from numpy.lib.npyio import load

def load_data_one_file(file):
    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first_col = []
        second_col = []
        for row in reader:
            first_col.append(int(row[0]))
            second_col.append(float(row[1]))
    return first_col, second_col

def load_data_two_files(first_file, second_file):
    f1_c1, f1_c2 = load_data_one_file(first_file)
    f2_c1, f2_c2 = load_data_one_file(second_file)
    return f1_c1, f1_c2, f2_c1, f2_c2

def plot_xy_labels(xlog, ylog):
    if xlog and ylog:
        plt.xlabel("Size of square matrix  (log)")
        plt.ylabel("Average number of cache misses per item (log)")
        plt.loglog()
    elif xlog:
        plt.xlabel("Size of square matrix (log)")
        plt.ylabel("Average number of cache misses per item")
        plt.xscale("log")
    elif ylog:
        plt.xlabel("Size of square matrix")
        plt.ylabel("Average number of cache misses per item (log)")
        plt.yscale("log")
    else:
        plt.xlabel("Size of square matrix")
        plt.ylabel("Average number of cache misses per item")
    plt.grid()

def plot_m1024(xlog=False, ylog=False):
    n_N, n_misses, my_N, my_misses = load_data_two_files('m1024-b16_naive.csv', 'm1024-b16.csv')

    plt.figure(figsize=(8,6))
    plt.plot(n_N, n_misses, label='trivial', marker='.')
    plt.plot(my_N, my_misses, label='divide-and-conquer', marker='.', color='orange')

    plt.title("Cache of 1024 items organized in 16-item blocks")

    plot_xy_labels(xlog, ylog)

    plt.legend()
    plt.show()

def plot_m8192(xlog=False, ylog=False):
    n_N, n_misses, my_N, my_misses = load_data_two_files('m8192-b64_naive.csv', 'm8192-b64.csv')

    plt.figure(figsize=(8,6))
    plt.plot(n_N, n_misses, label='trivial', marker='.')
    plt.plot(my_N, my_misses, label='divide-and-conquer', marker='.', color='orange')

    plt.title("Cache of 8192 items organized in 64-item blocks")

    plot_xy_labels(xlog, ylog)
    
    plt.legend()
    plt.show()

def plot_m65536b256(xlog=False, ylog=False):
    n_N, n_misses, my_N, my_misses = load_data_two_files('m65536-b256_naive.csv', 'm65536-b256.csv')

    plt.figure(figsize=(8,6))
    plt.plot(n_N, n_misses, label='trivial', marker='.')
    plt.plot(my_N, my_misses, label='divide-and-conquer', marker='.', color='orange')

    plt.title("Cache of 65536 items organized in 256-item blocks")

    plot_xy_labels(xlog, ylog)

    plt.legend()
    plt.show()

def plot_m65536b4096(xlog=False, ylog=False):
    n_N, n_misses, my_N, my_misses = load_data_two_files('m65536-b4096_naive.csv', 'm65536-b4096.csv')

    plt.figure(figsize=(8,6))
    plt.plot(n_N, n_misses, label='trivial', marker='.')
    plt.plot(my_N, my_misses, label='divide-and-conquer', marker='.', color='orange')

    plt.title("Cache of 65536 items organized in 4096-item blocks")
    
    plot_xy_labels(xlog, ylog)

    plt.legend()
    plt.show()

def plot_m65536b128(xlog=False, ylog=False):
    n_N, n_misses, my_N, my_misses = load_data_two_files('m65536-b128_naive.csv', 'm65536-b128.csv')

    plt.figure(figsize=(8,6))
    plt.plot(n_N, n_misses, label='trivial', marker='.')
    plt.plot(my_N, my_misses, label='divide-and-conquer', marker='.', color='orange')

    plt.title("Cache of 65536 items organized in 128-item blocks")
    
    plot_xy_labels(xlog, ylog)

    plt.legend()
    plt.show()

def plot_m65536b256_9x9(xlog=False, ylog=False):
    n_N, n_misses, my_N, my_misses = load_data_two_files('m65536-b256_9x9_naive.csv', 'm65536-b256_9x9.csv')

    plt.figure(figsize=(8,6))
    plt.plot(n_N, n_misses, label='trivial', marker='.')
    plt.plot(my_N, my_misses, label='divide-and-conquer', marker='.', color='orange')

    plt.title("Cache of 65536 items organized in 256-item blocks (naive 9x9)")

    plot_xy_labels(xlog, ylog)

    plt.legend()
    plt.show()

def plot_m65536b4096_9x9(xlog=False, ylog=False):
    n_N, n_misses, my_N, my_misses = load_data_two_files('m65536-b4096_9x9_naive.csv', 'm65536-b4096_9x9.csv')

    plt.figure(figsize=(8,6))
    plt.plot(n_N, n_misses, label='trivial', marker='.')
    plt.plot(my_N, my_misses, label='divide-and-conquer', marker='.', color='orange')

    plt.title("Cache of 65536 items organized in 4096-item blocks doing (naive 9x9)")
    
    plot_xy_labels(xlog, ylog)

    plt.legend()
    plt.show()

def plot_m65536b256_20x20(xlog=False, ylog=False):
    n_N, n_misses, my_N, my_misses = load_data_two_files('m65536-b256_20x20_naive.csv', 'm65536-b256_20x20.csv')

    plt.figure(figsize=(8,6))
    plt.plot(n_N, n_misses, label='trivial', marker='.')
    plt.plot(my_N, my_misses, label='divide-and-conquer', marker='.', color='orange')

    plt.title("Cache of 65536 items organized in 256-item blocks (naive 20x20)")

    plot_xy_labels(xlog, ylog)

    plt.legend()
    plt.show()

def plot_m65536b4096_20x20(xlog=False, ylog=False):
    n_N, n_misses, my_N, my_misses = load_data_two_files('m65536-b4096_20x20_naive.csv', 'm65536-b4096_20x20.csv')

    plt.figure(figsize=(8,6))
    plt.plot(n_N, n_misses, label='trivial', marker='.')
    plt.plot(my_N, my_misses, label='divide-and-conquer', marker='.', color='orange')

    plt.title("Cache of 65536 items organized in 4096-item blocks doing (naive 20x20)")
    
    plot_xy_labels(xlog, ylog)

    plt.legend()
    plt.show()

def plot_m65536b4096_1x1(xlog=False, ylog=False):
    n_N, n_misses, my_N, my_misses = load_data_two_files('m65536-b4096_1x1_naive.csv', 'm65536-b4096_1x1.csv')

    plt.figure(figsize=(8,6))
    plt.plot(n_N, n_misses, label='trivial', marker='.')
    plt.plot(my_N, my_misses, label='divide-and-conquer', marker='.', color='orange')

    plt.title("Cache of 65536 items organized in 4096-item blocks doing (naive 1x1)")
    
    plot_xy_labels(xlog, ylog)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    #plot_m1024(xlog=True, ylog=False)
    #plot_m8192(xlog=True, ylog=False)
    #plot_m65536b256(xlog=True, ylog=False)
    #plot_m65536b4096(xlog=False, ylog=False)
    #plot_m65536b256_9x9(xlog=True, ylog=False)
    #plot_m65536b4096_9x9(xlog=True, ylog=False)
    #plot_m65536b256_20x20(xlog=True, ylog=False)
    #plot_m65536b4096_20x20(xlog=False, ylog=False)
    #plot_m65536b128(xlog=True, ylog=False)
    plot_m65536b4096_1x1(xlog=False, ylog=False)