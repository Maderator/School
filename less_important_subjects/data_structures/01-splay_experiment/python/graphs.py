from matplotlib import pyplot as plt
import csv
import numpy as np

def plot_sequential(loglog=True):
    with open('sequential_naive.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        n_n = []
        rot_per_op_n = []
        for row in reader:
            n_n.append(int(row[0]))
            rot_per_op_n.append(float(row[1]))
    with open('sequential_std.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        n_s = []
        rot_per_op_s = []
        for row in reader:
            n_s.append(int(row[0]))
            rot_per_op_s.append(float(row[1]))
    #plt.plot(n_n, rot_per_op_n, label='naive')
    plt.plot(n_s, rot_per_op_s, label='standard', color='orange')

    plt.title("Sequential test average rotations")

    #plt.yscale('log')
    if loglog:
        plt.xlabel("Number of elements in splay tree (log. scale)")
        plt.ylabel("Average number of rotations per operation (log. scale)")
        plt.loglog()
    else:
        plt.xlabel("Number of elements in splay tree")
        plt.ylabel("Average number of rotations per operation")


    plt.legend()

    plt.show()

def plot_random():
    with open('random_naive.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        n_n = []
        rot_per_op_n = []
        for row in reader:
            n_n.append(int(row[0]))
            rot_per_op_n.append(float(row[1]))
    with open('random_std.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        n_s = []
        rot_per_op_s = []
        for row in reader:
            n_s.append(int(row[0]))
            rot_per_op_s.append(float(row[1]))
    plt.plot(n_n, rot_per_op_n, label='naive')
    plt.plot(n_s, rot_per_op_s, label='standard')

    plt.xlabel("Number of elements in splay tree")
    plt.ylabel("Average number of rotations per operation")
    plt.title("Random test average rotations")

    #plt.yscale('log')
    plt.legend()

    plt.show()

def plot_subset():
    sub = [100]
    #sub = [1000, 4000, 10000, 50000]
    for s in sub:
        with open('subset_naive_{}.csv'.format(s)) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            n_n = []
            rot_per_op_n = []
            for row in reader:
                n_n.append(int(row[1]))
                rot_per_op_n.append(float(row[2]))
        with open('subset_std_{}.csv'.format(s)) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            n_s = []
            rot_per_op_s = []
            for row in reader:
                n_s.append(int(row[1]))
                rot_per_op_s.append(float(row[2]))
        plt.plot(n_n, rot_per_op_n, label='naive_{}'.format(s))
        plt.plot(n_s, rot_per_op_s, label='standard_{}'.format(s))

    #plt.xlabel("Number of elements in splay tree")
    plt.xlabel("Number of lookups")
    plt.ylabel("Average number of rotations per operation (log. scale)")
    plt.title("Subset test dependence of average number of rotations on number of lookups in splay tree with 1024 nodes")

    #plt.yscale('log')
    #plt.xscale('log')
    #plt.loglog()
    plt.legend()

    plt.show()

if __name__ == "__main__":
    #plot_sequential(loglog=False)
    #plot_sequential(loglog=True)
    #plot_random()
    plot_subset()