from matplotlib import pyplot as plt
import csv
import numpy as np

def plot_insert(loglog=True):
    with open('insert2_3.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        n_23 = []
        changes_per_op_23 = []
        for row in reader:
            n_23.append(int(row[0]))
            changes_per_op_23.append(float(row[1]))
    with open('insert2_4.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        n_24 = []
        changes_per_op_24 = []
        for row in reader:
            n_24.append(int(row[0]))
            changes_per_op_24.append(float(row[1]))
    plt.plot(n_23, changes_per_op_23, label='2-3', marker='.')
    plt.plot(n_24, changes_per_op_24, label='2-4', marker='.', color='orange')

    plt.title("Insert test average changes")

    #plt.yscale('log')
    if loglog:
        plt.xlabel("Number of elements in ab-tree (log. scale)")
        plt.ylabel("Average number of changes per operation (log. scale)")
        plt.loglog()
    else:
        plt.xlabel("Number of elements in ab-tree")
        plt.ylabel("Average number of changes per operation")


    plt.legend()

    plt.show()

def plot_min():
    with open('min2_3.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        n_23 = []
        changes_per_op_23 = []
        for row in reader:
            n_23.append(int(row[0]))
            changes_per_op_23.append(float(row[1]))
    with open('min2_4.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        n_24 = []
        changes_per_op_24 = []
        for row in reader:
            n_24.append(int(row[0]))
            changes_per_op_24.append(float(row[1]))
    plt.plot(n_23, changes_per_op_23, label='2-3', marker='.')
    plt.plot(n_24, changes_per_op_24, label='2-4', marker='.', color='orange')

    plt.xlabel("Number of elements in ab-tree")
    plt.ylabel("Average number of changes per operation")
    plt.title("Min test average changes")

    #plt.yscale('log')
    plt.legend()

    plt.show()

def plot_random():
    with open('random2_3.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        n_23 = []
        changes_per_op_23 = []
        for row in reader:
            n_23.append(int(row[0]))
            changes_per_op_23.append(float(row[1]))
    with open('random2_4.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        n_24 = []
        changes_per_op_24 = []
        for row in reader:
            n_24.append(int(row[0]))
            changes_per_op_24.append(float(row[1]))
    plt.plot(n_23, changes_per_op_23, label='2-3', marker='.')
    plt.plot(n_24, changes_per_op_24, label='2-4', marker='.', color='orange')

    #plt.xlabel("Number of elements in splay tree")
    plt.xlabel("Number of elements in ab-tree")
    plt.ylabel("Average number of changes per operation")
    plt.title("Random test average changes")

    #plt.yscale('log')
    #plt.xscale('log')
    #plt.loglog()
    plt.legend()

    plt.show()

if __name__ == "__main__":
    plot_insert(loglog=False)
    #plot_insert(loglog=True)
    #plot_min()
    #plot_random()