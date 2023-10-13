#!/usr/bin/env python3

import sys
import random
import csv


from splay_operation import Tree

class BenchmarkingTree(Tree):
    """ A modified Splay tree for benchmarking.

    We inherit the implementation of operations from the Tree class
    and extend it by keeping statistics on the number of splay operations
    and the total number of rotations. Also, if naive is turned on,
    splay uses only single rotations.
    """

    def __init__(self, naive=False):
        Tree.__init__(self)
        self.do_naive = naive
        self.reset()

    def reset(self):
        """Reset statistics."""
        self.num_rotations = 0;
        self.num_operations = 0;

    def rotate(self, node):
        self.num_rotations += 1
        Tree.rotate(self, node)

    def splay(self, node):
        self.num_operations += 1
        if self.do_naive:
            while node.parent is not None:
                self.rotate(node)
        else:
            Tree.splay(self, node)

    def rot_per_op(self):
        """Return the average number of rotations per operation."""
        if self.num_operations > 0:
            return self.num_rotations / self.num_operations
        else:
            return 0

def test_sequential(naive=False):
    if naive:
        filename = 'sequential_naive.csv'
    else:
        filename = 'sequential_std.csv'
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for n in range(100, 3001, 100):
            tree = BenchmarkingTree(naive)
            for elem in range(n):
                tree.insert(elem)

            for _ in range(5):
                for elem in range(n):
                    tree.lookup(elem)

            writer.writerow([n] + [tree.rot_per_op()])
            print(n, tree.rot_per_op())

def test_random(naive=False):
    if naive:
        filename = 'random_naive.csv'
    else:
        filename = 'random_std.csv'
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for exp in range(32, 64):
            n = int(2**(exp/4))
            tree = BenchmarkingTree(naive)

            for elem in random.sample(range(n), n):
                tree.insert(elem)

            for _ in range(5*n):
                tree.lookup(random.randrange(n))

            writer.writerow([n] + [tree.rot_per_op()])
            print(n, tree.rot_per_op())

def make_progression(seq, A, B, s, inc):
    """An auxiliary function for constructing arithmetic progressions.

    The array seq will be modified to contain an arithmetic progression
    of elements in interval [A,B] starting from position s with step inc.
    """
    for i in range(len(seq)):
        while seq[i] >= A and seq[i] <= B and s + inc*(seq[i]-A) != i:
            pos = s + inc*(seq[i]-A)
            seq[i], seq[pos] = seq[pos], seq[i]

def test_subset(naive=False):
    for sub in [100]:
        if naive:
            filename = 'subset_naive_{}.csv'.format(sub)
        else:
            filename = 'subset_std_{}.csv'.format(sub)
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for lookup in range (100,20000,30):
            #for exp in range(32,64):
                exp = 40
                n = int(2**(exp/4))
                if n < sub:
                    continue

                # We will insert elements in order, which contain several
                # arithmetic progressions interspersed with random elements.
                seq = random.sample(range(n), n)
                make_progression(seq, n//4, n//4 + n//20, n//10, 1)
                #make_progression(seq, n//10, n//2 + n//4, n//10, 1)
                make_progression(seq, n//2, n//2 + n//20, n//10, -1)
                make_progression(seq, 3*n//4, 3*n//4 + n//20, n//2, -4)
                make_progression(seq, 17*n//20, 17*n//20 + n//20, 2*n//5, 5)

                tree = BenchmarkingTree(naive)
                for elem in seq:
                    tree.insert(elem)
                tree.reset()

                #for _ in range(40000):
                for _ in range(lookup):
                    tree.lookup(seq[random.randrange(sub)])

                #writer.writerow([sub] + [n] + [tree.rot_per_op()])
                writer.writerow([sub] + [lookup] + [tree.rot_per_op()])
                #print(sub, n, tree.rot_per_op())
                print(sub, lookup, tree.rot_per_op())

tests = {
    #"sequential": test_sequential,
    #"random": test_random,
    "subset": test_subset,
}

#35
for test, func in tests.items():
    student_id = sys.argv[1]
    random.seed(student_id)
    func(naive=False)
    random.seed(student_id)
    func(naive=True)

#if len(sys.argv) == 4:
#    test, student_id = sys.argv[1], sys.argv[2]
#    if sys.argv[3] == "std":
#        naive = False
#    elif sys.argv[3] == "naive":
#        naive = True
#    else:
#        raise ValueError("Last argument must be either 'std' or 'naive'")
#    random.seed(student_id)
#    if test in tests:
#        tests[test](naive)
#    else:
#        raise ValueError("Unknown test {}".format(test))
#else:
#    raise ValueError("Usage: {} <test> <student-id> (std|naive)".format(sys.argv[0]))
