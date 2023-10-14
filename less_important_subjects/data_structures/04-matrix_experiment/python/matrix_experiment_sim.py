#!/usr/bin/env python3
import csv
import sys

from matrix_tests import TestMatrix


def simulated_test(M, B, naive, filename):
    if naive:
        filename = filename + "_naive.csv"
    else:
        filename = filename + ".csv"
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for e in range(10, 25):
            N = int(2 ** (e/2))
            print("    ", N, M, B, file=sys.stderr)
            m = TestMatrix(N, M, B, 0)
            m.fill_matrix()
            m.reset_stats()
            if naive:
                m.naive_transpose()
            else:
                m.transpose()
            misses_per_item = m.stat_cache_misses / (N*(N-1))

            writer.writerow([N] + [misses_per_item])
            print(N, misses_per_item, flush=True)
            m.check_result()

tests = {
#                                                M     B
   # "m1024-b16":    lambda n: simulated_test( 1024,   16, n, "m1024-b16"),
   # "m8192-b64":    lambda n: simulated_test( 8192,   64, n, "m8192-b64"),
   # "m65536-b256":  lambda n: simulated_test(65536,  256, n, "m65536-b256"),
   # "m65536-b4096": lambda n: simulated_test(65536, 4096, n, "m65536-b4096"),
   # "m65536-b": lambda n: simulated_test(65536, 128, n, "m65536-b128"),
   # "m65536-b256":  lambda n: simulated_test(65536,  256, n, "m65536-b256_20x20"),
   "m65536-b4096": lambda n: simulated_test(65536, 4096, n, "m65536-b4096_1x1"),
}

# Random seed = 35

def do_tests(tests_names, student_id):
    for test in tests_names:
        if test in tests:
            tests[test](True)
            tests[test](False)

do_tests(tests.keys(), 35)