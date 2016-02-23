#!/usr/bin/env python2.7

import os
import re
import sys
import matplotlib.pyplot as plt


def main():
    # get log file from argv
    argv = sys.argv
    if len(argv) != 2:
        print "Wrong Argv:", argv
        sys.exit()
    log_file = argv[-1]
    print "Parsing", log_file
    if not os.path.exists(log_file):
        print "No log file, Please Check it"
        sys.exit()
    # parse log
    fin = open(log_file, 'r')
    tree_index = 0
    offset = 0
    result = []
    for line in fin.readlines():
        res = re.search("Train (\d+) th stages", line)
        if res is not None:
            offset += tree_index
            continue
        res = re.search("Train (\d+) th Cart", line)
        if res is not None:
            tree_index = int(res.groups()[0]) + offset
            continue
        res = re.search("Average number of cart to reject is (\d+\.\d+), FP = (\d+\.\d+)%", line)
        if res is not None:
            avg = float(res.groups()[0])
            fp = float(res.groups()[1]) / 100.
            result.append((tree_index, avg, fp))
        else:
            continue
    fin.close()
    # result
    tree = [_[0] for _ in result]
    avg_tree = [_[1] for _ in result]
    fp = [_[2] for _ in result]
    plt.plot(tree, fp)
    plt.yscale('log')
    plt.show()


if __name__ == '__main__' :
    main()
