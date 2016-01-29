#!/usr/bin/env python2.7
import re


def main():
    with open('face.txt', 'r') as fin:
        fout = open('listfile.txt', 'w')
        fout.write("202599\n")
        for line in fin:
            line = line.strip()
            line = line.strip()
            components = line.split(' ')[:5]
            components[0] = re.search('/(\d+.jpg)', components[0]).groups()[0]
            fout.write(' '.join(components))
            fout.write('\n')
        fout.close()


if __name__ == '__main__':
    main()
