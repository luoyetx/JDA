#!/usr/bin/env python2.7
import os
import random
import cv2
import numpy as np


IMG_ROOT = 'F:/JDA_BG/CelebA/img_celeba/img_celeba/'
LANDMARK_TXT = 'F:/JDA_BG/CelebA/list_landmarks_celeba.txt'
PRE_NEG_ROOT = 'F:/JDA_BG/CelebA/pre_neg/'
NEG_ROOT = 'F:/JDA_BG/CelebA/neg/'


def adj2square(rect):
    x, y, w, h = rect
    if w < h:
        delta = (h-w) / 2
        x -= delta
        w = h
    elif w > h:
        delta = (w-h) / 2
        y -= delta
        h = w
    return (x, y, w, h)

def scale2twice(rect, shape):
    x, y, w, h = rect
    r, c = shape
    x -= w/2
    y -= h/2
    w *= 2
    h *= 2
    if x < 0:
        x = 0
    if x + w > c-1:
        w = c-x
    if y < 0:
        y = 0
    if y+h > r-1:
        h = r-y
    return (x, y, w, h)


def main():
    bg_patches = [cv2.imread(PRE_NEG_ROOT+_, 0) for _ in os.listdir(PRE_NEG_ROOT)]

    def random_patch(w, h):
        bg_patch = random.choice(bg_patches)
        r, c = bg_patch.shape
        if r < h or c < w:
            return cv2.resize(bg_patch, (w, h))
        else:
            x = random.randint(0, c - w)
            y = random.randint(0, r - h)
            return bg_patch[y:y+h, x:x+w]

    fout = open('face.txt', 'w')
    fout_neg = open('bg.txt', 'w')
    with open(LANDMARK_TXT, 'r') as fin:
        fin.readline()
        fin.readline()
        counter = 0
        for line in fin:
            line = line.strip()
            components = filter(lambda x: x!='', line.split(' '))
            fpath = IMG_ROOT + components[0]
            landmark = np.asarray([int(_) for _ in components[1:]]).reshape((5, 2))
            img = cv2.imread(fpath, 0)
            max_x, max_y = landmark.max(0)
            min_x, min_y = landmark.min(0)

            rect = adj2square((min_x, min_y, max_x-min_x+1, max_y-min_y+1))
            x, y, w, h = scale2twice(rect, img.shape)
            fout.write("%s %d %d %d %d"%(fpath, x, y, w, h))
            for p in landmark:
                fout.write(" %d %d"%(p[0], p[1]))
            fout.write("\n")
            # neg
            # face = img[y:y+h, x:x+w]
            # cv2.imshow('face', face)
            # cv2.waitKey(0)
            img[y:y+h, x:x+w] = random_patch(w, h)
            cv2.imwrite(NEG_ROOT+components[0], img)
            fout_neg.write("%s\n"%(NEG_ROOT+components[0]))

            # process
            counter += 1
            if counter % 100 == 0:
                print "done with %d images" % counter
    fout.close()
    fout_neg.close()

if __name__ == '__main__':
    main()
