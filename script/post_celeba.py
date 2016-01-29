#!/usr/bin/env python2.7
import os
import random
import cv2
import numpy as np


IMG_ROOT = 'F:/JDA_BG/CelebA/img_celeba/img_celeba/'
LANDMARK_TXT = 'results.80k.txt' # result from TCDCN


def get_bbox(size, landmark):
    img_h, img_w = size
    x_max, y_max = landmark.max(0)
    x_min, y_min = landmark.min(0)
    w = x_max - x_min
    h = y_max - y_min
    w = h = max(w, h)
    x = x_min
    y = y_min

    ratio = 0.2
    x_new = x - w*ratio
    y_new = y - h*ratio
    w_new = w*(1+2*ratio)
    h_new = h*(1+2*ratio)
    if x_new < 0 or y_new < 0 or x_new+w_new > img_w or y_new+h_new > img_h:
        # print x_new, y_new, x_new+w_new, y_new+h_new, img_w, img_h
        # x_new = max(x_new, 0)
        # y_new = max(y_new, 0)
        # if x_new+w_new > img_w: w_new = img_w - x_new
        # if y_new+h_new > img_h: h_new = img_h - y_new
        return (0, 0, 0, 0)
    return (x_new, y_new, w_new, h_new)


def main():
    fout = open('face.txt', 'w')
    with open(LANDMARK_TXT, 'r') as fin:
        counter = 0
        while True:
            line = fin.readline()
            if line == '': break
            fpath = IMG_ROOT+line.strip()
            line = fin.readline().strip()

            components = filter(lambda x: x!='', line.split(' '))
            landmark_68 = np.asarray([float(_) for _ in components]).reshape((68, 2))
            landmark_27 = np.zeros((27, 2))

            # select landmarks from 68
            landmark_27[1 - 1] = landmark_68[18 - 1]
            landmark_27[2 - 1] = landmark_68[20 - 1]
            landmark_27[3 - 1] = landmark_68[22 - 1]
            landmark_27[4 - 1] = landmark_68[23 - 1]
            landmark_27[5 - 1] = landmark_68[25 - 1]
            landmark_27[6 - 1] = landmark_68[26 - 1]
            landmark_27[7 - 1] = landmark_68[37 - 1]
            landmark_27[8 - 1] = (landmark_68[38 - 1]+landmark_68[39 - 1]) / 2.
            landmark_27[9 - 1] = landmark_68[40 - 1]
            landmark_27[10 - 1] = (landmark_68[41 - 1]+landmark_68[42 - 1]) / 2.
            landmark_27[11 - 1] = (landmark_68[38 - 1]+landmark_68[39 - 1]+landmark_68[41 - 1]+landmark_68[42 - 1]) / 4.
            landmark_27[12 - 1] = landmark_68[43 - 1]
            landmark_27[13 - 1] = (landmark_68[44 - 1]+landmark_68[45 - 1]) / 2.
            landmark_27[14 - 1] = landmark_68[46 - 1]
            landmark_27[15 - 1] = (landmark_68[47 - 1]+landmark_68[48 - 1]) / 2.
            landmark_27[16 - 1] = (landmark_68[44 - 1]+landmark_68[45 - 1]+landmark_68[47 - 1]+landmark_68[48 - 1]) / 4.
            landmark_27[17 - 1] = landmark_68[29 - 1]
            landmark_27[18 - 1] = landmark_68[31 - 1]
            landmark_27[19 - 1] = landmark_68[32 - 1]
            landmark_27[20 - 1] = landmark_68[34 - 1]
            landmark_27[21 - 1] = landmark_68[36 - 1]
            landmark_27[22 - 1] = landmark_68[49 - 1]
            landmark_27[23 - 1] = landmark_68[55 - 1]
            landmark_27[24 - 1] = landmark_68[52 - 1]
            landmark_27[25 - 1] = landmark_68[63 - 1]
            landmark_27[26 - 1] = landmark_68[67 - 1]
            landmark_27[27 - 1] = landmark_68[58 - 1]


            img = cv2.imread(fpath, 0)
            x, y, w, h = get_bbox(img.shape, landmark_27)
            if x == 0 and y == 0 and w == 0 and h == 0:
                continue

            fout.write("%s %d %d %d %d"%(fpath, x, y, w, h))
            for p in landmark_27:
                fout.write(" %lf %lf"%(p[0], p[1]))
            fout.write("\n")

            # # face
            # for p in landmark_27:
            #     cv2.circle(img, (int(p[0]), int(p[1])), 2, (0,0,0), -1)
            # face = img[y:y+h, x:x+w]
            # cv2.imshow('face', face)
            # cv2.waitKey(0)

            # process
            counter += 1
            if counter % 100 == 0:
                print "done with %d images" % counter
    fout.close()

if __name__ == '__main__':
    main()
