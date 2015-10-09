#!/usr/bin/env python2.7
import os
from os.path import exists
from random import shuffle
import cv2


TRAIN = 'data/cuhk/train.txt'
TEST = 'data/cuhk/test.txt'
NEGA = 'data/nega'


def load_txt(txt):
    data_points = []
    with open(txt, 'r') as fd:
        for line in fd.readlines():
            line = line.strip()
            components = line.split(' ')
            img_path = components[0]
            # bounding box, (left, right, top, bottom)
            bbox = [int(_) for _ in components[1:5]]
            landmark = [float(_) for _ in components[5:]]
            data_points.append((img_path, bbox, landmark))
    return data_points

def main():
    assert(exists(TRAIN) and exists(TEST) and exists(NEGA))
    # train
    data_points = load_txt(TRAIN)
    train = []
    for img_path, bbox, landmark in data_points:
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        left, right, top, bottom = bbox
        face = img[top:bottom+1, left:right+1]
        face_o = cv2.resize(face, (80, 80))
        w, h = right-left, bottom-top
        x_t = lambda x: (x-left)*80.0 / w
        y_t = lambda y: (y-top)*80.0 / h
        landmark_o = [x_t(landmark[0]), y_t(landmark[1]),
                      x_t(landmark[2]), y_t(landmark[3]),
                      x_t(landmark[4]), y_t(landmark[5]),
                      x_t(landmark[6]), y_t(landmark[7]),
                      x_t(landmark[8]), y_t(landmark[9]),]
        train.append((face_o, landmark_o))
        face_f = cv2.flip(face_o, 1) # flip
        landmark_f = [80-x_t(landmark[2]), y_t(landmark[3]),
                      80-x_t(landmark[0]), y_t(landmark[1]),
                      80-x_t(landmark[4]), y_t(landmark[5]),
                      80-x_t(landmark[8]), y_t(landmark[9]),
                      80-x_t(landmark[6]), y_t(landmark[7]),]
        train.append((face_f, landmark_f))
    # random shuffle
    shuffle(train)
    with open('data/train.txt', 'w') as fd:
        idx = 0
        for face, landmark in train:
            idx += 1
            print 'Save %05d.jpg'%idx
            fn = '../data/train/%05d.jpg'%idx
            cv2.imwrite(fn, face)
            fd.write(fn)
            for _ in landmark:
                fd.write(' %lf'%_)
            fd.write('\n')

    # test
    data_points = load_txt(TEST)
    test = []
    for img_path, bbox, landmark in data_points:
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        left, right, top, bottom = bbox
        face = img[top:bottom+1, left:right+1]
        face_o = cv2.resize(face, (80, 80))
        w, h = right-left, bottom-top
        x_t = lambda x: (x-left)*80.0 / w
        y_t = lambda y: (y-top)*80.0 / h
        landmark_o = [x_t(landmark[0]), y_t(landmark[1]),
                      x_t(landmark[2]), y_t(landmark[3]),
                      x_t(landmark[4]), y_t(landmark[5]),
                      x_t(landmark[6]), y_t(landmark[7]),
                      x_t(landmark[8]), y_t(landmark[9]),]
        test.append((face_o, landmark_o))
    # random shuffle
    shuffle(test)
    with open('data/test.txt', 'w') as fd:
        idx = 0
        for face, landmark in test:
            idx += 1
            print 'Save %05d.jpg'%idx
            fn = '../data/test/%05d.jpg'%idx
            cv2.imwrite(fn, face)
            fd.write(fn)
            for _ in landmark:
                fd.write(' %lf'%_)
            fd.write('\n')
    # negative
    with open('data/nega.txt', 'w') as fd:
        fs = os.listdir(NEGA)
        shuffle(fs)
        for f in fs:
            fd.write('../data/nega/%s\n'%f)


if __name__ == '__main__':
    main()
