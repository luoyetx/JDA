#!/usr/bin/env python2.7
import os
from os.path import exists
from random import shuffle
import cv2


def load_txt(txt):
    data_points = []
    with open(txt, 'r') as fd:
        for line in fd.readlines():
            line = line.strip()
            components = line.split(' ')
            img_path = components[0]
            # bounding box, (left, right, top, bottom)
            bbox = [int(_) for _ in components[1:5]]
            w, h = bbox[1]-bbox[0], bbox[3]-bbox[2]
            w = h = max(w, h)
            bbox[1], bbox[3] = bbox[0]+w, bbox[2]+h
            landmark = [float(_) for _ in components[5:]]
            data_points.append((img_path, bbox, landmark))
    return data_points

def main():
    TRAIN = 'data/cuhk/train.txt'
    assert(exists(TRAIN))
    # train
    data_points = load_txt(TRAIN)
    train = []
    for img_path, bbox, landmark in data_points:
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        left, right, top, bottom = bbox
        face = img[top:bottom+1, left:right+1]
        x_t = lambda x: x-left
        y_t = lambda y: y-top
        landmark = [x_t(landmark[0]), y_t(landmark[1]),
                    x_t(landmark[2]), y_t(landmark[3]),
                    x_t(landmark[4]), y_t(landmark[5]),
                    x_t(landmark[6]), y_t(landmark[7]),
                    x_t(landmark[8]), y_t(landmark[9]),]
        train.append((face, landmark))
    # random shuffle
    shuffle(train)
    with open('data/face.txt', 'w') as fd:
        idx = 0
        for face, landmark in train:
            idx += 1
            print 'Save %05d.jpg'%idx
            fn = '../data/face/%05d.jpg'%idx
            cv2.imwrite('data/face/%05d.jpg'%idx, face)
            fd.write(fn)
            for _ in landmark:
                fd.write(' %lf'%_)
            fd.write('\n')


if __name__ == '__main__':
    main()
