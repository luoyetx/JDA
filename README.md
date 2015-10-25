JDA
===

C++ implementation of Joint Cascade Face Detection and Alignment.

### Build

```
$ git clone https://github.com/luoyetx/JDA.git
$ cd JDA
$ mkdir build && cd build
$ cmake ..
$ make
```

If you are on Windows, make sure you have set environment variable `OpenCV_DIR` to OpenCV's build directory like `D:/3rdparty/opencv2.4.11/build`. You may also need [Visual Studio](https://www.visualstudio.com/) to compile the source code. If you are on Linux or Unix, install the development packages of OpenCV via your system's Package Manager like `apt-get` on Ubuntu or `yum` on CentOS. However, Compile the source code of OpenCV will be the best choice of all.

### Data

You should prepare your own data and all data should be under `data` directory. You need two kinds of data, face with landmarks and background images. You also need to create two text file `train.txt` and `nega.txt` (you can change the name of these two text file by editing the code in `common.cpp`).

Every line of `train.txt` stores a face image's path with its landmarks. The number of landmarks can be changed in `common.cpp` and the order of landmarks does not matter.

```
../data/train/00001.jpg x1 y1 x2 y2 ........
../data/train/00002.jpg x1 y1 x2 y2 ........
....
....
```

The face images should be resized to the pre-defined size and you should do any data augmentation by yourself, the code will exactly use the face images you provide.

`nega.txt` is much more simpler. Every line stores where the background image in the file system.

```
../data/nega/000001.jpg
../data/nega/000002.jpg
../data/nega/000003.jpg
....
....
```

Background images should have no face and we will do data augmentation during the hard negative mining.

You may refer to `script/gen.py` for more detail.

### Train

```
$ ./jda train
```

If you are using Visual Studio, make sure you know how to pass command line arguments to the program. The model will be saved to `model` directory. The model file is stored in binary form and I may change the data format later, so training the model on your own risk. However, I will try to make sure that the further code can load the model parameter correctly.

### Attention

This project is not completed yet and may has some hidden bugs. Welcome any question or idea through the [issues](https://github.com/luoyetx/JDA/issues).

### License

BSD 3-Clause

### References

- [Joint Cascade Face Detection and Alignment](http://home.ustc.edu.cn/~chendong/JointCascade/ECCV14_JointCascade.pdf)
- [Face Alignment at 3000 FPS via Regressing Local Binary Features](http://research.microsoft.com/en-us/people/yichenw/cvpr14_facealignment.pdf)
- [FaceDetect/jointCascade_py](https://github.com/FaceDetect/jointCascade_py)
- [luoyetx/face-alignment-at-3000fps](https://github.com/luoyetx/face-alignment-at-3000fps)
- [cjlin1/liblinear](https://github.com/cjlin1/liblinear)
