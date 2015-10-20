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

If you are on Windows, make sure you have set environment variable `OpenCV_DIR` to OpenCV's build directory like `D:/3rdparty/opencv2.4.11/build`. If you are on Linux or Unix, install the development packages of OpenCV via your system's Package Manager like `apt-get` on Ubuntu or `yum` on CentOS. However, Compile the source code of OpenCV will be the best choice of all.

### Attention

This project is not completed yet, welcome any question or idea through the [issues](https://github.com/luoyetx/JDA/issues).

### License

BSD 3-Clause

### References

- [Joint Cascade Face Detection and Alignment](http://home.ustc.edu.cn/~chendong/JointCascade/ECCV14_JointCascade.pdf)
- [Face Alignment at 3000 FPS via Regressing Local Binary Features](http://research.microsoft.com/en-us/people/yichenw/cvpr14_facealignment.pdf)
- [FaceDetect/jointCascade_py](https://github.com/FaceDetect/jointCascade_py)
- [luoyetx/face-alignment-at-3000fps](https://github.com/luoyetx/face-alignment-at-3000fps)
- [cjlin1/liblinear](https://github.com/cjlin1/liblinear)
