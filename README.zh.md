JDA
===

这个项目旨在复现 Joint Cascade Face Detection and Alignment 这篇论文。

### 获取代码

我强烈建议你通过 [Git](https://git-scm.com/) 软件来获取代码。如果你对 Git 不是很熟悉希望了解下，这个[教程](https://git-scm.com/book/en/v2)或许会帮助到你。

你可以通过以下的 Git 命令获取源代码

```
$ git clone --recursive https://github.com/luoyetx/JDA.git
```

或者

```
$ git clone https://github.com/luoyetx/JDA.git
$ cd JDA
$ git submodule update --init
```

如果你是直接从 Github 上下载 zip 压缩包，你同时需要从 [luoyetx/liblinear][luoyetx/liblinear] 和 [luoyetx/jsmnpp][luoyetx/jsmnpp] 这两个项目下载 zip 压缩包，然后把代码解压到 3rdparty 目录下。liblinear 这个库用来做回归，jsmnpp 这个库用来解析 json 配置文件。

### 构建代码

项目使用 [CMake][cmake] 来构建代码，同时依赖 [OpenCV][opencv] 库。Windows 平台下需要设置 `OpenCV_DIR` 环境变量，指出 OpenCV 的安装路径，比如 `D:/3rdparty/opencv2.4.11/build`，同时需要 [Visual Studio][vs] 来编译代码。如果是在 Linux/Unix 下，可以使用包管理工具安装 OpenCV 或者直接源码编译。

下面的命令可以用于 Linux 下的构建，强烈建议创建 build 目录，将编译的中间产物和结果都和源代码隔离开来，Windows 平台下也是如此。

```
$ cd JDA
$ mkdir build && cd build
$ cmake ..
$ make
```

同时有一份 Windows 平台下的[编译指南](http://pan.baidu.com/s/1mhWXlqw)。

### 配置文件

项目使用 `config.json` 这个文件来配置算法参数，`config.template.json` 是一个模板，由于 json 格式的限制，json 文件中无法添加注释。关于文件路径的注意点，请尽量不要出现非 ASCII 码字符，同时不要有空格，路径分隔符最好全部使用 `/`。

### 训练数据

准备训练数据，这个确实很痛苦。大概需要两种图片数据，人脸和背景图。人脸同时需要附带关键点信息。人脸数据写在一个 text 文件中，然后在 config.json 中指出文件的路径，这样算法就可以加载到人脸数据了。text 文件中一行表示一张人脸，第一项是图片路径，后面紧接在人脸框在图片中的位置信息，后面是所有关键点在图片中的位置信息。关键点的个数需要在 config.json 中设置。同时算法可以支持没有关键点的人脸数据，把所有 x, y 的值设置成 -1 就行了。text 文件的格式如下。

```
../data/face/00001.jpg bbox_x bbox_y bbox_w bbox_h x1 y1 x2 y2 ........
../data/face/00002.jpg bbox_x bbox_y bbox_w bbox_h x1 y1 x2 y2 ........
....
....
```

训练时可以对人脸做翻转操作来增加数据量，需要在 config 中开启，同时因为翻转人脸同时需要翻转对称的关键点，还需要在 config 中指出哪些关键点的位置是相互对称的。如果人脸框超出了图片的边框，算法会将多余部分用 0 填充。

背景图的 text 比较简单，一行一个图片路径，背景图中最好不要出现人脸。

```
../data/bg/000001.jpg
../data/bg/000002.jpg
../data/bg/000003.jpg
....
....
```

背景图用来生成负样本，当负样本不足时，算法会从背景图中挖掘负样本，将算法分错的负样本（难例）添加到训练数据中。

为了减少图片数据加载的时间开销，代码会在初始训练数据生成完后将数据导出到文件系统中，所有中间导出的数据都在 `data/dump` 目录下，每个文件都会带有时间戳。同时算法运行时会先去找 `data/jda_train_data.data` 文件，如果存在，则直接加载训练数据，不会再去读取 face.txt 文件中的数据，也就不会再去加载原始图片，这样可以节省不少加载的时间。

我共享了自己收集的数据，如果你感兴趣请参看这个 [issue][jda-data]。

### 训练

Linux 下直接运行下面的命令

```
$ ./jda train
```

Visual Studio 下需要设置命令行参数，具体操作请自行搜索。训练过程中产生的模型文件均在 `model` 目录下。

### 模型文件

模型文件是一个二进制文件，参数类型主要有两种，4 字节 `int` 和 8 字节 `double`，同时注意下自己 CPU 的大小端。

```
|-- mask (int)
|-- meta
|    |-- T (int)
|    |-- K (int)
|    |-- landmark_n (int)
|    |-- tree_depth (int)
|    |-- current_stage_idx (int) // training status
|    |-- current_cart_idx (int)
|-- mean_shape (double, size = 2*landmark_n)
|-- stages
|    |-- stage_1
|    |    |-- cart_1
|    |    |-- cart_2
|    |    |-- ...
|    |    |-- cart_K
|    |    |-- global regression weight
|    |-- stage_2
|    |-- ...
|    |-- stage_T
|-- mask (int)
```

更多模型文件的细节可以参考这两个文件 `cascador.cpp` 和 `cart.cpp`。

### FDDB Benchmark

[FDDB][fddb] 数据集经常用于人脸检测算法的评价, 将数据下载解压到 `data` 目录下。

```
|-- data
|    |-- fddb
|         |-- images
|         |    |-- 2002
|         |    |-- 2003
|         |-- FDDB-folds
|         |    |-- FDDB-fold-01.txt
|         |    |-- FDDB-fold-01-ellipseList.txt
|         |    |-- ....
|         |-- result
```

准备好数据和模型，在 config 中设置检测参数，运行测试。生成的 text 文件可以给这个评价[程序]([npinto/fddb-evaluation][npinto/fddb-evaluation])使用。

```
$ ./jda fddb
```

### QQ 群

欢迎加入 QQ 群 347185749 来讨论人脸相关的算法。

### License

BSD 3-Clause

### References

- [Joint Cascade Face Detection and Alignment](http://home.ustc.edu.cn/~chendong/JointCascade/ECCV14_JointCascade.pdf)
- [Face Alignment at 3000 FPS via Regressing Local Binary Features](http://research.microsoft.com/en-us/people/yichenw/cvpr14_facealignment.pdf)
- [FaceDetect/jointCascade_py](https://github.com/FaceDetect/jointCascade_py)
- [luoyetx/face-alignment-at-3000fps](https://github.com/luoyetx/face-alignment-at-3000fps)
- [cjlin1/liblinear](https://github.com/cjlin1/liblinear)
- [luoyetx/jsmnpp](https://github.com/luoyetx/jsmnpp)


[opencv]: http://opencv.org/
[luoyetx/jsmnpp]: https://github.com/luoyetx/jsmnpp
[luoyetx/liblinear]: https://github.com/luoyetx/liblinear
[cmake]: https://cmake.org/
[vs]: https://www.visualstudio.com/
[endianness]: https://en.wikipedia.org/wiki/Endianness
[qq]: http://im.qq.com/
[fddb]: http://vis-www.cs.umass.edu/fddb/
[npinto/fddb-evaluation]: https://github.com/npinto/fddb-evaluation
[jda-data]: https://github.com/luoyetx/JDA/issues/22
