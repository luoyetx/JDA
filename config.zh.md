config.zh.md
=========

config 文件的注解。

```
{
    "T": 5, // stage 的个数
    "K": 1080, // 每个 stage 中分类回归树的个数
    "landmark_n": 27, // 关键点数目
    "tree_depth": 4, // 分类回归树的深度
    "random_shift": 0.02, // 对每个样本的初始形状做随机偏移的振幅
    "image_size": {
        "multi_scale": true, // 是否使用多尺度
        "origin_size": 64, // 原始大小
        "half_size": 46, // 二分之一大小
        "quarter_size": 32 // 四分之一大小
    },
    "hard_negative_mining": {
        "mining_th": [0.5, 0.5, 0.5, 0.5, 0.5], // 当负样本比例小于这个阈值时，开始负样本挖掘
        "min_size": 32, // 没有用到
        "factor": 1.3, // 没有用到
        "step_ratio": 0.5 // 没有用到
    },
    "stages": { // 每个 stage 都设置
        "feature_pool_size": [2000, 2000, 2000, 2000, 2000], // 随机生成的特征个数
        "random_sample_radius": [0.3, 0.2, 0.15, 0.12, 0.1], // 生成特征点时采样的半径
        "classification_p": [0.9, 0.8, 0.7, 0.6, 0.5], // 树中节点做分类的概率
        "recall": [0.99, 0.99, 0.99, 0.99, 0.99], // 没有用到
        "drop_n": [1, 1, 1, 1, 1], // 树在计算 score 阈值时，丢弃掉的正样本个数
        "neg_pos_ratio": [1.0, 1.0, 1.0, 1.0, 1.0], // 训练时负样本与正样本的比例
        "score_normalization_step": [5, 5, 5, 5, 5] // 每隔 step*landmark_n 棵树，会对训练数据的 score 做归一化操作，如果不想做这个操作，请把 step 的值设大，比如 100
    },
    "data": {
        "use_hard": false, // 是否准备了初始负样本
        "face": "../data/face.txt", // 人脸 text文件
        "background": ["../data/hd.txt", "../data/background1.txt", "../data/background2.txt"], // 第一项的初始负样本的 text 文件，如果你没有准备初始负样本，也不要删掉这一项。后面的全是背景图的 text 文件，代码会把它们的内容合并起来
        "test": "../data/test.txt" // 简单测试时的 text 文件
    },
    "fddb": {
        "dir": "../data/fddb", // fddb 数据的目录
        "out": false, // 是否将检测结果输出到文件系统中
        "nms": true, // 是否开启 nms 操作
        "draw_score": true, // 是否在结果上画 score 值
        "draw_shape": true, // 是否在结果上画 shape
        "minimum_size": 20, // 检测时的人脸框最小大小
        "step": 5, // 滑动窗口的 step
        "scale": 1.3, // 图像缩放时的 scale
        "overlap": 0.3, // nms 的 overlap 参数
        "method": 0 // 选择哪种检测方式，可以是 0 或者 1
    },
    "cart": {
        "restart": {
            "on": false, // 是否在训练时开启 restart 操作，当弱分类器（分类回归树）丢弃的负样本太少时，重新训练弱分类器
            "th": [0.0025, 0.0025, 0.0025, 0.0025, 0.0025], // 丢弃负样本比例的阈值
            "times": 5 // restart 最多的次数，restart 太多次之后，会选择最好的一个弱分类器
        }
    },
    "face": {
        "online_augment": false, // 是否将训练数据的人脸做翻转
        "symmetric_landmarks": { // 翻转人脸同时训练翻转对称的关键点
            "offset": 1, // 关键点下标的偏移量，方便后书写
            "left": [1, 2, 5, 6, 7, 8, 9, 19, 22], // 左边对称点下标
            "right": [4, 3, 12, 11, 10, 13, 14, 21, 23] // 右边对称点下标
        },
        "pupils": { // 瞳孔
            "offset": 1, // 同上
            "left": [9], // 左边瞳孔的下标，如果没有瞳孔点，可以用多个点来代表瞳孔点
            "right": [14] // 右边瞳孔点的下标
        },
        "similarity_transform": false // 是否做相似性变换
    },
    "resume_model": "../model/jda_xxx.model", // 没有用到
    "snapshot_iter": 600 // 每隔多少棵树暂存下当前的模型和数据
}
```
