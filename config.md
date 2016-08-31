config.md
=========

explain the config.

```
{
    "T": 5, // number of stages
    "K": 1080, // number of carts in every stage
    "landmark_n": 27, // number of landmarks
    "tree_depth": 4, // depth of a cart
    "random_shift": 0.02, // random shift to the initial shape of every face/non-face
    "image_size": {
        "multi_scale": true, // whether use multi scale or not
        "origin_size": 64, // original size of a face
        "half_size": 46, // half size of a face
        "quarter_size": 32 // quarter size of a face
    },
    "hard_negative_mining": {
        "mining_th": [0.5, 0.5, 0.5, 0.5, 0.5], // if negative sample ratio < mining_th, hard mining starts, this value can be set for every stage
        "min_size": 32, // not used
        "factor": 1.3, // not used
        "step_ratio": 0.5 // not used
    },
    "stages": { // these values are set for every stage
        "feature_pool_size": [2000, 2000, 2000, 2000, 2000], // feature pool size
        "random_sample_radius": [0.3, 0.2, 0.15, 0.12, 0.1], // max offset of generated feature points
        "classification_p": [0.9, 0.8, 0.7, 0.6, 0.5], // possibility of a node to do classification
        "recall": [0.99, 0.99, 0.99, 0.99, 0.99], // not used
        "drop_n": [1, 1, 1, 1, 1], // number of positive samples to drop when calculate score threshold for a cart
        "neg_pos_ratio": [1.0, 1.0, 1.0, 1.0, 1.0], // negative / positive ratio
        "score_normalization_step": [5, 5, 5, 5, 5] // normalize the training data's score distribution after training step*landmark_n carts
    },
    "data": {
        "use_hard": false, // wheter to use initial negative samples
        "face": "../data/face.txt", // text file for face
        "background": ["../data/hd.txt", "../data/background1.txt", "../data/background2.txt"], // the first text file is for initial negative samples (never should remove this item even if you don't use initial negative samples), text file followed are all background text file, they will be merged
        "test": "../data/test.txt" // test text file
    },
    "fddb": {
        "dir": "../data/fddb", // fddb dataset
        "out": false, // output the result image
        "nms": true, // turn on nms
        "draw_score": true, // draw face score on result image
        "draw_shape": true, // draw face shape on result image
        "minimum_size": 20, // minimun face size when detection
        "step": 5, // sliding window step
        "scale": 1.3, // image pyramid scale
        "overlap": 0.3, // nms overlap
        "method": 0 // detection method, can be 0 or 1
    },
    "cart": {
        "restart": {
            "on": false, // whether to restart training a cart when it drops little negative samples
            "th": [0.0025, 0.0025, 0.0025, 0.0025, 0.0025], // drop ratio threshold
            "times": 5 // maximum restart times
        }
    },
    "face": {
        "online_augment": false, // whether to flip the training face
        "symmetric_landmarks": {
            "offset": 1, // landmark index offset
            "left": [1, 2, 5, 6, 7, 8, 9, 19, 22], // left symmetric landmarks
            "right": [4, 3, 12, 11, 10, 13, 14, 21, 23] // right symmetric landmarks
        },
        "pupils": {
            "offset": 1,
            "left": [9], // left pupils, if there's no pupil landmark, use some landmarks to calculate pupils
            "right": [14] // right pupils
        }
    },
    "resume_model": "../model/jda_xxx.model", // not used
    "snapshot_iter": 600 // snapshot after 600 carts
}
```
