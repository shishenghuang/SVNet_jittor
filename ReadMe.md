

# Supervoxel-CNN with [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/)

This is the Supervoxel-CNN network impremented by [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/). Supervoxel-CNN is the backbone network used in the SVNet system, which is an online 3D semantic segmentation approach. SVNet is a online 3D semantic segmentation system, which contains both a online 3D reconstruction system and online 3D semantic prediction system (Supervoxel-CNN in this part). Here we only provide the Supervoxel-CNN part to show how we implement the network we used in the paper. If you are interesting with the whole system, please contact [Shi-Sheng Huang](https://shishenghuang.github.io/index/). For more details about SVNet, please check our paper 

```
@article{SupervoxelConv2021,
author = {Shi{-}Sheng Huang and Ze{-}Yu Ma and Tai{-}Jiang Mu and Hongbo Fu and Shi{-}Min Hu},
title = {Supervoxel Convolution for Online 3D Semantic Segmentation},
journal = {ACM Transactions on Graphics},
volume = {40},
number = {3},
article = {34}
year = {2021}
}
```

# network.py 

This file contains the Supervoxel-CNN network

# dataset.py 

This file contains the data preparation method, the input is the training data we have preprocessed described in the paper. For details, please contact [Shi-Sheng Huang](https://shishenghuang.github.io/index/)

# loss.py 

This file contains the weighted_cross_entropy we used to train Supervoxel-CNN

# train_val.py 

This file contains the method to train and evaluate Supervoxel-CNN

