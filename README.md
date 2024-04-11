Real-Time Monocular Skeleton-Based Hand Gesture Recognition Using 3D-Jointsformer
=======
<!-- 
This repository contains PyTorch implementation for 3D-Jointsformer.
Automatic hand gesture recognition in video sequences has widespread applications, ranging from home automation to sign language interpretation and clinical operations. The primary challenge lies in achieving real-time recognition while managing temporal dependencies that can impact performance. Existing methods employ 3D convolutional or Transformer-based architectures with hand skeleton estimation, but both have limitations. To address these challenges, a hybrid approach that combines 3D Convolutional Neural Networks (3D-CNNs) and Transformers is proposed. The method involves using a 3D-CNN to compute high-level semantic skeleton embeddings, capturing local spatial and temporal characteristics of hand gestures. A Transformer network with a self-attention mechanism is then employed to efficiently capture long-range temporal dependencies in the skeleton sequence. Evaluation of the Briareo and Multimodal Hand Gesture datasets resulted in accuracy scores of 95.49% and 97.25%, respectively. Notably, this approach achieves real-time performance using a standard CPU, distinguishing it from methods that require specialized GPUs. The hybrid approach’s real-time efficiency and high accuracy demonstrate its superiority over existing state-of-the-art methods. In summary, the hybrid 3D-CNN and Transformer approach effectively address real-time recognition challenges and efficient handling of temporal dependencies, outperforming existing methods in both accuracy and speed.
-->
This repository hosts our PyTorch implementation of 3D-Jointsformer, a novel approach for real-time hand gesture recognition in video sequences. Traditional methods struggle with managing temporal dependencies while maintaining real-time performance. To address this, we propose a hybrid approach combining 3D-CNNs and Transformers. Our method utilizes a 3D-CNN to compute high-level semantic skeleton embeddings, capturing local spatial and temporal characteristics. A Transformer network with self-attention then efficiently captures long-range temporal dependencies. Evaluation of the Briareo and Multimodal Hand Gesture datasets yielded accuracy scores of 95.49% and 97.25%. Importantly, our approach achieves real-time performance on standard CPUs, distinguishing it from GPU-dependent methods. The hybrid 3D-CNN and Transformer approach outperforms existing methods in both accuracy and speed, effectively addressing real-time recognition challenges.


### Installation

```shell
conda create -n 3DJointsformer python=3.9 -y
conda activate 3DJointsformer
conda install pytorch=1.11.0 torchvision=0.12.0 cudatoolkit=11.3 -c pytorch -y
pip install 'mmcv-full==1.5.0' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmaction2  # tested mmaction2 v0.24.0
```


### Data Preparation
In this work we have tested the proposed model on two datasets : the [Briareo](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=31) and [Multi-Modal Hand Gesture Dataset](http://gti.ssr.upm.es/data/MultiModalHandGesture_dataset) . The hand keypoints are obtained by [Mediapipe](https://developers.google.com/mediapipe/solutions), we have included code to generate these hand keypoints. 
* You can use this script [**img_to_coord.py**](https://github.com/Enminxo/handgesture_2/blob/c0dd52be999e3dc7525cae5a77620fa0e59de40b/images_to_coordinates/img_to_coord.py) to generate hand keypoints from the images or videos. 
* Save the coordinates into .pkl file using [**my_gen_ntu_rgbd.py**](https://github.com/Enminxo/handgesture_2/blob/2831f468f56c986c19f3f183a93a1e1942d685a5/tools/data/my_gen__ntu_rgbd.py)



### Train

You can use the following command to train a model.

```shell
./tools/run.sh ${CONFIG_FILE} ${GPU_IDS} ${SEED}
```

Example: train the model on the joint data of Briareo dataset using 2 GPUs with seed 0.

```shell
./tools/run.sh configs/transformer/jointsformer3d_briareo.py 0,1 0
```

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: inference on the joint data of Briareo dataset.

```shell
python tools/test.py configs/transformer/jointsformer3d_briareo.py \
    work_dirs/jointsformer3d/best_top1_acc_epoch_475.pth \
    --eval top_k_accuracy --cfg-options "gpu_ids=[0]"
```

### Bibtex
If this project is useful for you, please consider citing our paper.
```shell
@Article{s23167066,
AUTHOR = {Zhong, Enmin and del-Blanco, Carlos R. and Berjón, Daniel and Jaureguizar, Fernando and García, Narciso},
TITLE = {Real-Time Monocular Skeleton-Based Hand Gesture Recognition Using 3D-Jointsformer},
JOURNAL = {Sensors},
VOLUME = {23},
YEAR = {2023},
NUMBER = {16},
ARTICLE-NUMBER = {7066},
URL = {https://www.mdpi.com/1424-8220/23/16/7066},
PubMedID = {37631602},
ISSN = {1424-8220},
DOI = {10.3390/s23167066}
}
```

### Acknowledgements
Our code is based on [SkelAct](https://github.com/hikvision-research/skelact) , [MMAction2](https://github.com/open-mmlab/mmaction2/) , [SlowFast](https://github.com/facebookresearch/SlowFast/tree/2090f2918ac1ce890fdacd8fda2e590a46d5c734) Sincere thanks to their wonderful works.

### License

This project is released under the [Apache 2.0 license](LICENSE).
