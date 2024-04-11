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

The dataset (images) is downloaded from [GTI- Grupo Tratamiento de Imágenes](http://gti.ssr.upm.es/data/MultiModalHandGesture_dataset)
* To run the script [**img_to_coord.py**](https://github.com/Enminxo/handgesture_2/blob/c0dd52be999e3dc7525cae5a77620fa0e59de40b/images_to_coordinates/img_to_coord.py) you should place the downloaded dataset in the folder [images_to_coordinates](https://github.com/Enminxo/handgesture_2/blob/6d639c08f6dbfec4b820d67f9e61916d04cb2e4a/images_to_coordinates)
* transform the coordinates to .pkl file using [**my_gen_ntu_rgbd.py**](https://github.com/Enminxo/handgesture_2/blob/2831f468f56c986c19f3f183a93a1e1942d685a5/tools/data/my_gen__ntu_rgbd.py)
----
From this point onwards, we use the "dataset" to refer the hand joints coordinates in .pkl format.
* **_media_test.py_**: script to run the real-time hand gesture recognition.
  * Set the paths to the dataset and the trained weights,
  * The script will open the webcam to inference the hand gesture in real time, so make sure this is available when you run the script
  
Use [gen_ntu_rgbd_raw.py](tools/data/my_gen_ntu_rgbd_raw.py) to preprocess the NTU RGB+D dataset. Put the dataset in `data/` with the following structure.

```
data/
└── ntu
    └── nturgb+d_skeletons_60_3d
        ├── xsub
        │   ├── train.pkl
        │   └── val.pkl
        └── xview
            ├── train.pkl
            └── val.pkl
```

### Train

You can use the following command to train a model.

```shell
./tools/run.sh ${CONFIG_FILE} ${GPU_IDS} ${SEED}
```

Example: train TSCNN model on the joint data of NTU RGB+D using 2 GPUs with seed 0.

```shell
./tools/run.sh configs/tscnn/tscnn_ntu60_xsub_joint.py 0,1 0
```

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSCNN model on the joint data of NTU RGB+D.

```shell
python tools/test.py configs/tscnn/tscnn_ntu60_xsub_joint.py \
    work_dirs/tscnn_ntu60_xsub_joint/best_top1_acc_epoch_475.pth \
    --eval top_k_accuracy --cfg-options "gpu_ids=[0]"
```

### Bibtex
If this project is useful for you, please consider citing our paper 
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
Our code is based on [SkelAct](https://github.com/hikvision-research/skelact),[MMAction2](https://github.com/open-mmlab/mmaction2/),[SlowFast](https://github.com/facebookresearch/SlowFast/tree/2090f2918ac1ce890fdacd8fda2e590a46d5c734) Sincere thanks to their wonderful works.

### License

This project is released under the [Apache 2.0 license](LICENSE).
