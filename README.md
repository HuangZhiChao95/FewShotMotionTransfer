# Few-Shot Human Motion Transfer by Personalized Geometry and Texture Modeling
Pytorch implementation of our paper "Few-Shot Human Motion Transfer by Personalized Geometry and Texture Modeling". 

[[Paper](https://arxiv.org/abs/2103.14338)] [[Video](https://youtu.be/ZJ15X-sdKSU)]


## Requirements

The code is tested under python3.6 and Ubuntu 18.04. Please run the following command to install package.
```
pip install -r requirements.txt
```

If you want to generate your own data for training and inference, you also need to install [youtube-dl](https://youtube-dl.org/), [ffmpeg](https://www.ffmpeg.org/), [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), [Densepose](https://github.com/facebookresearch/DensePose) and the human segmentation model. We use a commercial human segmentation model that can not be released. Such models can also be found at github: [Human-Segmentation-Pytorch](https://github.com/thuyngch/Human-Segmentation-PyTorch).


## Training

### Preparing the Data


1. We provide the links for downloading and processing the videos from youtube. Note that we do not hold the copyright of these videos and they may be deleted from youtube at any time. Videos can be downloaded and splitted into frames with the following command
```
python data_preprocess/download_video.py /path/to/data-root
```
2. Then you should use [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), [Densepose](https://github.com/facebookresearch/DensePose) and [Human-Segmentation-Pytorch](https://github.com/thuyngch/Human-Segmentation-PyTorch) to generate the keypoint, densepose and foreground segmentation of the human. After generating  the keypoint, densepose and foreground segmentation of the human, The structure of the data should be 

```
├── data-train
│   ├──001                      #any name is okay
│   │  ├── image                #folder for storing image
│   │  ├── body                 #folder for storing pose
│   │  ├── texture              #folder for storing texture input
│   │  ├── segmentation         #folder for storing foreground mask of the human
│   │  ├── densepose            #folder for storing densepose
│   │  ├── background.png       #the background image
│   │  └── image_list.txt       #list of all images in the folder
│   ├──002                      #any name is okay
│   │  ├── image                #folder for storing image
│   │  ├── body                 #folder for storing pose
│   │  ├── texture              #folder for storing texture input
│   │  ├── segmentation         #folder for storing foreground mask of the human
│   │  ├── densepose            #folder for storing densepose
│   │  ├── background.png       #the background image
│   │  └── image_list.txt       #list of all images in the folder
|   ├──...

```
3. Run the following script to generate additional data for training
```
python data_preprocess/UVToTexture.py /path/to/data-root
python data_preprocess/merge_background.py /path/to/data-root
python data_preprocess/connect_body.py /path/to/data-root
```

4. Generate *image_list.txt* with
```
python data_preprocess/generate_list.py /path/to/data-root
```

5. Split the dataset into training set and validation set with
```
python data_preprocess/split_train_test.py --root /path/to/data-root --train /path/to/data-train --test /path/to/data-test --n_train 50
```
Note that we randomly shuffle the training data and the test data, it may not be the same as the split in the paper.

6. Generate list for finetunes samples
```
python data_preprocess/generate_finetune_list.py /path/to/data-test --n_finetune 20
```

### Initialization

Change the *dataroot* in *config/config_pretrain.yaml* and *config/config_pretrain_texture.yaml* to */path/to/data-train* and run
```
python pretrain.py config/config_pretrain.yaml --device 0 1
python pretrain_texture.py config/config_pretrain_texture.yaml --device 0 1
```
You may change the number of gpus by selecting different index of gpu after *--device*. And the *batchsize* in the config file may also need to be changed according to your memory of gpu.

### Multi-video Training

Make sure you have run *pretrain.py* and *pretrain_texture.py*. Then run the following command to perform multi-video training

```
python train.py config/config_train.yaml --device 0 1
```

## Finetune and Inference

### Preparing data for inference and finetune

You can also use your own data for generating human motion transfer. The folder structure of the data is:

```
├── ...
├── dataset-test-source
│   ├── image                       #folder for storing image
│   ├── body                        #folder for storing pose
│   ├── texture                     #folder for storing texture input
│   ├── segmentation                #folder for storing foreground mask of the human
│   ├── densepose                   #folder for storing densepose
│   ├── background.png              #the background image
│   ├── finetune_samples.txt        #list of images used for finetuning
│   └── image_list.txt              #list of all images in the folder
├── dataset-test-target
│   ├── image
│   ├── body
│   ├── texture
│   ├── segmentation
│   ├── densepose
│   ├── background.png
│   ├── finetune_samples.txt
│   └── image_list.txt
```

I provide one example of the dataset structure on *dataset-test-example* and pretrained weights are available from [Google Drive](https://drive.google.com/drive/folders/1w8Vkfdto13gDzo7X8NkDps1GGRbjJXNd?usp=sharing). Please save the pretrained weights under *checkpoint/selfatt*. The generated videos will be saved under *video/selfatt*

### Inference with Fine-tuning

```
python finetune.py config/config_finetune.yaml --source /path/to/source-person --target /path/to/target-pose --epochs 300
```

### Inference without Fine-tuning

```
python finetune.py config/config_finetune.yaml --source /path/to/source-person --target /path/to/target-pose --epochs 0
```

## Citation
```
@inproceedings{huang2021few,
  title={Few-Shot Human Motion Transfer by Personalized Geometry and Texture Modeling},
  author={Huang, Zhichao and Han, Xintong and Xu, Jia and Zhang, Tong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```