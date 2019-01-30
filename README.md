author: YadongWei(github:https://github.com/XJTUWYD) & ZhijunTU(github:https://github.com/Ironteen) from XJTU if you have any questions please contact use with the authors email(yadongwei2@gmail.com)

## Tips:
this work is based on https://github.com/BichenWuUCB/squeezeDet, what different is that:
- We changed squeezenet part in squeedet into shufflenetv2(we changed the filter number of each layer to multiples of 8 to keep it hardware friendly) which is more efficient.
- We build the classification task on ImageNet so you can pretrain the shufflenetv2 by yourself.
- We do the quantization of parameters and activitions and we fusion the BN layer into convolution layer, so you don't need BN when inference.


## Installation:
  ```Shell
  git clone https://github.com/XJTUWYD/shuffledet
  ```
- Use pip to install required Python packages:
    ```Shell
    pip install -r requirements.txt
    ```
## Training shuffleNet(pretrain):
    ```Shell
    ./train_shuffleNet.sh
    ```
you need to change the imagenet_path into your ImageNet path

## Training shuffleDet:
- Download KITTI object detection dataset: [images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip) and [labels](http://www.cvlibs.net/download.php?file=data_object_label_2.zip). Put them under `$SQDT_ROOT/data/KITTI/`. Unzip them, then you will get two directories:  `$SQDT_ROOT/data/KITTI/training/` and `$SQDT_ROOT/data/KITTI/testing/`. 

- Now we need to split the training data into a training set and a vlidation set. 

  ```Shell
  cd $SQDT_ROOT/data/KITTI/
  mkdir ImageSets
  cd ./ImageSets
  ls ../training/image_2/ | grep ".png" | sed s/.png// > trainval.txt
  ```
  `trainval.txt` contains indices to all the images in the training data. In our experiments, we randomly split half of indices in `trainval.txt` into `train.txt` to form a training set and rest of them into `val.txt` to form a validation set. For your convenience, we provide a script to split the train-val set automatically. Simply run
  
    ```Shell
  cd $SQDT_ROOT/data/
  python random_split_train_val.py
  ```
  
  then you should get the `train.txt` and `val.txt` under `$SQDT_ROOT/data/KITTI/ImageSets`. 

  When above two steps are finished, the structure of `$SQDT_ROOT/data/KITTI/` should at least contain:

  ```Shell
  $SQDT_ROOT/data/KITTI/
                    |->training/
                    |     |-> image_2/00****.png
                    |     L-> label_2/00****.txt
                    |->testing/
                    |     L-> image_2/00****.png
                    L->ImageSets/
                          |-> trainval.txt
                          |-> train.txt
                          L-> val.txt
  ```
- train_shuffleDet
    ```Shell
  ./train_shuffleDet.sh
  ```

- eval_shuffleDet
    ```Shell
  ./eval_shuffleDet.sh
  ```
- save input activation and parameters
    ```Shell
  ./save_parameters.sh
  ```
  when you run "save_parameters.sh",you will get the "activation&parameters8bit" file
   shuffledet/activation&parameters8bit/
                    |->input/
                    |     L-> input.txt
                    |->parameters/
                    |     |-> conv1_bias.txt
                    |     |-> conv1_weight.txt
                    |     |-> ratio_key.txt 
                    |     |-> ratio_value.txt 
                    |     |-> ... 
                    |     |-> ...
                    |     |-> stage_4_3_right_a_w_fold.txt
                    L     L-> stage_6_3_right_c_w_fold.txt
