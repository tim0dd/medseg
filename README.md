# MedSeg
MedSeg is a framework for medical image segmentation that I developed for my [master's thesis](https://drive.google.com/file/d/1fFE3lcJ5UbJ4h-i5zD7G85opg6-SgT9S/view?usp=sharing). It is based on PyTorch and provides a variety of models and support for various segmentation datasets.

The framework offers a lot of versatility in the configuration of data augmentation pipelines, evaluations and logging, and training modes like hyperparameter tuning or k-fold cross-validation. Furthermore, scripts for dataset analytics and validation are available. E.g., class density heatmaps showing the distribution of class pixels with various data augmentation configs can be generated. Additionally, datasets can be scanned for duplicate images via byte-wise comparison of the images, as well as a perceptual similarity metric with a configurable threshold. 

More detailed information about the features can be found in the [thesis](https://drive.google.com/file/d/1fFE3lcJ5UbJ4h-i5zD7G85opg6-SgT9S/view?usp=sharing) under Section 4.3. Note that this framework was created under strict time-constraints and is decently large (~13k lines of Python). The code in some places is therefore not as clean as I would have liked it to be (for ML standards, it's ok though :D).

## Getting started
Dependencies can be installed with the following commands, using an anaconda or miniconda environment
    
```
conda create -n "medseg" python=3.10.9
conda activate medseg
pip3 install -e ./src/main
pip3 install -r requirements.txt
```

## Datasets
Datasets have to be downloaded manually and in many cases require a registration.
However, converter scripts for a variety of supported datasets can be found in `src/main/medseg/data/converters`.
The scripts are usually adapted to the folder structure of the downloaded dataset and should require no additional 
configuration. Please refer to the documentation in the respective converter script for more information.

Example:

```python3 ./src/main/medseg/training/convert_fuseg21.py in_path=/home/user/Downloads --out_path="./data/datasets/fuseg21"```

The `out_path` argument should be set to the default path used in the respective dataset definition in `src/main/medseg/data/datasets` for the framework to find them without additional configuration.

## Pretrained Weights
The framework by default expects pretrained weights to be located in `./data/pretrained`. 

The weights, while automatically downloaded on first use in the case of U-Net and HiFormer, for most models have to be downloaded manually. The following links can be used for this purpose:
- [HarDNet-DFUS](https://drive.google.com/drive/folders/1UbuMKLUlCsZAusUVLJqwcBaXiwe0ZUe8?usp=sharing)
- [FCBFormer](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth)
- [Segformer](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/Ept_oetyUGFCsZTKiL_90kUBy5jmPV65O5rJInsnRCDWJQ?e=CvGohw)
- [SegNeXt](https://cloud.tsinghua.edu.cn/d/c15b25a6745946618462/)
## Example Configs
Example configs can be found in `./configs`. They can be used as a starting point for custom configurations.

## Training
New training runs can be started with the following command:
`python3 ./src/main/medseg/training/train.py from_config --path="./configs/some_config.yaml"`
The training mode is determined by the config.

For logging and checkpoints saving, the `./out` folder is used, with subfolders according to the training type.
Within the subfolder corresponding to the training type, a new folder is created for each training run, according to the
`model_name` set in the training config and the timestamp at the start of the run. In this folder, the training log 
is saved along with checkpoints, a metric summary for each saved checkpoint, a model summary detailing the 
model architecture and parameter counts, the tensorboard event files for visualizing training and evaluation metrics, 
and more.

Existing checkpoints from previous runs can be used for further training as follows:
`python3 ./src/main/medseg/training/train.py from_checkpoint --path="./some_folder/example_checkpoint.pt" --add_epochs=10`

Interrupted hyperparameter optimization runs can be resumed with the following command:
`python3 ./src/main/medseg/training/train.py from_hyperopt_state --path="./some_folder/hyperopt_state.pkl"`

For resuming a k-fold cross-validation run, the following command can be used:
`python3 ./src/main/medseg/training/train.py from_kfold_state --path="./some_folder/kfold_state.pkl"`

## Evaluation
Evaluations are automatically performed during and after training, however, separate evaluations can be performed with the following commands:

`python3 ./src/main/medseg/evaluation/evaluate.py from_checkpoint --path="./some_folder/example_checkpoint.pt" --split="test"`

`python3 ./src/main/medseg/evaluation/evaluate.py from_kfold --path="./some_folder/example_checkpoint.pt --add_aux_test_set="KvasirSeg"`

## Tools
A variety of dataset and model analysis tools can be found in `./src/main/medseg/tools`. 

Please refer to the documentation in the respective files for more information.

The feature for sending an email when training is finished or encounters an exception can be used by including a file `./data/mail/mail_config.json`. 
The feature currently only supports sending mails from gmail accounts. The file should have the following format:
```
{
  "sender_email": "some_sender@gmail.com",
  "receiver_email": "some_receiver@gmail.com",
  "app_password": "sender_app_password"
}
```
Note that the `app_password` is not the normal account password. It can be generated in the account settings and
is required by Google for third-party applications.

## Adding new architectures
New architectures can be added by creating a new python file `./src/main/medseg/models/segmentors/`
and inheriting the `Segmentor` class in the main module. Additionally, an import of the main module has be added to 
the `./src/main/medseg/segmentors/__init__.py` file. The new architecture can then be used for training by specifying the name of the
main class in the training config file under `architecture -> arch_type`. Examples can be found in the provided configs under `./configs`.
