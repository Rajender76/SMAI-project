# Cricket Shot Type Classification

A deep learning system for classifying 10 types of cricket shots from video clips using ResNet50 feature extraction and Bidirectional LSTM with attention mechanism.

## Requirements
- Python 3.8+
- NVIDIA GPU (Recommended) with CUDA 11.8
- Required packages:
  ```
  pip install torch torchvision opencv-python numpy scikit-learn seaborn matplotlib gdown ```

### Steps to run training_prediction.py(train) file

1) Replace the DATASET_DIR with your dataset. The dataset folder structure should be 

   ``` dataset_folder/
    ├── cover/       ## videos
    ├── defense/
    ├── flick/
    ├── hook/
    ├── late_cut/
    ├── lofted/
    ├── pull/
    ├── square_cut/
    ├── straight/
    └── sweep/ ```

2) Now the features will be extracted and saved at the given location.

3) Now the model will be trained to get the results.

### Steps to run infer.py file

``` 
python infer.py \
  --input_zip ./path/to/input_videos.zip \
  --model_path ./model_weights.pth
  ```



the input_videos.zip contains the test videos, model_weights.pth is the location of best model saved.

4) You will be able to see the prediction of each video with the confidence level and the prediction metrics at the end.

## DATASET DIRECTORY- https://drive.google.com/drive/folders/11cya49yjZlAXOEEgz0Cn6VoNQZfmBGLD?usp=drive_link

## INFER SET DIRECTORY- https://drive.google.com/drive/folders/1i25JVrS3VuuxGDuiGRLxE9oCPJPiucNx

## MODEL PATH - https://drive.google.com/file/d/1cjqbVYFLIXRrx75z7QMEI_uoa6Oylpw6/view?usp=sharing
