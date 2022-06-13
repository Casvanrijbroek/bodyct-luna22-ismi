# bodyct-luna22-Team1
Repo for training Team 1's algorithms for LUNA22-ismi challenge


## Preparing the dataset
Instructions to download the dataset are available [here](https://luna22-ismi.grand-challenge.org/training-dataset/). Download the nodule dataset and place them in a directory called `LUNA22 prequel/`. Make sure to extract the zip file in the same directory. 

## Training
To train the algorithm, just execute the following script:
```bash
python train.py
```

The defaul model is currently ResNet18. If you want to run another model you need to change the model declaration in train.py
