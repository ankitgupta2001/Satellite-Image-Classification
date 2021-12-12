# Major Project (Satellite Image Classification using CNN)

In this project we are classifying `EuroSAT` satelite data of RGB class to create a classifier which will classify the dataset into 10 classes:
- AnnualCrop
- Forest 
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

Dataset consists of 27,000 images and can be downloaded using this [link](http://madm.dfki.de/files/sentinel/EuroSAT.zip).

The Classification is based on `wide_resnet50_2` model of PyTorch and is modified accordingly to improve its accuracy. It consists of 50 convolutional layer.

### How to run the code
- Create an anaconda environment using `conda create -n env_name python=3.7`
- Install all dependencies using `pip install -r requirements.txt`
- Copy the dataset in the `Image_dataset` folder
- Run the `Land_Cover_Classification_using_Sentinel_2_Satellite_Imagery_and_Deep_Learning_10_epochs.ipynb` file
- Run all the cells in a sequential manner, suggested via `Google colab` for higher processing power (May get bottleneck by Personal Computer)
- After successful execution, a `lulc.pth` and `lulc_max_acc.pth` model will get saved, which can be loaded and then used to predict the class of the test dataset.

### Our Model 
- We trained our model with the following parameters:
```
## Hyper Parameters
max_epochs_stop = 10
max_lr = 1e-4
grad_clip = 0.1
weight_decay = 1e-3
batch_size = 64
criterion = nn.CrossEntropyLoss()
epochs = 10
opt_func = torch.optim.Adam
```
- We trained the model for 10 epochs on Goole Colab, each epoch taking about `~45 minutes` on Google Colab. Total time taken `~8 hours`. 
- We were able to achieve an accuracy of `0.9901111111111112` upon predicting on `27000` training and test images.



<br>
Following is the confusion matrix generated:

<br>
<p align="center">
<img align="center" src ="Confusion%20Matrix%20%20Test%20Data.png">
</p>




## References
If you have used the EuroSAT dataset, please cite the following papers:

[1] Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.
```
@article{helber2019eurosat,
  title={Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2019},
  publisher={IEEE}
}
```
