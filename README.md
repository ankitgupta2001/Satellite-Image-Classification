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
- Run the `Satellite_Image_Classification_using_CNN.ipynb` file




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
