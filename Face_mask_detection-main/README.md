# Face_mask_detection

In this project, we aimmed at buidling an automatic detection system that can classify people wearing masks correctly, people wearing masks but wrongly and people not wearing masks. We got the motivation from the outbreak of COVID-19 starting in Spring 2020. 

##Structure

The whole project is made up of five jupyter notebooks. Including:

* Annotation acquisition.ipynb: The notebook is used to extract raw xml data and transform into csv files.

* Data Visualization, Preprocess and Pilot Test with SVM.ipynb: This notebook is used to explore the given image data, preprocess to get better features and do a pilot test of classification using a simple SVM model.

* Face Detecion.ipynb: The notebook is used to extract the bounding boxes of human faces from the given images using a Cascade Classifier.

* CNNtest.ipynb: The notebook is used to build a CNN model to extract features and make classifications.

* ResNet.ipynb: The notebook is used to build a ResNet model with the last layer to be SVM. This is the optimal model we have for the project.

And also the data for the project can be found in the below links:

https://www.kaggle.com/datasets/andrewmvd/face-mask-detection?select=images

https://www.kaggle.com/datasets/prasoonkottarathil/face-mask-lite-dataset

https://esigelec-my.sharepoint.com/personal/cabani_esigelec_fr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcabani%5Fesigelec%5Ffr%2FDocuments%2FMaskedFaceNetDataset%2FIMFD&ga=1
