# **Detection Algorithm**

In order to train the detection model run the `detection.py` script.

The [training data](https://tiger.grand-challenge.org/Data/) is composed of 1744 ROIs from the TCGA dataset where 20727 TILs are annotated, 81 ROIs from the RUMC dataset (we will call it TC from now on) where 4728 TILs are annotated, and 54 ROIs from the JB dataset where 5523 TILs are annotated. Each TIL annotation marks the centroid position of the cells: [x_c , y_c] where x_c and y_c are the centroid positions along the x and y axis correspondingly.

## Examples from the TCGA, TC and JB datasets are provided here (blue dots are the TILs annotations):

| JB Example | TC Example | TCGA Exmaple |
| ---------- | ---------- | ------------ |
| ![image](https://user-images.githubusercontent.com/68286434/181012474-85dfc8a6-5673-4f23-a6e3-c21aa7dc938c.png) | ![image](https://user-images.githubusercontent.com/68286434/181012510-91319e12-f336-4349-9992-2e569a41cb68.png) | ![image](https://user-images.githubusercontent.com/68286434/181012537-d2acc5de-7fbe-4631-a2c8-1aa60e438d71.png) |

 - For TCGA dataset, most of the annotated ROIs are smaller than 256 pixels. For that, we intentionally extract larger ROIs of size of 500 pixels form the TIFF images (expanding the original ROIs). This will ensure that we can train a model with trainig patches of size of 256 pixels. We split the ROIs randomly into a train and a test set. Train set contains of 1578 ROIs with 18585 TILs. Test set contains of 166 ROIs with 2142 TILs.

 - For TC dataset, we split the ROIs randomly into a train and a test set. Train set contains of 76 ROIs with 4456 TILs. Test set contains of 5 ROIs with 272 TILs.

 - For JB dataset, we split the ROIs randomly into a train and a test set. Train set contains of 49 ROIs with 4891 TILs. Test set contains of 5 ROIs with 632 TILs.

A binary [segmentation model](https://github.com/qubvel/segmentation_models) based on the U-Net model with InceptionV3 as backend is developed to train the detection model.

## Steps below describe the pipeline to develpe the detection algorithm:

1) Load all the TCGA, TC and JB images and TIL annotations. Split the data into train/test (this is emmbeded within the load functions).

2) For TCGA, extract 9 patches of size of 256 pixels. This can be done by extracting the patch at the center of the larger ROI of size of 500 pixels along with 8 other patches along the x and y axis with a stride of 32:

![image](https://user-images.githubusercontent.com/68286434/181013196-47d046a8-ca94-4cf9-8a6c-324e2d2eafc8.png)

For TC and JB, using the sliding window technique, extract patches of size 256 with a stride of 128:

![image](https://user-images.githubusercontent.com/68286434/181013256-13ac77c1-5852-48ea-aa3a-668b8eafa1ee.png)

Combine all the extracted patches from TCGA, TC and JB. This will result in 23654 patches where 199223 TILs are annotated.

3) Augment the training patches by keeping the original patches, flipping the patches left/right, flipping the patches up/down and also tranposing each patch. This will increase the numebr of total patches to 94616 with 796892 TILs annotations. Randomly shuffle the training patches by fixing the random state.

4) For test ROIs repeat the steps as explained above. This will result in 2296 test patches where 22411 TILs are annotated. This time, instead of augmenting the test patches, randomly flip patches left/right, up/down and transpose the patches.

5) To train the detection algorithm, create a binary mask of TILs annotations by extending the centroid position of the TILs to a square of size of 12 pixels (mask values for squares are one and everything else is zero):

![image](https://user-images.githubusercontent.com/68286434/181013615-fd2da0d8-ebca-4ce9-b57b-500086eab126.png)
![image](https://user-images.githubusercontent.com/68286434/181013640-4480a683-07ff-4da3-9d3b-5a8fec08846f.png)

We train a binary segmentation U-Net model with the InceptionV3 as backend. Imagenet pretrained weights are used to initialize the weights and biases. The input patches to the network are rescaled from [0,255] to [0,1]. The loss function is a binary cross entropy loss. We train the network for 1 epoch with a batch size of 32. ADAM is chosen as the optimizer with a fixed learning rate of 0.001.

## Examples of model's prediciton on three test patches:

![image](https://user-images.githubusercontent.com/68286434/181013799-de0fe60c-562f-45c4-b02a-9644b598918c.png)
![image](https://user-images.githubusercontent.com/68286434/181013818-0794c709-5978-4e27-a343-61eecb1e225e.png)
![image](https://user-images.githubusercontent.com/68286434/181013825-bd5673b4-c282-49f2-8ff6-5bffabf1c34a.png)

In order to extract the x, y location of the detected TILs from the prediction masks, we first filter the mask values below a threshold value of 0.1. Anything with probability less than 0.1 will be discarded. Furthermore, by applying a non-max suppression on the distance, we can obtain the centroid position of the TILs. The non-max suppression distance threshlold is set to 12 pixels which is equal to the suqare size in the traning masks:

![image](https://user-images.githubusercontent.com/68286434/184369672-585ac796-f808-47fc-8465-f5b686facbc6.png)

Figure below is the FROC plot of the model's prediction on all of the test patches. We can compute the FROC score with a hit distance set to 8 pixels. This distance is chosen by the organizers of the TIGER challenge (A detection will be a True-Positive if it lies within a distance of 8 pixels with respect to the ground-truth).

![image](https://user-images.githubusercontent.com/68286434/181014064-f11804a6-4ee5-4f61-9aac-e1e8153f3929.png)

Below is the FROC plot of our detection model on the experimental hidden dataset:

![image](https://user-images.githubusercontent.com/68286434/181014140-6c2d494e-367f-4984-9377-a2c5600d2c67.png)

Below is the FROC plot of our detection model on the final hidden dataset:

![image](https://user-images.githubusercontent.com/68286434/181014165-06d5977c-5ea1-4a9b-98f9-efc372e85e51.png)
