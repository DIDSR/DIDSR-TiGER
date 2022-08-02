## **Detection Algorithm**

In order to train the detection model run the "detection.py" script.

The training data is composed of 1744 ROIs from the TCGA dataset where 20727 TILs are annotated, 81 ROIs from the RUMC dataset (we will call it TC from now on) where 4728 TILs are annotated, and 54 ROIs from the JB dataset where 5523 TILs are annotated (https://tiger.grand-challenge.org/Data/). Below examples from TCGA, TC and JB are provided (blue dots are the TILs positions):

JB Example

![image](https://user-images.githubusercontent.com/68286434/181012474-85dfc8a6-5673-4f23-a6e3-c21aa7dc938c.png)

TC Example:

![image](https://user-images.githubusercontent.com/68286434/181012510-91319e12-f336-4349-9992-2e569a41cb68.png)

TCGA Exmaple:

![image](https://user-images.githubusercontent.com/68286434/181012537-d2acc5de-7fbe-4631-a2c8-1aa60e438d71.png)

For TCGA, the annotated ROIs have size smaller than 256 pixels. We intentionally extracted ROIs form the original TIFF images of fixed sizes of 500*500 (expanding the original ROIs). This will ensure that we can train a model where trainig patches of size of 256 are possible to be extracted from the traning ROIs. We should note that not all the TILs in the extracted ROIs are annotated (as can be seen in the example above). This way for TCGA we are training a model where the annoatations are semi-annotated.

Each of the TIL annotations marks the centroid position of the TILs: [x_c , y_c] where x_c is the centroid position along the x-axis and correspondingly for y_c along the y-axis.

For TCGA dataset, we split the ROIs randomly into a train and a test set. Train set contains of 1578 ROIs with 18585 TILs. Test set contains of 166 ROIs with 2142 TILs.

For TC dataset, we split the ROIs randomly into a train and a test set. Train set contains of 76 ROIs with 4456 TILs. Test set contains of 5 ROIs with 272 TILs.

For JB dataset, we split the ROIs randomly into a train and a test set. Train set contains of 49 ROIs with 4891 TILs. Test set contains of 5 ROIs with 632 TILs.

We developed a binary segmentation model using a U-Net model with InceptionV3 as backend. The following library is used to train the detection model:

https://github.com/qubvel/segmentation_models

Steps below describes the pipeline to develpe the detection algorithm:

1) Load all the TCGA, TC and JB images along with the TILs annotations. We splitted data into train/test when we loaded the TCGA, TC and JB images and TILs (embeded within the load functions).

2) For TCGA, we extract 9 patches of size 256 from the original ROI of size of 500 pixels. We do this by extracting the patch at the center of the ROI along with 8 other patches along the x and y axis with a stride of 32, as shown in the figure below:

![image](https://user-images.githubusercontent.com/68286434/181013196-47d046a8-ca94-4cf9-8a6c-324e2d2eafc8.png)

For TC and JB, we extract patches of size 256 with a stride of 128 as shown in the figure below in sliding window technique:

![image](https://user-images.githubusercontent.com/68286434/181013256-13ac77c1-5852-48ea-aa3a-668b8eafa1ee.png)

After combining all the extracted patches from TCGA, TC and JB, we will obtain 23654 patches where 199223 TILs are annotated.

3) We then augment the training patches by keeping the original patches, flipping the patches left/right, flipping the patches up/down and also tranposing each patche. This will increase the numebr of total patches to 94616 with 796892 TILs annotated. We then shuffle the training patches by fixing the random state.

4) For the test patches, we do the same operations as explained above. This will result in 2296 test patches where 22411 TILs are annotated. This time instead of augmenting the test patches, we randomly flip patches left/right, up/down and taking the transpose of the patches. We then suffle the test patches by fixing the random state.

5) For training, from the train TILs annotations, we create a mask by extending the centroid position of the TILs to a square of size of 12 pixels as shown below:
The mask values of the squares are one and everything else is set to zero.

![image](https://user-images.githubusercontent.com/68286434/181013615-fd2da0d8-ebca-4ce9-b57b-500086eab126.png)
![image](https://user-images.githubusercontent.com/68286434/181013640-4480a683-07ff-4da3-9d3b-5a8fec08846f.png)

For training we train a U-Net model using the InceptionV3 as backend. Imagenet pretrained weights are used. The input patches are first normalized from values between 0 to 255 to values between 0 to 1. The loss function is a binary cross entropy loss. We train the network for 1 epoch using a batch size of 32. ADAM is chosen as the optimizer with a fixed learning rate of 0.001. After trainig the model weights are saved.

Below some examples of the model predicitons on the test patches are shown:

![image](https://user-images.githubusercontent.com/68286434/181013799-de0fe60c-562f-45c4-b02a-9644b598918c.png)
![image](https://user-images.githubusercontent.com/68286434/181013818-0794c709-5978-4e27-a343-61eecb1e225e.png)
![image](https://user-images.githubusercontent.com/68286434/181013825-bd5673b4-c282-49f2-8ff6-5bffabf1c34a.png)

In order to extract locations of the detected TILs from the prediction masks, we first filter the mask values below a threshold value. In this case we chose the threshold value to 0.1, as a result, anything with probability less than 0.1 will be discarded. Furthermore, by applying a non-max suppression on the distance we can obtain the centroid position of the detected TILs. Here we chose the non-max suppression distance to 12 pixels which is equal to the suqare size chosen in the traning mask:

![image](https://user-images.githubusercontent.com/68286434/181013935-269968a5-b4b5-4bcf-a356-6d405ff4b615.png)

By predicting TILs from the test patches, we can compute the FROC score with a distance threshold set to 8 pixels which is the hit-distance set by the organizers in the TIGER challenge (A detection will be a True-Positive if it lies within a distance of 8 pixels with respect to the ground-truth).

![image](https://user-images.githubusercontent.com/68286434/181014064-f11804a6-4ee5-4f61-9aac-e1e8153f3929.png)

Below is our detection model FROC plot on the experimental hidden dataset:

![image](https://user-images.githubusercontent.com/68286434/181014140-6c2d494e-367f-4984-9377-a2c5600d2c67.png)

Below is our detection model result on the final hidden dataset:

![image](https://user-images.githubusercontent.com/68286434/181014165-06d5977c-5ea1-4a9b-98f9-efc372e85e51.png)
