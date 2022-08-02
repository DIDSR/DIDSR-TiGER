## **Segmentation Algorithm**

To train the segmentation algorithm run the "segmentation.py" script.

The training data is composed of 151 slides from the TCGA dataset, 26 slides from the RUMC dataset (we will call it TC from now on), and 18 slides from the JB dataset (https://tiger.grand-challenge.org/Data/).

For TCGA dataset, each slide has exacly one region of interest (ROI) annotated by the pathologist, resulting in 151 ROIs.

For TC (RUMC) dataset, 81 ROIs are annotated. 25 slides have 3 ROIs and one slide has 6 ROIs, resulting in total of 25*3+6 = 81 ROIs.

For JB dataset, each slide has 3 ROIs, resulting in 18*3 = 54 ROIs.

The segmentation masks contain values from [0,1,2,3,4,5,6,7]. 0 corresponds to the regions in the ROI not annotated by the pathologist. 1 corresponds to the "invasive-tumor". 2 corresponds to the "tumor-associated stroma". 3 correponds to "in-situ tumor". 4 corresponds to "healthy glands". 5 corresponds to "necrosis not in-situ". 6 corresponds to "inflammed stroma". 7 corresponds to the "rest" class, not falling into any categories described above.

For the segmentation evaluation, participants have to only segment the tissue into stroma and tumor regions. For that, we relabel the mask values to train a 3-call segmentation model:

![image](https://user-images.githubusercontent.com/68286434/181014711-78027965-0c48-4c63-a938-dad981dfae3e.png)

As a result, we are segmenting the tissue into "Rest class", "Tumor class" and "Stroma class". The mask values of 0 correspond to the "Rest" class. Mask values of 1 correpond to the "Tumor" class and mask values of 2 corresponds to the "Stroma" class.

The following table shows the class imbalance between the three classes among the TCGA, TC and JB datasets:

![image](https://user-images.githubusercontent.com/68286434/181014748-7d3f4343-689a-43b5-92df-6350f2f03830.png)

Loading TCGA images, masks:

The function "extract_tcga_data.py" loads the TCGA images and masks form the local disk where the TIGER data is dowanloaded to. Some of the masks in the TCGA dataset are rotated. This function automatically corrects for this rotations. One example of this rotations is provided here:

![image](https://user-images.githubusercontent.com/68286434/181014785-b3061da8-37eb-48f6-917f-43b7f4d0a420.png)

Here you can see one example of TCGA ROI and the corresponding mask (mask values are relabled to 0,1,2):

![image](https://user-images.githubusercontent.com/68286434/181014817-106f1c69-8f99-4c74-ae92-b3b731c13535.png)

Loading TC images, masks:

The function "extract_tc_data.py" loads the TC images and masks form the local disk where the TIGER data is dowanloaded to.

Here you can see one example of TC ROI and the corresponding mask (mask values are relabled to 0,1,2):

![image](https://user-images.githubusercontent.com/68286434/181014877-25820a36-ecc7-4a2c-be6a-bf73994d470f.png)

Loading JB images, masks:

The function "extract_jb_data.py" loads the JB images and masks form the local disk where the TIGER data is dowanloaded to.

Here you can see one example of JB ROI and the corresponding mask (mask values are relabled to 0,1,2):

![image](https://user-images.githubusercontent.com/68286434/181014921-c9c09afb-bd66-4140-a465-3b9eaf2fd41f.png)

We developed a 3-class segmentation model using a U-Net model with InceptionV3 as backend. The following library is used to train the segmentatio model:

https://github.com/qubvel/segmentation_models

Steps below describes the pipeline to develpe the segmentation model:

1) Load all the TCGA, TC and JB images and masks.
2) Relabel the mask values to 0,1 and 2 ("change_masks.py" relables the mask values).
3) Combine all the images and masks.
4) Extract patches of size 256 with a stride of 128 (crop_img_msk.py extract patches of size "256" and a stride of "128"). This will result in 16039 patches. Extract patches of size 256 with a stride of 128 with angle of 45 degrees (extarct_patches_with_angle.py extract patches of size "256" and a stride of "128" with a specific angle of "45 degrees"). This will result in 9300 patches. Extract patches of size 256 with a stride of 128 with angle of minus 45 degrees. This will result in 9354 patches.
5) Combine all the extracted patches to obtain 34696 patches (16039+9300+9354 = 34693). Here you can see the patch extraction scheme:

![image](https://user-images.githubusercontent.com/68286434/181015792-51195300-61ae-48c7-b81c-1f6a7ab395e2.png)

6) Augment the patches spatially using the D4 symmetry group of a square. This will enhance the number of patches from 34693 to 34693*8 = 277544 patches. D4 symmetry of a square:

![image](https://user-images.githubusercontent.com/68286434/181015913-a26934d9-2496-4fc2-8569-b48415ac6c93.png)

7) Shuffle the patches using a fixed RandomState set to 42.
8) Find the calss imbalance between the 3 classes of the patches masks. This will result in a class weight of [1.6,1,0.76]. We will use this class weight in the dice loss function.
9) We will train a U-Net model using the InceptionV3 as backend. We will also use the pre-trained imagenet weights to train the segmentation model. In order to avoid over fitting we added a dropout layer before the softmax layer. The dropout value is set to 0.4. The InceptionV3 pre-processing unit is used to pre-process the training patches. This will normalize the patches ranging from 0 to 255 to -1 to 1. A batch-size of 32 is used and we will train the network for 30 epochs. ADAM is used as optimizer with a fixed learning rate of 0.0001. Loss function is the compound loss of dice_loss(class_weights) and the categorical focal loss.
Loss = dice_loss(class_weights)+focal_loss.

Training Intersection over Union for 30 epochs:

![image](https://user-images.githubusercontent.com/68286434/181016176-603128cb-bb27-4c7b-ae6b-65cf2cb9ef61.png)

10) The 30th epoch weights will be saved and used as the final model to segment the H&E images into 'rest', 'tumor' and 'stroma' class.

11) We will use the np.argmax function to find the prediction with the highest probaility. 0 corresponds to the rest class, 1 to tumor class and 2 to stroma class.

Here you can see some examples of the model predicting the segmentation mask:

![image](https://user-images.githubusercontent.com/68286434/181016537-5759b7f1-2f8d-42b6-9b5f-49189f439aed.png)
![image](https://user-images.githubusercontent.com/68286434/181016558-3c77633f-cb9a-4074-b0a7-01fc17ab523b.png)
![image](https://user-images.githubusercontent.com/68286434/181016594-8ed06c27-c07d-40ad-aa0e-b7b105b81639.png)

Here for our final model, we trained the network using all the images. One might split the data into train/test and observes the model's progress tracking the test performance. Above examples are drawn from some of the experiments we did by spliting the data into train/test with 80/20 split. For our final model we trained the network using all the training data at hand.

Our model's segmentation resutls on the experimental hidden set is: 

Stroma Dice = 0.7513

Tumor Dice = 0.7372

Our model's segmentation resutls on the final hidden set is: 

Stroma Dice = 0.7717

Tumor Dice = 0.7056


