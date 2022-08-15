# **Segmentation Algorithm**

To train the segmentation algorithm run the `segmentation.py` script.

The [training data](https://tiger.grand-challenge.org/Data/) is composed of 151 slides from the TCGA dataset, 26 slides from the RUMC dataset (we will call it TC from now on), and 18 slides from the JB dataset.

 - For TCGA dataset, each slide has exacly one region of interest (ROI) annotated by pathologists, resulting in 151 ROIs.

 - For TC (RUMC) dataset, 81 ROIs are annotated. 25 slides have 3 ROIs and one slide has 6 ROIs, resulting in total of 25*3+6 = 81 ROIs.

 - For JB dataset, each slide has 3 ROIs, resulting in 18*3 = 54 ROIs.

## Class labeling

The segmentation masks highlight 6 different tissue compartments (invasive tumor, tumor-associated stroma, in-situ tumor, healthy glands, necrosis not in-situ, and inflamed stroma) with label values from 1 to 6. Anything that does not fall into the above-mentioned categories will be labeled as the “rest” class with the label value of 7. Some parts of the ROIs are left without any annotations, those correspond to the label value of 0. To design a TILs-scoring algorithm, the key areas are the tumor and the stroma regions. Hence, we focused on those and combined all the others:

![image](https://user-images.githubusercontent.com/68286434/181014711-78027965-0c48-4c63-a938-dad981dfae3e.png)

As a result, we segment the tissue into three regions: "Rest class", "Tumor class" and "Stroma class". The mask values of 0 correspond to the "Rest" class. The mask values of 1 correspond to the "Tumor" class and mask values of 2 correspond to the "Stroma" class.

The following table shows the class imbalance between the three classes:

![image](https://user-images.githubusercontent.com/68286434/181014748-7d3f4343-689a-43b5-92df-6350f2f03830.png)

## Data loading

**Loading TCGA images, masks:**

The function `extract_tcga_data.py` loads the TCGA images and masks. Some of the masks in the TCGA dataset are rotated. This function automatically corrects for the rotations. One example of these rotations is provided here:

![image](https://user-images.githubusercontent.com/68286434/181014785-b3061da8-37eb-48f6-917f-43b7f4d0a420.png)

Here you can see one example of TCGA ROI and the corresponding mask (mask values are relabled to 0,1,2):

![image](https://user-images.githubusercontent.com/68286434/181014817-106f1c69-8f99-4c74-ae92-b3b731c13535.png)

**Loading TC images, masks:**

The function `extract_tc_data.py` loads the TC images and masks. Here you can see one example of TC ROI and the corresponding mask (mask values are relabled to 0,1,2):

![image](https://user-images.githubusercontent.com/68286434/181014877-25820a36-ecc7-4a2c-be6a-bf73994d470f.png)

**Loading JB images, masks:**

The function `extract_jb_data.py` loads the JB images and masks. Here you can see one example of JB ROI and the corresponding mask (mask values are relabled to 0,1,2):

![image](https://user-images.githubusercontent.com/68286434/181014921-c9c09afb-bd66-4140-a465-3b9eaf2fd41f.png)

## Model development
A 3-class [segmentation model](https://github.com/qubvel/segmentation_models) based on the U-Net model with InceptionV3 as backend is developed to train the segmentation model. Steps below describe the pipeline to develope the segmentation model:

1) Load all the TCGA, TC and JB images and masks.
2) Relabel the mask values to 0,1 and 2 (`change_masks.py` relables the mask values).
3) Combine all the images and masks.
4) Extract patches of size 256 with a stride of 128 (`crop_img_msk.py` extract patches of size "256" and a stride of "128"). This will result in 16039 patches. Extract patches of size 256 with a stride of 128 with angle of 45 degrees (`extarct_patches_with_angle.py` extract patches of size "256" and a stride of "128" with a specific angle of "45 degrees"). This will result in 9300 patches. Extract patches of size 256 with a stride of 128 with angle of minus 45 degrees. This will result in 9354 patches.
5) Combine all the extracted patches to obtain 34696 patches (16039+9300+9354 = 34693). Here you can see the patch extraction scheme:

![image](https://user-images.githubusercontent.com/68286434/181015792-51195300-61ae-48c7-b81c-1f6a7ab395e2.png)

6) Augment the patches spatially using the D4 symmetry group of a square. This will enhance the number of patches from 34693 to 34693*8 = 277544 patches. D4 symmetry of a square:

![image](https://user-images.githubusercontent.com/68286434/181015913-a26934d9-2496-4fc2-8569-b48415ac6c93.png)

7) Shuffle the patches using a fixed RandomState..
8) Find the calss imbalance between the 3 classes. This will result in a class weight of [1.6,1,0.76]. This class weight  will be used in the dice loss function.

The basic segmentation model we developed was U-Net with an InceptionV3 backend, which uses ImageNet pretrained weights to initialize its weights and biases. We also rescaled the RGB values of the training patches mapping the range [0, 255] to [-1, 1]. We added a Dropout layer with the rate of 0.4 before an output SoftMax layer to avoid overfitting. A compound loss function of Dice Loss and Categorical Focal Loss is used in model training. ADAM was chosen as the optimizer with a fixed learning rate of 0.0001.
Loss = dice_loss(class_weights)+focal_loss.

Training Intersection over Union (IOU) for 30 epochs:

![image](https://user-images.githubusercontent.com/68286434/181016176-603128cb-bb27-4c7b-ae6b-65cf2cb9ef61.png)

The 30th epoch weights will be saved and used as the final model to segment the H&E slides into 'rest', 'tumor' and 'stroma' classes.

We will use the np.argmax function to find the prediction with the highest probaility. 0 corresponds to the rest class, 1 corresponds to the tumor class, and 2 to corresponds to the stroma class.

Examples of model's predicitons on three test patches:

![image](https://user-images.githubusercontent.com/68286434/181016537-5759b7f1-2f8d-42b6-9b5f-49189f439aed.png)
![image](https://user-images.githubusercontent.com/68286434/181016558-3c77633f-cb9a-4074-b0a7-01fc17ab523b.png)
![image](https://user-images.githubusercontent.com/68286434/181016594-8ed06c27-c07d-40ad-aa0e-b7b105b81639.png)

Here, we trained the network using all the training data. One might split the data into train and test sets to study the generalizability errors. Above examples are drawn from some of the experiments we did by spliting the data into train and test sets with 80/20 split. 

Final model's performance on the experimental hidden test set:

Stroma Dice = 0.7513

Tumor Dice = 0.7372

Final model's performance on the final hidden test set:

Stroma Dice = 0.7717

Tumor Dice = 0.7056


