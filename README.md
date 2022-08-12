## **DIDSR-TiGER Algorithm**

This repository contains the algorithm and the codes developed to predict a "TILs-score" in breast cancer histhopathology H&E slides. The training and the test data is provided by the TiGER challenge team (https://tiger.grand-challenge.org/). Our model is ranked 5th in the final leaderboard of the challenge predicting a TILs-score with a C-index (concordance index) of 0.6034. The winning algorithm resulted in a C-index of 0.6388.

![image](https://user-images.githubusercontent.com/68286434/181017984-7b545385-7203-4c74-8dc8-6592c22b6bb9.png)

To design a TILs-score, we first need to segment the tissue into stromal and tumor regions. Then we need to identify the TILs on the stromal regions. One example is shown in the figure below:

![image](https://user-images.githubusercontent.com/68286434/181020487-3b1ad0cb-91fe-4b2b-8ea6-6fd07f41baf1.png)

Using the following equation we can obtain the TILs-score which is a number between 0 to 100. We just calculate the ratio between the TILs area and the stromal area:

![image](https://user-images.githubusercontent.com/68286434/181020219-354f255f-1d17-4268-b89d-377b7fedd86f.png)

To scan a whole slide image, we extract patches of size 256 with a stride of 256 in a for loop along the x-axis and another for loop along the y-axis. For each of the extracted patches, we predict a segmentation mask and detect the TILs. We then count the number of TILs on the stromal region and we also calculate the area of the stromal regions. Here, as an example, one slide image along with the tissue mask is shown. Tissue mask identifies the region of the tissue, everything else is the slide background. On the tissue region the algorithm identifies the stromal and tumor regions along with the TILs. As a result, the TILs-score can be obtained:

![image](https://user-images.githubusercontent.com/68286434/181023515-2135f75d-1736-4420-b127-6009d2c67d8e.png)

## **Authors**

This code is made by Arian Arab as part of the Division of Imaging, Diagnostics and Software Reliability of the United States Food and Drug Administration. This code is based on the code developed by the TIGER challenge organizers.
