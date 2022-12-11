# CS639_Photo_Video_Anonymization
Final Projecgt for CS639 UW-Madison

## Instructions for Face Blur

### demo_face_blur.ipynb is a demo that illustrates how to use face blur in Goolge Colab.

#### Please write the path of the target image after --source. By default we will use the data folder to store the target image. 

In order to upload the image to be blurred, please manually upload it yourself in google colab. 

How to upload the image to data folder:
1. Clone our github repository.
2. Click the file icon on the left side.

![InkedScreenshot 2022-12-11 163512](https://user-images.githubusercontent.com/94937314/206932922-ec09b7db-8e9c-4053-adaa-66a20798d37b.jpg)

3. Find the data folder in the CS639_Photo_Video_Anonymization folder, then put your mouse on it and click the three dots at the end.

![InkedScreenshot 2022-12-11 163646](https://user-images.githubusercontent.com/94937314/206932970-e6493368-c6de-4353-a365-64fdcf3680e9.jpg)

4. Click Upload, then select the image you want to process on your local computer.

![image](https://user-images.githubusercontent.com/94937314/206933026-ebcf6a5e-ce09-45fa-a9f4-c50deff45a73.png)

You can upload the images elsewhere, as long as the path is correct.


#### Change the number after --blur-level to change the blur level of the image. There are four levels in total from 1-4.


#### --save-img is used to set the program to save the resulting images. The images are stored in the CS639_Photo_Video_Anonymization/runs/blur/exp folder

Each run creates a new exp folder but with different numbers. The largest number is the result of the latest run.

To ensure that the latest run results can be found, please click Refresh after each run. 

![InkedScreenshot 2022-12-11 164850](https://user-images.githubusercontent.com/94937314/206933500-8a5b89a8-9a0f-4a97-9cea-cb7bd238eb65.jpg)


#### Other features 
The user can change the size of the resulting image by changing the number after --img.
By adding --vieww-img to the run command, the program is expected to display the result image, but now there are some bugs.
