# CS639_Photo_Video_Anonymization
Final Project for CS639 UW-Madison

Website: https://cs639-anonymization.netlify.app/

## Result
![image](https://user-images.githubusercontent.com/94937314/206935040-1c215770-14f3-46b9-a7d9-14082d16b7e8.png)
![reult_diff_obj](https://user-images.githubusercontent.com/94937314/207505674-464b653c-6ff2-44ec-965c-a8780171ed3d.png)

https://user-images.githubusercontent.com/94937314/207507661-b1980fdb-0205-46ef-82cf-898c4eeffa3f.mp4

## Demo.ipynb is a demo that illustrates how to use face and plate blur in Goolge Colab.

#### Please write the path of the target image after --source. By default we will use the data folder to store the target image. 

In order to upload the image to be blurred, please manually upload it yourself in google colab. 

How to upload the image to data folder:
1. Clone our github repository.
2. Click the file icon on the left side.

![InkedScreenshot 2022-12-11 163512](https://user-images.githubusercontent.com/94937314/206932922-ec09b7db-8e9c-4053-adaa-66a20798d37b.jpg)

3. Find the data folder in the CS639_Photo_Video_Anonymization folder, then put your mouse on it and click the three dots at the end.

![InkedScreenshot 2022-12-12 225154](https://user-images.githubusercontent.com/94937314/207229731-5d56a376-14a0-4b7d-9c2b-ae4e16f37cc5.jpg)

4. Click Upload, then select the image you want to process on your local computer.

![image](https://user-images.githubusercontent.com/94937314/207229819-ac309c5f-1335-4b0c-b03f-fc6cb93d6427.png)

You can upload the images elsewhere, as long as the path is correct.


#### Change the number after --blur-level to change the blur level of the image. There are four levels in total from 1-4.


#### If you are using the demo, the resulting images are stored in the /CS639_Photo_Video_Anonymization/runs/plate_blur/exp

--save-img is used to set the program to save the resulting images. 

Since we are using different models for our face blur and license plate blur, we have to manually blur the images twice. In the demo, you can see that the source of blur_plate.py is the resulting photo generated by blur_face.py. 

The plate_blur folder contains images of blurred license plates, and the face_blur folder contains images of blurred faces.

Each run creates a new exp folder but with different numbers. The largest number is the result of the latest run. To ensure that the latest run results can be found, please click Refresh after each run. 

![InkedScreenshot 2022-12-12 225454](https://user-images.githubusercontent.com/94937314/207230036-dfabc229-e67f-49d9-91c0-3546aeb23da4.jpg)


#### Other features 

The user can change the size of the resulting image by changing the number after --img.
By adding --vieww-img to the run command, the program is expected to display the result image, but now there are some bugs.
