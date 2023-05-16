# Multiscale-cross-connection-network
MCCNet: Multi-Scale Cross Connection Network for Low-Light Image Enhancement

Our technique utilizes a multi-scale cross-connection network (MCCNet) that employs region-specific analysis to improve feature extraction. Moreover, our method also includes a calibrated network to fine-tune the features to get improved results. The critical section of the model is the region selection of the images, where some overlapping regions help provide contextual information between the extracted features. The cross-connection network helps to share the distinct features extracted and concatenate them to a single feature block, further improving the model's performance. Substantial simulations show our model's haughty performance compared with the state-of-the-art benchmark methods. 


<p float="left">
  <img src="/Images/197_low.png" width="200" />
  <img src="/Images/197_high.png" width="200" />
  <img src="/Images/278_low.png" width="200" />
  <img src="/Images/278_high.png" width="200" /> 
</p>

<p float="left">
  <img src="/Images/Low_25.JPG" width="200" />
  <img src="/Images/Ours_25.png" width="200" />
  <img src="/Images/low_722.png" width="200" />
  <img src="/Images/ours_722.png" width="200" /> 
</p>

**Fig:** Results of our approach, where we compared to the original low light image and our enhanced image.
<hr style="border-top: 3px dotted #998143">



Here I am giving the steps to be followed in Google colab.
1) To run in colab, first of all clone this repo
```
!git clone https://github.com/santoshpanda1995/Multiscale-cross-connection-network.git
```
The requirements.txt file contains all the library files needed for the program to run.

2) Then copy the contents of requirements.txt and just run in a cell in colab.

**Note:** There may be some extra or repeated library files present in the requirements file, usually I maintain a single requirement file so sometimes i copy the existing command which may already be there. It will never effect anything to the program.

3) After that we have to load our trained model, you can directly download the pretrained model which I have provided and put it in the colab to import it.
- [x] MCCNet.h5 ( Our pretrained model, trained on our synthetic paired dataset, for 600 epochs with 32 steps per epoch. Adam Optimizer and L1 loss function has been used.)
```
from tensorflow.keras.models import load_model
# load model
Model = load_model(path to the model)
# summarize model
Model.summary()
```

Now our model is loaded in **Model**. , We can test low light images from this model. Some of the low light images on which i have tested my model , I will provide here.

4) To test image, define this function
```
from google.colab.patches import cv2_imshow
import cv2
def test(img,og):
  height, width, channels = img.shape
  image_for_test= cv.resize(img,(600,400))
  og= cv.resize(og,(600,400))
  image_for_test= image_for_test[np.newaxis,:, :,:]
  Prediction = Model.predict(image_for_test)
  Prediction = Prediction.reshape(400,600,3)
  Prediction = np.array(Prediction)
  #Write the Low light image, the predictiona image and groundtruth image
  cv2.imwrite('root path', img)
  cv2.imwrite('root path', Prediction)
  cv2.imwrite('root path', og)
  original = cv2.imread("/content/low.png")
  compressed = cv2.imread("/content/high.png")
  Hori = np.concatenate((cv.resize(img,(600,400)),cv.resize(og,(600,400)),cv.resize(Prediction,(600,400))), axis=1)
  cv2_imshow(Hori)
  ```
6) Then 
```
img = cv.imread(path of low light image)
og = cv.imread(path of groundtruth image)
test(img,og)
```
