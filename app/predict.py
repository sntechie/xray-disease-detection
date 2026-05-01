import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("model/model.h5")

img = cv2.imread("/Users/snehabhateja/Medical_Image_Processing/app/test.jpg")
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.reshape(img, (1,224,224,3))

result = model.predict(img)

if result > 0.5:
    print("PNEUMONIA")
else:
    print("NORMAL")