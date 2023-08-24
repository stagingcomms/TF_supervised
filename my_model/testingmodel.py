import tensorflow as tf
import numpy as np
import cv2

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


cv2.imwrite("first.jpg",x_train[0])
model = tf.keras.models.load_model('model.h5')
img = cv2.imread("first.jpg")
size = [28,28]
img = cv2.resize(img,size) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = (np.expand_dims(img,0))
print(np.argmax(model(img)))

cv2.imwrite("second.jpg",x_train[5])
model = tf.keras.models.load_model('model.h5')
img = cv2.imread("second.jpg")
size = [28,28]
img = cv2.resize(img,size) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = (np.expand_dims(img,0))
print(np.argmax(model(img)))


cv2.imwrite("third.jpg",x_train[14])
model = tf.keras.models.load_model('model.h5')
img = cv2.imread("third.jpg")
size = [28,28]
img = cv2.resize(img,size) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = (np.expand_dims(img,0))
print(np.argmax(model(img)))



"""
model = tf.keras.models.load_model('model.h5')
img = cv2.imread("drawed_zero.png")
img= cv2.bitwise_not(img)
cv2.imwrite("negative_big_zero.jpg",img)
size = [28,28]
img = cv2.resize(img,size) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = (np.expand_dims(img,0))


print(np.argmax(model(img))) """