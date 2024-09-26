import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
from keras.applications import VGG19 #or any other CNN model


print(os.listdir("/content/drive/MyDrive/Labelled/Train and Validation"))

SIZE = 256 #Resize images

images = []
labels = []

for directory_path in glob.glob("/path_to_data/*"):
    label = directory_path.split("/")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)


from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels = train_test_split(images,labels, test_size = 0.2, random_state = 91234)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded
x_train, x_test = x_train / 255.0, x_test / 255.0


VGG_model = VGG19(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

for layer in VGG_model.layers:
	layer.trainable = False

#VGG_model.summary()  


feature_extractor=VGG_model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_ML = features #This is our X input to ML

#RANDOM FOREST/Use any other ML model
from sklearn.ensemble import RandomForestClassifier
ML_model = RandomForestClassifier(n_estimators = 50, random_state = 42)


# Train the model on training data
ML_model.fit(X_for_ML, y_train) #For sklearn no one hot encoding

X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

prediction_ML = ML_model.predict(X_test_features)
prediction_ML = le.inverse_transform(prediction_ML)


from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_ML))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(test_labels, prediction_ML)
print(cm)
print(classification_report(test_labels, prediction_ML))


#Check results on a few select images
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=VGG_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction_ML = ML_model.predict(input_img_features)[0]
prediction_ML = le.inverse_transform([prediction_ML])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_ML)
print("The actual label for this image is: ", test_labels[n])