#%%
import cv2
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digit_img = cv2.imread("Images/digits.png", cv2.IMREAD_GRAYSCALE)

cells = [np.hsplit(row, 100) for row in np.vsplit(digit_img, 50)]

digits = np.array(cells)

labels = np.repeat(np.arange(10), 500)

hog_descriptors = []
for row in digits:
    for img in row:
        hog_descriptor = hog(img, pixels_per_cell=(
            10, 10), cells_per_block=(1, 1), visualize=False)
        hog_descriptors.append(hog_descriptor)

hog_descriptors = np.array(hog_descriptors).reshape(-1, 36)

train_input, test_input, train_target, test_target = train_test_split(
    hog_descriptors, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(train_input)

train_scaled = scaler.transform(train_input)
test_scaled = scaler.transform(test_input)

sc = SGDClassifier(loss='hinge', max_iter=200, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

predicted_target = sc.predict(test_scaled)

score = accuracy_score(test_target, predicted_target)
print(score)

def detect_digit(image, classifier, scaler, cells_per_block=(1,1), pixels_per_cell=(10,10)):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholds = cv2.threshold(gray_img, 10,255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresholds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.imshow("thresholds", thresholds)
    
    hog_descriptors = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        digit_roi = thresholds[y:y+h, x:x+w]
        
        resized_digit = cv2.resize(digit_roi, (20, 20), interpolation=cv2.INTER_AREA)

        hog_descriptor = hog(resized_digit, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=False)
        hog_descriptors.append(hog_descriptor)
    
    hog_descriptors = np.array(hog_descriptors).reshape(-1, 36)
    hog_descriptors = scaler.transform(hog_descriptors)
    return classifier.predict(hog_descriptors)

digit_image = cv2.imread("Images/new_data.png")
predicted_digit = detect_digit(digit_image, sc, scaler)

print(predicted_digit)
    
'''
sc = SGDClassifier(loss="log_loss" ,max_iter=200, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
# print(sc.score(train_scaled, train_target))
# print(sc.score(test_scaled, test_target))

sc.partial_fit(train_scaled, train_target)
# print(sc.score(train_scaled, train_target))
# print(sc.score(test_scaled, test_target))

sc = SGDClassifier(loss='log_loss', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)

for _ in range(0,1000):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))
    
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
'''
