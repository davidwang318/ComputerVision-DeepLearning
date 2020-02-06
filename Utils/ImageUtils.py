import cv2
import numpy as np

def preprocess(img, label, imgContainer, labelContainer):
	imgContainer.append(img)
	labelContainer.append(label)
	return imgContainer, labelContainer

# Augment the data in batch by flipping the image
def preprocessAug(img, label, imgContainer, labelContainer):
	imgContainer.append(img)
	labelContainer.append(label)
	imgContainer.append(np.fliplr(img))
	labelContainer.append(label)
	return imgContainer, labelContainer