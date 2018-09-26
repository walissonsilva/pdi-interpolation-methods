import cv2
import numpy as np 


img = cv2.imread('Imagens/sunflower-small.jpg', cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Image', img)
cv2.waitKey()

sh = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
sv = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)

SIr = np.sqrt(sh**2 + sv**2)

SI_mean = np.sum(SIr) / (SIr.shape[0] * SIr.shape[1])
SI_rms = np.sqrt(np.sum(SIr**2) / (SIr.shape[0] * SIr.shape[1]))
SI_stdev = np.sqrt(np.sum(SIr**2 - SI_mean**2) / (SIr.shape[0] * SIr.shape[1]))

print("> Informação espacial - MÉDIA: %f" % SI_mean)
print("> Informação espacial - RMS: %f" % SI_rms)
print("> Informação espacial - DESV. PADRÃO: %f" % SI_stdev)