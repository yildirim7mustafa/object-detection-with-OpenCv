import cv2
import matplotlib.pyplot as plt
import numpy as np

#resmi içe aktar
img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img) # ondalıklı sayılara çeviriyoruz
print(img.shape)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")

#harris corner detection -> blocksize= komsuluk boyutu, kernelsize= simdiye kadar kullandığımız kutucuğun boyutu, k = free parametre
dst = cv2.cornerHarris(img, blockSize = 2, ksize = 3, k = 0.04)
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")

#dst'nin max indan büyükse 1 yapalım daha belirgin olsun
dst = cv2.dilate(dst, None)
img[dst>0.2*dst.max()] = 1
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")

#shi tomsai detection
img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)
#100 = kaç köşe tespit etmek istediğimiz, 0.01 quality level, 10 = min distance
corners = cv2.goodFeaturesToTrack(img, 120, 0.01, 10)
corners = np.int64(corners)

for i in corners:
    x,y = i.ravel() #düzleştirme
    cv2.circle(img,(x,y),3,(125,125,125),cv2.FILLED) # resim, kordinat, yarıçap , renk , içini doldur
    
plt.imshow(img), plt.axis("off")    