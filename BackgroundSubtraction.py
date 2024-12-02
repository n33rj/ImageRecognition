import numpy as np
import cv2
from matplotlib import pyplot as plt
  
image = cv2.imread('./Desktop/BackgroundSubtraction/sunflower.jpg')

mask = np.zeros(image.shape[:2], np.uint8)

backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)
  

rectangle = (10, 10, 780, 580)

cv2.grabCut(image, mask, rectangle,  
            backgroundModel, foregroundModel,
            3, cv2.GC_INIT_WITH_RECT)
  

mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
  
image_segmented = image * mask2[:, :, np.newaxis]

plt.figure()
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.figure()
plt.imshow(cv2.cvtColor(image_segmented, cv2.COLOR_BGR2RGB))
plt.axis('off')
 
plt.show()