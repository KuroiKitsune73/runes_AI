import cv2
import numpy as np


#Download images to list

images = []
for i in range(100):

    #Create 20 runes in each folder before

    image = cv2.imread(f'image{i}.jpg')
    images.append(image)


def random_deformation(image):
    # Random deformation func
    scale = np.random.uniform(0.8, 1.2)
    angle = np.random.uniform(-15, 15)
    translation_x = np.random.uniform(-10, 10)
    translation_y = np.random.uniform(-10, 10)

    # Deform images
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, scale)
    M[0, 2] += translation_x
    M[1, 2] += translation_y
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # return result
    return image


#append func to all list

for i in range(100):
    image = images[i]
    for j in range(120):
        deformed_image = random_deformation(image)

        #rename it!!

        cv2.imwrite(f'deformed_image{i}_{j}.jpg', deformed_image)

########################
#CREATE NPZ AFTER!!!
########################