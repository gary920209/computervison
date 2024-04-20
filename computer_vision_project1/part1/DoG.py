import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)

        gaussian_images = []
        for octave in range(self.num_octaves):
            gaussian_images.append(image)
            for image_num in range(self.num_guassian_images_per_octave-1):
                # cv2.imwrite("octave" + str(octave) + "image" + str(image_num) + ".png", cv2.GaussianBlur(image, (0, 0), sigmaX = self.sigma**(image_num)))
                gaussian_images.append(cv2.GaussianBlur(image, (0, 0), sigmaX = self.sigma**(image_num+1)))  
            image = cv2.resize(gaussian_images[-1], (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(self.num_DoG_images_per_octave*2 + 1):
            if i != 4:
                dog_images.append(cv2.subtract(gaussian_images[i+1], gaussian_images[i]))

            #cv2.imwrite('output_image'+str(i)+'.jpg', img)


        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for i in range(1,self.num_DoG_images_per_octave - 1):
            for row in range(1,dog_images[i].shape[0] - 1):
                for col in range(1,dog_images[i].shape[1] - 1):
                    if abs(dog_images[i][row,col]) >= self.threshold:
                        is_max = True
                        is_min = True
                        for di in range(-1,2):
                            for x in range(-1,2):
                                for y in range(-1,2):
                                    if di == 0 and x == 0 and y == 0:
                                        continue
                                    if(is_max and (dog_images[i][row,col] <= dog_images[i+di][row+x,col+y])):
                                        is_max = False
                                    if(is_min and (dog_images[i][row,col] >= dog_images[i+di][row+x,col+y])):
                                        is_min = False
                        if is_max or is_min:
                            keypoints.append([row,col])
            
        for i in range(self.num_DoG_images_per_octave + 1, self.num_DoG_images_per_octave*2 - 1):
            for row in range(1,dog_images[i].shape[0] - 1):
                for col in range(1,dog_images[i].shape[1] - 1):
                    if abs(dog_images[i][row,col]) >= self.threshold:
                        is_max = True
                        is_min = True
                        for di in range(-1,2):
                            for x in range(-1,2):
                                for y in range(-1,2):
                                    if di == 0 and x == 0 and y == 0:
                                        continue
                                    if(is_max and (dog_images[i][row,col] <= dog_images[i+di][row+x,col+y])):
                                        is_max = False
                                    if(is_min and (dog_images[i][row,col] >= dog_images[i+di][row+x,col+y])):
                                        is_min = False
                        if is_max or is_min:
                            keypoints.append([2*row,2*col])
    
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        if len(keypoints) > 0:
            keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
