
import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        
        # x,y
        x,y = np.meshgrid(np.arange(-self.pad_w, self.pad_w+1), np.arange(-self.pad_w, self.pad_w+1))
        spatial_weight = np.exp( -((x**2+y**2)/(2*self.sigma_s**2)) )

        output = np.zeros_like(img, dtype=np.float64)
        if len(padded_guidance.shape) == 2:
            padded_guidance = np.expand_dims(padded_guidance, axis=2)
        padded_guidance = padded_guidance/255

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                patch_img = padded_img[i : i+self.wndw_size , j : j+self.wndw_size, :]                
                patch_guidance = padded_guidance[i : i+self.wndw_size, j : j+self.wndw_size, :]
                tmp = (patch_guidance - padded_guidance[i+self.pad_w, j+self.pad_w, :])**2
                range_weight = np.exp(-np.sum(tmp, axis=2)/(2*self.sigma_r**2))
                weight = np.multiply(spatial_weight, range_weight)
                weight_val = np.sum(weight)
                output[i, j, 0] = np.sum(np.multiply(weight, patch_img[:, :, 0])) / weight_val
                output[i, j, 1] = np.sum(np.multiply(weight, patch_img[:, :, 1])) / weight_val
                output[i, j, 2] = np.sum(np.multiply(weight, patch_img[:, :, 2])) / weight_val
                # output = sum_vals/weight_vals
                # output = sum_vals
        return np.clip(output, 0, 255).astype(np.uint8)