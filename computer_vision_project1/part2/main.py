import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    # set RGB
    RGB_values = [(0.1, 0.0, 0.9), (0.2, 0.0, 0.8), (0.2, 0.8, 0.0), (0.4, 0.0, 0.6), (1.0, 0.0, 0.0)]
    # RGB_values = [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.1, 0.0, 0.9), (0.1, 0.4, 0.5), (0.8, 0.2, 0.0)]
    i = 0
    cost = np.zeros(5)
    sigma_s = 1
    sigma_r = 0.05
    jbf = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_img = jbf.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    jbf_img = jbf.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
    cost1 = np.sum(np.abs(bf_img.astype('int32') - jbf_img.astype('int32')))
    print("Cost for COLOR_BGR2GRAY = :", cost1)

    for RGB in RGB_values:
        #img_filtered = cv2.merge((img_rgb[:, :, 0] * RGB[0], img_rgb[:, :, 1] * RGB[1], img_rgb[:, :, 2] * RGB[2]))
        img_gray = np.dot(img_rgb, RGB_values[i]) 
        bf_img = jbf.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
        jbf_img = jbf.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
        cost[i] = np.sum(np.abs(jbf_img.astype('int32') - bf_img.astype('int32')))
        print("Cost for R =", RGB, ":", cost[i])
        img_bgr = cv2.cvtColor(jbf_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(i) +'rgb.jpg', img_bgr)
        i = i + 1



    # joint bilateral filter
    # jbf = Joint_bilateral_filter(sigma_s, sigma_r)

    # cost = np.sum(np.abs(bf_img.astype('int32')-jbf_img.astype('int32')))

    

if __name__ == '__main__':
    main()