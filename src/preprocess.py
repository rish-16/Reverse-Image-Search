import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pprint import pprint
import numpy as np

class Extractor:
    def __init__(self):
        self.images = {}
        self.captions = {}
        
    def read_captions(self, path):
        with open(path) as f:
            data = f.read().strip().split('\n')
            for i in range(len(data)):
                cur = data[i].strip().split('#')
                image_id = cur[0].split('.')[0]
                caption = cur[1][1:]
                self.captions[image_id] = caption
        
    def preview_captions(self, n):
        i = 0
        for key, val in self.captions.items():
            print (key)
            pprint (val)
            if i != n:
                i += 1
            else:
                break
        
    def read_images(self, path):
        img_dir = os.listdir(path)
        for i in range(len(img_dir)):
            cur = path + img_dir[i]
            image_id = img_dir[i].split('.')[0]
            img = Image.open(cur)
            img = img.resize((128, 128), Image.ANTIALIAS)
            img = np.array(img)
            
            self.images[image_id] = img
            
    def preview_images(self, n):
        fig, axs = plt.subplots(n)
        for i in range(n):
            axs[i].imshow(list(self.images.values())[i])
        plt.show()
        
    def get_stats(self):
        print ('Total images: {}'.format(len(list(self.images.values()))))
        print ('Total captions: {}'.format(len(list(self.captions.values()))))
        
    def find_pairs(self):
        images = set(self.images)
        final_images = []
        final_captions = []
        for key, value in self.images.items():
            if key in self.captions:
                final_captions.append(self.captions[key].strip("\t"))
                final_images.append(value)
            
        return final_captions, final_images