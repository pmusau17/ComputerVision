# import the neccessary packages

import imgaug as ia 
import imgaug.augmenters as iaa 
import numpy as np 

seq = iaa.Sequential([
    iaa.Crop(px=(0,16)),            # Crop images from each side by 0 to 16 px (randomly chosen)
    iaa.Fliplr(0.5),                # Horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0,3.0)) # Blur images with a sigma of 0 to 3.0
])

def augment_seg(img, seg):
    """removes the randomness from all augmenters and makes them deterministic (e.g. for each parameter that comes from a distribution, 
    it samples one value from that distribution and then keeps reusing that value). That is useful for landmark augmentation, 
    because you want to transform images and their landmarks in the same way, e.g. rotate an image and its landmarks by 30 degrees. 
    If you don't have landmarks, then you most likely don't need the function."""
    aug_det = seg.to_deterministic()
    image_aug=aug_det.augment_image(img)

    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg)+1,shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()

    return image_aug, segmap_aug