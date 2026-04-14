#!/usr/bin/env python

### Import required modules
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

### return single image
def singleImage(img2d_under):
    
    # Create PIL image for overlay
    pil_under = Image.fromarray(np.uint8(cm.Greys_r(img2d_under, bytes=True)))
    return pil_under

### overlay an img on another
def overlayImage(img2d_under, img2d_over, transparency=0.6, is_edge=0):
    
    # Create PIL images for overlay and underlay images
    pil_under = Image.fromarray(np.uint8(cm.Greys_r(img2d_under, bytes=True)))
    pil_fused = Image.fromarray(np.uint8(cm.Greys_r(img2d_under, bytes=True)))
    
    ## tmpData =  np.uint8(np.tile(np.expand_dims(img2d_over,3),(1,1,4))*255)    ## Old version
    tmpData =  np.uint8(np.tile(np.expand_dims(img2d_over,2),(1,1,4))*255)
    tmpData[:,:,[1,2]] = 0        # Set RED color
    pil_over = Image.fromarray(tmpData)
    
    # Fing edges of overlay img
    if is_edge == 1:
        pil_over = pil_over.filter(ImageFilter.FIND_EDGES)
        pil_over = pil_over.point(lambda p: p > 0 and 255)        # to binarize edge image  

    # Set alpha=0 for the background (intensity=0) of the fg image
    red, green, blue, alpha = pil_under.split()

    if transparency < 1:
        pil_fused = Image.blend(pil_under, pil_over, transparency)
    else:
        pil_fused.paste(pil_over, (0,0), pil_over)
        pil_fused.putalpha(alpha)

    return pil_under, pil_fused
    #return pil_over, pil_fused


def labelToOverlay(img2d_over, rgb_color, is_edge=0):
    '''Create a colored overlay image for a binary or labeled mask.'''

    mask = (img2d_over > 0).astype(np.uint8) * 255
    pil_mask = Image.fromarray(mask, mode='L')

    if is_edge == 1:
        pil_mask = pil_mask.filter(ImageFilter.FIND_EDGES)
        pil_mask = pil_mask.point(lambda p: p > 0 and 255)

    mask = np.array(pil_mask, dtype=np.uint8)
    tmpData = np.zeros(mask.shape + (4,), dtype=np.uint8)
    tmpData[:, :, 0] = rgb_color[0]
    tmpData[:, :, 1] = rgb_color[1]
    tmpData[:, :, 2] = rgb_color[2]
    tmpData[:, :, 3] = mask
    pil_over = Image.fromarray(tmpData)

    return pil_over
    
### overlay two images on another
def overlayImageDouble(img2d_under, img2d_over1, img2d_over2, transparency=0.6, is_edge1=0,
                       is_edge2=0, over1_color=(255, 0, 0),
                       over2_label_colors=None):
    
    # Create PIL images for overlay and underlay images
    pil_under = Image.fromarray(np.uint8(cm.Greys_r(img2d_under, bytes=True)))
    pil_fused = Image.fromarray(np.uint8(cm.Greys_r(img2d_under, bytes=True)))

    pil_over1 = labelToOverlay(img2d_over1, over1_color, is_edge1)

    pil_over2_arr = np.zeros(img2d_over2.shape + (4,), dtype=np.uint8)
    if over2_label_colors:
        for label, rgb_color in over2_label_colors.items():
            label_img = (img2d_over2 == label).astype(np.uint8)
            if label_img.sum() == 0:
                continue

            pil_label = labelToOverlay(label_img, rgb_color, is_edge2)
            label_arr = np.array(pil_label)
            label_alpha = label_arr[:, :, 3] > 0
            pil_over2_arr[label_alpha] = label_arr[label_alpha]

        unmapped_mask = ((img2d_over2 > 0) & ~np.isin(img2d_over2, list(over2_label_colors.keys())))
        if np.any(unmapped_mask):
            fallback_arr = np.array(labelToOverlay(unmapped_mask.astype(np.uint8), (0, 255, 0), is_edge2))
            fallback_alpha = fallback_arr[:, :, 3] > 0
            pil_over2_arr[fallback_alpha] = fallback_arr[fallback_alpha]
    else:
        pil_over2_arr = np.array(labelToOverlay(img2d_over2, (0, 255, 0), is_edge2))

    pil_over2 = Image.fromarray(pil_over2_arr)

    # Set alpha=0 for the background (intensity=0) of the fg image
    red, green, blue, alpha = pil_under.split()

    transparency = float(transparency)
    if transparency < 1:
        pil_fused = Image.blend(pil_under, pil_over1, transparency)
        pil_fused = Image.blend(pil_fused, pil_over2, transparency)
    else:
        pil_fused.paste(pil_over1, (0,0), pil_over1)
        pil_fused.paste(pil_over2, (0,0), pil_over2)
        pil_fused.putalpha(alpha)

    return pil_under, pil_fused
