
from PIL import Image
import numpy as np
import torch
img1 = Image.open('segmentation.png')


image = np.array(Image.open('segmentation.png').convert("RGBA"))
image = image.astype(np.float32)/255.0
image = image[None].transpose(0,3,1,2)
image = torch.from_numpy(image)
print('image shape:', image.shape)


mask = np.array(Image.open('segmentation.png').convert("L"))
mask = mask.astype(np.float32)/255.0
mask = mask[None,None]
mask[mask < 0.5] = 0
mask[mask >= 0.5] = 1
mask = torch.from_numpy(mask)
print('mask shape:', mask.shape)
