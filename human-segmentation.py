
# TEST SCRIPT BASICALLY.

from fittingroom import segmentation_models as sm
from PIL import Image
import numpy as np
from fittingroom.utils import resizeImgMultipleEight

src_image_fn = 'fittingroom/sample_pics/woman1.jpg'
input_image = Image.open(src_image_fn)
input_image = resizeImgMultipleEight(input_image, MAXSIZE=500)

segmentation = sm.humanseg(np.array(input_image))
print('segmentation done')
print(segmentation.shape)

# save segmentation
segmentation[segmentation == 1] = 255
segmentation = Image.fromarray(segmentation)
print('unique values in segmentation:', np.unique(segmentation))
with open('segmentation.png', 'wb') as f:
    segmentation.save(f, 'PNG')
    
    
headmask = sm.headmask(np.array(input_image))

headmask_Im = Image.fromarray((headmask), mode='RGBA')
print('head segmentation done')
print('unique values in segmentation:', np.unique(headmask))
with open('headsegmentation.png', 'wb') as f:
    headmask_Im.save(f, 'PNG')
    
#%%
print(headmask_Im.size)
print(headmask_Im)
# make a transparant iamge that only contains the masked region
head_only= Image.new('RGB', input_image.size)
head_only.paste(input_image, (0, 0), headmask_Im)
with open('head_only.png', 'wb') as f:
    head_only.save(f, 'PNG')