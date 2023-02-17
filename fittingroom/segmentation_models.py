# create human segmenetation and head/hair segmentation here, at least!
from PIL import Image

from people_segmentation.pre_trained_models import create_model
model = create_model("Unet_2020-07-20")
# model.eval();

# %%
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from PIL import Image
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1


#%%

# def convert to 255 scale integer
def convert_to_255scale(image):
    # IMPORTANT, output masks should be in 0-255 scale
    if image.max() > 1:
        return image.astype(np.uint8)
    else:
        return (image * 255).astype(np.uint8)


def humanseg(image) -> np.array:
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]

    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]
    
    
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

    return convert_to_255scale(mask)




def headmask(image) -> np.array:
    
    mtcnn = MTCNN(select_largest=True, post_process=False, min_face_size=10)
    # Detect face
    boxes, probs, landmarks  = mtcnn.detect(image, landmarks=True)

    print('face detected: ', boxes, probs, landmarks)
    print('vertical coordinates for face: ', int(boxes[0][0].round()))


    # mask with the boxes where the heads are
    headmask_temp = np.zeros(image.shape) 
    
    # headmask_temp[int(boxes[0][3].round()):,:] = 0
    for box in boxes:
        headmask_temp[int(box[1].round()):int(box[3].round()),
                    int(box[0].round()):int(box[2].round())] = 1
    
    
    return convert_to_255scale(headmask_temp)
