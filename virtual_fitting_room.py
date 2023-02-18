# input image 
# pose model merged with dreambooth trained model for particular item, items. 


#%%
from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler


apply_openpose = OpenposeDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
# model.load_state_dict(load_state_dict('./models/control_dreamlike_openpose.pth', location='cuda'))
model.load_state_dict(load_state_dict('./models/control_zora2.pth', location='cuda'))

model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, scale, seed, eta, mask):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        print('input_image shape: ', input_image.shape)
        print('img shape: ', img.shape)
        print(' mask shape: ', mask.shape)
        print('detected_map.shape: ', detected_map.shape)

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        print('detected_map.shape after cv2 resize: ', detected_map.shape)



        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        print('control shape: ', control.shape)


        if False:
            # not sure about these lines - can maybe find this in another repo
            mask_torch = cv2.resize(mask, (W//8, H//8), interpolation=cv2.INTER_NEAREST)
            mask_torch = torch.from_numpy(mask_torch).float().cuda() / 255.0
            mask_torch = torch.stack([mask_torch for _ in range(num_samples)], dim=0)
            mask_torch = einops.rearrange(mask_torch, 'b h w c -> b c h w').clone()

            # make sure 2nd dimension is 1
            mask_torch = mask_torch[:, 0:1, :, :]
            # copy the 2nd dimension 4 times 
            mask_torch = torch.cat((mask_torch, mask_torch, mask_torch, mask_torch), dim=1)
            
            input_torch = cv2.resize(input_image, (W//8, H//8), interpolation=cv2.INTER_NEAREST)
            input_torch = torch.from_numpy(input_torch).float().cuda() / 255.0
            input_torch = torch.stack([input_torch for _ in range(num_samples)], dim=0)
            input_torch = einops.rearrange(input_torch, 'b h w c -> b c h w').clone()
            # make sure 2nd dimension is 4 (was 3, add zeros)
            input_torch = torch.cat((torch.zeros(input_torch.shape[0], 1, input_torch.shape[2], input_torch.shape[3]).cuda(), input_torch, ), dim=1)

            print('input_torch shape: ', input_torch.shape)
            print('mask_torch shape: ', mask_torch.shape)


        else:
                        
            def preprocess_image(image_path):
                image = Image.open(image_path)
                image.thumbnail((W, H))
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                image = np.array(image).astype(np.uint8)
                image = (image/127.5 - 1.0).astype(np.float32)
                return image

            def preprocess_mask(mask_path, h, w):
                mask = Image.open(mask_path).convert('1')
                mask_resize = mask.resize((w, h))
                return np.array(mask_resize).astype(np.float32)

            
            
            h = H//8
            w = W//8

            # code from inpainting SD
            from einops import rearrange, repeat
            image_fn = 'fittingroom/sample_pics/woman1.jpg'

            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            print('setting up image...')
            # image_prompt_input = preprocess_image(image_fn)
            image_prompt_input = (input_image/127.5 - 1.0).astype(np.float32)
            image_prompt_input = rearrange(image_prompt_input, 'h w c -> c h w')
            image_prompt_input = torch.from_numpy(image_prompt_input)
            # image_prompt_input = image_prompt_input.to(memory_format=torch.contiguous_format).to(torch.float16)
            image_prompt_input = repeat(image_prompt_input, 'c h w -> b c h w', b=num_samples) 
            print('input image shape:')
            print(image_prompt_input.shape)
            
            image_prompt_input = image_prompt_input.to(device)
            
            print('input image shape on device:')
            print(image_prompt_input)

            
            #print(image_prompt_input)
            encoder_posterior = model.encode_first_stage(image_prompt_input )
            x0 = model.get_first_stage_encoding(encoder_posterior).detach()
            
            # MASK PART
            print('setting up mask...')
            # mask_prompt_input = preprocess_mask(mask_prompt, h, w)
            mask_prompt_input = cv2.resize(mask, (W//8, H//8), interpolation=cv2.INTER_NEAREST)
            mask = torch.tensor(mask_prompt_input)
            mask = repeat(mask, 'h w -> b h w', b=num_samples).to(device)
            mask = mask[:, None, ...]

            # hack until I find out what is happening
            # input_torch = x0[:,:,1:-2,:]
            input_torch = x0[:,:,:,:]
            mask_torch = mask[:,:,:,:]
            print('input_torch shape: ', input_torch.shape)
            print('mask_torch shape: ', mask_torch.shape)




        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)
        print('shape: ', shape)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond, 
                                                     mask=mask_torch, x0=input_torch)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


#%%
from PIL import Image
from fittingroom import segmentation_models as sm
import numpy as np
from fittingroom.utils import resizeImgMultipleEight, pasteMaskedPart

src_image_fn = 'fittingroom/sample_pics/woman1.jpg'

input_image = Image.open(src_image_fn)
input_image = resizeImgMultipleEight(input_image, MAXSIZE=512)

# HACK
input_image = input_image.resize((512, 512), Image.ANTIALIAS)

input_image_numpy = np.array(input_image)
headmask = sm.headmask(input_image_numpy)

print('headmask made!', headmask.shape)
# print('humansegmask made!', humansegmask.shape)

prompt = "woman wearing a dress"
a_prompt = "4K, 8K, photography, high quality"
n_prompt = ""
num_samples = 1
image_resolution = 512
detect_resolution = 256
ddim_steps = 20
scale = 9.0
seed = -1
eta = 0.1
mask = headmask
mask = mask[:, :, 0]/255


test = process(input_image_numpy, prompt,
               a_prompt, n_prompt,
               num_samples, image_resolution,
               detect_resolution, ddim_steps,
               scale, seed, eta, mask)

# find head mask in original image


# paste head mask on top of generated image
output = Image.fromarray(test[1])


# paste the mask on top of the generated image
if False:
    output = pasteMaskedPart(headmask, output, input_image)


#%%
# save to file
filename = 'fittingroom/results/woman1_dress' + str(random.randint(0, 65535)) + '.jpg'
# save test to file
with open(filename, 'wb') as f:
    output.save(f, format='JPEG')
    f.close()