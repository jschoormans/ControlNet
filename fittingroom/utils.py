from PIL import Image





def pasteMaskedPart(headmask, output, input_image):
    mask_to_paste = Image.fromarray(headmask[:,:,0]).convert('L')
    output.paste(input_image, (0, 0), mask_to_paste)
    return output


def resizeImgMultipleEight(img: Image, MAXSIZE=500):
    print('resizing image to nearest multiple of 8')
    sz = img.size
    print('Original image size:', sz)
    nearest = (sz[0] + 7) & ~7, (sz[1] + 7) & ~7

    print('FIX ASPRATIO')
    if sz[0] > MAXSIZE or sz[1] > MAXSIZE:
        nearest = (MAXSIZE + 7) & ~7, (MAXSIZE + 7) & ~7
        if sz[0] > sz[1]:
            nearest = (int(nearest[0] * sz[0] / sz[1]), nearest[1])
        else:
            nearest = (nearest[0], int(nearest[1] * sz[1] / sz[0]))

    # resize image to nearest multiple of 8
    img = img.resize(nearest, Image.Resampling.LANCZOS)
    print('Image resized to', img.size)
    return img 
