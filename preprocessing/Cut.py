# Su dung thu vien PIL
from PIL import Image
# Tap dư lieu ban dau
data_set_src = "D:\Senior\Term20211\CVProject\SmileDetection\dataset\SMILEsmileD\\not_smile\\not_smile"
# Tap dư lieu sau khi cat anh
data_set_dst = "D:\Senior\Term20211\CVProject\SmileDetection\dataset\SMILEsmileD\\not_smile\\not_smile"

import os
left = 28
top = 24
right = 92
bottom = 88

# Cat anh tư nhieu kich thuoc ve kich thuoc 64x64
for count, filename in enumerate(os.listdir(data_set_src)):
    src = f"{data_set_src}\{str(filename)}"
    im = Image.open(src)
    # Dua anh ve kich thuoc 120x120 vi rat it anh nho hon kich thuoc nay
    newsize = (120, 120)
    im = im.resize(newsize)
    # Cat anh ve kich thuoc 64x64
    im1 = im.crop((left, top, right, bottom))
    # Luu anh vao tap du lieu sau khi cat anh
    dst = f"{data_set_dst}\{str(filename)}"
    print(dst)
    im1.save(dst) 

