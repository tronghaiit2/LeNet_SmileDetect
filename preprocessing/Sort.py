#Sap xep anh de tron lan voi nhau
data_set_src = "D:\Senior\Term20211\CVProject\SmileDetection\dataset\SMILEsmileD\SMILE\\negatives\\negatives2"
data_set_dst = "D:\Senior\Term20211\CVProject\SmileDetection\dataset\SMILEsmileD\SMILE\\negatives\\negatives"
cnt = 0
dem = 0
import os
for count, filename in enumerate(os.listdir(data_set_src)):
    src = f"{data_set_src}\{str(filename)}"
    print(src)
    dst = f"{str(cnt)}.jpg"
    print(dst)
    dst =f"{data_set_dst}\{dst}"
    print(dst)
    os.rename(src, dst)
    cnt += 2
    # if(dem < 3):
    #     cnt += 2
    #     dem += 1
    # else:
    #     cnt += 1
    #     dem = 0

    # if(dem < 1):
    #     cnt += 2
    #     dem += 1
    # else:
    #     cnt += 3
    #     dem = 0