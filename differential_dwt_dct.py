'''https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-015-0239-5'''

import cv2
import pywt
import numpy as np
from scipy.fftpack import fft2, ifft2, dct, idct

def apply_dct(image_array):
    size = image_array[0].__len__()
    all_subdct = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subpixels = image_array[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i+8, j:j+8] = subdct

    return all_subdct


# def inverse_dct(all_subdct):
#     size = all_subdct[0].__len__()
#     all_subidct = np.empty((size, size))
#     for i in range (0, size, 8):
#         for j in range (0, size, 8):
#             subidct = idct(idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
#             all_subidct[i:i+8, j:j+8] = subidct

#     return all_subidct

# def dct2(block):
#     return dct(dct(block.T, norm='ortho').T, norm='ortho')


# def idct2(block):
#     return idct(idct(block.T, norm='ortho').T, norm='ortho')

# def svd(input):
# 	return np.linalg.svd(input)

def encode_watermark(watermark, SEED=2019):
    h = watermark.shape[0]
    rand_H = np.random.RandomState(seed=SEED).permutation(h)
    # encoded_watermark = np.zeros((h,w,c))
    encoded_watermark = watermark.copy()
    for i in range(h):
        encoded_watermark[i] = watermark[rand_H[i]]
    # cv2.imwrite('encoded_watermark.jpg', encoded_watermark)
    # cv2.imwrite('encoded_watermark_ori.jpg', decode_watermark(encoded_watermark))
    return encoded_watermark

def decode_watermark(watermark, SEED=2019):
    h = watermark.shape[0]
    rand_H = np.random.RandomState(seed=SEED).permutation(h)
    decoded_watermark = watermark.copy()
    for i in range(h):
        decoded_watermark[rand_H[i]] = watermark[i]
    # cv2.imwrite('decoded_watermark.jpg', decoded_watermark)
    return decoded_watermark

def zigzag(matrix):
    '''https://www.geeksforgeeks.org/print-matrix-zag-zag-fashion/'''
    h,w = matrix.shape
    solution=[[] for i in range(w+h-1)]
    for i in range(h): 
        for j in range(w): 
            sum=i+j 
            if(sum%2 ==0): 
                solution[sum].insert(0,matrix[i][j]) 
            else:
                solution[sum].append(matrix[i][j])

    result = []
    for i in solution: 
        for j in i: 
            result.append(j)

    # print(result)
    return np.asarray(result)

def inverse_zigzag(input, vmax, hmax):
    
    #print input.shape

    # initializing the variables
    #----------------------------------
    h = 0
    v = 0
    vmin = 0
    hmin = 0
    output = np.zeros((vmax, hmax))
    i = 0
    #----------------------------------
    while ((v < vmax) and (h < hmax)): 
        #print ('v:',v,', h:',h,', i:',i)       
        if ((h + v) % 2) == 0:                 # going up
            if (v == vmin):
                #print(1)
                output[v, h] = input[i]        # if we got to the first line
                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        
                i = i + 1
            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                #print(2)
                output[v, h] = input[i] 
                v = v + 1
                i = i + 1
            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                #print(3)
                output[v, h] = input[i] 
                v = v - 1
                h = h + 1
                i = i + 1
        else:                                    # going down
            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                #print(4)
                output[v, h] = input[i] 
                h = h + 1
                i = i + 1
            elif (h == hmin):                  # if we got to the first column
                #print(5)
                output[v, h] = input[i] 
                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1          
            elif((v < vmax -1) and (h > hmin)):     # all other cases
                output[v, h] = input[i] 
                v = v + 1
                h = h - 1
                i = i + 1
        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            #print(7)           
            output[v, h] = input[i] 
            break
    return output.astype(int)

def compose(x1,x2):
    x=[]
    for i in range(len(x1)+len(x2)):
        if i%2 == 0:
            x.append(x1[int(i/2)])
        else:
            x.append(x2[int((i-1)/2)])
    print(len(x))
    return x

def decompose(x):
    x1=[]
    x2=[]
    # print('decomposed len',len(x)/2)
    for i in range(len(x)):
        if i%2 == 0:
            x1.append(x[i])
        else:
            x2.append(x[i])

    # print(x1,x2)
    return x1,x2

def convert_wm(wm):
    h,w = wm.shape
    for i in range(h):
        for j in range(w):
            if wm[i, j] > 127:
                wm[i, j] = 1
            else:
                wm[i, j] = -1
    return wm

def revert_wm(wm):
    h,w = wm.shape
    for i in range(h):
        for j in range(w):
            if wm[i, j] > 127:
                wm[i, j] = 255
            else:
                wm[i, j] = 0
    return wm

# start

alpha = 0.2
a = 200
# b = a + 4096

target_size = (256, 256)

Imau = cv2.imread('lena.bmp')
ori_img_h, ori_img_w = Imau.shape[:-1]
cv2.imwrite('ori.jpg', Imau[:,:,2])
oriG = Imau[:,:,1]
oriB = Imau[:,:,0]
# Imau = cv2.imread('ad.jpg')
# Imau = cv2.resize(Imau, target_size) # resize bad 
img_h, img_w = Imau.shape[:-1]
I1 = Imau[:,:,2]
IG = Imau[:,:,1]
IB = Imau[:,:,0]
# I1 = cv2.resize(I1, (256, 256))
# I1 = cv2.resize(I1, (img_w, img_h))
'''dwt'''
(LL, (LH, HL, HH)) = pywt.dwt2(I1, 'haar', axes=(0, 1))

'''debug'''
# matrix =[ 
#             [ 1, 2, 3,4], 
#             [ 5, 6,7,8 ], 
#             [ 9,10,11,12 ], 
#             [ 13,14,15,16 ], 
#         ] 
# matrix = np.asarray(matrix)
# zigzag_LL = zigzag(matrix)
# output = inverse_zigzag(zigzag_LL, 4,4)
# print(output)
# exit()
'''debug'''

x = zigzag(LL)
# print(zigzag_LL.shape)
x1,x2 = decompose(x)
X1 = dct(x1, norm='ortho')
X2 = dct(x2, norm='ortho')


# wm = cv2.imread('jamiecai_2.jpg')[:,:,0]
wm = cv2.imread('qrcode.jpg')[:,:,0]
# wm = revert_wm(wm)
# cv2.imwrite('qrcode.jpg', wm)
wm_ori = wm.copy()
# wm = cv2.resize(wm, (int(img_w/2),int(img_h/2)))
# wm = convert_wm(wm)
wm = np.reshape(wm, (wm.shape[0]*wm.shape[1], -1))
b = a + int(wm.shape[0])
wm_template = np.zeros(X1.shape[0])
for i in range(a,b):
    wm_template[i] = wm[i-a]

wm_template = encode_watermark(wm_template)

X1_hat = X1.copy()
X2_hat = X2.copy()
for i in range(X1_hat.shape[0]):
    # X1_hat[i] = X1[i] + alpha*wm_template[i]
    # X2_hat[i] = X2[i] - alpha*wm_template[i]
    X1_hat[i] = 0.5*(X1[i]+X2[i]) + alpha*wm_template[i] # better extraction but obvious
    X2_hat[i] = 0.5*(X1[i]+X2[i]) - alpha*wm_template[i]

x1_hat = idct(X1_hat, norm='ortho')
x2_hat = idct(X2_hat, norm='ortho')

x_hat = compose(x1_hat, x2_hat)
# x_hat = compose(x1, x2)
# print(compose([1,3,5,7], [2,4,6,8]))
new_LL = inverse_zigzag(x_hat, int(img_w/2),int(img_h/2))

wmed_img = pywt.idwt2([new_LL, (LH, HL, HH)], 'haar')
# wmed_img = cv2.resize(wmed_img, (ori_img_w, ori_img_h))
# wmed_img = cv2.merge([oriB, oriG, wmed_img])
cv2.imwrite('wmed_img.jpg', wmed_img)

'''detect'''
wmed_img = cv2.imread('wmed_img.jpg')
# wmed_img = cv2.resize(wmed_img, target_size)
img_h, img_w = wmed_img.shape[:-1]
I1 = wmed_img[:,:,2]
IG = wmed_img[:,:,1]
IB = wmed_img[:,:,0]
'''dwt'''
(LL, (LH, HL, HH)) = pywt.dwt2(I1, 'haar', axes=(0, 1))
x_hat = zigzag(LL)
x1_hat,x2_hat = decompose(x_hat)
X1_hat = dct(x1_hat, norm='ortho')
X2_hat = dct(x2_hat, norm='ortho')
wm_template = (X1_hat - X2_hat)/2./alpha
wm_template = decode_watermark(wm_template)
# print(wm_template.shape)

wm_template = wm_template[a:b]
# print(wm_template.shape)
wm_template = np.reshape(wm_template, (64, 64))
# wm_template = 255 - wm_template
# wm_template = revert_wm(wm_template)
cv2.imwrite('extracted_wm.jpg', wm_template)

cv2.imwrite('wm_diff.jpg', wm_ori - wm_template)
