import cv2
import numpy as np
from skimage.measure import compare_ssim, compare_psnr, compare_mse

dir_image = 'Imagens/berlin-large.jpg'

img = cv2.imread(dir_image, cv2.IMREAD_COLOR)
img_compare = cv2.imread('Imagens/berlin-small.jpg', cv2.IMREAD_COLOR)

cv2.namedWindow('Original - Gray Scale', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Original - Gray Scale', img)
cv2.waitKey()

# Função para calcular a informação espacial

# interpolação da imagem do original
nearest = cv2.resize(img, dsize=(640, 426), interpolation=cv2.INTER_NEAREST)
linear = cv2.resize(img, dsize=(640, 426), interpolation=cv2.INTER_LINEAR)
cubica = cv2.resize(img, dsize=(640, 426), interpolation=cv2.INTER_CUBIC)

print('### Compare - Nearest ###')
print('MSE: %.3f' % round(compare_mse(img_compare, nearest), 3))
print('PSNR: %.3f' % round(compare_psnr(img_compare, nearest), 3))
print('SSIM: %.3f' % round(compare_ssim(img_compare, nearest, multichannel=True), 3))
print('')

print('### Compare - Linear ###')
print('MSE: %.3f' % round(compare_mse(img_compare, linear), 3))
print('PSNR: %.3f' % round(compare_psnr(img_compare, linear), 3))
print('SSIM: %.3f' % round(compare_ssim(img_compare, linear, multichannel=True), 3))
print('')

print('### Compare - Cubic ###')
print('MSE: %.3f' % round(compare_mse(img_compare, cubica), 3))
print('PSNR: %.3f' % round(compare_psnr(img_compare, cubica), 3))
print('SSIM: %.3f' % round(compare_ssim(img_compare, cubica, multichannel=True), 3))
print('')

cv2.imshow('Nearest Interpolation', nearest)
cv2.imshow('Linear Interpolation', linear)
cv2.imshow('Cubic Interpolation', cubica)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('Imagens/Resultados/' + dir_image[8:dir_image.find('-')] + '-inter2small-nearest.jpg', nearest)
cv2.imwrite('Imagens/Resultados/' + dir_image[8:dir_image.find('-')] + '-inter2small-linear.jpg', linear)
cv2.imwrite('Imagens/Resultados/' + dir_image[8:dir_image.find('-')] + '-inter2small-cubic.jpg', cubica)