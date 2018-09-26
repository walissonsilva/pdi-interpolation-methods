import cv2
import numpy as np

def calc_inf_spacial(img):
	sh = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
	sv = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)

	SIr = np.sqrt(sh**2 + sv**2)

	SI_mean = np.sum(SIr) / (SIr.shape[0] * SIr.shape[1])
	SI_rms = np.sqrt(np.sum(SIr**2) / (SIr.shape[0] * SIr.shape[1]))
	SI_stdev = np.sqrt(np.sum(SIr**2 - SI_mean**2) / (SIr.shape[0] * SIr.shape[1]))

	print(SI_mean)
	print(SI_rms)
	print(SI_stdev)

def interp_large(dim1=1280, dim2=853):
	img = cv2.imread('Imagens/berlin-small.jpg', cv2.IMREAD_COLOR)
	#img = cv2.imread('Imagens/white.png', cv2.IMREAD_GRAYSCALE)

	cv2.namedWindow('Original - Gray Scale', cv2.WINDOW_AUTOSIZE)
	cv2.imshow('Original - Gray Scale', img)

	# Função para calcular a informação espacial
	calc_inf_spacial(img)

	# interpolação da imagem do original
	nearest = cv2.resize(img, dsize=(dim1, dim2), interpolation=cv2.INTER_NEAREST)
	linear = cv2.resize(img, dsize=(dim1, dim2), interpolation=cv2.INTER_LINEAR)
	cubica = cv2.resize(img, dsize=(dim1, dim2), interpolation=cv2.INTER_CUBIC)
	
	cv2.imshow('Nearest Interpolation', nearest)
	cv2.imshow('Linear Interpolation', linear)
	cv2.imshow('Cubic Interpolation', cubica)
	cv2.waitKey()
	cv2.destroyAllWindows()

def interp_small(dim1=640, dim2=426):
	img = cv2.imread('Imagens/berlin-large.jpg', cv2.IMREAD_GRAYSCALE)
	#img = cv2.imread('Imagens/white.png', cv2.IMREAD_GRAYSCALE)

	cv2.namedWindow('Original - Gray Scale', cv2.WINDOW_AUTOSIZE)
	cv2.imshow('Original - Gray Scale', img)

	# Função para calcular a informação espacial
	calc_inf_spacial(img)

	# interpolação da imagem do original
	nearest = cv2.resize(img, dsize=(dim1, dim2), interpolation=cv2.INTER_NEAREST)
	linear = cv2.resize(img, dsize=(dim1, dim2), interpolation=cv2.INTER_LINEAR)
	cubica = cv2.resize(img, dsize=(dim1, dim2), interpolation=cv2.INTER_CUBIC)
	
	cv2.imshow('Nearest Interpolation', nearest)
	cv2.imshow('Linear Interpolation', linear)
	cv2.imshow('Cubic Interpolation', cubica)
	cv2.waitKey()
	cv2.destroyAllWindows()

interp_large()