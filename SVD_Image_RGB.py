import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

# On a une matrice M, et on la décompose en M = U.Sigma.VT et ce que nous pourrions 
# faire estice par un rang inférieur "r", et de cette façon nous approximerons l'image.
# M est une image à deux dimensions, on va approximer cette image comme le produit de 
# quelques colonnes et lignes (pixels).
# Et ce serait une compression car au lieu de stocker t d'approximer cette matroutes ces n 
# lignes m colonnes, je n'ai plus qu'à stocker r.

plt.rcParams['figure.figsize'] = [12, 4]


#Téléchargement de l'image lena_couleurs:
M = imread('lena_couleurs.jpg')

R = M[:,:,0] / 0xff
G = M[:,:,1] / 0xff
B = M[:,:,2] / 0xff

R_U, R_S, R_VT = np.linalg.svd(R)
G_U, G_S, G_VT = np.linalg.svd(G)
B_U, B_S, B_VT = np.linalg.svd(B)
R_sigma = np.diag(R_S)
G_sigma = np.diag(G_S)
B_sigma = np.diag(B_S)
for k in (10, 20, 30, 150):
    compression_rouge = R_U[:,:k] @ R_sigma[0:k, :k] @ R_VT[:k, :]
    compression_jaune = G_U[:,:k] @ G_sigma[0:k, :k] @ G_VT[:k, :]
    compression_bleu = B_U[:,:k] @ B_sigma[0:k, :k] @ B_VT[:k, :]
    colormatrices = np.dstack((compression_rouge, compression_jaune, compression_bleu))
    compressee = (np.minimum(colormatrices, 1.0) * 0xff).astype(np.uint8)
    plt.title('k ='+str(k))
    plt.figure(1)
    img = plt.imshow(compressee)
    plt.show()
