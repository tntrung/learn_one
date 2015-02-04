import numpy as np
import matplotlib.pyplot as plt

def display_patches(A):

    	# rescale
    	A = A - np.mean(A.reshape(A.shape[0]*A.shape[1]))
	clim = max(abs(A.reshape(A.shape[0]*A.shape[1])))
    	# compute rows, cols
    	N = A.shape[0]
    	D = A.shape[1]
	nsz = int(np.sqrt(N))
	psz = int(np.sqrt(D))
	npi  = 1 # 5 boundary pixels

	# init figure size
	fsz = nsz * psz + (nsz-1)*npi
	M = [[0 for ifsz in range(fsz)] for ifsz in range(fsz)]

	# asign matrice to figure
	for iN in range(N):
		x = int(int(iN/nsz)*psz + int(iN/nsz)*npi)
		y = int((iN%nsz)*psz + (iN%nsz)*npi)
		for i in range(psz):
			for j in range(psz):
				M[x+j][y+i] = A[iN,i*psz+j]/clim
	plt.gray()	
	plt.imshow(M)
	#plt.show()
