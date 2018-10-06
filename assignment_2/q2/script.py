import numpy as np 
import matplotlib.pyplot as plt

files = ['G_a.npy', 'G_i.npy', 'G_n.npy', 'G_s.npy']
for file in files:
	fig = plt.figure()
	l = np.load(file)
	l = l[1:]*1000000
	l = np.square(l)
	plt.plot(l)
	plt.xlabel('Order')
	plt.ylabel('Error signal energy')
	plt.title('Error Signal Energy for /' + file[2] + ' /')
	plt.savefig('error_energy_' + file[2] + '.png')
