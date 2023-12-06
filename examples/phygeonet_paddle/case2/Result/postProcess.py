import numpy as np
import matplotlib.pyplot as plt 
import tikzplotlib
import pdb
res_c=np.loadtxt('MRes.txt')


err_v=np.loadtxt('EV.txt')


epoch=1000
interval=1
iteration=np.asarray([i for i in range(epoch)])
idx=[i for i in range(epoch) if i%interval==0]

plt.figure()
plt.plot(iteration[idx],res_c[idx],'o',label='Equation residual')
plt.xlabel('Epoch')
plt.ylabel('Residual')
plt.yscale('log')
plt.savefig('Res'+str(epoch)+'.pdf',
            bbox_inches='tight')
tikzplotlib.save('Res'+str(epoch)+'.tikz')

plt.figure()
plt.plot(iteration[idx],err_v[idx],'o',label=r'$T$')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.yscale('log')
plt.savefig('Err'+str(epoch)+'.pdf',
            bbox_inches='tight')
tikzplotlib.save('Err'+str(epoch)+'.tikz')
