import numpy as np
import seaborn as sns

for i in [10, 11,12,13]: #[10, 11,12,13]
    data=np.loadtxt('/home/amax/code/rl/DINOAYN-master/log/m1-Pendulum-v0/m1-Pendulum-v0_s0/A-output-'+str(i)+'.txt')
    #print(data)
    data = data[:, 50:]
    print(data.shape)
    
    ax = sns.tsplot(data=data)




#aa = np.array([[1,3,4,1,1],
#               [1,3,4,2,2],
#               [-6,3,0,4,3]])
#sns.barplot(data=aa)

#import seaborn as sns
#print('sadfasd')
#bb = np.array([[1,3,4,1,2,3],[3,3,4,1,2,4]]) -1
#sns.barplot(data=bb)
#ti = sns.load_dataset('titanic')
#print(ti)

