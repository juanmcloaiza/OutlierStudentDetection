import numpy as np
n_samples = 1000
x = np.random.normal(0,1,n_samples) 
y = np.random.normal(0,1,n_samples) 
tf1 = np.random.choice([0,1], n_samples).astype(int)
tf2 = np.random.choice([0,1], n_samples).astype(int)
x += -10 + 20*tf1 
y += -10 + 20*tf2
ID = np.arange(1,n_samples+1).astype(int)

columns = np.stack((ID,x,y,tf1,tf2))
cols = columns.T
np.savetxt("test_data.csv", cols)
