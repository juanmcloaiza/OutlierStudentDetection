import numpy as np

def create_test_cases(n_samples = 1000):
    x = np.random.normal(0,1,n_samples) 
    y = np.random.normal(0,1,n_samples) 
    tf1 = np.random.choice([0,1], n_samples).astype(int)
    tf2 = np.random.choice([0,1], n_samples).astype(int)
    x += -10 + 20*tf1 
    y += -10 + 20*tf2
    ID = np.arange(1,n_samples+1).astype(int)

    columns = np.stack((ID,x,y,tf1,tf2))
    cols = columns.T
    np.savetxt(f"test_data_{n_samples:05}.csv", cols)


for exponent in (np.random.uniform(2,3,20)):
    print(exponent)
    n_samples = int(10**exponent)
    create_test_cases(n_samples)