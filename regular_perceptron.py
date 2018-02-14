
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt

data = np.zeros((100, 3))
val = np.random.uniform(0, 2, 100)
diff = np.random.uniform(-1, 1, 100)
data[:,0], data[:,1], data[:,2] = val - diff, val + diff, np.ones(100)
target = np.asarray(val > 1, dtype = int) * 2 - 1


# In[12]:


#Visualizing the data before perceptron 
plt.figure()
for i in range(100):
    if(target[i] == 1):
        plt.plot(data[i,0],data[i,1],'rx')
    else:
        plt.plot(data[i,0],data[i,1],'bx')

plt.show()


# In[13]:


w = np.zeros(3)
for epoch in range(10):
    for i in range(len(target)):
        if(target[i]*(data[i,:].dot(w))> 0):
            continue
        else:
            w = w + (target[i] * data[i,:])


# In[14]:


#Computing the points for the line
x = np.linspace(-0.75,2.5,100)
y = np.zeros(100)
for i in range(100):
    y[i] = (-w[2] - w[0]*x[i])/w[1]


# In[15]:


plt.figure()
count1 = 0
count2 = 0
for i in range(100):
    if(target[i] == 1):
        if(count1 == 0):
            plt.plot(data[i,0],data[i,1],'rx',label ='label +1')
            count1 = 1
        else:
            plt.plot(data[i,0],data[i,1],'rx')
    else:
        if(count2 == 0):
            plt.plot(data[i,0],data[i,1],'bo',label ='label -1')
            count2 = 1
        else:
            plt.plot(data[i,0],data[i,1],'bo')

plt.plot(x,y,'g-',label = 'Decision boundary')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Scatter plot of data')
plt.legend()
plt.savefig('hw2p5a.png')
plt.show()
plt.close()


# In[16]:


target

