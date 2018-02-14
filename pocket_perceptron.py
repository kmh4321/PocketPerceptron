
# coding: utf-8

# In[103]:


import numpy as np
from matplotlib import pyplot as plt

data = np.ones((100, 3))
data[:50,0], data[50:,0] = np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)
data[:50,1], data[50:,1] = np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)
target = np.zeros(100)
target[:50], target[50:] = -1 * np.ones(50), np.ones(50)


# In[104]:


#Visualizing the data before perceptron 
pow_x = 1
pow_y = 1
plt.figure()
for i in range(100):
    if(target[i] == 1):
        plt.plot(data[i,0]**pow_x, data[i,1]**pow_y,'rx')
    else:
        plt.plot(data[i,0]**pow_x, data[i,1]**pow_y,'bx')
#plt.yscale('log')
#plt.xscale('log')
plt.show()


# In[105]:


w = np.zeros(3)
best_w = np.zeros(3)
best_acc = 0
for epoch in range(5000):
    for i in range(len(target)):
        x = data[i,0]
        y = data[i,1]
        if(target[i]*(data[i,:].dot(w))> 0):
            continue
        else:
            w = w + (target[i] * data[i,:])
        for j in range(len(target)):
            current_acc = 0
            if(target[j]*(data[j,:].dot(w))> 0):
                current_acc = current_acc + 1
        if(current_acc > best_acc):
            best_acc = current_acc
            best_w = w


# In[106]:


#Computing the points for the line
x = np.linspace(-2.5,5,100)
y = np.zeros(100)
for i in range(100):
    y[i] = (-best_w[2] - best_w[0]*x[i])/best_w[1]


# In[107]:


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
plt.title('Scatter plot of data w/ Pocket Algorithm')
plt.legend()
plt.savefig('hw2p5b_pocket.png')
plt.show()
plt.close()


# In[108]:


#Computing the points for the line
x = np.linspace(-2.5,5,100)
y = np.zeros(100)
for i in range(100):
    y[i] = (-w[2] - w[0]*x[i])/w[1]


# In[109]:


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
plt.title('Scatter plot of data WITHOUT Pocket Algorithm')
plt.legend()
plt.savefig('hw2p5b_nopocket.png')
plt.show()
plt.close()

