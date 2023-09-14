#!/usr/bin/env python
# coding: utf-8

# # Transportation Planning Project
# ### Instructor: Dr. Erfan Hassannayebi
# 
# ### Group Members:
# #### 1. Mahan Pourhosseini
# #### 2. Soroush Etminan Bakhsh
# #### 3. Iman Sherkat Bazazan
# #### 4. Mahdi Rahmani Talab

# ### Importing Libraries

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import seaborn as sns
import numpy as np
import scipy.stats
import math
from scipy.stats import chi2_contingency


# ### Loading Data

# In[3]:


data = pd.read_csv ('F:\\Semester 7\\Transportation\\Project\\q1final.csv')
data


# ### Data Info (1)

# In[4]:


data.info()


# ### Extracting Data Types (1)

# In[5]:


df1 = pd.DataFrame(data.dtypes,columns =['Data Type'])
df1


# #### No categorical encoding is required since all data types are either integer, float, or date-time. 

# ### Null Values (1)

# In[6]:


l1=[]
pnv = []

for i in data.columns:
    if data [i].count () < len (data):
        l1.append(i)
        pnv.append(100 * (len (data) - data [i].count ()) / len (data)) 

df2 = pd.DataFrame(list(zip(l1, pnv)),columns =['Attribute Name', 'Percentage of Null Values'])

df2


# #### We can fill the null values in distance and expected duration with each set's mean because their percentage of null values is low.

# In[7]:


data_imputer = IterativeImputer ()
data [['distance']] = data_imputer.fit_transform (data [['distance']])
data [['expected_duration']] = data_imputer.fit_transform (data [['expected_duration']])


# ### OD Matrix of Trip Requests (2)

# In[8]:


OD_Total = pd.DataFrame ({
    0: pd.Series ([0, 0, 0, 0, 0], index = [0, 1, 2, 3, 4]),
    1: pd.Series ([0, 0, 0, 0, 0], index = [0, 1, 2, 3, 4]),
    2: pd.Series ([0, 0, 0, 0, 0], index = [0, 1, 2, 3, 4]),
    3: pd.Series ([0, 0, 0, 0, 0], index = [0, 1, 2, 3, 4]),
    4: pd.Series ([0, 0, 0, 0, 0], index = [0, 1, 2, 3, 4])
    })

for i in range (0, 5):
    for j in range (0, 5):
        OD_Total.loc [i, j] = len (data.loc [(data ['origin'] == i) & (data ['destination'] == j)])
        
OD_Total1 = OD_Total.style.set_caption('OD Matrix of Trip Requests').set_table_styles([{'selector': 'caption','props':
        [('color', 'Black'),
        ('font-size', '13px'),
        ('font-weight', 'bold'),
        ('text-align', 'center')]}])

OD_Total1


# ### OD Matrix of Done Trips (3)

# In[9]:


OD_Done = pd.DataFrame ({
    0: pd.Series ([0, 0, 0, 0, 0], index = [0, 1, 2, 3, 4]),
    1: pd.Series ([0, 0, 0, 0, 0], index = [0, 1, 2, 3, 4]),
    2: pd.Series ([0, 0, 0, 0, 0], index = [0, 1, 2, 3, 4]),
    3: pd.Series ([0, 0, 0, 0, 0], index = [0, 1, 2, 3, 4]),
    4: pd.Series ([0, 0, 0, 0, 0], index = [0, 1, 2, 3, 4])
    })

for i in range (0, 5):
    for j in range (0, 5):
        OD_Done.loc [i, j] = len (data.loc [(data ['origin'] == i) & (data ['destination'] == j) & (data ['status'] == 1)])

OD_Done1 = OD_Done.style.set_caption('OD Matrix of Done Trips').set_table_styles([{'selector': 'caption','props':
        [('color', 'Black'),
        ('font-size', '13px'),
        ('font-weight', 'bold'),
        ('text-align', 'center')]}])

OD_Done1


# ### Unfulfilled Request (4)

# In[10]:


total = 0
done = 0
for i in range (0,5):
    done += sum(OD_Done[i])
    total += sum(OD_Total[i])
    
unfulfilled = total - done

l2 = ['Total','Done', 'Unfulfilled']
l3 = [total, done, unfulfilled]

df3 = pd.DataFrame(list(zip(l2, l3)),columns =['Type','Sum'])

df3 = df3.style.set_caption('Demand').set_table_styles([{'selector': 'caption','props':
        [('color', 'Black'),
        ('font-size', '13px'),
        ('font-weight', 'bold'),
        ('text-align', 'center')]}])

df3


# In[11]:


a = unfulfilled/total * 100
print('Percentage of unfulfilled trip requests is {0:.2f}%.'.format(a))


# In[12]:


df4 = (1 - (OD_Done / OD_Total)) * 100
df4 = df4.style.set_caption('Percentage of Requests Not Leading to Trips Per Each Origin and Destination').set_table_styles([{'selector': 'caption','props':
        [('color', 'Black'),
        ('font-size', '13px'),
        ('font-weight', 'bold'),
        ('text-align', 'center')]}])
df4


# In[13]:


M = 0
imax = 0
jmax = 0

for i in range (0, 5):
    for j in range (0, 5):
        if ((1 - (OD_Done / OD_Total)).fillna (0).loc [i, j] > M):
            M = (1 - (OD_Done / OD_Total)).loc [i, j]
            imax = i
            jmax = j
            

l5 = ['Origin (From)','Destination (To)','Unfulfilled Percentage']
l6 = [str(imax), str(jmax), (1 - (OD_Done / OD_Total)).loc [imax, jmax] * 100]

df5 = pd.DataFrame(list(zip(l5, l6)),columns =['',''])

df5


# ### Total Demand Chart (5)

# In[ ]:


copied_data = data.copy()
copied_data ['price_check_time'] = copied_data ['price_check_time'].apply (lambda x : pd.to_datetime (str (x)))
copied_data ['price_check_hour'] = copied_data ['price_check_time'].dt.hour
sns.histplot (copied_data, x = 'price_check_hour', kde = True, bins = 24).set (title = 'Total Demand Per Hour', xlabel = 'Hour', ylabel = 'Demand')


# ### Customer Behavior (6)

# In[14]:


price = data ['price']
status = data ['status']

def SS (x):
    average = x.mean()
    x = list (x)
    return sum (([(i-average) ** 2 for i in x]))


# In[15]:


Rprice = price [status == 2]
Aprice = price [status != 2]


# In[16]:


F = ((len (Rprice) * (Rprice.mean () - price.mean ()) ** 2 + len (Aprice) * (Aprice.mean () - price.mean ()) ** 2) / (2 - 1)) / ((SS (Rprice) + SS (Aprice)) / (data.shape [0] - 2))

f = scipy.stats.f.ppf (q= 0.95, dfn = 2 - 1, dfd = data.shape [0] - 2)


# In[17]:


if F <= f:
    print ("A meaningful correlation can be found between trip prices and trip cancellation by customers, based on fisher's test.")
else:
    print ("No meaningful correlation could be found between trip prices and trip cancellation by customers.")
print ('\nF Score: {0:.3f}'.format (F))
print ('\nF Critical Value: {0:.3f}'.format (f))


# In[ ]:


table = [list(pd.concat([Rprice,Aprice],ignore_index=True)),[2 for i in range(len(Rprice))]+[0 for i in range(len(Aprice))]]


# In[ ]:


from scipy.stats import chi2_contingency
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat , p))
if p>0.05:
    print("probably independent")
else:
    print("probably dependente")


# ### Drivers Behavior
# #### Effect of trip price on cancelling trip (7)

# In[18]:


Rprice = price [status == 3]
Aprice = price [status != 3]


# In[19]:


F = ((len (Rprice) * (Rprice.mean () - price.mean ()) ** 2 + len (Aprice) * (Aprice.mean () - price.mean ()) ** 2) / (2 - 1)) / ((SS (Rprice) + SS (Aprice)) / (data.shape [0] - 2))
f = scipy.stats.f.ppf (q= 0.95, dfn = 2 - 1, dfd = data.shape [0] - 2)


# In[20]:


if F <= f:
    print ("A meaningful correlation can be found between trip prices and trip cancellation by drivers, based on fisher's test.")
else:
    print ("No meaningful correlation could be found between trip prices and trip cancellation by drivers.")

print ('\nF Score: {0:.3f}'.format (F))
print ('\nF Critical Value: {0:.3f}'.format (f))


# In[ ]:


table = [list(pd.concat([Rprice,Aprice],ignore_index=True)),[3 for i in range(len(Rprice))]+[0 for i in range(len(Aprice))]]


# In[ ]:


stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat , p))
if p>0.05:
    print("probably independent")
else:
    print("probably dependente")


# In[ ]:


Rprice = pd.concat([price [status == 3] , price[status == 4]])
Aprice = pd.concat([price [status == 1] , price[status == 2]])


# In[ ]:


table = [list(pd.concat([Rprice,Aprice],ignore_index=True)),[3 for i in range(len(Rprice))]+[0 for i in range(len(Aprice))]]


# In[ ]:


stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat , p))
if p>0.05:
    print("probably independent")
else:
    print("probably dependente")


# #### Realtionship between cancelling trip in peak and nonpeak hours (8)

# In[21]:


hours = copied_data ['price_check_hour']


# In[22]:


Ptrips = hours [((hours > 6) & (hours < 8)) | ((hours > 16) & (hours < 19))]
Ntrips = hours [(hours < 6) | ((hours > 8) & (hours < 16)) | (hours > 19)]


# In[23]:


A = np.array ([1 - int (i == 3) for i in list(status)])


# In[24]:


PA = A [((hours > 6) & (hours < 8))| ((hours > 16) & (hours < 19))]
NA = A [(hours < 6) | ((hours > 8) & (hours < 16)) | (hours > 19)]


# In[25]:


p1 = 1 - (sum (PA) + sum (NA)) / (len (Ptrips) + len (Ntrips))
p2 = 1 - sum (PA) / sum (Ptrips)
p3 = 1 - sum (NA) / sum (Ntrips)


# In[26]:


z = (p2 - p3) / math.sqrt (p1 * (1 - p1) * (1 / len (Ptrips)))
n = scipy.stats.norm.ppf (0.95, 0, math.sqrt (p1 * (1 - p1) * (1 / len (Ptrips))))


# In[27]:


if abs (z) < abs (n):
    print ("No meaningful relationship could be found.")
else:
    print ("The probability of drivers' cancelling trips is affected by whether or not we are in the peak hours.")

print ('\n(z,n):','(',z, n,')')


# #### Effect of trip average price in different hours of day on cancelling trips (9)

# In[28]:


Newprice = price [status == 3]
hours = hours [status == 3]


# In[29]:


data2 = pd.DataFrame ({'New Price': list (Newprice),'Hours': list(hours)})
data2 = data2.groupby ('Hours') ['New Price'].mean ()


# In[30]:


data2.plot.line (x = "Hours", y = "New Price", rot = 70, title = "Average price of denied trips in each time period")

