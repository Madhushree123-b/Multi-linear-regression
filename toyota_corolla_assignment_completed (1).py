#!/usr/bin/env python
# coding: utf-8

# ### Importing the libraries:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot 
import statsmodels.formula.api as smf
import statsmodels.api as sm


# In[2]:


raw_data = pd.read_csv("ToyotaCorolla (1).csv", encoding="ISO-8859-1")
cols_needed = ["Price", "Age_08_04","KM", "HP", "cc", "Doors", "Gears", "Quarterly_Tax", "Weight"]
raw_data = raw_data[(cols_needed)]
raw_data


# ### EDA:

# In[3]:


raw_data.info()


# In[4]:


raw_data.describe()


# In[5]:


raw_data.skew()


# In[6]:


raw_data.corr()


# In[7]:


fig, ax = plt.subplots(figsize=(10,6))  
sns.heatmap(raw_data.corr(), annot=True , ax=ax)


# In[8]:


sns.pairplot(raw_data)


# ### Making the initial model:

# In[9]:


model = smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data=raw_data).fit()
model.summary()


# ###### Observation: The the p-values of the features CC & Doors are higher than 0.05 & therefore there is a weak evidence to reject the null hypothesis.

# ### Checking the pvalues off these features individualy:

# In[10]:


model_cc = smf.ols("Price~cc", data=raw_data).fit()
model_cc.summary()


# In[11]:


model_doors = smf.ols("Price~Doors", data= raw_data).fit()
model_doors.summary()


# ###### Observation: Both the features, CC & Doors do not make quite a difference to the model

# In[12]:


model_2 = smf.ols("Price~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight", data=raw_data).fit()
model_2.summary()


# In[13]:


def vif_calc(x):
    if x != 0:
        1/(1-x)
        return x
    else:
        return x       


# In[14]:


rsq_age = smf.ols("Age_08_04~KM+HP+Gears+Quarterly_Tax+Weight", data=raw_data).fit().rsquared
vif_age = vif_calc(rsq_age)

rsq_km = smf.ols("KM~+HP+Gears+Quarterly_Tax+Weight+Age_08_04", data=raw_data).fit().rsquared
vif_km = vif_calc(rsq_km)

rsq_hp = smf.ols("HP~+KM+Gears+Quarterly_Tax+Weight+Age_08_04", data=raw_data).fit().rsquared
vif_hp = vif_calc(rsq_hp)

rsq_gears = smf.ols("Gears~+HP+KM+Quarterly_Tax+Weight+Age_08_04", data=raw_data).fit().rsquared
vif_gears = vif_calc(rsq_gears)

rsq_quat_tax = smf.ols("Quarterly_Tax~+HP+Gears+KM+Weight+Age_08_04", data=raw_data).fit().rsquared
vif_quat_tax = vif_calc(rsq_quat_tax)

rsq_weight = smf.ols("Weight~+HP+Gears+KM+Quarterly_Tax+Age_08_04", data=raw_data).fit().rsquared
vif_weight = vif_calc(rsq_weight)

# Making a Dataframe
vif_dict = {"Features": ["Weight","HP","Gears","KM","Quarterly_Tax","Age_08_04"],
           "VIF": [vif_weight,vif_hp,vif_gears,vif_km,vif_quat_tax,vif_age]}

vif_box = pd.DataFrame(vif_dict)
vif_box


# ### Making a QQ plot to check Normal Distribution:

# In[15]:


qqplot=sm.qqplot(model_2.resid,line='q') 
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[16]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[17]:


plt.scatter(get_standardized_values(model_2.fittedvalues),
            get_standardized_values(model_2.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[18]:


model_influence = model_2.get_influence()
(c, _) = model_influence.cooks_distance


# In[19]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(raw_data)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[20]:


influence_plot(model_2)
plt.show()


# In[21]:


k = raw_data.shape[1]
n = raw_data.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# ###### Observation: The 221st row is highly influencing the data.

# In[22]:


raw_data[raw_data.index.isin([221])]


# In[23]:


raw_data.head()


# In[24]:


raw_data.tail()


# In[25]:


data = raw_data.copy()
data = data.drop(data.index[[221]], axis=0).reset_index()
data


# In[26]:


data=data.drop(['index'],axis=1)


# In[27]:


data


# ### Making a model without the influencing data:

# In[28]:


model_3 = smf.ols("Price~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight", data=data).fit()
model_3.summary()


# In[29]:


influence_plot(model_3)
plt.show()


# In[30]:


data1 = data.drop(data.index[[959]], axis=0).reset_index()
data1=data1.drop(['index'],axis=1)


# In[31]:


data1


# In[32]:


model_4 = smf.ols("Price~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight", data=data1).fit()
model_4.summary()


# In[33]:


model_5 = smf.ols("Price~Age_08_04+KM+HP+Gears+Weight", data=data1).fit()
model_5.summary()


# In[34]:


# Making a data frame to check which model is the best


model_dict = {
    "Model": ["model", "model_2", "model_3", "model_4", "model_5"],
    "Model Info": ["All features", "excluding cc & Doors", 
                   "removing the influential data row",
                  "removing another influential data row", 
                  "removing the Quaterly_tax feature"],
    
    "rsquared": [model.rsquared, model_2.rsquared, 
                 model_3.rsquared, model_4.rsquared,
                 model_5.rsquared],
    "Adj rsquared": [model.rsquared_adj, model_2.rsquared_adj, 
                 model_3.rsquared_adj, model_4.rsquared_adj,
                 model_5.rsquared_adj],
    "AIC": [model.aic, model_2.aic, 
                 model_3.aic, model_4.aic,
                 model_5.aic]
}

model_check = pd.DataFrame(model_dict)
model_check


# ###### Conclusion: model 5 is the best model.

# In[ ]:





# In[ ]:





# In[ ]:




