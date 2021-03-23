#!/usr/bin/env python
# coding: utf-8

# ### Importing the libraries required

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot 
import statsmodels.formula.api as smf
import statsmodels.api as sm


# ### importing the dataset

# In[4]:


raw_data = pd.read_csv("50_Startups (1).csv")
raw_data


# ## Exploratory Data Analysis:

# ### Checking if there are any null value

# In[5]:


raw_data.info()


#  ### Creating dummy variables for the State column

# In[6]:


data = raw_data.copy()


# In[7]:


data = pd.get_dummies(data, columns=["State"])


# In[8]:


data


# ### Checking the correlation

# In[8]:


data.corr()


# ### Visualizing the correlation

# In[9]:


fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, ax=ax)


# In[10]:


sns.set_style(style = "darkgrid")
sns.pairplot(data)


# In[13]:


sns.pairplot(data, kind='reg', diag_kind='kde')


# ### Reanaming the features for convenience

# In[14]:


data.rename(columns={'R&D Spend': 'RnD_Spend', 'Marketing Spend': 'Marketing_Spend'}, inplace=True)


# ## Preparing the initial Model: 

# ###### Using all the Features except State_New_York so that we avoid multi collinearity also called (Dummy Variable trap)

# In[15]:


model = smf.ols("Profit~RnD_Spend+Administration+Marketing_Spend+State_California+State_Florida", data=data).fit()


# In[16]:


model.summary()


# ###### Observation: The P-values of Administration, Marketing Spend and State is higher than 0.05

# ### Checking the P value of these features individually

# In[19]:


model_admin = smf.ols("Profit~Administration", data= data).fit()
model_admin.summary()


# In[20]:


model_ms = smf.ols("Profit~Marketing_Spend", data= data).fit()
model_ms.summary()


# In[21]:


model_state_cali = smf.ols("Profit~State_California", data= data).fit()
model_state_cali.summary()


# In[22]:


model_state_florida = smf.ols("Profit~State_Florida", data= data).fit()
model_state_florida.summary()


# ###### Observation: The P-value of the Marketing Spend feature is significant when individually taken

# #### Lets calculate the Variance inflation Factor for each feature

# In[23]:


# Function to calculate VIF
def vif(x):
    if x != 1:
        vif_val = 1/(1-x)
    else:
        vif_val = 0
    return vif_val


# In[24]:


rsq_admin = smf.ols("Administration~RnD_Spend+Administration+Marketing_Spend+State_California+State_Florida",data=data).fit().rsquared
vif_admin = vif(rsq_admin)

rsq_ms = smf.ols("Marketing_Spend~RnD_Spend+Administration+State_California+State_Florida",data=data).fit().rsquared
vif_ms = vif(rsq_ms)

rsq_rnd = smf.ols("RnD_Spend~Marketing_Spend+Administration+State_California+State_Florida",data=data).fit().rsquared
vif_rnd = vif(rsq_rnd)

rsq_cali = smf.ols("State_California~RnD_Spend+Marketing_Spend+Administration+State_Florida",data=data).fit().rsquared
vif_cali = vif(rsq_cali)

rsq_flo = smf.ols("State_Florida~RnD_Spend+Marketing_Spend+Administration+State_California",data=data).fit().rsquared
vif_flo = vif(rsq_flo)

vif_table = pd.DataFrame({
            "Features":["State_Florida", "RnD_Spend",
                         "Marketing_Spend", "Administration", 
                        "State_California"],
            "VIF":[vif_flo, vif_rnd, vif_ms, vif_admin, vif_cali]
})

vif_table


# ###### Observation: The VIF of the features are all less then 4 which tells us that there is very less collinearity between the independant variables (Features) and therefore will not cause the problem of multi-collinearity  

# ### Making the Q-Q plot:

# In[25]:


qqplot = sm.qqplot(model.resid, line='q')
plt.title("Normal QQ plot of residuals")
plt.show


# In[26]:


model.resid


# In[27]:


# function to Standardize the values
def standardized_values(values):
    return (values - values.mean()/values.std())


# In[28]:


#residual plot for Homoscedasticity
plt.scatter(standardized_values(model.fittedvalues), standardized_values(model.resid))


# ### Residual vs regressors:

# In[29]:


# Administration
fig = plt.figure(figsize=(15,8))
fif = sm.graphics.plot_regress_exog(model, "Administration", fig = fig)
plt.show


# In[30]:


# R&D Spend
fig = plt.figure(figsize=(15,8))
fif = sm.graphics.plot_regress_exog(model, "RnD_Spend", fig = fig)
plt.show


# In[31]:


# Marketing_Spend
fig = plt.figure(figsize=(15,8))
fif = sm.graphics.plot_regress_exog(model, "Marketing_Spend", fig = fig)
plt.show


# In[32]:


# State_California
fig = plt.figure(figsize=(15,8))
fif = sm.graphics.plot_regress_exog(model, "State_California", fig = fig)
plt.show


# In[33]:


# State_Florida
fig = plt.figure(figsize=(15,8))
fif = sm.graphics.plot_regress_exog(model, "State_Florida", fig = fig)
plt.show


# ## Model deletion Diagnostics

# ### Detecting the influencers/ outliers

# ### Cook's distance:

# In[34]:


model_influence = model.get_influence()
(c,_) = model_influence.cooks_distance


# In[35]:


c


# In[36]:


### Plotting the Cooks Distance:
figure = plt.subplots(figsize = (20,7))
plt.stem(np.arange(len(data)), np.round(c,3))
plt.xlabel("Row index")
plt.ylabel("Cook's distance")
plt.show


# In[37]:


np.argmax(c), np.max(c)


# ### High influence plot:

# In[38]:


influence_plot(model)
plt.show()


# In[39]:


# finding the cut off value
k = data.shape[1]
n = data.shape[0]
leverage_cutoff = 3 * ((k+1)/ n)
leverage_cutoff


# ###### The cut off value is higher so we are not dropping any rows

# #### Checking the AIC of the model

# In[40]:


model.aic


# #### The p value of State California was higher than 0.05, let us remove the column and see if there is any improvement in the model

# In[41]:


model_2 = smf.ols("Profit~RnD_Spend+Administration+Marketing_Spend+State_Florida", data = data).fit()


# In[42]:


model_2.summary()


# ###### Observation: When we removed the whole row of State California we see that the adj rsquared value has increased just a little but the F Statistic has increased tremendously showing that the features are more relevant using the ANOVA test

# ### Backward Elimination:

# In[43]:


# The pvalue of State_florida is high too, lets remove that and see if there is any improvement

model_3 = smf.ols("Profit~RnD_Spend+Administration+Marketing_Spend", data = data).fit()
model_3.summary()


# ##### Observation: The adj rsquared value has become better and also the F Statistic has shot up showing that we have remmoved variables that are irrelevant

# In[44]:


# removing the Administration Feature as the Pvalue is too high
model_4 = smf.ols("Profit~RnD_Spend+Marketing_Spend", data = data).fit()
model_4.summary()


# In[45]:


qqplot = sm.qqplot(model_4.resid, line='q')
plt.title("Normal QQ plot of residuals")
plt.show


# In[46]:


#residual plot for Homoscedasticity
plt.scatter(standardized_values(model_4.fittedvalues), standardized_values(model_4.resid))


# In[47]:


model_influence = model_4.get_influence()
(c,_) = model_influence.cooks_distance


# In[48]:


c


# In[49]:


### Plotting the Cooks Distance:
figure = plt.subplots(figsize = (20,7))
plt.stem(np.arange(len(data)), np.round(c,3))
plt.xlabel("Row index")
plt.ylabel("Cook's distance")
plt.show


# In[50]:


np.argmax(c), np.max(c)


# ##### Observation: We can see that the 49th argument though not above the value of 1 is significantly different from others

# In[51]:


influence_plot(model_4)
plt.show()


# ###### Observation: Here too we can see that the 49th observation is highly influential.

# In[52]:


# removing the 49th observation and checking if the model is getting any better:
data[data.index.isin([49])]


# In[53]:


data_copy = data.copy()
data_copy = data_copy.drop(data_copy.index[49], axis=0).reset_index()
data_copy


# ### Model without the 49th row

# In[54]:


model_5 = smf.ols("Profit~RnD_Spend+Marketing_Spend", data = data_copy).fit()
model_5.summary()


# ###### The pvalue of Marketing Spend (feature) is higher than 0.05 therefore lets remove and check the model

# In[55]:


# Making a data frame to check which model is the best


model_dict = {
    "Model": ["model", "model_2", "model_3", "model_4", "model_5"],
    "Model Info": ["All features", "excluding State_California", 
                   "excluding State_California & State_Florida",
                  "Using only R&D Spend and Marketing Spend", 
                  "Using only R&D Spend and Marketing Spend & removing influencing data"],
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


# ###### Obsevation: Model_5 is the best model as the adj rsquared and AIC are better than others

# In[ ]:




