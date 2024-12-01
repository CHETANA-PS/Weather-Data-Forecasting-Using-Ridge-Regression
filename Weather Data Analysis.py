#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


weather=pd.read_csv(r"C:\Users\VISALAKSHI.P.S\Downloads\3730345.csv")
weather


# In[5]:


weather=pd.read_csv(r"C:\Users\VISALAKSHI.P.S\Downloads\3730345.csv",index_col="DATE")


# In[6]:


weather


# In[7]:


null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]


# In[8]:


null_pct


# In[9]:


valid_columns = weather.columns[null_pct <.05]


# In[10]:


valid_columns


# In[11]:


weather = weather[valid_columns].copy()


# In[12]:


weather.columns = weather.columns.str.lower()


# In[13]:


weather


# In[14]:


weather = weather.ffill()


# In[15]:


weather.apply(pd.isnull).sum()


# In[16]:


weather.dtypes


# In[17]:


weather.index


# In[18]:


weather.index = pd.to_datetime(weather.index)


# In[19]:


weather.index


# In[21]:


weather.index.year.value_counts().sort_index()


# In[22]:


weather["snow"].plot()


# In[23]:


weather


# In[24]:


weather["target"] = weather.shift(-1)["tmax"]a


# In[25]:


weather


# In[26]:


weather = weather.ffill()


# In[27]:


weather


# In[31]:


from sklearn.linear_model import Ridge

rr = Ridge(alpha=.1)


# In[33]:


predictors = weather.columns[~weather.columns.isin(["target","name","station"])]


# In[34]:


predictors


# In[63]:


def backtest(weather,model,predictors,start=3650,step=90):
    all_predictions = []
    
    for i in range(start,weather.shape[0],step):
        train = weather.iloc[:i,:]
        test = weather.iloc[i:(i+step),:]
        
        model.fit(train[predictors],train["target"])
        
        preds = model.predict(test[predictors])
        
        preds = pd.Series(preds,index=test.index)
        combined = pd.concat([test["target"],preds],axis=1)
        
        combined.columns = ["actual","predictions"]
        
        combined["diff"] = (combined["predictions"] - combined["actual"]).abs()
        
        all_predictions.append(combined)
        
    return pd.concat(all_predictions)


# In[65]:


predictions = backtest(weather, rr, predictors)


# In[66]:


predictions


# In[67]:


from sklearn.metrics import mean_absolute_error

mean_absolute_error(predictions["actual"],predictions["predictions"])


# In[68]:


predictions["diff"].mean()


# In[69]:


def pct_diff(old,new):
    return(new - old) / old

def compute_rolling(weather,horizon,col):
    label = f"rolling_{horizon}_{col}"
    
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label],weather[col])
    return weather

rolling_horizons = [3,14]

for horizon in rolling_horizons:
    for col in ["tmax", "tmin","prcp"]:
        weather = compute_rolling(weather,horizon,col)
    


# In[70]:


weather


# In[71]:


weather = weather.iloc[14:,:]


# In[72]:


weather


# In[73]:


weather = weather.fillna(0)


# In[75]:


def expand_mean(df):
    return df.expanding(1).mean()

for col in ["tmax","tmin","prcp"]:
    weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month,group_keys=False).apply(expand_mean)
    weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year,group_keys=False).apply(expand_mean)


# In[76]:


weather


# In[78]:


predictors = weather.columns[~weather.columns.isin(["target","name","station"])]


# In[79]:


predictorspredictors


# In[86]:


predictions


# In[87]:


mean_absolute_error(predictions["actual"],predictions["predictions"])


# In[88]:


predictions.sort_values("diff",ascending=False)


# In[89]:


predictions["diff"].round().value_counts().sort_index()


# In[90]:


predictions["diff"].round().value_counts().sort_index().plot()


# In[ ]:




