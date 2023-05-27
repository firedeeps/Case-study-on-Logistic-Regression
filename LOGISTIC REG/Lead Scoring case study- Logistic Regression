#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import time, warnings
import datetime as dt


# 
# # Reading the data

# In[2]:


xleads = pd.read_csv("Leads.csv")


# In[3]:


xleads.head()


# In[4]:


xleads.shape


# In[5]:


xleads.info()


# In[6]:


xleads.describe()


# # Data cleaning and preparation

# In[7]:


#check for null values
xleads.isnull().sum()


# As there are lot of null values that to in many columns it is not possible to impute the missing values so instead we delete/ drop the columns
# them as there are 9000 datapoints.

# In[8]:


#Dropping the columns which have null values
for col in xleads.columns:
        if xleads[col].isnull().sum()>3000:
            xleads.drop(col, 1 , inplace  = True)


# In[9]:


xleads.isnull().sum()


# Now lets at the columns individually to remove the columns aren't much of a use we drop them

# In[10]:


#Columns like Country and city are really not of use in the leads
xleads.drop(['City'], axis = 1 , inplace = True)


# In[11]:


xleads.drop(['Country'], axis = 1 , inplace = True)


# In[12]:


#Lets check the percentage of missing values 
round(100*(xleads.isnull().sum()/len(xleads.index)), 2)


# As we know the selection of the leads are done so the columns that have select lets have a value count check to get an idea.

# In[13]:


for column in xleads:
    print(xleads[column].astype('category').value_counts())
    print('___________________________________________________')


# The select is in three columns now.

# In[14]:


xleads['Lead Profile'].astype('category').value_counts()


# In[15]:


xleads['How did you hear about X Education'].astype('category').value_counts()


# In[16]:


xleads['Specialization'].astype('category').value_counts()


# In[17]:


#lets drop the select columns
xleads.drop(['Lead Profile', 'How did you hear about X Education'], axis = 1, inplace = True)


# There is one more information that we got from cell 21 i.e, There are so many columns that have only one value dominantly so maybe we can remove them .Lets do that.

# In[18]:


xleads.drop(['Do Not Call', 'Search', 'Magazine', 'Newspaper Article', 'X Education Forums', 'Newspaper', 
            'Digital Advertisement', 'Through Recommendations', 'Receive More Updates About Our Courses', 
            'Update me on Supply Chain Content', 'Get updates on DM Content', 
            'I agree to pay the amount through cheque'], axis = 1, inplace = True)


# In[19]:


xleads.drop(['What matters most to you in choosing a course'], axis = 1, inplace = True)


# In[20]:


xleads.isnull().sum()


# What is your current occupation have more than 2500 null values As already so many columns have been lost Now we remove rows.

# In[21]:


xleads = xleads[~pd.isnull(xleads['What is your current occupation'])]


# In[22]:


xleads.isnull().sum()


# In[23]:


# Drop the null value rows in the column 'TotalVisits'
xleads = xleads[~pd.isnull(xleads['TotalVisits'])]


# In[24]:


xleads.isnull().sum()


# In[25]:


# Drop the null value rows in the column 'lead source'
xleads = xleads[~pd.isnull(xleads['Lead Source'])]


# In[26]:


xleads.isnull().sum()


# In[27]:


# Drop the null values rows in the column 'Specialization'

xleads = xleads[~pd.isnull(xleads['Specialization'])]


# In[28]:


xleads.isnull().sum()


# In[29]:


len(xleads.index)


# In[30]:


xleads.head()


# As we look at the data the first two columns i.e, Prospect ID and Lead Number are of no particular use in the analysis so lets drop those columns.

# In[31]:


xleads.drop(['Prospect ID'], 1, inplace = True)


# In[32]:


xleads.drop(['Lead Number'], 1, inplace = True)


# In[33]:


xleads.head()


# # Data modelling 

# In[34]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[35]:


sns.pairplot(xleads, diag_kind = 'kde' , hue='Converted')
plt.show()


# In[36]:


xedu = xleads[['TotalVisits','Total Time Spent on Website','Page Views Per Visit','Converted']]
sns.pairplot(xedu,diag_kind='kde',hue='Converted')
plt.show()


# In[37]:


from sklearn.preprocessing import PowerTransformer


# In[38]:


pt = PowerTransformer()
transformedxedu = pd.DataFrame(pt.fit_transform(xedu))
transformedxedu.columns = xedu.columns
transformedxedu.head()


# In[39]:


sns.pairplot(transformedxedu,diag_kind='kde',hue='Converted')
plt.show()


# # Dummy variables creation
# The next is to deal with categorical variables in the dataset.

# In[40]:


temp = xleads.loc[:, xleads.dtypes=='object']
temp.columns


# In[41]:


dummy = pd.get_dummies(xleads[['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
       'Specialization', 'What is your current occupation',
       'A free copy of Mastering The Interview', 'Last Notable Activity']], drop_first = True)

xleads = pd.concat([xleads, dummy], axis=1)


# In[42]:


# Creating dummy variable separately for the variable 'Specialization' since it has the level 'Select' which is useless so we
# drop that level by specifying it explicitly

dummy_spl = pd.get_dummies(xleads['Specialization'], prefix = 'Specialization')
dummy_spl = dummy_spl.drop(['Specialization_Select'], 1)
xleads = pd.concat([xleads, dummy_spl], axis = 1)


# In[43]:


# Drop the variables for which the dummy variables have been created

xleads = xleads.drop(['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                   'Specialization', 'What is your current occupation',
                   'A free copy of Mastering The Interview', 'Last Notable Activity'], 1)


# In[44]:


xleads.head()


# # TEST-TRAIN SPLIT

# In[45]:


#Tmporting the library

from sklearn.model_selection import train_test_split


# In[46]:


X = xleads.drop(['Converted'],1)
X.head()


# In[47]:


# y is the target variable
y = xleads['Converted']
y.head()


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# # Scaling

# In[49]:


from sklearn.preprocessing import MinMaxScaler


# In[50]:


scaler = MinMaxScaler()
X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])

X_train.head()


# In[51]:


#Heatmap to check corelations
sns.heatmap(xleads.corr())
plt.show()
plt.figure(figsize= (25,15))


# # MODEL BUILDING

# In[52]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[53]:


#using RFE we'll select 15 variables
from sklearn.feature_selection import RFE


# In[54]:


rfe = RFE(logreg,n_features_to_select= 15)
rfe = rfe.fit(X_train, y_train)


# In[55]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[56]:


col = X_train.columns[rfe.support_]


# In[57]:


X_train = X_train[col]


# In[58]:


import statsmodels.api as sm


# In[59]:


# Fit a logistic Regression model on X_train after adding a constant and output the summary

X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[60]:


# Import 'variance_inflation_factor'

from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[61]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[62]:


X_train.drop('Last Notable Activity_Had a Phone Conversation', axis = 1, inplace = True)


# In[63]:


logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[64]:


X_train.drop('What is your current occupation_Housewife', axis = 1, inplace = True)


# In[65]:


logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[66]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[67]:


X_train.drop('What is your current occupation_Working Professional', axis = 1, inplace = True)


# In[68]:


logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# All the p-values are now in the appropriate range. Let's also check the VIFs again in case we had missed something.

# In[69]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# # MODEL EVALUATION

# As we see both p=values and VIF are appropriate now we can proceed now.
# 

# In[70]:


# Getting the Predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[71]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[72]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[73]:


y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[74]:


from sklearn import metrics


# In[75]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[76]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[77]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives

# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# In[79]:


# Let us calculate specificity
TN / float(TN+FP)


# In[80]:


#Calculate false postive rate 
print(FP/ float(TN+FP))


# In[81]:


# positive predictive value 
print (TP / float(TP+FP))


# In[82]:


# Negative predictive value
print (TN / float(TN+ FN))


# # ROC CURVE

# In[83]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[84]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[85]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[86]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[87]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[88]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[89]:


#### From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[90]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final[['Converted','Converted_prob','Prospect ID','final_Predicted','Lead_Score']].head()


# In[91]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[92]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion2


# In[93]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[94]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[95]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[96]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[97]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[98]:


#Looking at the confusion matrix again

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion


# In[99]:


##### Precision
TP / TP + FP

confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[100]:


##### Recall
TP / TP + FN

confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[101]:


from sklearn.metrics import precision_score, recall_score


# In[102]:


precision_score(y_train_pred_final.Converted , y_train_pred_final.final_Predicted)


# In[103]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[104]:


from sklearn.metrics import precision_recall_curve


# In[105]:


y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[106]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[107]:


#scaling test set

num_cols=X_test.select_dtypes(include=['float64', 'int64']).columns

X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

X_test.head()


# In[108]:


X_test = X_test[col]
X_test.head()


# In[109]:


X_test_sm = sm.add_constant(X_test)


# In[110]:


y_test_pred = res.predict(X_test_sm)


# In[111]:


y_test_pred[:10]


# In[112]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[113]:


# Let's see the head
y_pred_1.head()


# In[114]:


# Converting y_test to dataframe\
y_test_df = pd.DataFrame(y_test)


# In[115]:


# Putting CustID to index
y_test_df['Prospect ID'] = y_test_df.index


# In[116]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[117]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[118]:


y_pred_final.head()


# In[119]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})


# In[120]:


y_pred_final.head()


# In[121]:


# Rearranging the columns
y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))


# In[122]:


y_pred_final.head()


# In[123]:


y_pred_final['final_Predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.3 else 0)


# In[124]:


y_pred_final.head()


# In[125]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[126]:


confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion2


# In[127]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[128]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[129]:


# Let us calculate specificity
TN / float(TN+FP)


# In[130]:


precision_score(y_pred_final.Converted , y_pred_final.final_Predicted)


# In[131]:


recall_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[132]:


hot_leads=y_pred_final.loc[y_pred_final["Lead_Score"]>=85]
hot_leads


# In[133]:


print("The Prospect ID of the customers which should be contacted are :")

hot_leads_ids = hot_leads["Prospect ID"].values.reshape(-1)
hot_leads_ids


# In[134]:


res.params.sort_values(ascending=False)


# In[ ]:




