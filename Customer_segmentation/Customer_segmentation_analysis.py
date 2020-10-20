#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pylab as plt

from datetime import datetime

# pd.set_option('display.max_columns', 25)


# In[27]:


df_transactions = pd.read_csv('transactions_n100000.csv')
df = df_transactions.pivot(index='ticket_id', columns='item_name', values='item_count').fillna(0)
df_transactions.reset_index(inplace=True, drop=True)
df = df.merge(df_transactions[['ticket_id', 'location', 'order_timestamp']].drop_duplicates(), how='left', on='ticket_id')

df['order_timestamp'] = df['order_timestamp'].apply(lambda st: datetime.strptime(st, '%Y-%m-%d %H:%M:%S'))
df['hour'] = df.order_timestamp.apply(lambda x: x.hour)


# In[28]:


print(df.shape)
df.head()


# In[29]:


dummy_store = pd.get_dummies(df['location'])
store_name_dic = {i:'store_'+str(i) for i in dummy_store.columns}
dummy_store.rename(columns=store_name_dic, inplace=True)

# dummy_hour = pd.get_dummies(df['hour'])
# hour_name_dic = {i:'hour'+str(i) for i in dummy_hour.columns}
# dummy_hour.rename(columns=hour_name_dic, inplace=True)


# In[30]:


# df_cluster = pd.concat([df[['burger', 'fries', 'salad', 'shake']], dummy_store, dummy_hour], axis=1)
df_cluster = pd.concat([df[['burger', 'fries', 'salad', 'shake', 'hour']], dummy_store], axis=1)

df_cluster.head()


# In[31]:


elbow_loss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, max_iter=300).fit(df_cluster.to_numpy())
    elbow_loss.append(kmeans.inertia_ / df_cluster.shape[0] / i)
    
plt.plot(range(1, 10), elbow_loss)
plt.xlabel('cluster number')
plt.ylabel('elbow loss')
plt.show()


# In[32]:


n_clusters = 3
init_point_selection_method = 'k-means++'
attribute_summary_method_dict = {'burger': np.mean, 'fries': np.mean, 'salad': np.mean, 'shake': np.mean, 'hour': np.mean, 'store_1': sum, 'store_4': sum, 'store_6': sum, 'store_3': sum, 'store_9': sum, 'store_2': sum, 'store_8': sum, 'store_5': sum, 'store_7': sum}
col_output_order = ['burger', 'fries', 'salad', 'shake', 'hour', 'store_1', 'store_2', 'store_3', 'store_4', 'store_5', 'store_6', 'store_7', 'store_8', 'store_9'] 

def run_kmeans(n_clusters_f, init_f, df_f):
    k_means_model_f = KMeans(n_clusters=n_clusters_f, max_iter=300, random_state=5).fit(df_f.to_numpy())

    df_f['predict_cluster_kmeans'] = k_means_model_f.labels_ 
    
    k_means_model_f_summary = df_f.groupby('predict_cluster_kmeans').agg(attribute_summary_method_dict)
    return k_means_model_f, k_means_model_f_summary

model, model_summary = run_kmeans(n_clusters, init_point_selection_method, df_cluster)


# In[33]:


store_col_names = ['store_1', 'store_2', 'store_3', 'store_4', 'store_5', 'store_6', 'store_7', 'store_8', 'store_9']
df_cluster['store'] = None
for t_col in store_col_names:
    df_cluster.loc[df_cluster[t_col] == 1, 'store'] = t_col.split('_')[1]


# In[34]:


df_cluster


# In[35]:


t_df = df_cluster.groupby('store')['predict_cluster_kmeans'].apply(lambda x: x.mode()).reset_index()[['store', 'predict_cluster_kmeans']]
t_df['store'] = t_df['store'].astype(int)
original_df = df_transactions[['location', 'lat', 'long']].drop_duplicates().sort_values(['location']).reset_index(drop=True)
store_df = original_df.merge(t_df, how='left', left_on='location', right_on='store')#.to_csv('store_locations.csv')
store_df


# In[36]:


cluster1 = df_cluster[df_cluster.predict_cluster_kmeans==0].reset_index(drop=True)
cluster2 = df_cluster[df_cluster.predict_cluster_kmeans==1].reset_index(drop=True)
cluster3 = df_cluster[df_cluster.predict_cluster_kmeans==2].reset_index(drop=True)


# In[37]:


df1 = pd.DataFrame.from_dict(dict(cluster1.store.value_counts()), orient='index')
df2 = pd.DataFrame.from_dict(dict(cluster2.store.value_counts()), orient='index')
df3 = pd.DataFrame.from_dict(dict(cluster3.store.value_counts()), orient='index')


# In[38]:


store_df_cluster = pd.merge(pd.merge(df1, df2, left_index=True, right_index=True), df3, left_index=True, right_index=True)
# store_df = store_df.reset_index().sort_values(by='index').reset_index(drop=True)
store_df_cluster.rename(columns={'0_x':'cluster_1', '0_y':'cluster_2', 0:'cluster_3'}, inplace=True)


# In[42]:


store_df_cluster


# In[43]:


store_df_cluster.to_excel('store_percentage.xlsx')


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(data=store_df_cluster['cluster_1'], legend='brief', label='cluster 1')
sns.lineplot(data=store_df_cluster['cluster_2'], legend='brief', label='cluster 2')
sns.lineplot(data=store_df_cluster['cluster_3'], legend='brief', label='cluster 3')
plt.xlabel('store_number')
plt.ylabel('order_quantity')
# plt.legend() 
# cluster1: 4, 7, 9
# cluster2: 2, 6
# cluster3: 1, 3, 5, 8


# In[23]:


store_df_cluster


# In[16]:


store_df


# In[22]:


plt.figure(figsize=(20,10))
plt.scatter(x=store_df.long, y=store_df.lat)
for i in range(len(store_df['store'])):
    txt = str(store_df['store'][i])
    plt.annotate(txt,(store_df.long[i],store_df.lat[i]))
plt.show()


# In[17]:


plt.figure(figsize=(20,10))
plt.scatter(x=store_df.lat, y=store_df.long)
for i in range(len(store_df['store'])):
    txt = str(store_df['store'][i])
    plt.annotate(txt,(store_df.lat[i],store_df.long[i]))
plt.show()


# In[18]:


df1 = pd.DataFrame.from_dict(dict(cluster1.hour.value_counts()), orient='index')
df2 = pd.DataFrame.from_dict(dict(cluster2.hour.value_counts()), orient='index')
df3 = pd.DataFrame.from_dict(dict(cluster3.hour.value_counts()), orient='index')

store_df_cluster = pd.merge(pd.merge(df1, df2, left_index=True, right_index=True, how='outer'), df3, left_index=True, right_index=True, how='outer')
store_df_cluster.rename(columns={'0_x':'cluster_1', '0_y':'cluster_2', 0:'cluster_3'}, inplace=True)
store_df_cluster.fillna(0, inplace=True)

sns.lineplot(data=store_df_cluster['cluster_1'], legend='brief', label='cluster 1')
sns.lineplot(data=store_df_cluster['cluster_2'], legend='brief', label='cluster 2')
sns.lineplot(data=store_df_cluster['cluster_3'], legend='brief', label='cluster 3')
plt.xlabel('hour')
plt.ylabel('order_quantity')


# In[48]:


cluster1['order4'] = [(cluster1.burger[i], cluster1.fries[i], cluster1.salad[i], cluster1.shake[i]) for i in range(cluster1.shape[0])]
cluster1['order3'] = [(cluster1.burger[i], cluster1.fries[i], cluster1.shake[i]) for i in range(cluster1.shape[0])]
cluster1['order1'] = [(cluster1.burger[i]) for i in range(cluster1.shape[0])]

cluster2['order4'] = [(cluster2.burger[i], cluster2.fries[i], cluster2.salad[i], cluster2.shake[i]) for i in range(cluster2.shape[0])]
cluster2['order3'] = [(cluster2.burger[i], cluster2.fries[i], cluster2.shake[i]) for i in range(cluster2.shape[0])]
cluster2['order1'] = [(cluster2.burger[i]) for i in range(cluster2.shape[0])]

cluster3['order4'] = [(cluster3.burger[i], cluster3.fries[i], cluster3.salad[i], cluster3.shake[i]) for i in range(cluster3.shape[0])]
cluster3['order3'] = [(cluster3.burger[i], cluster3.fries[i], cluster3.shake[i]) for i in range(cluster3.shape[0])]
cluster3['order1'] = [(cluster3.burger[i]) for i in range(cluster3.shape[0])]


# In[52]:


cluster1['order10'] = [(cluster1.fries[i]) for i in range(cluster1.shape[0])]
cluster2['order10'] = [(cluster2.fries[i]) for i in range(cluster2.shape[0])]
cluster3['order10'] = [(cluster3.fries[i]) for i in range(cluster3.shape[0])]

cluster1['order11'] = [(cluster1.shake[i]) for i in range(cluster1.shape[0])]
cluster2['order11'] = [(cluster2.shake[i]) for i in range(cluster2.shape[0])]
cluster3['order11'] = [(cluster3.shake[i]) for i in range(cluster3.shape[0])]

cluster1['order12'] = [(cluster1.salad[i]) for i in range(cluster1.shape[0])]
cluster2['order12'] = [(cluster2.salad[i]) for i in range(cluster2.shape[0])]
cluster3['order12'] = [(cluster3.salad[i]) for i in range(cluster3.shape[0])]


# In[53]:


df1 = pd.DataFrame.from_dict(dict(cluster1.order1.value_counts()), orient='index')
df2 = pd.DataFrame.from_dict(dict(cluster2.order1.value_counts()), orient='index')
df3 = pd.DataFrame.from_dict(dict(cluster3.order1.value_counts()), orient='index')

store_df_cluster = pd.merge(pd.merge(df1, df2, left_index=True, right_index=True, how='outer'), df3, left_index=True, right_index=True, how='outer')
store_df_cluster.rename(columns={'0_x':'cluster_1', '0_y':'cluster_2', 0:'cluster_3'}, inplace=True)
store_df_cluster.fillna(0, inplace=True)

store_df_cluster
# burger


# In[54]:


df1 = pd.DataFrame.from_dict(dict(cluster1.order10.value_counts()), orient='index')
df2 = pd.DataFrame.from_dict(dict(cluster2.order10.value_counts()), orient='index')
df3 = pd.DataFrame.from_dict(dict(cluster3.order10.value_counts()), orient='index')

store_df_cluster = pd.merge(pd.merge(df1, df2, left_index=True, right_index=True, how='outer'), df3, left_index=True, right_index=True, how='outer')
store_df_cluster.rename(columns={'0_x':'cluster_1', '0_y':'cluster_2', 0:'cluster_3'}, inplace=True)
store_df_cluster.fillna(0, inplace=True)

store_df_cluster
# fries


# In[55]:


df1 = pd.DataFrame.from_dict(dict(cluster1.order11.value_counts()), orient='index')
df2 = pd.DataFrame.from_dict(dict(cluster2.order11.value_counts()), orient='index')
df3 = pd.DataFrame.from_dict(dict(cluster3.order11.value_counts()), orient='index')

store_df_cluster = pd.merge(pd.merge(df1, df2, left_index=True, right_index=True, how='outer'), df3, left_index=True, right_index=True, how='outer')
store_df_cluster.rename(columns={'0_x':'cluster_1', '0_y':'cluster_2', 0:'cluster_3'}, inplace=True)
store_df_cluster.fillna(0, inplace=True)

store_df_cluster
# shake


# In[56]:


df1 = pd.DataFrame.from_dict(dict(cluster1.order12.value_counts()), orient='index')
df2 = pd.DataFrame.from_dict(dict(cluster2.order12.value_counts()), orient='index')
df3 = pd.DataFrame.from_dict(dict(cluster3.order12.value_counts()), orient='index')

store_df_cluster = pd.merge(pd.merge(df1, df2, left_index=True, right_index=True, how='outer'), df3, left_index=True, right_index=True, how='outer')
store_df_cluster.rename(columns={'0_x':'cluster_1', '0_y':'cluster_2', 0:'cluster_3'}, inplace=True)
store_df_cluster.fillna(0, inplace=True)

store_df_cluster
# salad


# In[20]:


df1 = pd.DataFrame.from_dict(dict(cluster1.order3.value_counts()), orient='index')
df2 = pd.DataFrame.from_dict(dict(cluster2.order3.value_counts()), orient='index')
df3 = pd.DataFrame.from_dict(dict(cluster3.order3.value_counts()), orient='index')

store_df_cluster = pd.merge(pd.merge(df1, df2, left_index=True, right_index=True, how='outer'), df3, left_index=True, right_index=True, how='outer')
store_df_cluster.rename(columns={'0_x':'cluster_1', '0_y':'cluster_2', 0:'cluster_3'}, inplace=True)
store_df_cluster.fillna(0, inplace=True)

store_df_cluster
# sns.lineplot(data=store_df_cluster['cluster_1'], legend='brief', label='cluster 1')
# sns.lineplot(data=store_df_cluster['cluster_2'], legend='brief', label='cluster 2')
# sns.lineplot(data=store_df_cluster['cluster_3'], legend='brief', label='cluster 3')
# plt.xlabel('hour')
# plt.ylabel('order_quantity')


# In[21]:


df1 = pd.DataFrame.from_dict(dict(cluster1.salad.value_counts()), orient='index')
df2 = pd.DataFrame.from_dict(dict(cluster2.salad.value_counts()), orient='index')
df3 = pd.DataFrame.from_dict(dict(cluster3.salad.value_counts()), orient='index')

store_df_cluster = pd.merge(pd.merge(df1, df2, left_index=True, right_index=True, how='outer'), df3, left_index=True, right_index=True, how='outer')
store_df_cluster.rename(columns={'0_x':'cluster_1', '0_y':'cluster_2', 0:'cluster_3'}, inplace=True)
store_df_cluster.fillna(0, inplace=True)

store_df_cluster
# sns.lineplot(data=store_df_cluster['cluster_1'], legend='brief', label='cluster 1')
# sns.lineplot(data=store_df_cluster['cluster_2'], legend='brief', label='cluster 2')
# sns.lineplot(data=store_df_cluster['cluster_3'], legend='brief', label='cluster 3')
# plt.xlabel('hour')
# plt.ylabel('order_quantity')

