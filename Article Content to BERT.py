#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ


# In[2]:


import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
import pandas as pd


# In[3]:


# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

## Set CPU as available physical device
#devices = tf.config.experimental.list_physical_devices(device_type='CPU')
#tf.config.experimental.set_visible_devices(devices= devices, device_type='CPU')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


# In[4]:


# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# model
model = TFBertModel.from_pretrained('bert-base-uncased')
model.summary()


# In[5]:


dataset = pd.read_csv('dataset.csv', index_col=0)
#display(dataset.loc['Content'].values)


# In[6]:


dataset_content = dataset.loc['Content'].fillna('').values
#print(dataset_content)


# In[7]:


encoded_strings = [tokenizer.encode(content, max_length=512, pad_to_max_length=True) for content in dataset_content]


# In[8]:


encoded_vectors = tf.constant(encoded_strings)

outputs = model(encoded_vectors)
last_hidden_states = outputs[0]


# In[9]:


print(np.shape(outputs))
print(np.shape(outputs[0]))
print(np.shape(outputs[1]))


# In[10]:


_, num_articles = np.shape(outputs)
article_names = [f'Article {idx}' for idx in range(num_articles)]


# In[11]:


article_embeddings = outputs[1]


# # Look at some dimensionality reduction plots

# In[12]:


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'widget')
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd


# In[13]:


# vector embeddings of each token
vec_embeddings = article_embeddings.numpy()

# color for each token for visuals
token_color = [np.random.rand(3,) for _ in range(num_articles)]


# In[14]:


pca = PCA(n_components=2)
pca_result = pca.fit_transform(vec_embeddings)
print(np.shape(pca_result))


# In[15]:


df = pd.DataFrame(pca_result, columns=['pca_0', 'pca_1'], index=article_names)
#display(df)


# In[16]:


plt.figure(figsize=(16,8))
sns.scatterplot(
    x='pca_0', y='pca_1',
    data=df,
    legend='full'
)
plt.grid()

ax = plt.gca()
for (k, v), color in zip(df.iterrows(), token_color):
    ax.annotate(k, v, c=color)

#for idx, (article, color) in enumerate(zip(article_names, token_color)):
#    plt.text(-7 + idx*1, -10, article, fontsize='medium', color=color)


# In[17]:


tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(vec_embeddings)
print(np.shape(tsne_results))


# In[18]:


df = pd.DataFrame(tsne_results, columns=['tsne_0', 'tsne_1'], index=article_names)
#display(df)


# In[19]:


plt.figure(figsize=(16,8))
sns.scatterplot(
    x="tsne_0", y="tsne_1",
    data=df,
    legend="full"
)
plt.grid()

ax = plt.gca()
for (k, v), color in zip(df.iterrows(), token_color):
    ax.annotate(k, v, c=color)

#for idx, (token, color) in enumerate(zip(tokens, token_color)):
#    plt.text(-400 + idx*50, -400, token, fontsize='medium', color=color)


# # Dimensionality Reduction projection into 3D space

# In[20]:


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[21]:


pca = PCA(n_components=3)
pca_result = pca.fit_transform(vec_embeddings)
print(np.shape(pca_result))


# In[22]:


df = pd.DataFrame(pca_result, columns=['pca_0', 'pca_1', 'pca_2'], index=article_names)
#display(df)


# In[23]:


fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(df['pca_0'], df['pca_1'], df['pca_2'], c=token_color)
plt.grid()


# In[24]:


print(np.shape(vec_embeddings))


# In[25]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Conv1D, MaxPooling1D, InputLayer


# In[26]:


model = Sequential([
    InputLayer(input_shape=(768, 1)),
    Conv1D(16, 3, padding='valid', activation='relu', input_shape=(768, 1)),
    MaxPooling1D(),
    Conv1D(32, 5, padding='valid', activation='relu', input_shape=(383, 16)),
    MaxPooling1D(),
    Conv1D(64, 3, padding='valid', activation='relu', input_shape=(383, 16)),
    MaxPooling1D(),
    Conv1D(128, 3, padding='valid', activation='relu', input_shape=(383, 16)),
    MaxPooling1D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])
model.summary()


# In[27]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[28]:


np.shape(vec_embeddings[0:10].T)


# In[29]:


print(vec_embeddings[0:10])


# In[30]:


input_vectors = vec_embeddings[0]
print(np.shape(input_vectors[None,:]))


# In[31]:


output_vectors = np.array([i for i in range(1)])
print(np.shape(output_vectors))


# In[32]:


model.fit(x=input_vectors[None,:,None], y=np.array(output_vectors))


# In[ ]:


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)


# In[ ]:


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(1, 768, 1)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

