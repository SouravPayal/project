#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the file
import pandas as pd
df= pd.read_csv("train.csv",encoding="unicode_escape")
df.sample(5)


# In[10]:


#cleaning data set
df = df.dropna(subset=['text'])


# In[ ]:


# making one hot encoding of sentiments
df_encoded = pd.get_dummies(df, columns=['sentiment', ])
df_target= df_encoded[['sentiment_negative','sentiment_neutral','sentiment_positive']]
df_target


# In[23]:


#spliting data for training and validation
import sklearn.model_selection
tv_data, test_data, tv_labels, test_labels = sklearn.model_selection.train_test_split(df.text, df_target)
train_data, validation_data, train_labels, validation_labels = sklearn.model_selection.train_test_split(tv_data, tv_labels)


# In[24]:


max_tokens=2000
output_sequence_length=50


# In[ ]:





# In[25]:


import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import TextVectorization, Embedding, Dense, GlobalAveragePooling1D, Input

# Text vectorization
vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length)
vectorizer.adapt(train_data)


# In[26]:


from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
import keras.callbacks
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


# In[32]:


train_labels.shape


# In[33]:


inputs = Input(shape=(1,), dtype=tf.string)
vectorized = vectorizer(inputs)
thinking = Dense(128, activation='relu')(vectorized)
output = Dense(3, activation='softmax')(thinking)
model = Model(inputs=[inputs], outputs=[output])
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                   patience=50, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model', save_best_only=True)

history = model.fit(train_data, train_labels, epochs=100,
                   validation_data=(validation_data, validation_labels),
                   callbacks=[early_stopping,model_checkpoint])

