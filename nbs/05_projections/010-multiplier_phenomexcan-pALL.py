#!/usr/bin/env python
# coding: utf-8

# # Description

# For this projection, this notebook takes **ALL non-zero gene loadings** from each latent variable (LV) in the MultiPLIER model.

# # Modules loading

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import conf
from data.cache import read_data


# # Settings

# In[3]:


# The percentile name indicates the top percentage of genes retained
PERCENTILE_NAME = 'pALL'

display(PERCENTILE_NAME)


# In[4]:


RESULTS_PROJ_OUTPUT_DIR = Path(
    conf.RESULTS['PROJECTIONS_DIR']
)

RESULTS_PROJ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_PROJ_OUTPUT_DIR)


# # Read gene mappings

# In[5]:


GENE_ID_TO_NAME_MAP = read_data(conf.PHENOMEXCAN["GENE_MAP_ID_TO_NAME"])
GENE_NAME_TO_ID_MAP = read_data(conf.PHENOMEXCAN["GENE_MAP_NAME_TO_ID"])


# # Read MultiPLIER model metadata

# In[6]:


input_file = Path(
    conf.MULTIPLIER['BASE_DIR'], 'multiplier_model_metadata.pkl'
).resolve()
display(input_file)

multiplier_model_metadata = pd.read_pickle(input_file)


# In[7]:


multiplier_model_metadata


# # Read MultiPLIER Z (loadings)

# In[8]:


input_file = Path(
    conf.MULTIPLIER['BASE_DIR'],
    'multiplier_model_z.pkl'
).resolve()
display(input_file)

multiplier_model_z = pd.read_pickle(input_file)


# In[9]:


multiplier_model_z.shape


# In[10]:


multiplier_model_z.head()


# # Load PhenomeXcan data (S-MultiXcan)

# In[11]:


smultixcan_results_filename = Path(
    conf.PHENOMEXCAN['GENE_ASSOC_DIR'],
    'smultixcan-mashr-zscores.pkl'
).resolve()

display(smultixcan_results_filename)


# In[12]:


smultixcan_results = pd.read_pickle(smultixcan_results_filename)


# In[13]:


smultixcan_results.shape


# In[14]:


smultixcan_results.head()


# ## Gene IDs to Gene names

# In[15]:


smultixcan_results = smultixcan_results.rename(index=GENE_ID_TO_NAME_MAP)


# In[16]:


smultixcan_results.shape


# In[17]:


smultixcan_results.head()


# ## Remove duplicated gene entries

# In[18]:


smultixcan_results.index[smultixcan_results.index.duplicated(keep='first')]


# In[19]:


smultixcan_results = smultixcan_results.loc[~smultixcan_results.index.duplicated(keep='first')]


# In[20]:


smultixcan_results.shape


# ## Remove NaN values

# **TODO**: it might be better to try to impute this values

# In[21]:


smultixcan_results = smultixcan_results.dropna(how='any')


# In[22]:


smultixcan_results.shape


# # Project S-MultiXcan data into MultiPLIER latent space

# In[23]:


from multiplier import MultiplierProjection


# In[24]:


mproj = MultiplierProjection()


# In[25]:


smultixcan_into_multiplier = mproj.transform(smultixcan_results)


# In[26]:


smultixcan_into_multiplier.shape


# In[27]:


smultixcan_into_multiplier.head()


# # Quick analysis

# In[28]:


(
    smultixcan_into_multiplier.loc['LV603']
    .sort_values(ascending=False)
    .head(20)
)


# In[29]:


(
    smultixcan_into_multiplier.loc['LV136']
    .sort_values(ascending=False)
    .head(20)
)


# # Save

# In[30]:


output_file = Path(
    RESULTS_PROJ_OUTPUT_DIR,
    f'projection-smultixcan_zscores-{PERCENTILE_NAME}.pkl'
).resolve()

display(output_file)


# In[31]:


smultixcan_into_multiplier.to_pickle(output_file)


# In[ ]:




