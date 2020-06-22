#!/usr/bin/env python
# coding: utf-8

# # Description

# This notebook reads all matrices from the MultiPLIER model (https://github.com/greenelab/multi-plier) trained in recount2, like gene loadings (Z) or the
# latent space (B), and saves them into a Python friendly format (Pandas DataFrames in pickle format).

# # Modules loading

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import os
import pickle

import numpy as np
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
# pandas2ri.activate()

import settings as conf


# In[3]:


readRDS = ro.r['readRDS']


# # Read MultiPLIER model

# In[4]:


conf.MULTIPLIER_SETTINGS['RECOUNT2_FULL_MODEL_FILE']


# In[5]:


multiplier_full_model = readRDS(conf.MULTIPLIER_SETTINGS['RECOUNT2_FULL_MODEL_FILE'])


# # Matrix Z (loadings; genes x LVs)

# In[6]:


multiplier_model_matrix = multiplier_full_model.rx2('Z')


# In[7]:


multiplier_model_matrix


# In[8]:


multiplier_model_matrix.rownames


# In[9]:


multiplier_model_matrix.colnames


# In[10]:


with localconverter(ro.default_converter + pandas2ri.converter):
  multiplier_model_matrix_values = ro.conversion.rpy2py(multiplier_model_matrix)


# In[11]:


multiplier_model_matrix_df = pd.DataFrame(
    data=multiplier_model_matrix_values,
    index=multiplier_model_matrix.rownames,
    columns=[f'LV{i}' for i in range(1, multiplier_model_matrix.ncol + 1)]
)


# In[12]:


display(multiplier_model_matrix_df.shape)
assert multiplier_model_matrix_df.shape == (6750, 987)


# In[13]:


multiplier_model_matrix_df.head()


# In[14]:


# make sure I'm seeing the same when loaded with R
assert multiplier_model_matrix_df.loc['GAS6', 'LV2'] == 0
assert multiplier_model_matrix_df.loc['GAS6', 'LV3'] == 0.039437739697954444
assert multiplier_model_matrix_df.loc['GAS6', 'LV984'] == 0.3473620915326928
assert multiplier_model_matrix_df.loc['GAS6', 'LV987'] == 0

assert multiplier_model_matrix_df.loc['SPARC', 'LV981'] == 0
assert multiplier_model_matrix_df.loc['SPARC', 'LV986'].round(8) == 0.12241734


# ## Save

# In[15]:


output_dir = os.path.dirname(conf.MULTIPLIER_SETTINGS['RECOUNT2_FULL_MODEL_FILE'])
display(output_dir)


# In[16]:


output_file = os.path.join(output_dir, 'multiplier_model_z.pkl')
display(output_file)


# In[17]:


multiplier_model_matrix_df.to_pickle(output_file)


# # Matrix B (latent space; LVs x samples)

# In[18]:


multiplier_model_matrix = multiplier_full_model.rx2('B')


# In[19]:


multiplier_model_matrix


# In[20]:


multiplier_model_matrix.rownames


# In[21]:


multiplier_model_matrix.colnames


# In[22]:


with localconverter(ro.default_converter + pandas2ri.converter):
  multiplier_model_matrix_values = ro.conversion.rpy2py(multiplier_model_matrix)


# In[23]:


multiplier_model_matrix_df = pd.DataFrame(
    data=multiplier_model_matrix_values,
#    Look like the rows have a special meaning, so no overriding it.
#    index=[f'LV{i}' for i in range(1, multiplier_model_matrix.nrow + 1)],
    index=multiplier_model_matrix.rownames,
    columns=multiplier_model_matrix.colnames,
)


# In[24]:


display(multiplier_model_matrix_df.shape)
assert multiplier_model_matrix_df.shape == (987, 37032)


# In[25]:


multiplier_model_matrix_df.head()


# In[26]:


# make sure I'm seeing the same when loaded with R
assert multiplier_model_matrix_df.loc['1,REACTOME_MRNA_SPLICING', 'SRP000599.SRR013549'].round(9) == -0.059296689
assert multiplier_model_matrix_df.loc['1,REACTOME_MRNA_SPLICING', 'SRP000599.SRR013553'].round(9) == -0.036394186

assert multiplier_model_matrix_df.loc['2,SVM Monocytes', 'SRP000599.SRR013549'].round(9) == 0.006212678
assert multiplier_model_matrix_df.loc['2,SVM Monocytes', 'SRP004637.SRR073776'].round(9) == -0.008800153

assert multiplier_model_matrix_df.loc['LV 9', 'SRP004637.SRR073774'].round(9) == 0.092318955
assert multiplier_model_matrix_df.loc['LV 9', 'SRP004637.SRR073776'].round(9) == 0.100114294


# ## Make sure no GTEx samples are included

# In[27]:


# Test search string first
_tmp = multiplier_model_matrix_df.columns.str.contains('SRP000599.', regex=False)
assert _tmp[0]
assert _tmp[1]
assert not _tmp[-1]


# In[28]:


GTEX_ACCESSION_CODE = 'SRP012682'


# In[29]:


_tmp = multiplier_model_matrix_df.columns.str.contains(GTEX_ACCESSION_CODE, regex=False)
assert not _tmp.any()


# ## Save

# In[30]:


output_dir = os.path.dirname(conf.MULTIPLIER_SETTINGS['RECOUNT2_FULL_MODEL_FILE'])
display(output_dir)


# In[31]:


output_file = os.path.join(output_dir, 'multiplier_model_b.pkl')
display(output_file)


# In[32]:


multiplier_model_matrix_df.to_pickle(output_file)


# # Matrix U (gene sets x LVs)

# In[33]:


multiplier_model_matrix = multiplier_full_model.rx2('U')


# In[34]:


multiplier_model_matrix


# In[35]:


multiplier_model_matrix.rownames


# In[36]:


multiplier_model_matrix.colnames


# In[37]:


with localconverter(ro.default_converter + pandas2ri.converter):
  multiplier_model_matrix_values = ro.conversion.rpy2py(multiplier_model_matrix)


# In[38]:


multiplier_model_matrix_df = pd.DataFrame(
    data=multiplier_model_matrix_values,
    index=multiplier_model_matrix.rownames,
    columns=multiplier_model_matrix.colnames,
)


# In[39]:


display(multiplier_model_matrix_df.shape)
assert multiplier_model_matrix_df.shape == (628, 987)


# In[40]:


multiplier_model_matrix_df.head()


# In[41]:


# make sure I'm seeing the same when loaded with R
assert multiplier_model_matrix_df.loc['IRIS_Bcell-Memory_IgG_IgA', 'LV1'] == 0
assert multiplier_model_matrix_df.loc['IRIS_Bcell-Memory_IgG_IgA', 'LV898'].round(7) == 0.5327689
assert multiplier_model_matrix_df.loc['IRIS_Bcell-Memory_IgG_IgA', 'LV977'].round(7) == 0.1000158
assert multiplier_model_matrix_df.loc['IRIS_Bcell-Memory_IgG_IgA', 'LV986'] == 0
assert multiplier_model_matrix_df.loc['IRIS_Bcell-Memory_IgG_IgA', 'LV987'] == 0

assert multiplier_model_matrix_df.loc['IRIS_Bcell-naive', 'LV851'].round(8) == 0.01330388
assert multiplier_model_matrix_df.loc['IRIS_Bcell-naive', 'LV977'].round(7) == 0.3966446


# ## Save

# In[42]:


output_dir = os.path.dirname(conf.MULTIPLIER_SETTINGS['RECOUNT2_FULL_MODEL_FILE'])
display(output_dir)


# In[43]:


output_file = os.path.join(output_dir, 'multiplier_model_u.pkl')
display(output_file)


# In[44]:


multiplier_model_matrix_df.to_pickle(output_file)


# # Matrix U - AUC

# In[45]:


multiplier_model_matrix = multiplier_full_model.rx2('Uauc')


# In[46]:


multiplier_model_matrix


# In[47]:


multiplier_model_matrix.rownames


# In[48]:


multiplier_model_matrix.colnames


# In[49]:


with localconverter(ro.default_converter + pandas2ri.converter):
  multiplier_model_matrix_values = ro.conversion.rpy2py(multiplier_model_matrix)


# In[50]:


multiplier_model_matrix_df = pd.DataFrame(
    data=multiplier_model_matrix_values,
    index=multiplier_model_matrix.rownames,
    columns=multiplier_model_matrix.colnames,
)


# In[51]:


display(multiplier_model_matrix_df.shape)
assert multiplier_model_matrix_df.shape == (628, 987)


# In[52]:


multiplier_model_matrix_df.head()


# In[53]:


# make sure I'm seeing the same when loaded with R
assert multiplier_model_matrix_df.loc['PID_FASPATHWAY', 'LV136'] == 0
assert multiplier_model_matrix_df.loc['PID_INTEGRIN1_PATHWAY', 'LV136'].round(7) == 0.8832853
assert multiplier_model_matrix_df.loc['REACTOME_COLLAGEN_FORMATION', 'LV136'].round(7) == 0.8707412

assert multiplier_model_matrix_df.loc['PID_FASPATHWAY', 'LV603'] == 0
assert multiplier_model_matrix_df.loc['IRIS_Neutrophil-Resting', 'LV603'].round(7) == 0.9057506
assert multiplier_model_matrix_df.loc['SVM Neutrophils', 'LV603'].round(7) == 0.9797889


# ### Save

# In[54]:


output_dir = os.path.dirname(conf.MULTIPLIER_SETTINGS['RECOUNT2_FULL_MODEL_FILE'])
display(output_dir)


# In[55]:


output_file = os.path.join(output_dir, 'multiplier_model_u_auc.pkl')
display(output_file)


# In[56]:


multiplier_model_matrix_df.to_pickle(output_file)


# # Model metadata

# In[57]:


model_names = list(multiplier_full_model.names)
display(model_names)


# In[58]:


with localconverter(ro.default_converter + pandas2ri.converter):
    model_metadata = {k: ro.conversion.rpy2py(multiplier_full_model.rx2(k))[0] for k in ('L1', 'L2', 'L3')}


# In[59]:


model_metadata


# In[60]:


assert len(model_metadata) == 3


# In[61]:


assert model_metadata['L2'] == 241.1321740143624


# ## Save

# In[62]:


output_dir = os.path.dirname(conf.MULTIPLIER_SETTINGS['RECOUNT2_FULL_MODEL_FILE'])
display(output_dir)


# In[63]:


output_file = os.path.join(output_dir, 'multiplier_model_metadata.pkl')
display(output_file)


# In[64]:


with open(output_file, 'wb') as handle:
    pickle.dump(model_metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




