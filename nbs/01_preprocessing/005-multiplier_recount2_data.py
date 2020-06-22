#!/usr/bin/env python
# coding: utf-8

# # Description

# This notebook reads 1) the normalized gene expression and 2) pathways from the data processed by
# MultiPLIER scripts (https://github.com/greenelab/multi-plier) and saves it into a more friendly Python
# format (Pandas DataFrames as pickle files).

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


# # Read entire recount data prep file

# In[4]:


conf.MULTIPLIER_SETTINGS['RECOUNT2_PREP_GENE_EXP_FILE']


# In[5]:


recount_data_prep = readRDS(conf.MULTIPLIER_SETTINGS['RECOUNT2_PREP_GENE_EXP_FILE'])


# # Read recount2 gene expression data

# In[6]:


recount2_rpkl_cm = recount_data_prep.rx2('rpkm.cm')


# In[7]:


recount2_rpkl_cm


# In[8]:


recount2_rpkl_cm.rownames


# In[9]:


recount2_rpkl_cm.colnames


# In[10]:


with localconverter(ro.default_converter + pandas2ri.converter):
  recount2_rpkl_cm = ro.conversion.rpy2py(recount2_rpkl_cm)


# In[11]:


# recount2_rpkl_cm = pd.DataFrame(
#     data=pandas2ri.ri2py(recount2_rpkl_cm).values,
#     index=recount2_rpkl_cm.rownames,
#     columns=recount2_rpkl_cm.colnames,
# )


# In[12]:


assert recount2_rpkl_cm.shape == (6750, 37032)


# In[13]:


recount2_rpkl_cm.shape


# In[14]:


recount2_rpkl_cm.head()


# ## Testing

# Test whether what I load from a plain R session is the same as in here.

# In[15]:


recount2_rpkl_cm.loc['GAS6', 'SRP000599.SRR013549']


# In[16]:


assert recount2_rpkl_cm.loc['GAS6', 'SRP000599.SRR013549'].round(4) == -0.3125


# In[17]:


assert recount2_rpkl_cm.loc['GAS6', 'SRP045352.SRR1539229'].round(7) == -0.2843801


# In[18]:


assert recount2_rpkl_cm.loc['CFL2', 'SRP056840.SRR1951636'].round(7) == -0.3412832


# In[19]:


recount2_rpkl_cm.iloc[9, 16]


# In[20]:


assert recount2_rpkl_cm.iloc[9, 16].round(7) == -0.4938852


# ## Save

# In[21]:


output_filename = os.path.join(conf.MULTIPLIER_SETTINGS['RECOUNT2_DATA_DIR'], 'recount_data_prep_PLIER.pkl')
display(output_filename)


# In[22]:


recount2_rpkl_cm.to_pickle(output_filename)


# In[23]:


# from utils.hdf5 import simplify_string_for_hdf5


# In[24]:


# output_filename = os.path.join(conf.DATA_DIR, 'recount_data_prep_PLIER.h5')
# display(output_filename)


# In[25]:


# with pd.HDFStore(output_filename, mode='w', complevel=1) as store:
#     for idx, gene in enumerate(recount2_rpkl_cm.index):
#         if idx % 100:
#             print(f'', flush=True, end='')
        
#         clean_gene = simplify_string_for_hdf5(gene)
#         store[clean_gene] = recount2_rpkl_cm.loc[gene]


# In[26]:


del recount2_rpkl_cm


# # Read recount2 pathways

# In[27]:


recount2_all_paths_cm = recount_data_prep.rx2('all.paths.cm')


# In[28]:


recount2_all_paths_cm


# In[29]:


recount2_all_paths_cm.rownames


# In[30]:


recount2_all_paths_cm.colnames


# In[31]:


with localconverter(ro.default_converter + pandas2ri.converter):
  recount2_all_paths_cm_values = ro.conversion.rpy2py(recount2_all_paths_cm)


# In[32]:


recount2_all_paths_cm_values


# In[33]:


recount2_all_paths_cm = pd.DataFrame(
    data=recount2_all_paths_cm_values,
    index=recount2_all_paths_cm.rownames,
    columns=recount2_all_paths_cm.colnames,
    dtype=bool,
)


# In[34]:


assert recount2_all_paths_cm.shape == (6750, 628)


# In[35]:


recount2_all_paths_cm.shape


# In[36]:


recount2_all_paths_cm.dtypes.unique()


# In[37]:


recount2_all_paths_cm.head()


# ## Testing

# In[38]:


recount2_all_paths_cm.loc['CTSD', 'REACTOME_SCFSKP2_MEDIATED_DEGRADATION_OF_P27_P21']


# In[39]:


assert not recount2_all_paths_cm.loc['CTSD', 'REACTOME_SCFSKP2_MEDIATED_DEGRADATION_OF_P27_P21']


# In[40]:


assert recount2_all_paths_cm.loc['CTSD', 'PID_P53DOWNSTREAMPATHWAY']


# In[41]:


assert recount2_all_paths_cm.loc['MMP14', 'PID_HIF2PATHWAY']


# ## Save

# In[42]:


output_filename = os.path.join(conf.MULTIPLIER_SETTINGS['RECOUNT2_DATA_DIR'], 'recount_all_paths_cm.pkl')
display(output_filename)


# In[43]:


recount2_all_paths_cm.to_pickle(output_filename)


# In[44]:


del recount2_all_paths_cm


# In[ ]:




