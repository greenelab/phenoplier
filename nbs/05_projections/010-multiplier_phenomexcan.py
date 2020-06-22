#!/usr/bin/env python
# coding: utf-8

# # Description

# This notebook projects the PhenomeXcan data (MultiXcan results) into the MultiPLIER latent space. It takes only the top 1% of genes from each latent variable.
# 
# *Technical debt:* the notebook is usable and does its job, but will be refactored in the future, since other configurations are also useful (for instance, take different percentages of top genes to make the projection). Ideas for future refactoring:
# 
# 1. Move projection code into libs.

# # Modules loadings

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import os
import pickle

import numpy as np
import pandas as pd

import settings as conf
from metadata import GENE_ID_TO_NAME_MAP, GENE_NAME_TO_ID_MAP


# # Read MultiPLIER model metadata

# In[3]:


metadata_dir = os.path.dirname(conf.MULTIPLIER_SETTINGS['RECOUNT2_FULL_MODEL_FILE'])
metadata_file = os.path.join(metadata_dir, 'multiplier_model_metadata.pkl')
display(metadata_file)


# In[4]:


with open(metadata_file, 'rb') as handle:
    multiplier_model_metadata = pickle.load(handle)


# In[5]:


multiplier_model_metadata


# # Read MultiPLIER Z (loadings)

# In[6]:


multiplier_model_z = pd.read_pickle(os.path.join(conf.MULTIPLIER_SETTINGS['RECOUNT2_DATA_DIR'], 'multiplier_model_z.pkl'))


# In[7]:


multiplier_model_z.shape


# In[8]:


multiplier_model_z.head()


# # Load PhenomeXcan data for S-MultiXcan results

# In[9]:


smultixcan_results_filename = os.path.join(conf.PHENOMEXCAN_SETTINGS['GENE_ASSOC_DIR'], 'smultixcan-mashr-zscores.pkl')
display(smultixcan_results_filename)


# In[10]:


smultixcan_results = pd.read_pickle(smultixcan_results_filename)


# In[11]:


smultixcan_results.shape


# In[12]:


smultixcan_results.head()


# # Get genes in common

# In[13]:


lvs_gene_ids = [
    GENE_NAME_TO_ID_MAP[g]
    for g in multiplier_model_z.index
    if g in GENE_NAME_TO_ID_MAP and GENE_NAME_TO_ID_MAP[g] in smultixcan_results.index
]


# In[14]:


len(lvs_gene_ids)


# In[15]:


lvs_gene_names = [GENE_ID_TO_NAME_MAP[g] for g in lvs_gene_ids]


# In[16]:


len(lvs_gene_names)


# In[17]:


assert len(lvs_gene_ids) == len(lvs_gene_names)


# ## MultiPLIER Z: select common genes and rename them to IDs

# In[18]:


multiplier_model_z_common = multiplier_model_z.loc[lvs_gene_names].rename(index=GENE_NAME_TO_ID_MAP)


# In[19]:


multiplier_model_z_common.shape


# In[20]:


multiplier_model_z_common.head()


# ## Quick look at LV603 genes

# In[21]:


t = multiplier_model_z_common['LV603']


# In[22]:


t.quantile(0.90)


# In[23]:


t.sort_values(ascending=False).head()


# In[24]:


t = t[t > 0]


# In[25]:


t.sort_values(ascending=False).rename(index=GENE_ID_TO_NAME_MAP).head(10)


# # MultiPLIER Z: keep only those genes with high weight

# In[26]:


PERCENTILE_SELECTED = 0.99


# In[27]:


def zero_nonimportant_genes(x, perc=PERCENTILE_SELECTED):
#     x = x.copy()
    x_gt_zero = x[x > 0]
    q = x_gt_zero.quantile(perc)
    x[x < q] = 0.0
    return x


# In[28]:


multiplier_model_z_common_important_genes = multiplier_model_z_common.copy().apply(zero_nonimportant_genes)


# In[29]:


multiplier_model_z_common_important_genes.shape


# In[30]:


multiplier_model_z_common_important_genes.head()


# In[31]:


multiplier_model_z_common_important_genes.describe()


# ### Some testing on LV603

# In[32]:


_lv_prev = multiplier_model_z_common['LV603']


# In[33]:


_lv_prev.shape


# In[34]:


_lv_prev_nonzero = _lv_prev[_lv_prev > 0.0]


# In[35]:


_lv_prev_nonzero.shape


# In[36]:


_lv_prev_q = _lv_prev_nonzero.quantile(PERCENTILE_SELECTED)
display(_lv_prev_q)


# In[37]:


_tmp = _lv_prev[_lv_prev > _lv_prev_q].shape
display(_tmp)


# In[38]:


assert _tmp[0] == 22


# How many non-zero genes are in any LV? There must be around 22 (1 percent of nonzero loadings genes)

# In[39]:


_any_lv = multiplier_model_z_common_important_genes['LV603']


# In[40]:


_any_lv.shape


# In[41]:


_any_lv.head()


# In[42]:


_any_lv[_any_lv > 0].shape


# In[43]:


assert _tmp[0] == _any_lv[_any_lv > 0].shape[0]


# ### Some testing on LV136

# In[44]:


_any_lv = multiplier_model_z_common_important_genes['LV136']


# In[45]:


_any_lv.shape


# In[46]:


_any_lv.head()


# In[47]:


_any_lv[_any_lv > 0].shape


# ### What's the coverage of important genes?

# In[48]:


genes_covered = set()

for lv_col in multiplier_model_z_common_important_genes.columns:
    lv_values = multiplier_model_z_common_important_genes[lv_col]
    lv_values = lv_values[lv_values > 0.0]
    genes_covered.update(lv_values.index.tolist())


# In[49]:


len(genes_covered)


# In[50]:


len(genes_covered) / multiplier_model_z_common_important_genes.shape[0]


# Even considering only the top 1% of genes in LVs, we still have that most genes considered are part of some LV.

# ## How many genes, on average, are nonzero for each LV?

# In[51]:


multiplier_model_z_common_important_genes.head()


# In[52]:


_tmp = multiplier_model_z_common_important_genes.apply(lambda x: x[x > 0].shape[0])


# In[53]:


_tmp.describe()


# ## Save MultiPLIER Z with only important genes selected

# In[54]:


suffix = f'{PERCENTILE_SELECTED}'.replace('.', '_')
display(suffix)


# In[55]:


output_filename = os.path.join(conf.LVS_MULTIPLIER_PHENOMEXCAN_SETTINGS['LVS_PROJECTIONS'], f'multiplier_z_important_genes_{suffix}.pkl')
display(output_filename)


# In[56]:


multiplier_model_z_common_important_genes.to_pickle(output_filename)


# ## S-MultiXcan: keep common genes

# In[57]:


smultixcan_results_common = smultixcan_results.loc[multiplier_model_z_common_important_genes.index]


# In[58]:


smultixcan_results_common.shape


# In[59]:


smultixcan_results_common.head()


# # Multiply to obtains LVs associations to all traits

# In[60]:


from numpy.linalg import pinv


# In[61]:


# smultixcan_results_common_final = smultixcan_results_common_stded
smultixcan_results_common_final = smultixcan_results_common


# In[62]:


# row z-score standardization
smultixcan_results_common_final = (
        # remove mean
        smultixcan_results_common_final.sub(smultixcan_results_common_final.mean(1), axis=0)
        # divide by std
        .div(smultixcan_results_common_final.std(1), axis=0)
)#.fillna(0)


# In[63]:


# # row minmax standardization
# smultixcan_results_common_final = \
# (
#         # remove mean
#         smultixcan_results_common_final.sub(smultixcan_results_common_final.min(1), axis=0)
#         # divide by std
#         .div(smultixcan_results_common_final.max(1) - smultixcan_results_common_final.min(1), axis=0)
# )#.fillna(0)


# In[64]:


# # row minmax total standardization
# smultixcan_results_common_final = \
# (
#         # remove mean
#         smultixcan_results_common_final.div(smultixcan_results_common_final.sum(1), axis=0)
# )#.fillna(0)


# In[65]:


smultixcan_results_common_final.head(2)


# In[66]:


_tmp = smultixcan_results_common_final.T.describe()


# In[67]:


_tmp


# In[68]:


assert (_tmp.loc['mean'] < 1e-10).all()
assert np.allclose(_tmp.loc['std'].values, 1.0)


# In[69]:


smultixcan_results_common_final = smultixcan_results_common_final.fillna(0)


# In[70]:


multiplier_model_metadata['L2']


# In[71]:


prev_mat = (
    multiplier_model_z_common_important_genes.T.dot(multiplier_model_z_common_important_genes)
    + (multiplier_model_metadata['L2'] * np.identity(multiplier_model_z_common_important_genes.shape[1]))
)


# In[72]:


prev_mat.shape


# In[73]:


prev_mat = pd.DataFrame(pinv(prev_mat), index=prev_mat.index.copy(), columns=prev_mat.columns.copy())


# In[74]:


lvs_traits_df = (
    prev_mat.dot(multiplier_model_z_common_important_genes.T).dot(smultixcan_results_common_final)
)


# In[75]:


lvs_traits_df.shape


# In[76]:


lvs_traits_df.head()


# # Quick analysis

# In[77]:


lvs_traits_df.loc['LV603'].sort_values(ascending=False).head(20)


# In[78]:


lvs_traits_df.loc['LV136'].sort_values(ascending=False).head(20)


# # Save

# In[79]:


lvs_traits_df.shape


# In[80]:


lvs_traits_df.head()


# In[81]:


os.makedirs(conf.LVS_MULTIPLIER_PHENOMEXCAN_SETTINGS['LVS_PROJECTIONS'], exist_ok=True)


# In[82]:


lvs_filename = os.path.join(conf.LVS_MULTIPLIER_PHENOMEXCAN_SETTINGS['LVS_PROJECTIONS'], 'lvs_x_smultixcan.pkl')
display(lvs_filename)


# In[83]:


lvs_traits_df.to_pickle(lvs_filename)


# In[ ]:




