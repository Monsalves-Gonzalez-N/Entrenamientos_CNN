#!/usr/bin/env python
# coding: utf-8

# In[1]:


from CNN_2dhist_function import *


# In[2]:


# Establecer la semilla para TensorFlow
tf.random.set_seed(42)

# Obtén el número de CPUs
num_cpus = psutil.cpu_count(logical=False)

path_data = "/home/nicolas/nico/Data/data_Paper_OGLE/"

datos = f"{path_data}Data/datos_ogle/datos"

path_datos_4 = datos + "/datos_ogle_4/I"

path_datos_3 = datos + "/datos_ogle_3/I"
path_datos = ["_","_","_",path_datos_3,path_datos_4]

rng = np.random.default_rng(42)

gyr = ["#ffa600",
        '#003f5c',
       "#58508d",
       "#ff6361",
       "#ffd380",
       "#bc5090",
       "#129675"
      ]
palet = sns.palplot(sns.color_palette(gyr))
sns.set_context("paper")
path = '/project/nico/Data/data_Paper_OGLE/7_01_2024/'


# In[4]:


train_number_ELL = pd.read_csv(f"{path}/train_number_ELL.csv")
train_number_DST = pd.read_csv(f"{path}/train_number_DST.csv")
train_number_M = pd.read_csv(f"{path}/train_number_M.csv")
prueba_8mil = pd.read_csv(f"{path}/prueba_8mil.csv")


# In[5]:


data = h5py.File(f"{path}/Data.hdf5", 'r+')


# In[6]:


df_lista = [prueba_8mil,train_number_ELL,train_number_DST,train_number_M]
keys_lista = ['Number_CEP','Number_ELL','Number_DST', 'Number_M']


# In[7]:


train_models(df_lista, keys_lista, data, prueba_8mil,path,epochs=1000, use_balanced_generator=False)


# In[12]:


#train_models(df_lista, keys_lista, data, prueba_8mil,path,epochs=1000, use_balanced_generator=True)

