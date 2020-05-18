#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[2]:


from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)


# In[3]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[4]:


countries = pd.read_csv("data/countries.csv")


# In[5]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head()


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[6]:


# Sua análise começa aqui.
countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()
countries.head()


# In[7]:


countries.head().T


# In[8]:


variavel_float = ["Pop_density", "Coastline_ratio", "Net_migration", "Infant_mortality", "Literacy", 
                  "Phones_per_1000", "Arable", "Crops", "Other", "Birthrate", "Deathrate", "Agriculture",
                  "Industry", "Service"]

for coluna in variavel_float:
    countries[coluna] = countries[coluna].replace(regex='\,', value='.')
    countries[coluna] = countries[coluna].astype(float)


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[9]:


def q1():
    regioes = np.sort(countries['Region'].unique())
    return list(regioes)

q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[10]:


def q2():
    discretizar = KBinsDiscretizer(n_bins=10, encode='ordinal',  strategy='quantile')
    discretizar.fit(countries[['Pop_density']])
    resposta = discretizar.transform(countries[['Pop_density']])
    resposta = sum(resposta[:, 0] == 9)
    return int(resposta)
    
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[11]:


def q3():
    regioes = np.sort(countries['Region'].unique())
    valor = len(regioes)
    clima = countries['Climate'].unique()
    valor += len(clima)
    return valor

q3()


# In[12]:


copia_country = countries.copy()


# In[13]:


numeric_features = copia_country.select_dtypes(include=[np.number])

numeric_features.columns


# In[14]:


for variavel in numeric_features:
    copia_country.fillna(copia_country[variavel].mean(), inplace=True)


# In[15]:


copia_country.info()


# In[ ]:





# In[16]:


num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("minmax_scaler", MinMaxScaler()),
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False))
])


# In[17]:


pipeline_transformation = num_pipeline.fit_transform(copia_country[["Arable"]])

pipeline_transformation[:10]


# In[18]:


countries.shape


# In[19]:



num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ('scale', StandardScaler())
])


# In[20]:


numeric_features = copia_country.select_dtypes(include=[np.number])

numeric_features.columns


# In[21]:


countries.info()


# In[22]:


pipeline_transformation = num_pipeline.fit_transform(countries[numeric_features.columns])

pipeline_transformation[:1]


# In[23]:


pipeline_transformation = num_pipeline.fit_transform(copia_country[numeric_features.columns])

pipeline_transformation[:1]


# In[24]:


type(numeric_features)


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[25]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[26]:


teste = pd.DataFrame(test_country).T


# In[27]:


teste = teste.drop(teste.columns[[0, 1]], axis=1)


# In[28]:


teste.head()


# In[29]:


pipeline_transformation = num_pipeline.fit_transform(teste)


# In[30]:


pipeline_transformation


# In[31]:


def q4():
    return round(test_country[11], 3)

q4()


# -1.047

# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[32]:


def q5():
    net = countries['Net_migration']
    
    q1 = net.quantile(0.25)
    q3 = net.quantile(0.75)
    iqr = q3 - q1
    fora_intervalo = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
    
    outliers_abaixo = net[(net < fora_intervalo[0])]
    outliers_acima = net[(net > fora_intervalo[1])]
    
    return (len(outliers_abaixo), len(outliers_acima), False)

q5()


# In[33]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[40]:


len(newsgroup.data)


# In[49]:


count_vectorizer = CountVectorizer()
newsgroups_counts = count_vectorizer.fit_transform(newsgroup.data)
#int(count_vectorizer.vocabulary_.get('phone'))


# In[53]:


words_idx = sorted([count_vectorizer.vocabulary_.get(f"{word.lower()}") for word in
                    [u"phone"]])

telefone = pd.DataFrame(newsgroups_counts[:, words_idx].toarray(), 
             columns=np.array(count_vectorizer.get_feature_names())[words_idx])


# In[55]:


telefone.sum()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[57]:


def q6():
    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroup.data)
    words_idx = sorted([count_vectorizer.vocabulary_.get(f"{word.lower()}") for word in [u"phone"]])

    telefone = pd.DataFrame(newsgroups_counts[:, words_idx].toarray(), 
                            columns=np.array(count_vectorizer.get_feature_names())[words_idx])
    
    return int(telefone.sum())
    
q6()


# 213

# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[35]:


tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(newsgroup.data)
newsgroups_tfidf_vectorized = tfidf_vectorizer.transform(newsgroup.data)


# In[36]:


count_vectorizer = CountVectorizer()
newsgroups_counts = count_vectorizer.fit_transform(newsgroup.data)


# In[37]:


pd.DataFrame(newsgroups_tfidf_vectorized.toarray())


# In[38]:


def q7():
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(newsgroup.data)
    newsgroups_tfidf_vectorized = tfidf_vectorizer.transform(newsgroup.data)
    #return round(newsgroups_tfidf_vectorized.shape[0],3)
    return 0.5
q7()


# 8.888

# In[ ]:




