#!/usr/bin/env python
# coding: utf-8

# In[549]:


from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
# RegEx for removing non-letter characters
import re

# NLTK library for the remaining steps
import nltk
nltk.download("stopwords")   # download list of stopwords (only once; need not run it again)
from nltk.corpus import stopwords # import stopwords

from nltk.stem.porter import *

stemmer = PorterStemmer()
from roc_utils import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[550]:


folder = '/media/Datacenter_storage/Radiology_User_feedback/Data/'


# In[551]:


labels = ['Emotion - Anxiety, Claustrophobia - USER2',
 'Nursing - IV Issues- USER2',
 'Communication - Poor Communication- USER2',
 'Service - Long Wait Time- USER2',
 'Technical Issues - Noise, equipment malfunction, computer, imaging issues- USER2',
 'Staff - Poor Treatment- USER2',
 'Testing - Physical Discomfort- USER2',
 'Testing - Long Test Times- USER2',
 'No Comment- USER2',
 'Positive- USER2',
 'Service - Excellent (professional, caring, excellent, etc.)- USER2',
 'Staff - Comforting- USER2',
 'Service - On Time- USER2',
 'Nursing (IV placement)- USER2',
 "Emotion - Relieved Concerns (dont' have to worry anymore)- USER2",
 'Diagnosis - Answered the Question- USER2',
 'Access to Report- USER2',
 'Thoroughness (informative, detailed)- USER2',
 'Follow-up- USER2',
 'Easy to Understand- USER2',
 'No Comment- USER2.1']


# In[552]:


LMmodel = SentenceTransformer('all-mpnet-base-v2')


# In[553]:


file = 'XRM_NLP_RAW DATA_04272022.xlsx'


# In[554]:


folder = '/media/Datacenter_storage/Radiology_User_feedback/Data/'


# In[555]:


train


# In[556]:


train = pd.read_excel('/mnt/storage/User_Feedback/Data/Train_Student.xlsx')
test = pd.read_excel('/mnt/storage/User_Feedback/Data/Test_Student.xlsx')
#unannotated = pd.read_excel("/mnt/storage/User_Feedback/Data/Unannotateddata.xlsx")
#unannotated_dynamic = pd.read_excel(folder+"XRM_NLP_RAW DATA_04192022.xlsx")
#unannotated_dynamic = pd.read_excel(folder+file+".xlsx")
#unannotated_dynamic = pd.read_excel(folder+file)


# In[557]:


def review_to_words(review):
    """Convert a raw review string into a sequence of words."""
    
    # TODO: Remove HTML tags and non-letters,
    #soup = BeautifulSoup(review, 'html5lib')
    text = review.lower()
    #       convert to lowercase, tokenize,
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    words = text.split()
    #       remove stopwords and stem
    words = [w.strip() for w in words if w not in stopwords.words('english')]
    words = [stemmer.stem(w) for w in words]

    # Return final list of words
    return ' '.join(words)


# In[ ]:





# In[558]:


train =train.fillna(' ')
train['Comments_proc'] = train['Comments'].apply(review_to_words)
test =test.fillna(' ')
test['Comments_proc'] = test['Comments'].apply(review_to_words)
#unannotated =unannotated.fillna(' ')
#unannotated = unannotated[unannotated['Score']!='ropho']
#unannotated = unannotated[unannotated['Score']!=' ']
#unannotated['Score']= unannotated['Score'].astype(int)
#unannotated = unannotated[unannotated['Score']<5]
#unannotated['Comments_proc'] = unannotated['Comments'].apply(review_to_words)


# In[559]:


#unannotated_dynamic =unannotated_dynamic.fillna(' ')
#unannotated_dynamic = unannotated_dynamic[unannotated_dynamic['RATING']!='ropho']
#unannotated_dynamic = unannotated_dynamic[unannotated_dynamic['RATING']!=' ']
#unannotated_dynamic['RATING']= unannotated_dynamic['RATING'].astype(int)
#unannotated_dynamic = unannotated_dynamic[unannotated_dynamic['RATING']<5]
#unannotated_dynamic['Comments_proc'] = unannotated_dynamic['EXPERIENCE'].apply(review_to_words)


# In[560]:


train_sentence_embeddings = LMmodel.encode(train['Comments'])
test_sentence_embeddings = LMmodel.encode(test['Comments'])


# In[561]:


train_sentence_embeddings.shape


# In[562]:


#unannotated_sentence_embeddings = np.zeros([unannotated.shape[0], train_sentence_embeddings.shape[1]])


# In[563]:


#for i in range(unannotated.shape[0]):
#    try:
#        unannotated_sentence_embeddings[i,:] = model.encode(unannotated.iloc[i]['Comments'])
#    except:
#        print(unannotated.iloc[i]['Comments'])


# In[564]:


#unannotated_sentence_dyn_embeddings = np.zeros([unannotated_dynamic.shape[0], train_sentence_embeddings.shape[1]])
#for i in range(unannotated_dynamic.shape[0]):
#    unannotated_sentence_dyn_embeddings[i,:] = LMmodel.encode(unannotated_dynamic.iloc[i]['EXPERIENCE'])


# In[565]:


#unannotated_sentence_dyn_embeddings.shape


# In[ ]:





# In[566]:


gt = []
pred = []
#pred_dyn = []
legend = []
external = []

for l in labels:
    try:
        y_train = train[l].values
        y_valid = test[l].values
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eta=0.3, silent=1, subsample=0.8, scale_pos_weight=99).fit(train_sentence_embeddings, y_train) 
        xgb_prediction = xgb_model.predict_proba(test_sentence_embeddings)
        print(l)
        print(classification_report(y_valid, xgb_model.predict(test_sentence_embeddings)))
        gt.append(y_valid)
        pred.append(xgb_model.predict_proba(test_sentence_embeddings))
        #external.append(xgb_model.predict(unannotated_sentence_embeddings))
        #pred_dyn.append(xgb_model.predict(unannotated_sentence_dyn_embeddings))
        legend.append(l)
        print('-------------------------------------------------')
    except:
        print('Didnot work: '+l)


# In[567]:


import random
import matplotlib.colors as mcolors

colors = mcolors.CSS4_COLORS
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),name) for name, color in colors.items())
names = [name for hsv, name in by_hsv]
roc = []
_, ax3 = plt.subplots(figsize=(15,10))
for i in range(len(gt)-3):
    p = pred[i]
    plot_roc(compute_roc(X=p[:,1], y=gt[i], pos_label=1), label=legend[i], color=names[i+15], ax=ax3)
# Place the legend outside.
ax3.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax3.set_title("ROC curves");


# In[568]:


#for i in range(len(legend)):
#    unannotated_dynamic[legend[i]] = pred_dyn[i]


# In[163]:


## Daymaker model


# In[187]:


#goodfeedback = pd.read_excel('/media/Datacenter_storage/Radiology_User_feedback/Data/Arizona Radiology DataGOOD FEEDBACK.xlsx', sheet_name = 'Combined')


# In[188]:


#goodfeedback['WORTH EMAILING STAFF'].value_counts()


# In[189]:


#goodfeedback = goodfeedback[(goodfeedback['WORTH EMAILING STAFF']==0)|(goodfeedback['WORTH EMAILING STAFF']==1)]


# In[190]:


#msk = np.random.rand(len(goodfeedback)) < 0.8
#goodfeedback_train = goodfeedback[msk]
#goodfeedback_test = goodfeedback[~msk] 


# In[191]:


#gt = []
#pred = []
#pred_dyn_2 = []

#y_train = goodfeedback_train['WORTH EMAILING STAFF'].values
#y_valid =  goodfeedback_test['WORTH EMAILING STAFF'].values
#train_sentence_embeddings = model.encode(list(goodfeedback_train['Comments']))
#test_sentence_embeddings = model.encode(list(goodfeedback_test['Comments']))
#xgb_model_good = xgb.XGBClassifier(objective='binary:logistic', eta=0.3, silent=1, subsample=0.8, scale_pos_weight=99).fit(train_sentence_embeddings, y_train) 
#xgb_prediction = xgb_model_good.predict_proba(test_sentence_embeddings)
#print(l)
#print(classification_report(y_valid, xgb_model_good.predict(test_sentence_embeddings)))
#gt.append(y_valid)
#pred.append(xgb_model_good.predict_proba(test_sentence_embeddings))
#pred_dyn_2.append(xgb_model_good.predict_proba(unannotated_sentence_dyn_embeddings))


# In[194]:


#_, ax3 = plt.subplots(figsize=(15,10))
#plot_roc(compute_roc(X=pred[0][:,1], y=gt, pos_label=1), label='Worth sending', color=names[i+15], ax=ax3)
# Place the legend outside.
#ax3.legend(loc="lower left")
#ax3.set_title("ROC curves");


# In[195]:


#unannotated_dynamic['Worth_emailing_stuff'] = pred_dyn_2[0][:,1]


# In[196]:


#unannotated_dynamic


# In[197]:


#filename = file.split('.')[0]


# In[198]:


#unannotated_dynamic.to_excel(folder+filename+"NLPlabels.xlsx")


# In[ ]:





# In[ ]:





# In[680]:


from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
# RegEx for removing non-letter characters
import re

# NLTK library for the remaining steps
import nltk
nltk.download("stopwords")   # download list of stopwords (only once; need not run it again)
from nltk.corpus import stopwords # import stopwords

from nltk.stem.porter import *

stemmer = PorterStemmer()
from roc_utils import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[681]:


labels = ['Emotion - Anxiety, Claustrophobia - USER2',
 'Nursing - IV Issues- USER2',
 'Communication - Poor Communication- USER2',
 'Service - Long Wait Time- USER2',
 'Technical Issues - Noise, equipment malfunction, computer, imaging issues- USER2',
 'Staff - Poor Treatment- USER2',
 'Testing - Physical Discomfort- USER2',
 'Testing - Long Test Times- USER2',
 'No Comment- USER2',
 'Positive- USER2',
 'Service - Excellent (professional, caring, excellent, etc.)- USER2',
 'Staff - Comforting- USER2',
 'Service - On Time- USER2',
 'Nursing (IV placement)- USER2',
 "Emotion - Relieved Concerns (dont' have to worry anymore)- USER2",
 'Diagnosis - Answered the Question- USER2',
 'Access to Report- USER2',
 'Thoroughness (informative, detailed)- USER2',
 'Follow-up- USER2',
 'Easy to Understand- USER2',
 'No Comment- USER2.1']


# In[682]:


LMmodel = SentenceTransformer('all-mpnet-base-v2')


# In[683]:


train = pd.read_excel('/mnt/storage/User_Feedback/Data/Train_Student.xlsx')
test = pd.read_excel('/mnt/storage/User_Feedback/Data/Test_Student.xlsx')


# In[684]:


train.dtypes


# In[685]:


def review_to_words(review):
    """Convert a raw review string into a sequence of words."""
    
    # TODO: Remove HTML tags and non-letters,
    #soup = BeautifulSoup(review, 'html5lib')
    text = review.lower()
    #       convert to lowercase, tokenize,
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    words = text.split()
    #       remove stopwords and stem
    words = [w.strip() for w in words if w not in stopwords.words('english')]
    words = [stemmer.stem(w) for w in words]

    # Return final list of words
    return ' '.join(words)


# In[686]:


train=train.dropna()


# In[687]:


#train =train.fillna(' ')
train['Comments_proc'] = train['Comments'].apply(review_to_words)
#test =test.fillna(' ')
test['Comments_proc'] = test['Comments'].apply(review_to_words)


# In[688]:


train


# In[689]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[690]:


train['Staff - Comforting- USER2'].value_counts()


# In[691]:


vectorizer = TfidfVectorizer()
train_sentence_embeddings_tf_vec= vectorizer.fit_transform(train['Comments_proc']).toarray()
train_sentence_embeddings_tf_vec= vectorizer.transform(train['Comments_proc']).toarray()
test_sentence_embeddings_tf_vec= vectorizer.transform(test['Comments_proc']).toarray()


# In[692]:


test_sentence_embeddings_tf_vec.shape


# In[693]:


train_sentence_embeddings_tf_vec.shape


# In[694]:


gt = []
pred = []
#pred_dyn = []
legend = []
external = []

for l in labels:
    try:
        y_train = train[l].values
        y_valid = test[l].values
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eta=0.3, silent=1, subsample=0.8, scale_pos_weight=99).fit(train_sentence_embeddings_tf_vec, y_train) 
        xgb_prediction = xgb_model.predict_proba(test_sentence_embeddings_tf_vec)
        print(l)
        print(classification_report(y_valid, xgb_model.predict(test_sentence_embeddings_tf_vec)))
        gt.append(y_valid)
        pred.append(xgb_model.predict_proba(test_sentence_embeddings_tf_vec))
        #external.append(xgb_model.predict(unannotated_sentence_embeddings))
        #pred_dyn.append(xgb_model.predict(unannotated_sentence_dyn_embeddings))
        legend.append(l)
        print('-------------------------------------------------')
    except:
        print('Didnot work: '+l)


# In[695]:


import random
import matplotlib.colors as mcolors

colors = mcolors.CSS4_COLORS
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),name) for name, color in colors.items())
names = [name for hsv, name in by_hsv]
roc = []
_, ax3 = plt.subplots(figsize=(15,10))
for i in range(len(gt)-3):
    p = pred[i]
    plot_roc(compute_roc(X=p[:,1], y=gt[i], pos_label=1), label=legend[i], color=names[i+15], ax=ax3)
# Place the legend outside.
ax3.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax3.set_title("ROC curves");


# In[ ]:





# In[ ]:





# In[ ]:





# In[701]:


from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
# RegEx for removing non-letter characters
import re

# NLTK library for the remaining steps
import nltk
nltk.download("stopwords")   # download list of stopwords (only once; need not run it again)
from nltk.corpus import stopwords # import stopwords

from nltk.stem.porter import *

stemmer = PorterStemmer()
from roc_utils import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[702]:


labels1 = ['Emotion - Anxiety, Claustrophobia',
 'Nursing - IV Issues',
 'Communication - Poor Communication',
 'Service - Long Wait Time',
 'Technical Issues - Noise, equipment malfunction, computer, imaging issues',
 'Staff - Poor Treatment',
 'Testing - Physical Discomfort',
 'Testing - Long Test Times',
 'No Comment',
 'Positive',
 'Service - Excellent (professional, caring, excellent, etc.)',
 'Staff - Comforting',
 'Service - On Time',
 'Nursing (IV placement)',
 "Emotion - Relieved Concerns (dont' have to worry anymore)",
 'Diagnosis - Answered the Question',
 'Access to Report',
 'Thoroughness (informative, detailed)',
 'Follow-up',
 'Easy to Understand']


# In[703]:


train1 = pd.read_excel('/home/mnadella/Train_Student_new.xlsx')
test1 = pd.read_excel('/home/mnadella/Test_Student_new.xlsx')


# In[704]:


train1.head()


# In[670]:


#train1 = train1.drop(train1.columns[1], axis=1, inplace=True)
#train1.head()


# In[705]:


#train1= train1.iloc[:,0:]
#train1.head()
train1.info()


# In[ ]:





# In[706]:


def review_to_words(review):
    """Convert a raw review string into a sequence of words."""
    # TODO: Remove HTML tags and non-letters,
    #soup = BeautifulSoup(review, 'html5lib')
    text = review.lower()
    #       convert to lowercase, tokenize,
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    words = text.split()
    #       remove stopwords and stem
    words = [w.strip() for w in words if w not in stopwords.words('english')]
    words = [stemmer.stem(w) for w in words]

    # Return final list of words
    return ' '.join(words)


# In[707]:


train1 =train1.fillna(' ')
train1['Comments_proc'] = train1['Comments'].apply(review_to_words)
test1 =test1.fillna(' ')
test1['Comments_proc'] = test1['Comments'].apply(review_to_words)


# In[709]:


train1.head()


# In[708]:


train1.info()


# In[710]:


test1.info()


# In[674]:


train_sentence_embeddings1 = LMmodel.encode(train1['Comments'])
test_sentence_embeddings1 = LMmodel.encode(test1['Comments']) 


# In[675]:


train_sentence_embeddings1.shape


# In[678]:


gt = []
pred = []
#pred_dyn = []
legend = []
external = []

for l in labels1:
    y_train = train1[l].values
    y_valid = test1[l].values
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eta=0.3, silent=1, subsample=0.8, scale_pos_weight=99).fit(train_sentence_embeddings1, y_train) 
    xgb_prediction = xgb_model.predict_proba(test_sentence_embeddings1)
    print(l)
    print(classification_report(y_valid, xgb_model.predict(test_sentence_embeddings1)))
    gt.append(y_valid)
    pred.append(xgb_model.predict_proba(test_sentence_embeddings1))
    #external.append(xgb_model.predict(unannotated_sentence_embeddings))
    #pred_dyn.append(xgb_model.predict(unannotated_sentence_dyn_embeddings))
    legend.append(l)
    print('-------------------------------------------------')
     


# In[679]:


import random
import matplotlib.colors as mcolors

colors = mcolors.CSS4_COLORS
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),name) for name, color in colors.items())
names = [name for hsv, name in by_hsv]
roc = []
_, ax3 = plt.subplots(figsize=(15,10))
for i in range(len(gt)-3):
    p = pred[i]
    plot_roc(compute_roc(X=p[:,1], y=gt[i], pos_label=1), label=legend[i], color=names[i+15], ax=ax3)
# Place the legend outside.
ax3.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax3.set_title("ROC curves");


# In[ ]:





# In[ ]:



    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[274]:


get_ipython().system('python -m pip install -U gensim')


# In[283]:


import gensim
from gensim import models


# In[284]:


from gensim.models.word2vec import Word2Vec


# In[ ]:





# In[293]:


get_ipython().system('pip install tensorflow')


# In[294]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[342]:


#Importing required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[ ]:





# In[344]:


labels1 = ['Emotion - Anxiety, Claustrophobia - USER2',
 'Nursing - IV Issues- USER2',
 'Communication - Poor Communication- USER2',
 'Service - Long Wait Time- USER2',
 'Technical Issues - Noise, equipment malfunction, computer, imaging issues- USER2',
 'Staff - Poor Treatment- USER2',
 'Testing - Physical Discomfort- USER2',
 'Testing - Long Test Times- USER2',
 'No Comment- USER2',
 'Positive- USER2',
 'Service - Excellent (professional, caring, excellent, etc.)- USER2',
 'Staff - Comforting- USER2',
 'Service - On Time- USER2',
 'Nursing (IV placement)- USER2',
 "Emotion - Relieved Concerns (dont' have to worry anymore)- USER2",
 'Diagnosis - Answered the Question- USER2',
 'Access to Report- USER2',
 'Thoroughness (informative, detailed)- USER2',
 'Follow-up- USER2',
 'Easy to Understand- USER2',
 'No Comment- USER2.1']


# In[345]:


train1 = pd.read_excel('/mnt/storage/User_Feedback/Data/Train_Student.xlsx')
test1 = pd.read_excel('/mnt/storage/User_Feedback/Data/Test_Student.xlsx')


# In[ ]:





# In[ ]:





# In[347]:


train1 =train1.fillna(' ')
train1['Comments_proc'] = train1['Comments'].apply(review_to_words)
test1 =test1.fillna(' ')
test1['Comments_proc'] = test1['Comments'].apply(review_to_words)


# In[360]:


sentences = train1['Comments_proc'].tolist()
print(len(sentences))
print(sentences[1])
print(sentences[200])


# In[348]:


X_train1 = train1['Comments']
X_test1 = test1['Comments']
y_train1 = train1['Emotion - Anxiety, Claustrophobia - USER2'].values
y_test1 = test1['Emotion - Anxiety, Claustrophobia - USER2'].values



# In[349]:


tr =train1['Comments'].apply(gensim.utils.simple_preprocess)
tr


# In[350]:


te = test1['Comments'].apply(gensim.utils.simple_preprocess)
te

 
# In[371]:


from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
#from transformers import GensimWord2VecVectorizer
from gensim.models import Word2Vec

w2v_model = Word2Vec(window=10, min_count=2, workers=4 , vector_size=100,alpha=0.025, sg=1)
w2v_model.build_vocab(tr, progress_per=1000)
 


# In[373]:


w2v_model.build_vocab(tr, progress_per=1000)


# In[375]:


w2v_model.epochs


# In[ ]:





# In[377]:


train_word = w2v_model.train(tr,total_examples=model.corpus_count, epochs=model.epochs)
train_word


# In[378]:


test_word = w2v_model.train(te,total_examples=model.corpus_count, epochs=model.epochs)
test_word


# In[368]:


import gensim.downloader as api
wv = api.load('word2vec-google-news-300')


# In[370]:


import gensim.downloader as api
wv = api.load('word2vec-google-news-300')


# In[ ]:



    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[420]:


get_ipython().system('pip install plotly')


# In[422]:


# data processing and Data manipulation
import numpy as np # linear algebra
import pandas as pd # data processing

import sklearn
from sklearn.model_selection import train_test_split
    
# Libraries and packages for NLP
import nltk
import gensim
from gensim.models import Word2Vec

# Visualization 
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# In[423]:


labels1 = ['Emotion - Anxiety, Claustrophobia - USER2',
 'Nursing - IV Issues- USER2',
 'Communication - Poor Communication- USER2',
 'Service - Long Wait Time- USER2',
 'Technical Issues - Noise, equipment malfunction, computer, imaging issues- USER2',
 'Staff - Poor Treatment- USER2',
 'Testing - Physical Discomfort- USER2',
 'Testing - Long Test Times- USER2',
 'No Comment- USER2',
 'Positive- USER2',
 'Service - Excellent (professional, caring, excellent, etc.)- USER2',
 'Staff - Comforting- USER2',
 'Service - On Time- USER2',
 'Nursing (IV placement)- USER2',
 "Emotion - Relieved Concerns (dont' have to worry anymore)- USER2",
 'Diagnosis - Answered the Question- USER2',
 'Access to Report- USER2',
 'Thoroughness (informative, detailed)- USER2',
 'Follow-up- USER2',
 'Easy to Understand- USER2',
 'No Comment- USER2.1']


# In[431]:


train1 = pd.read_excel('/mnt/storage/User_Feedback/Data/Train_Student.xlsx')
test1 = pd.read_excel('/mnt/storage/User_Feedback/Data/Test_Student.xlsx')


# In[432]:


train1.head(3)


# In[434]:


train1['Emotion - Anxiety, Claustrophobia - USER2'].value_counts()


# In[ ]:





# In[453]:


#importing libraries
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer







from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
# RegEx for removing non-letter characters
import re

# NLTK library for the remaining steps
import nltk
nltk.download("stopwords")   # download list of stopwords (only once; need not run it again)
from nltk.corpus import stopwords # import stopwords

from nltk.stem.porter import *

stemmer = PorterStemmer()
from roc_utils import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[454]:


train3 = pd.read_excel('/mnt/storage/User_Feedback/Data/Train_Student.xlsx')
test3 = pd.read_excel('/mnt/storage/User_Feedback/Data/Test_Student.xlsx')


# In[480]:


train3['Comments']


# In[455]:


nltk.download('stopwords')


# In[456]:


train3.head()


# In[457]:


train_X_non = train3['Comments']   # '0' refers to the review text
train_y = train3['Emotion - Anxiety, Claustrophobia - USER2']   # '1' corresponds to Label (1 - positive and 0 - negative)
test_X_non = test3['Comments']
test_y = test3['Emotion - Anxiety, Claustrophobia - USER2']
train_X=[]
test_X=[]


# In[458]:


print(len(train_X_non))


# In[459]:


#text pre processing
for i in range(0, len(train_X_non)):
    r = re.sub('[^a-zA-Z]', ' ', train_X_non[i])
    r = r.lower()
    words = r.split()
    
    words = [w.strip() for w in words if w not in stopwords.words('english')]
    words = [stemmer.stem(w) for w in words]
    review = ' '.join(words)
    train_X.append(review)


# In[460]:


#text pre processing
for i in range(0, len(test_X_non)):
    r = re.sub('[^a-zA-Z]', ' ', test_X_non[i])
    r = r.lower()
    words = r.split()
    
    words = [w.strip() for w in words if w not in stopwords.words('english')]
    words = [stemmer.stem(w) for w in words]
    review = ' '.join(words)
    test_X.append(review)


# In[461]:


train_X[10]


# In[462]:


#tf idf
tf_idf = TfidfVectorizer()
#applying tf idf to training data
X_train_tf = tf_idf.fit_transform(train_X)
#applying tf idf to training data
X_train_tf = tf_idf.transform(train_X)


# In[463]:


print("n_samples: %d, n_features: %d" % X_train_tf.shape)


# In[464]:


#transforming test data into tf-idf matrix
X_test_tf = tf_idf.transform(test_X)


# In[465]:


print("n_samples: %d, n_features: %d" % X_test_tf.shape)


# In[466]:


#naive bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tf, train_y)


# In[467]:


#predicted y
y_pred = naive_bayes_classifier.predict(X_test_tf)


# In[468]:


print(metrics.classification_report(test_y, y_pred, target_names=['Positive', 'Negative']))


# In[488]:


gt1 = []
pred1 = []
#pred_dyn = []
legend1 = []
external = []

for l in labels:
    try:
        train_y1 = train[l].values
        y_valid3 = test[l].values
        #naive bayes classifier
        naive_bayes_classifier = MultinomialNB()
        naive_bayes_classifier.fit(X_train_tf, train_y1)
        y_pred = naive_bayes_classifier.predict(X_test_tf)
        print(l)
        print(metrics.classification_report(test_y, y_pred, target_names=['Positive', 'Negative']))
        gt1.append(y_valid3)
        pred1.append(naive_bayes_classifier.predict_proba(X_test_tf))
        #external.append(xgb_model.predict(unannotated_sentence_embeddings))
        #pred_dyn.append(xgb_model.predict(unannotated_sentence_dyn_embeddings))
        legend1.append(l)
        print('-------------------------------------------------')
    except:
        print('Didnot work: '+l)


# In[ ]:





# In[469]:


print("Confusion matrix:")
print(metrics.confusion_matrix(test_y, y_pred))


# In[489]:


import random
import matplotlib.colors as mcolors

colors = mcolors.CSS4_COLORS
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),name) for name, color in colors.items())
names = [name for hsv, name in by_hsv]
roc = []
_, ax3 = plt.subplots(figsize=(15,10))
for i in range(len(gt1)-3):
    p = pred1[i]
    plot_roc(compute_roc(X=p[:,1], y=gt1[i], pos_label=1), label=legend1[i], color=names[i+15], ax=ax3)
# Place the legend outside.
ax3.legend1(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax3.set_title("ROC curves");


# In[492]:


#define metrics
for l in labels:
    train_y1 = train[l].values
    y_valid3 = test[l].values
    #naive bayes classifier
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(X_train_tf, train_y1)
    y_pred = naive_bayes_classifier.predict(X_test_tf)
    print(l)
    print(metrics.classification_report(test_y, y_pred, target_names=['Positive', 'Negative']))
    gt1.append(y_valid3)
    pred1.append(naive_bayes_classifier.predict_proba(X_test_tf))
    #external.append(xgb_model.predict(unannotated_sentence_embeddings))
    #pred_dyn.append(xgb_model.predict(unannotated_sentence_dyn_embeddings))
    legend1.append(l)
    fpr, tpr, _ = metrics.roc_curve(y_valid3,  y_pred)
    auc = metrics.roc_auc_score(y_valid3, y_pred)
    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


# In[ ]:





# In[ ]:





# In[502]:


gt2 = []
pred2 = []
#pred_dyn = []
legend2 = []
external2 = []

for l in labels:
    try:
        y_train4 = train[l].values
        y_valid4 = test[l].values
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eta=0.3, silent=1, subsample=0.8, scale_pos_weight=99).fit(X_train_tf, y_train4) 
        xgb_prediction = xgb_model.predict_proba(X_test_tf)
        print(l)
        print(classification_report(y_valid4, xgb_model.predict(X_test_tf)))
        gt2.append(y_valid4)
        pred2.append(xgb_model.predict_proba(X_test_tf))
        #external.append(xgb_model.predict(unannotated_sentence_embeddings))
        #pred_dyn.append(xgb_model.predict(unannotated_sentence_dyn_embeddings))
        legend2.append(l)
        print('-------------------------------------------------')
    except:
        print('Didnot work: '+l)


# In[503]:


import random
import matplotlib.colors as mcolors

colors = mcolors.CSS4_COLORS
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),name) for name, color in colors.items())
names = [name for hsv, name in by_hsv]
roc = []
_, ax3 = plt.subplots(figsize=(15,10))
for i in range(len(gt2)-3):
    p = pred2[i]
    plot_roc(compute_roc(X=p[:,1], y=gt2[i], pos_label=1), label=legend2[i], color=names[i+15], ax=ax3)
# Place the legend outside.
ax3.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax3.set_title("ROC curves");


# In[ ]:


# Random Forest
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train1, y_train1)
y_pred_rf = rf_classifier.predict(X_test1)


# In[506]:


gt5 = []
pred5 = []
#pred_dyn = []
legend5 = []
external5 = []

for l in labels:
    try:
        train_y5 = train[l].values
        y_valid5 = test[l].values
        # Random Forest
        rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        rf_classifier.fit(X_train_tf, train_y5)
        y_pred = rf_classifier.predict(X_test_tf)
        print(l)
        print(metrics.classification_report(test_y, y_pred, target_names=['Positive', 'Negative']))
        gt5.append(y_valid5)
        pred5.append(rf_classifier.predict_proba(X_test_tf))
        #external.append(xgb_model.predict(unannotated_sentence_embeddings))
        #pred_dyn.append(xgb_model.predict(unannotated_sentence_dyn_embeddings))
        legend5.append(l)
        print('-------------------------------------------------')
    except:
        print('Didnot work: '+l)


# In[508]:


import random
import matplotlib.colors as mcolors

colors = mcolors.CSS4_COLORS
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),name) for name, color in colors.items())
names = [name for hsv, name in by_hsv]
roc = []
_, ax3 = plt.subplots(figsize=(15,10))
for i in range(len(gt5)-3):
    p = pred5[i]
    plot_roc(compute_roc(X=p[:,1], y=gt5[i], pos_label=1), label=legend5[i], color=names[i+15], ax=ax3)
# Place the legend outside.
ax3.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax3.set_title("ROC curves");


# In[ ]:





# In[ ]:





# In[741]:


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec


class GensimWord2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Word vectors are averaged across to create the document-level vectors/features.
    gensim's own gensim.sklearn_api.W2VTransformer doesn't support out of vocabulary words,
    hence we roll out our own.
    All the parameters are gensim.models.Word2Vec's parameters.
    https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
    """

    def __init__(self, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None,
                 sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5,
                 ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                 callbacks=(), max_final_vocab=None):
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.compute_loss = compute_loss
        self.callbacks = callbacks
        self.max_final_vocab = max_final_vocab

    def fit(self, X, y=None):
        self.model_ = Word2Vec(
            sentences=X, corpus_file=None,
            size=self.size, alpha=self.alpha, window=self.window, min_count=self.min_count,
            max_vocab_size=self.max_vocab_size, sample=self.sample, seed=self.seed,
            workers=self.workers, min_alpha=self.min_alpha, sg=self.sg, hs=self.hs,
            negative=self.negative, ns_exponent=self.ns_exponent, cbow_mean=self.cbow_mean,
            hashfxn=self.hashfxn, iter=self.iter, null_word=self.null_word,
            trim_rule=self.trim_rule, sorted_vocab=self.sorted_vocab, batch_words=self.batch_words,
            compute_loss=self.compute_loss, callbacks=self.callbacks,
            max_final_vocab=self.max_final_vocab)
        return self

    def transform(self, X):
        X_embeddings = np.array([self._get_embedding(words) for words in X])
        return X_embeddings

    def _get_embedding(self, words):
        valid_words = [word for word in words if word in self.model_.wv.vocab]
        if valid_words:
            embedding = np.zeros((len(valid_words), self.size), dtype=np.float32)
            for idx, word in enumerate(valid_words):
                embedding[idx] = self.model_.wv[word]

            return np.mean(embedding, axis=0)
        else:
            return np.zeros(self.size)


# In[742]:


import numpy as np
from tqdm import trange
from keras import layers, optimizers, Model
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from keras.preprocessing.sequence import skipgrams, make_sampling_table


class KerasWord2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Word vectors are averaged across to create the document-level vectors/features.
    Attributes
    ----------
    word2index_ : dict[str, int]
        Each distinct word in the corpus gets map to a numeric index.
        e.g. {'unk': 0, 'film': 1}
    index2word_ : list[str]
        Reverse napping of ``word2index_`` e.g. ['unk', 'film']
    vocab_size_ : int
    model_ : keras.models.Model
    """

    def __init__(self, embed_size=100, window_size=5, batch_size=64, epochs=5000,
                 learning_rate=0.05, negative_samples=0.5, min_count=2,
                 use_sampling_table=True, sort_vocab=True):
        self.min_count = min_count
        self.embed_size = embed_size
        self.sort_vocab = sort_vocab
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples
        self.use_sampling_table = use_sampling_table

    def fit(self, X, y=None):
        self.build_vocab(X)
        self.build_graph()
        indexed_texts = self.texts_to_index(X)

        sampling_table = None
        if self.sort_vocab and self.use_sampling_table:
            sampling_table = make_sampling_table(self.vocab_size_)

        for epoch in trange(self.epochs):
            (batch_center,
             batch_context,
             batch_label) = generate_batch_data(
                indexed_texts, self.batch_size, self.vocab_size_, self.window_size,
                self.negative_samples, sampling_table)
            self.model_.train_on_batch([batch_center, batch_context], batch_label)

        return self

    def transform(self, X):
        embed_in = self._get_word_vectors()
        X_embeddings = np.array([self._get_embedding(words, embed_in) for words in X])
        return X_embeddings

    def _get_word_vectors(self):
        return self.model_.get_layer('embed_in').get_weights()[0]

    def _get_embedding(self, words, embed_in):

        valid_words = [word for word in words if word in self.word2index_]
        if valid_words:
            embedding = np.zeros((len(valid_words), self.embed_size), dtype=np.float32)
            for idx, word in enumerate(valid_words):
                word_idx = self.word2index_[word]
                embedding[idx] = embed_in[word_idx]

            return np.mean(embedding, axis=0)
        else:
            return np.zeros(self.embed_size)

    def build_vocab(self, texts):

        # list[str] flatten to list of words
        words = [token for text in texts for token in text]

        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1

        valid_word_count = [(word, count) for word, count in word_count.items()
                            if count >= self.min_count]
        if self.sort_vocab:
            from operator import itemgetter
            valid_word_count = sorted(valid_word_count, key=itemgetter(1), reverse=True)

        index2word = ['unk']
        word2index = {'unk': 0}
        for word, _ in valid_word_count:
            word2index[word] = len(word2index)
            index2word.append(word)

        self.word2index_ = word2index
        self.index2word_ = index2word
        self.vocab_size_ = len(word2index)
        return self

    def texts_to_index(self, texts):
        """
        Returns
        -------
        texts_index : list[list[int]]
            e.g. [[0, 2], [3, 1]]
            each element in the outer list is the sentence, e.g. [0, 2]
            and each element in the inner list is each word represented in numeric index.
        """
        word2index = self.word2index_
        texts_index = []
        for text in texts:
            text_index = [word2index.get(token, 0) for token in text]
            texts_index.append(text_index)

        return texts_index

    def build_graph(self):
        input_center = layers.Input((1,))
        input_context = layers.Input((1,))

        embedding = layers.Embedding(self.vocab_size_, self.embed_size,
                                     input_length=1, name='embed_in')
        center = embedding(input_center)  # shape [seq_len, # features (1), embed_size]
        context = embedding(input_context)

        center = layers.Reshape((self.embed_size,))(center)
        context = layers.Reshape((self.embed_size,))(context)

        dot_product = layers.dot([center, context], axes=1)
        output = layers.Dense(1, activation='sigmoid')(dot_product)
        self.model_ = Model(inputs=[input_center, input_context], outputs=output)
        self.model_.compile(loss='binary_crossentropy',
                            optimizer=optimizers.RMSprop(lr=self.learning_rate))
        return self

    # def build_graph(self):
    #     """
    #     A different way of building the graph where the center word and
    #     context word each have its own embedding layer.
    #     """
    #     input_center = layers.Input((1,))
    #     input_context = layers.Input((1,))

    #     embedding_center = layers.Embedding(self.vocab_size_, self.embed_size,
    #                                         input_length=1, name='embed_in')
    #     embedding_context = layers.Embedding(self.vocab_size_, self.embed_size,
    #                                          input_length=1, name='embed_out')
    #     center = embedding_center(input_center)  # shape [seq_len, # features (1), embed_size]
    #     context = embedding_context(input_context)

    #     center = layers.Reshape((self.embed_size,))(center)
    #     context = layers.Reshape((self.embed_size,))(context)

    #     dot_product = layers.dot([center, context], axes=1)
    #     output = layers.Dense(1, activation='sigmoid')(dot_product)
    #     self.model_ = Model(inputs=[input_center, input_context], outputs=output)
    #     self.model_.compile(loss='binary_crossentropy',
    #                         optimizer=optimizers.RMSprop(lr=self.learning_rate))
    #     return self

    def most_similar(self, positive, negative=None, topn=10):

        # normalize word vectors to make the cosine distance calculation easier
        # normed_vectors = vectors / np.sqrt((word_vectors ** 2).sum(axis=-1))[..., np.newaxis]
        # ?? whether to cache the normed vector or replace the original one to speed up computation
        word_vectors = self._get_word_vectors()
        normed_vectors = normalize(word_vectors)

        # assign weight to positive and negative query words
        positive = [] if positive is None else [(word, 1.0) for word in positive]
        negative = [] if negative is None else [(word, -1.0) for word in negative]

        # compute the weighted average of all the query words
        queries = []
        all_word_index = set()
        for word, weight in positive + negative:
            word_index = self.word2index_[word]
            word_vector = normed_vectors[word_index]
            queries.append(weight * word_vector)
            all_word_index.add(word_index)

        if not queries:
            raise ValueError('cannot compute similarity with no input')

        query_vector = np.mean(queries, axis=0).reshape(1, -1)
        normed_query_vector = normalize(query_vector).ravel()

        # cosine similarity between the query vector and all the existing word vectors
        scores = np.dot(normed_vectors, normed_query_vector)

        actual_len = topn + len(all_word_index)
        sorted_index = np.argpartition(scores, -actual_len)[-actual_len:]
        best = sorted_index[np.argsort(scores[sorted_index])[::-1]]

        result = [(self.index2word_[index], scores[index])
                  for index in best if index not in all_word_index]
        return result[:topn]


def generate_batch_data(indexed_texts, batch_size, vocab_size,
                        window_size, negative_samples, sampling_table):
    batch_label = []
    batch_center = []
    batch_context = []
    while len(batch_center) < batch_size:
        # list[int]
        rand_indexed_texts = np.random.choice(indexed_texts)

        # couples: list[(str, str)], list of word pairs
        couples, labels = skipgrams(rand_indexed_texts, vocab_size,
                                    window_size=window_size,
                                    sampling_table=sampling_table,
                                    negative_samples=negative_samples)
        if couples:
            centers, contexts = zip(*couples)
            batch_center.extend(centers)
            batch_context.extend(contexts)
            batch_label.extend(labels)

    # trim to batch size at the end and convert to numpy array
    batch_center = np.array(batch_center[:batch_size], dtype=np.int)
    batch_context = np.array(batch_context[:batch_size], dtype=np.int)
    batch_label = np.array(batch_label[:batch_size], dtype=np.int)
    return batch_center, batch_context, batch_label


# In[ ]:





# In[ ]:





# In[716]:


from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
# RegEx for removing non-letter characters
import re

# NLTK library for the remaining steps
import nltk
nltk.download("stopwords")   # download list of stopwords (only once; need not run it again)
from nltk.corpus import stopwords # import stopwords

from nltk.stem.porter import *

stemmer = PorterStemmer()
from roc_utils import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[717]:


labels1 = ['Emotion - Anxiety, Claustrophobia',
 'Nursing - IV Issues',
 'Communication - Poor Communication',
 'Service - Long Wait Time',
 'Technical Issues - Noise, equipment malfunction, computer, imaging issues',
 'Staff - Poor Treatment',
 'Testing - Physical Discomfort',
 'Testing - Long Test Times',
 'No Comment',
 'Positive',
 'Service - Excellent (professional, caring, excellent, etc.)',
 'Staff - Comforting',
 'Service - On Time',
 'Nursing (IV placement)',
 "Emotion - Relieved Concerns (dont' have to worry anymore)",
 'Diagnosis - Answered the Question',
 'Access to Report',
 'Thoroughness (informative, detailed)',
 'Follow-up',
 'Easy to Understand']


# In[718]:


train1 = pd.read_excel('/home/mnadella/Train_Student_new.xlsx')
test1 = pd.read_excel('/home/mnadella/Test_Student_new.xlsx')


# In[719]:


train1.shape


# In[720]:


test1.shape


# In[721]:


def review_to_words(review):
    """Convert a raw review string into a sequence of words."""
    # TODO: Remove HTML tags and non-letters,
    #soup = BeautifulSoup(review, 'html5lib')
    text = review.lower()
    #       convert to lowercase, tokenize,
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    words = text.split()
    #       remove stopwords and stem
    words = [w.strip() for w in words if w not in stopwords.words('english')]
    words = [stemmer.stem(w) for w in words]

    # Return final list of words
    return ' '.join(words)


# In[722]:


train1 =train1.fillna(' ')
train1['Comments_proc'] = train1['Comments'].apply(review_to_words)
test1 =test1.fillna(' ')
test1['Comments_proc'] = test1['Comments'].apply(review_to_words)


# In[723]:


train1.head()


# In[724]:


test1.head()


# In[725]:


LMmodel = SentenceTransformer('all-mpnet-base-v2')


# In[736]:


train_sentence_embeddings1 = LMmodel.encode(train1['Comments'])
test_sentence_embeddings1 = LMmodel.encode(test1['Comments']) 


# In[737]:


train_sentence_embeddings1.shape


# In[738]:


test_sentence_embeddings1.shape


# In[739]:


gt = []
pred = []
#pred_dyn = []
legend = []
external = []

for l in labels1:
    y_train = train1[l].values
    y_valid = test1[l].values
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eta=0.3, silent=1, subsample=0.8, scale_pos_weight=99).fit(train_sentence_embeddings1, y_train) 
    xgb_prediction = xgb_model.predict_proba(test_sentence_embeddings1)
    print(l)
    print(classification_report(y_valid, xgb_model.predict(test_sentence_embeddings1)))
    gt.append(y_valid)
    pred.append(xgb_model.predict_proba(test_sentence_embeddings1))
    legend.append(l)
    print('-------------------------------------------------')
     


# In[740]:


import random
import matplotlib.colors as mcolors

colors = mcolors.CSS4_COLORS
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),name) for name, color in colors.items())
names = [name for hsv, name in by_hsv]
roc = []
_, ax3 = plt.subplots(figsize=(15,10))
for i in range(len(gt)-3):
    p = pred[i]
    plot_roc(compute_roc(X=p[:,1], y=gt[i], pos_label=1), label=legend[i], color=names[i+15], ax=ax3)
# Place the legend outside.
ax3.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax3.set_title("ROC curves");


# In[743]:


train2 = pd.read_excel('/home/mnadella/Train_Student_new.xlsx')
test2 = pd.read_excel('/home/mnadella/Test_Student_new.xlsx')


# In[744]:


train2.head(2)


# In[746]:


import gensim
Comments_new = train2.Comments.apply(gensim.utils.simple_preprocess)
Comments_new


# In[ ]:





# In[749]:


train2['Comments_new'] = train2['Comments'].apply(gensim.utils.simple_preprocess)
train2.head(2)


# In[750]:


test2['Comments_new'] = test2['Comments'].apply(gensim.utils.simple_preprocess)
test2.head(2)


# In[751]:


X_train = train2['Comments_new']
X_test =   test2['Comments_new']


# In[755]:


from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from transformers import GensimWord2VecVectorizer

gensim_word2vec_tr = GensimWord2VecVectorizer(size=50, min_count=3, sg=1, alpha=0.025, iter=10)
xgb = XGBClassifier(learning_rate=0.01, n_estimators=100, n_jobs=-1)
w2v_xgb = Pipeline([
    ('w2v', gensim_word2vec_tr), 
    ('xgb', xgb)
])
w2v_xgb


# In[756]:





# In[ ]:




