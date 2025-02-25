# ODIN
Repository for ODIN project

Notes for users:
* **DO NOT PUSH ANY HOSPITAL DATA TO GIT, EVER**
* **THIS ALSO MEANS THAT YOU HAVE TO CLEAR YOU NOTEBOOK CELLS**
* Put jsons/yaml etc. with configuration information in the *asset* folder



We have three models:
* base model: pipeline([TFIDF vectorizer (vanilla settings) for the text->PCA(0.90), standardize the rest], SVM/XGBoost)
* ResNet1D
* H-Transformer 1D


```
from sklearn.pipeline import Pipeline

il_tubo = Pipeline([
    ('Vectorizer', tfidf_vectorizer(**vkwargs)),
    ('PCA', PCA(**pkwargs)), 
    ('Clf', SVM(**clfkwargs))
])
```
Difficulty is putting the cumulative variance restriction IN il_tubo during inference(il_tubo.predict).
