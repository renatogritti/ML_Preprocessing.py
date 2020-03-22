import pandas as pd
import numpy as np


base = pd.read_csv('census.csv')

df_previsores = base.iloc[:, 1:3]
previsores = base.iloc[:, 1:3].values 

classe = base.iloc[:, 14].values
                
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# ENCONDIG Categories
#labelencoder_previsores = LabelEncoder()
#previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])

##### OR

# HOT ENCONDING WITH LABEL ENCONDING
campo1 = pd.get_dummies(previsores[:,0])
# campo2 = pd.get_dummies(previsores[:,0])

previsores = np.delete(previsores,0, axis=1)
# previsores = np.delete(previsores,0, axis=1)


prev = pd.DataFrame(previsores)
prev = pd.concat([campo1,prev], axis =1)
previsores = prev.values



##STANDARD SCALER

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


#### SPLIT DATA

from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

# importação da biblioteca
# criação do classificador
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)