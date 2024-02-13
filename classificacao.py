import pandas as pd 
import sklearn
from ydata_profiling import ProfileReport
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

pd.set_option('display.max_columns', None)

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

#Análise exploratória

#tamanho do dataset
print(df_train.shape)

#colunas
print(df_train.columns)

#verificando os NA
print(df_train.isna().sum())

#verificando tipos
print(df_train.dtypes)

#verificando informações do dataframe
print(df_train.info())

#buscando por duplicatas
print(df_train.duplicated().sum())

#estatistica básica
print(df_train.describe())

#contando as classes
print(df_train.groupby('Exited').agg({'Age':'mean', 'EstimatedSalary':'mean'}))

#Profile Report
#profile = ProfileReport(df_train, title = "Profiling Report")
#profile.to_file("report.html")

#analisando as features

#fig = px.scatter(df_train, x='Age', y='EstimatedSalary')
#fig.show()

##

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,8))

sns.histplot(df_train.loc[df_train.Exited == 1][["Age"]], kde=True, color='grey',ax = axes[0])
axes[0].set_title('Distribuicao de Idade das pessoas que sairam do banco')
axes[0].set_ylabel('\nDensidade\n')
axes[0].set_xlabel('\nIdade\n')

sns.histplot(df_train.loc[df_train.Exited == 0][["Age"]], kde=True, color="grey", ax=axes[1])
axes[1].set_title('Distribuicao de Idade das pessoas que nao sairam do banco')
axes[1].set_ylabel('\nDensidade\n')
axes[1].set_xlabel('\nIdade\n')

plt.tight_layout()
#plt.show()

#colunas
print(df_test.columns)


#Criando o modelo
target = df_train['Exited']
features = df_train.drop(['Surname','id','CustomerId','Exited'], axis=1)

#para converter variáveis categóricas em numéricas
le = LabelEncoder()

features['Geography'] = le.fit_transform(features['Geography'])
features['Gender'] = le.fit_transform(features['Gender'])


#Dividindo o dataset
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

#para normalizar os dados
#fit_transform() no dataset de treino e transform() no dataset de teste

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#modelos

lr = LogisticRegression()
knn = KNeighborsClassifier()
rnd = RandomForestClassifier()

#Regressão Logística
'''print("Regressão Logística")
lr.fit(x_train,y_train)
predict = lr.predict(x_test)
accuracy = accuracy_score(y_test, predict)
precision = precision_score(y_test, predict)
recall = recall_score(y_test, predict)
f1 = f1_score(y_test, predict)
print("Acurácia: %.2f%%" % accuracy)
print("Precisão: %.2f%%" % precision)
print("Recall: %.2f%%" % recall)
print("F1: %.2f%%" % f1)'''

print("-------------------------")

#KNN
print("KNN")
knn.fit(x_train,y_train)
predict = knn.predict(x_test)
accuracy = accuracy_score(y_test, predict)
precision = precision_score(y_test, predict)
recall = recall_score(y_test, predict)
f1 = f1_score(y_test, predict)
print("Acurácia: %.2f%%" % accuracy)
print("Precisão: %.2f%%" % precision)
print("Recall: %.2f%%" % recall)
print("F1: %.2f%%" % f1)

print("-------------------------")

#Random Forest
'''print("Random Forest")
rnd.fit(x_train,y_train)
predict = rnd.predict(x_test)
accuracy = accuracy_score(y_test, predict)
precision = precision_score(y_test, predict)
recall = recall_score(y_test, predict)
f1 = f1_score(y_test, predict)
print("Acurácia: %.2f%%" % accuracy)
print("Precisão: %.2f%%" % precision)
print("Recall: %.2f%%" % recall)
print("F1: %.2f%%" % f1)'''

# Como o caso aqui é prever o churn de clientes, o interessante é termos um baixo valor para
# Falsos Negativos. Não tem problema classificar um cliente que não vai sair como um que vai sair
# mas tem problema classificar um cliente que vai sair como um que não vai.
# Por isso, o melhor modelo é aquele que alcançou o maior valor de Recall: KNN e Random Forest.

###################################

# Agora, com a base de teste, vamos gerar a previsão para os clientes.

df_test = df_test.drop(['Surname','id','CustomerId'], axis=1)
df_test['Geography'] = le.fit_transform(df_test['Geography'])
df_test['Gender'] = le.fit_transform(df_test['Gender'])

#KNN

predict = knn.predict(df_test)
df_test['predicted_churn'] = predict
print(df_test.head())

print(df_test.groupby('predicted_churn').count())