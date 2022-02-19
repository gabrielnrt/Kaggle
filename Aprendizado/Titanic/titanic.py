from pandas import read_csv


df = read_csv('train.csv')

df

#-----------------------------------------------------
df2 = df.pop('Survided')

df['Survided'] = df2


ColunasExcluidas = ['Name', 'Ticket', 'Cabin']

df.drop(columns = ColunasExcluidas,
        inplace = True)
#------------------------------------------------------

df[ df.duplicated(keep = False) ]

df[df['PassengerId'].isnull()]
df[df['Pclass'].isnull()]
df[df['Sex'].isnull()]
df[df['Age'].isnull()]
df[df['Sibsp'].isnull()]

IdadeMedia = df['Age'].mean()
df['Age'].fillna(value = IdadeMedia, inplace = True)

moda = df['Embarked'].mode()
df['Embarked'].fillna(value = 'S', inplace = True)
# Obs: a moda é 'S', e coloquei direto pois se coloco
# value=moda o método não realiza a substituição.

#-----------------------------------------------------------

atributos = df.iloc[:,0:8].values
classe = df.iloc[:,8].values

from sklearn.preprocessing import LabelEncoder

codificador1 = LabelEncoder()
codificador2 = LabelEncoder()

atributos[:,2] = codificador1.fit_transform(atributos[:,2])
atributos[:,7] = codificador2.fit_transform(atributos[:,7])

atributos

#--------------------------------------------------------------

from sklearn.model_selection import train_test_split

xTreino, xTeste, yTreino, yTeste = train_test_split(atributos, classe,
                                                    test_size = 0.3,
                                                    random_state = 0)
xTreino
#-------------------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier

arvore = DecisionTreeClassifier()

arvore.fit(xTreino, yTreino)

previsoes = arvore.predict(xTeste)

previsoes

#------------------------------------------------------------------------------

from sklearn.metrics import confusion_matrix, accuracy_score

matriz = confusion_matrix(previsoes, yTeste)

matriz

TaxaAcerto = accuracy_score(previsoes,yTeste)

TaxaAcerto

#-------------------------------------------------------------------------------

df2 = read_csv('test.csv')

df2

df2.drop(columns = ColunasExcluidas,
        inplace = True)

df2

df2[df2['PassengerId'].isnull()]
df2[df2['Pclass'].isnull()]
df2[df2['Sex'].isnull()]
df2[df2['Age'].isnull()]
df2[df2['SibSp'].isnull()]
df2[df2['Parch'].isnull()]
df2[df2['Fare'].isnull()]
df2[df2['Embarked'].isnull()]


MediaDasIdades = df2['Age'].mean()
df2['Age'].fillna(value = MediaDasIdades, inplace = True)

MediaFare = df2['Fare'].mean()
df2['Fare'].fillna(value = MediaFare, inplace = True)


X = df2.iloc[:,0:8].values

codificador3 = LabelEncoder()
codificador4 = LabelEncoder()

X[:,2] = codificador3.fit_transform(X[:,2])
X[:,7] = codificador4.fit_transform(X[:,7])


Y = arvore.predict(X)

Y

dicionario = {'PassengerId': X[:,0], 'Survided': Y}

from pandas import DataFrame

RespostaObtida = DataFrame(dicionario)

RespostaObtida
RespostaObtida['Survided'].sum()

RespostaObtida.to_csv('RespostaTitanic.csv', index = False)
