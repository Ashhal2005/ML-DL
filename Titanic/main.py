import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


df = sns.load_dataset("titanic")
#print (df.info())
#print(df.head(10))
print (df.describe)

X = df[["pclass", "sex", "age", "fare", "embarked"]]
y = df.survived

X_test, y_test, X_train, y_train = train_test_split(X, y , test_size=0.2, random_state=42)

mean_transform = Pipeline ([("mean_impute", SimpleImputer)])
print (mean_transform)


