# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Kaggle-Titanic
# - https://www.kaggle.com/code/makarakinishin/first-steps-eda-fe-models-enhancement/notebook

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Round 0. EDA
#
# ### Data dictionary
# - survived : 생존=1, 죽음=0
# - pclass : 승객 등급. 1등급=1, 2등급=2, 3등급=3
# - sibsp : 함께 탑승한 형제 또는 배우자 수
# - parch : 함께 탑승한 부모 또는 자녀 수
# - ticket : 티켓 번호
# - cabin : 선실 번호
# - embarked : 탑승장소 S=Southhampton, C=Cherbourg, Q=Queenstown
# -

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv", index_col="PassengerId")

df.head(5)

df.info()

df.describe()

df.describe(include='O')

print('Overall survival rate:', sep='\n')
round(df['Survived'].value_counts() / df['Survived'].count(), 2)


def show_destibution_by_target(col, df=df, FE=False, inp=None):
    tt = df.copy()
    if FE == True:
        tt[col] = inp
    return tt[[col, 'Survived']].groupby([col], as_index=False).mean().sort_values('Survived', ascending=False)


show_destibution_by_target('Sex')

show_destibution_by_target('Pclass')

show_destibution_by_target('SibSp')

show_destibution_by_target('Parch')

sns.pairplot(df, hue='Survived')


def cat_plotter(df, cats, plot_type, hue='Survived', xplots=None, yplots=None):
    n_cats = len(cats)
    if not xplots:
        xplots = yplots = int(np.ceil(np.sqrt(n_cats)))
    
    fig = plt.figure(figsize=(10, 6), layout='constrained') if xplots > 1 else plt.figure(figsize=(5, 3))
    
    for n, i in zip(range(n_cats), cats):
        plt.subplot(xplots, yplots, n+1)
        if plot_type == sns.histplot:
            plot_type(df, x=i, hue=hue, multiple='dodge')
        else: 
            plot_type(df[i])


cat_plotter(df, ['Pclass', 'SibSp', 'Parch', 'Sex', 'Embarked'], sns.histplot)

cat_plotter(df, ['Pclass', 'Age'], sns.histplot, hue='Sex')

# +
continious = ['Age', 'Fare']

fig = plt.figure(figsize=(10, 4), layout='constrained')

for i in range(len(continious)):
    plt.subplot(2,2,i+1)
    sns.histplot(df, x=continious[i], hue='Survived', multiple='dodge', cbar=True)
# -

# ### Observations:
# The majority of passengers:
# - Died (~61%)
# - Were male between 18 and 35 y.o.
# - Were of low-fare category
#
# More likely survivers were:
# - Women
# - People of higher class (1st)
# - Children before 7 y.o. and very old people
# - People who paid fare more than 50
# - People who had family on board (1-3 parents/children; 1-2 siblings/spouse)
#
# df observations:
# - People had different titles. Might be related to the task
# - Too many missing values on Cabin. It seems logical that cabin info were missing for people who couldn't survive
# - Also seems logical that missing age might be related to survival

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Round 1. Intuitive Feature Engineering + Preprocessing
# -

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA

# ### Age

df[df['Age'].isna()].head()

# +
# Impute missing age values
imputer = KNNImputer(missing_values=np.nan, n_neighbors=3, add_indicator=True).fit_transform(df[['Age']])

age_pred = imputer[:, 0]

# binarize missing age
age_missing = [int(i) for i in imputer[:, 1]]

# categorize all ages
age_cat = [int(i // 10) for i in age_pred]
# -

show_destibution_by_target('age_missing', df=df, FE=True, inp=age_missing)

# - Although rates are different, this category doesnt seems to be very important
#
# ### Cabin

# See if the assumption about missing cabin is valid
df[df['Cabin'].isna()]['Survived'].value_counts()

# binarize cabin 0 - has cabin info, 1 - cabin was missing
miss_cabin = [abs(int(type(i) is str)-1) for i in df['Cabin']]

show_destibution_by_target('miss_cabin', df=df, FE=True, inp=miss_cabin)

# - People with missing Cabin had lower survival rate (~30%)
# - On the other hand, if cabin was known, survival rate was almost twice higher
#
# ### Titles

# Get all the titles
titles = [i.split('.')[0].split(' ')[-1] for i in df['Name']]

show_destibution_by_target('title', df=df, FE=True, inp=titles)

# ### Family on board

family_size = [s+p for s, p in zip(df['SibSp'], df['Parch'])]

show_destibution_by_target('family_size', df=df, FE=True, inp=family_size)

# - peolpe with very big families tended to die
#
# ### Insert new features and visualize destribution

temp_df = df.copy()

# +
# Fill NaN for Age column
temp_df['Age'] = age_pred

# Inserting new categories
temp_df['age_cat'] = age_cat
temp_df['age_missing'] = age_missing
temp_df['miss_cabin'] = miss_cabin
temp_df['title'] = titles
temp_df['family_size'] = family_size

# Label Encoder
le_cats = ['Sex', 'Embarked']
for i in le_cats:
    temp_df[i] = le().fit_transform(temp_df[i])

# +
to_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']

temp_df = temp_df.drop(to_drop, axis=1)
# -

# - Let's work a little bit with the title

# +
higher_noble = ['Sir', 'Countess', 'Ms', 'Mme', 'Lady', 'Mlle']
simple_people = ['Mr', 'Mrs', 'Miss']

titles_to_leave = [('noble' if i in higher_noble else 'other' if i not in simple_people else i) for i in temp_df['title']]
# -

# - Statistics of title

# +
from collections import Counter

titles_counts = Counter(titles_to_leave)

for title, count in titles_counts.items():
    print(f"{title}: {count}")

# +
from collections import Counter

titles_counts = Counter(titles)

for title, count in titles_counts.items():
    print(f"{title}: {count}")
# -

temp_df['title'] = titles_to_leave

dum_temp_df = pd.get_dummies(temp_df, columns=['Pclass', 'Sex', 'miss_cabin', 'title'])

dum_temp_df.info()

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Round 2. Baseline model(s)
#
# - The majority of the features are categorical, so we're not gonna use liniear models or KNN
# -

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate


def make_split(df, target, size=.2):
    X = df.copy().drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, random_state=5, test_size=size)


def eval_model(model, X, y):
    """
    model: should be passed already fitted with X_train, y_train
    X, y: should be passed as X_train, y_train
    """
    cvc = cross_validate(model, X, y, scoring=['accuracy', 'f1', 'roc_auc'], cv=10)

    return {key_ : round(val_.mean(), 4) for key_, val_ in cvc.items()}


X_train, X_test, y_train, y_test = make_split(dum_temp_df, target='Survived')

# ### Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear', random_state=5)
lr.fit(X_train, y_train)

# ### SVM RBF

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

svc = make_pipeline(StandardScaler(), 
                    SVC(kernel='rbf', random_state=5)
)
svc.fit(X_train, y_train)

# ### KNN
# - Expected to work poorly on dummied DF due to curse of dimensionality
# - More important result would be after Feature selection and oversampling

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# ### RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=15)
rfc.fit(X_train, y_train)

# ### XGBoost

# - Install xgboost
#
# ```python
# import sys
# # !{sys.executable} -m pip install xgboost
# ```
#
# - If need, install libomp
# ```shell
# brew install libomp
# ```

from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=5)
xgb.fit(X_train, y_train)

# ### HistGradientBoostingClassifier

from sklearn.ensemble import HistGradientBoostingClassifier

# Mark categories since it works better with this
# categories = [i for i in X_train.columns if X_train[i].dtype != 'float64']
hgbc = HistGradientBoostingClassifier(random_state=5)
hgbc.fit(X_train, y_train)

# ### Preparing for scoring

models = [lr, svc, knn, rfc, xgb, hgbc]
names = ['LR', 'SVM', 'KNN', 'RFC', 'XGB', 'HGBC']


# +
def internal_scorer(estimators: list, estim_names, mode, X_train, y_train, X_test, y_test):
    scores = []
    for estim, name in zip(estimators, names):
        estim.fit(X_train, y_train)
        scores.append(round(estim.score(X_test, y_test) * 100, 2))

    dfdf = pd.DataFrame(scores, columns=[mode])
    dfdf.index = estim_names
    return dfdf

def cv_table_score(X, y, models=models, names=names):
    dicts = [(name, eval_model(model.fit(X, y), X, y)) for name, model in zip(names, models)]
    total = {k : [dic[1][k] for dic in dicts] for k, v in dicts[0][1].items() if 'time' not in k}

    total_df = pd.DataFrame.from_dict(total)
    total_df.index = names

    return total_df


# -

baseline = internal_scorer(models, names, 'Baseline', X_train, y_train, X_test, y_test)

y_train.value_counts()

# - Lets try to stick with decision trees
#
# ### Extra: Try balancing target class
#
# ```python
# y_train.value_count()
# ```
#
# - y_train의 값이 불균형해서 오버샘플링 기법을 통해 밸런싱을 하는 코드라는데 y_train의 값 분포를 보면 0=438, 1=274로 나온다. 이게 그렇게 불균형인가 의문이다.
#

from imblearn.over_sampling import SMOTE

ROS = SMOTE(random_state=5)
X_OS_train, y_OS_train = ROS.fit_resample(X_train, y_train)

base_OS = pd.concat([baseline, internal_scorer(models, names, 'OverSampling', X_OS_train, y_OS_train, X_test, y_test)], axis=1)

# ### Comparing

cv_table_score(X_train, y_train)

cv_table_score(X_OS_train, y_OS_train)

base_OS

# - The majority of metrics doesnt drop neither on CrossVal nor on internal scoring, so lets have some Over-sampling

X_train, y_train = ROS.fit_resample(X_train, y_train)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Round 3. Feature Selection
# -

from sklearn.feature_selection import f_classif, chi2
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest

# ### KBest

# +
kbest = SelectKBest(chi2)
kbest.fit(X_train, y_train)

kbest_leave = kbest.get_feature_names_out()
# -

# ### RFECV

# +
rfecv = RFECV(rfc)
rfecv.fit(X_train, y_train)

rfecv_leave = rfecv.get_feature_names_out()


# -

# ### Feature Importance by random

# +
def get_feature_importances(model, X, y, thresh=None):
    model = model(min_samples_leaf=10, random_state=5)
    temp_X = X.copy()

    if thresh == None:
        np.random.seed(1)
        temp_X['random_feature'] = np.random.random(X.shape[0])
        model.fit(temp_X, y)
        thresh = model.feature_importances_[-1]

    model.fit(temp_X, y)
    features = dict(sorted({name:val for name, val in zip(temp_X.columns, model.feature_importances_)}.items(), key=lambda x: x[1]))
    to_leave = [key for key, val in features.items() if val >= thresh]
    if thresh == None:
        to_leave.remove('random_feature')

    return thresh, features, to_leave

def show_FI(features, thresh=None):
    fig = plt.figure(figsize=(8, 6))

    plt.plot([thresh, thresh], [0, len(features)+.2], 'ro--')
    plt.barh([i for i in features.keys()], features.values())


# +
thresh, features, FI_leave = get_feature_importances(RandomForestClassifier, X_train, y_train)

show_FI(features, thresh)
# -

# ### Hard selection
# - Scores on leaving features voted by all of the models

# +
hard_selection = list(set(kbest_leave) & set(rfecv_leave) & set(FI_leave))

FS_hard = pd.concat([
    base_OS,
    internal_scorer(models, names, 'Hard Selection', X_train[hard_selection], y_train, X_test[hard_selection], y_test)
], axis=1)
# -

# ### Soft selection
# - Scores on leaving features voted by at least one model

# +
soft_selection = list(set(kbest_leave) | set(rfecv_leave) | set(FI_leave))
soft_selection.remove('random_feature')

FS_soft = pd.concat([
    FS_hard,
    internal_scorer(models, names, 'Soft Selection', X_train[soft_selection], y_train, X_test[soft_selection], y_test)
], axis=1)
# -

# ### Comparing the results

voting = pd.concat(
    map(pd.Series,
        map(sorted, [kbest_leave, rfecv_leave, FI_leave, hard_selection, soft_selection])
    ), axis=1
).fillna('-')
voting.rename(columns={0: 'KBest', 1: 'RFECV', 2: 'Manual FI', 3: 'Hard Selection', 4: 'Soft Selection'})

cv_table_score(X_train[hard_selection], y_train)

cv_table_score(X_train[soft_selection], y_train)

FS_soft

# ### Let's stick with the soft selection because:
# - It gives us more balanced results
# - It affect almost every model in a positive way (internal model scoring, not CV)
# - Also the majority of models benefit from Soft selection more on CV
# - Gives us less synthetic dataset on the return
# - Hard selection drops almost every CrossVal metrics
#
# ### By the way:
# - As expected, minimizing n_features with the Soft selection gave us a positive result on KNN-model
#
# <i><b>We're not afraid of you naymore, you curse of dimensionality!</b></i>

X_train = X_train[soft_selection]
X_test = X_test[soft_selection]

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Round 4. Playing with Hyper-params
#
# ### Find the best parameters for multiple basic models
# - GridSearchCV
# - RandomizedSearchCV
# -

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV

# ### LogisticRegression

lr.get_params()

# +
lr_params = {
    'penalty': ['l2', 'l1'],
    'solver': ['liblinear'],
    'C': [0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1],
    'warm_start': [True, False],
    'class_weight': ['balanced', None],
    'max_iter': [500]
}

HRCV = HalvingGridSearchCV(lr, lr_params, factor=10, cv=10, scoring='f1', n_jobs=-1, random_state=5)
HRCV.fit(X_train, y_train)
lr_opt = HRCV.best_estimator_
# -

lr_opt

# ### SVM
#
# <i>This one would be a little fussy:</i>
# - We have imitate to the model by scaling data and refitting before GridSearch

svc.get_params()['steps'][1][1].get_params()

# +
X_scaled = StandardScaler().fit_transform(X_train)

svc101 = SVC(random_state=5)
svc101.fit(X_scaled, y_train)

svc_params = {
    'C' : [0.1, 0.5, 1],
    'kernel' : ['rbf', 'linear', 'poly', 'sigmoid'],
    'degree' : [1, 2, 3, 4],
    'gamma' : ['scale', 'auto'],
    'class_weight' : [None, 'balanced'],
    'decision_function_shape' : ['ovr', 'ovo'],
    'probability' : [True],
    'max_iter' : [20000]
}

HRCV = HalvingGridSearchCV(svc101, svc_params, factor=5, cv=10, scoring='f1', n_jobs=-1, random_state=5)
HRCV.fit(X_scaled, y_train)
svc_opt = make_pipeline(StandardScaler(), HRCV.best_estimator_)
# -

svc_opt

# ### KNN

knn.get_params()

# +
knn_params = {
    'n_neighbors' : [1, 3, 5, 7, 10, 13, 15],
    'algorithm' : ['auto', 'kd_tree', 'ball_tree', 'brute'],
    'leaf_size' : [10, 20, 30, 50],
    'p' : [1, 2, 3],
    'weights' : ['uniform', 'distance']
}

HRCV = HalvingGridSearchCV(knn, knn_params, factor=5, cv=10, scoring='f1', n_jobs=-1, random_state=5)
HRCV.fit(X_train, y_train)
knn_opt = HRCV.best_estimator_
# -

knn_opt

# ### RandomForest

rfc.get_params()

# +
rfc_params = {
    'n_estimators' : range(50, 121, 10),
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'max_depth' : range(1, 20, 4),
    'bootstrap' : [True, False],
    'min_samples_split' : [15, 20, 25, 30],
    'min_samples_leaf' : [1, 2, 3, 5],
    'max_features' : ['sqrt', 'log2'],
    'class_weight' : ['balanced', 'balanced_subsample'],
    'ccp_alpha' : [0.0, 0.1, 0.3, 0.5]
}

HRCV = HalvingRandomSearchCV(rfc, rfc_params, factor=5, cv=10, scoring='f1', n_jobs=-1, random_state=5)
HRCV.fit(X_train, y_train)
rfc_opt = HRCV.best_estimator_
# -

rfc_opt

# ### XGB

xgb.get_params()

# +
xgb_params = {
    'n_estimators' : range(90, 131, 10),
    'max_depth' : range(1, 20, 5),
    'max_leaves' : range(0, 51, 10),
    'gamma' : [0.0, 0.2, 0.3, 0.5],
    'min_child_weight' : [0.0, 0.2, 0.5],
    'sampling_method': ['uniform']
}

HRCV = HalvingRandomSearchCV(xgb, xgb_params, factor=5, cv=10, scoring='f1', n_jobs=-1, random_state=5)
HRCV.fit(X_train, y_train)
xgb_opt = HRCV.best_estimator_
# -

xgb_opt

# ### HGBC

hgbc.get_params()

# +
hgbc_params = {
    'max_leaf_nodes' : [None, 5, 10, 30, 50, 70],
    'max_depth' : [None, 100, 120, 150],
    'min_samples_leaf' : [10, 20, 50, 90],
    'l2_regularization' : [.0, .1, .2, .5],
    'max_iter' : [200, 500],
    'n_iter_no_change' : [2, 5, 10]
}

HRCV = HalvingRandomSearchCV(hgbc, hgbc_params, factor=5, cv=10, scoring='accuracy', n_jobs=-1)
HRCV.fit(X_train, y_train)
hgbc_opt = HRCV.best_estimator_
# -

hgbc_opt

# ### Comparing

models1 = [lr_opt, svc_opt, knn_opt, rfc_opt, xgb_opt, hgbc_opt]
names1 = ['LR', 'SVM', 'KNN', 'RFC', 'XGB', 'HGBC']

# - Old results

cv_table_score(X_train, y_train, models, names)

# - New results after optimization

cv_table_score(X_train, y_train, models=models1, names=names1)

optimizing = pd.concat([
    FS_soft,
    internal_scorer(models1, names1, 'SearchCV', X_train, y_train, X_test, y_test)
], axis=1)

optimizing

# - Very doubtful that we need this

# ## Round 5. Hard models. Ensembling
#
# Train ensemble models
# - Bagging
# - Voting classifier
# - Stacking

from sklearn.ensemble import BaggingClassifier, StackingClassifier, VotingClassifier

# ### Voting

estimators = [(name, model) for name, model in zip(names1, models1)]
voting = VotingClassifier(estimators, voting='soft')
voting.fit(X_train, y_train)

voting.score(X_test, y_test)

# ## Test prediction

test_df = pd.read_csv('test.csv', index_col='PassengerId')

test_df


def preprocess(df):
    # Impute missing age values
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=3, add_indicator=True).fit_transform(df[['Age']])
    
    # Fill NaN for Age column
    df['Age'] = imputer[:, 0]

    # Inserting new categories
    df['age_cat'] = [int(i // 10) for i in df['Age']]
    df['age_missing'] = [int(i) for i in imputer[:, 1]]
    df['miss_cabin'] = [abs(int(type(i) is str)-1) for i in df['Cabin']]
    df['title'] = [i.split('.')[0].split(' ')[-1] for i in df['Name']]
    df['family_size'] = [s+p for s, p in zip(df['SibSp'], df['Parch'])]
    df.loc[df['Fare'].isna(), 'Fare'] = df['Fare'].median()

    # Label Encoder
    le_cats = ['Sex', 'Embarked']
    for i in le_cats:
        df[i] = le().fit_transform(df[i])

    to_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
    df = df.drop(to_drop, axis=1)

    higher_noble = ['Sir', 'Countess', 'Ms', 'Mme', 'Lady', 'Mlle']
    simple_people = ['Mr', 'Mrs', 'Miss']

    titles_to_leave = [('noble' if i in higher_noble else 'other' if i not in simple_people else i) for i in df['title']]
    df['title'] = titles_to_leave

    return pd.get_dummies(df, columns=['Pclass', 'Sex', 'miss_cabin', 'title'])


test_data_X = preprocess(test_df)[soft_selection]

test_data_X

prediction = test_df.drop([i for i in test_df.columns], axis=1)
prediction['Survived'] = voting.predict(test_data_X)

prediction.head()

prediction.to_csv('submission_v2.csv')
