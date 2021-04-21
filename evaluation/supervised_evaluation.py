from train.supervised_train import Supervised

from utils import *

from model_performance import *

# 70:30
LAST_TRAIN_TIMESTEP = 34
LAST_TIMESTEP = 49

clf_with_score_dict = {
    'lgbm': {'score': [], 'clf': None},
    'rf': {'score': [], 'clf': None},
    'lr': {'score': [], 'clf': None},
    'xgboost': {'score': [], 'clf': None}
}

X_train_df, X_test_df, y_train, y_test = run_elliptic_preprocessing_pipeline(LAST_TRAIN_TIMESTEP, LAST_TIMESTEP)

X_train = X_train_df.values
X_test = X_test_df.values

for k, v in clf_with_score_dict.items():
    print(f'classifier {k}:')
    clf = Supervised(model=k,
                     task='binary',
                     X_train=X_train,
                     y_train=y_train)
    clf_with_score_dict[k]['clf'] = clf
    clf_with_score_dict[k]['score'] = clf.train_cv(cv=5)
    print(f'scores: {clf_with_score_dict[k]["score"]}')

for k in clf_with_score_dict.keys():
    pred = clf_with_score_dict[k]['clf'].predict(X_test)
    f1_micro = calculate_model_score(y_test, pred, 'f1_micro')
    print(f'{k} - f1_micro test: {f1_micro}')