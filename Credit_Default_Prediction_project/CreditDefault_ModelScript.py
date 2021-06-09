import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


def save(X, path):
    X.to_csv(path, index=False, encoding='utf-8', sep=';')


class DataPreprocessing:
    """Подготовка исходных данных"""

    def __init__(self):
        """Параметры класса"""
        self.medians = None
        self.modes = None
        self.gp_annual_income = None

        # """Parameters for model Annual Income"""
        # self.target_AI = 'Annual Income'
        # self.features_AI = None
        # # self.features_AI = ['Mean Income By Stage', 'Years in current job', 'Tax Liens', 'Number of Open Accounts',
        # #                     'Maximum Open Credit', 'Number of Credit Problems', 'Bankruptcies', 'Term',
        # #                     'Current Loan Amount', 'Monthly Debt',
        # #                     'Home Ownership_Home Mortgage', 'Home Ownership_Own Home', 'Home Ownership_Rent']
        # self.X_train_AI = None
        # self.y_train_AI = None
        # self.X_test_AI = None
        # self.catb_for_AI = None

        # """Parameters for model Credit Score"""
        # self.target_CS = 'Credit Score'
        # self.features_CS = None
        # # self.features_CS = ['Annual Income', 'Years in current job',
        # #                     'Number of Open Accounts', 'Years of Credit History', 'Maximum Open Credit',
        # #                     'Months since last delinquent', 'Term', 'Current Loan Amount',
        # #                     'Current Credit Balance', 'Monthly Debt', 'Mean Income By Stage',
        # #                     'Is Credit Score too large']
        # self.X_train_CS = None
        # self.y_train_CS = None
        # self.X_test_CS = None
        # self.catb_for_CS = None

    @staticmethod
    def fill_to_zero(X, columns):
        for column in columns:
            X[column].fillna(0, inplace=True)

    @staticmethod
    def gap_binary_feature(X, column):
        X[f'Nan {column}'] = 0
        X[f'Nan {column}'].loc[X[column].isnull()] = 1

    def fit(self, X):
        """Save statistics"""
        self.medians = X.median()
        self.modes = X.mode().loc[0]
        self.gp_annual_income = X.groupby(['Years in current job'])['Annual Income'].mean()

        """for filling Annual Income NaN"""
        # self.X_train_AI = X[self.features_AI].loc[X['Annual Income'].notnull()]
        # self.y_train_AI = X[self.target_AI].loc[X['Annual Income'].notnull()]
        # self.catb_for_AI = CatBoostRegressor(eval_metric='R2',
        #                                      cat_features=['Home Ownership'],
        #                                      n_estimators=350,
        #                                      l2_leaf_reg=2,
        #                                      max_depth=4,
        #                                      learning_rate=0.026,
        #                                      silent=True,
        #                                      random_state=21)
        #
        # self.catb_for_AI.fit(self.X_train_AI, self.y_train_AI)

        """for filling Credit Score NaN"""
        # self.X_train_CS = X[self.features_AI].loc[X['Credit Score'].notnull()]
        # self.y_train_CS = X[self.target_CS].loc[X['Credit Score'].notnull()]
        # self.catb_for_CS = CatBoostRegressor(eval_metric='R2',
        #                                      n_estimators=400,
        #                                      max_depth=5,
        #                                      learning_rate=0.045,
        #                                      silent=True,
        #                                      leaf_estimation_iterations=280,
        #                                      random_state=21)
        # self.catb_for_CS.fit(self.X_train_CS, self.y_train_CS)

    def transform(self, X):
        """Transform data"""
        # Home Ownership
        X['Home Ownership'].loc[X['Home Ownership'] == 'Have Mortgage'] = 'Home Mortgage'
        #
        # Purpose
        X.loc[X['Purpose'].isin(['business loan', 'small business', 'renewable energy']), 'Purpose'] = 'business'
        X.loc[X['Purpose'].isin(['medical bills', 'other']), 'Purpose'] = 'medical'
        X.loc[X['Purpose'].isin(
            ['take a trip', 'debt consolidation', 'wedding', 'buy house', 'home improvements', 'buy a car',
             'vacation']), 'Purpose'] = 'personal'
        X.loc[X['Purpose'].isin(['educational expenses', 'moving', 'major purchase']), 'Purpose'] = 'other'

        # Maximum Open Credit
        for i in range(0, len(X['Maximum Open Credit'].values)):
            if X['Maximum Open Credit'].values[i] == 0:
                X['Maximum Open Credit'].values[i] = X['Current Loan Amount'].values[i]
        #
        # Years in current job
        X.loc[X['Years in current job'].isin(['< 1 year', '1 year']), 'Years in current job'] = 1
        X.loc[X['Years in current job'].isin(['2 years', '3 years', '8 years']), 'Years in current job'] = 2
        X.loc[X['Years in current job'].isin(['4 years', '5 years', '6 years', '7 years']), 'Years in current job'] = 3
        X.loc[X['Years in current job'].isin(['10+ years', '9 years']), 'Years in current job'] = 4
        X.loc[X['Years in current job'] == '10+ years', 'Years in current job'] = 4
        print(1)

        # Current Loan Amount
        X['Current Loan Amount'].loc[X['Current Loan Amount'] == 99999999.0] = 0

        # Credit Score - outliers column
        X['Is Credit Score too large'] = 0
        X['Is Credit Score too large'].loc[X['Credit Score'] > 1000] = 1
        #
        # Credit Score
        try:
            X['Credit Score'].loc[(X['Credit Score'] > 750) & (X['Credit Default'] == 1)] = 550
            X['Credit Score'].loc[(X['Credit Score'] > 750) & (X['Credit Default'] == 0)] = 750
        except KeyError:
            X['Credit Score'].loc[X['Credit Score'] > 750] = 550
        #
        """Gap processing"""
        # Years in current job
        X['Years in current job'].loc[X['Years in current job'].isnull()] = self.modes['Years in current job']
        X['Years in current job'] = X['Years in current job'].astype(int)
        print(1)

        # Months since last delinquent, Bankruptcies
        self.fill_to_zero(X, ['Months since last delinquent', 'Bankruptcies'])

        # Term
        X['Term'] = X['Term'].map({'Short Term': '1', 'Long Term': '0'}).astype(int)

        # Home Ownership, Purpose
        for cat_colname in X.select_dtypes(include='object'):
            X = pd.concat([X, pd.get_dummies(X[cat_colname], prefix=cat_colname)], axis=1)
        #
        """NaN filling for Annual Income"""
        # Nan feature
        self.gap_binary_feature(X, 'Annual Income')
        #
        # New feature Mean Income By Stage
        X['Mean Income By Stage'] = X['Years in current job'].map(self.gp_annual_income.to_dict())

        # # Model for Annual Income
        # self.features_AI = ['Mean Income By Stage', 'Years in current job', 'Tax Liens', 'Number of Open Accounts',
        #                     'Maximum Open Credit', 'Number of Credit Problems', 'Bankruptcies', 'Term',
        #                     'Current Loan Amount', 'Monthly Debt',
        #                     'Home Ownership_Home Mortgage', 'Home Ownership_Own Home', 'Home Ownership_Rent']
        # self.X_test_AI = X[self.features_AI].loc[X['Annual Income'].isnull()]
        # X.loc[X['Annual Income'].isnull(), 'Annual Income'] = self.catb_for_AI.predict(self.X_test_AI)
        #
        """NaN filling for Credit Score"""
        # Nan feature
        self.gap_binary_feature(X, 'Credit Score')
        # Model for Credit Score
        # self.features_CS = ['Annual Income', 'Years in current job',
        #                     'Number of Open Accounts', 'Years of Credit History', 'Maximum Open Credit',
        #                     'Months since last delinquent', 'Term', 'Current Loan Amount',
        #                     'Current Credit Balance', 'Monthly Debt', 'Mean Income By Stage',
        #                     'Is Credit Score too large']
        # self.X_test_CS = X[self.features_CS].loc[X['Credit Score'].isnull()]
        # X.loc[X['Credit Score'].isnull(), 'Credit Score'] = self.catb_for_CS.predict(self.X_test_CS)

        X.fillna(self.medians, inplace=True)
        return X


class AnnualIncomeFilling:

    def __init__(self):
        """Parameters for model Annual Income"""
        self.target_AI = 'Annual Income'
        self.features_AI = ['Mean Income By Stage', 'Years in current job', 'Tax Liens', 'Number of Open Accounts',
                            'Maximum Open Credit', 'Number of Credit Problems', 'Bankruptcies', 'Term',
                            'Current Loan Amount', 'Monthly Debt',
                            'Home Ownership_Home Mortgage', 'Home Ownership_Own Home', 'Home Ownership_Rent']
        self.X_train_AI = None
        self.y_train_AI = None
        self.X_test_AI = None
        self.catb_for_AI = None

    def fit(self, X):
        self.X_train_AI = X[self.features_AI].loc[X['Annual Income'].notnull()]
        self.y_train_AI = X[self.target_AI].loc[X['Annual Income'].notnull()]
        self.catb_for_AI = CatBoostRegressor(eval_metric='R2',
                                             cat_features=['Home Ownership'],
                                             n_estimators=350,
                                             l2_leaf_reg=2,
                                             max_depth=4,
                                             learning_rate=0.026,
                                             silent=True,
                                             random_state=21)

        self.catb_for_AI.fit(self.X_train_AI, self.y_train_AI)

    def predict(self, X):
        # Model for Annual Income
        self.X_test_AI = X[self.features_AI].loc[X['Annual Income'].isnull()]
        X.loc[X['Annual Income'].isnull(), 'Annual Income'] = self.catb_for_AI.predict(self.X_test_AI)


class CreditScoreFilling:

    def __init__(self):
        """Parameters for model Credit Score"""
        self.target_CS = 'Credit Score'
        self.features_CS = ['Annual Income', 'Years in current job',
                            'Number of Open Accounts', 'Years of Credit History', 'Maximum Open Credit',
                            'Months since last delinquent', 'Term', 'Current Loan Amount',
                            'Current Credit Balance', 'Monthly Debt', 'Mean Income By Stage',
                            'Is Credit Score too large']
        self.X_train_CS = None
        self.y_train_CS = None
        self.X_test_CS = None
        self.catb_for_CS = None

    def fit(self, X):
        self.X_train_CS = X[self.features_CS].loc[X['Credit Score'].notnull()]
        self.y_train_CS = X[self.target_CS].loc[X['Credit Score'].notnull()]
        self.catb_for_CS = CatBoostRegressor(eval_metric='R2',
                                             n_estimators=400,
                                             max_depth=5,
                                             learning_rate=0.045,
                                             silent=True,
                                             leaf_estimation_iterations=280,
                                             random_state=21)
        self.catb_for_CS.fit(self.X_train_CS, self.y_train_CS)

    def predict(self, X):
        # Model for Credit Score
        self.X_test_CS = X[self.features_CS].loc[X['Credit Score'].isnull()]
        X.loc[X['Credit Score'].isnull(), 'Credit Score'] = self.catb_for_CS.predict(self.X_test_CS)


class FeatureGenerator:
    """Generate new features"""

    def __init__(self):
        self.gp_credit_score = None

    def fit(self, X):
        X = X.copy()
        self.gp_credit_score = X.groupby(['Class Credit Score'])['Credit Score'].mean()
        return X

    def transform(self, X):
        # Class Credit Score
        X['Class Credit Score'] = 0
        X['Class Credit Score'].loc[X['Credit Score'] < 600] = 1
        X['Class Credit Score'].loc[(X['Credit Score'] >= 600) & (X['Credit Score'] < 650)] = 2
        X['Class Credit Score'].loc[(X['Credit Score'] >= 650) & (X['Credit Score'] < 700)] = 3
        X['Class Credit Score'].loc[(X['Credit Score'] >= 700) & (X['Credit Score'] < 725)] = 4
        X['Class Credit Score'].loc[(X['Credit Score'] >= 725) & (X['Credit Score'] < 750)] = 5
        X['Class Credit Score'].loc[X['Credit Score'] >= 750] = 6

        # Monthly Income
        X['Monthly Income'] = X['Annual Income'] / 12

        # Mean Credit Score in Class
        X["Mean Credit Score In Class"] = X['Class Credit Score'].map(self.gp_credit_score.to_dict())

        # Class Current Loan
        X['Class Current Loan'] = 0
        X['Class Current Loan'].loc[X['Current Loan Amount'] < 100000] = 1
        X['Class Current Loan'].loc[(X['Current Loan Amount'] >= 100000) & (X['Current Loan Amount'] < 200000)] = 2
        X['Class Current Loan'].loc[(X['Current Loan Amount'] >= 200000) & (X['Current Loan Amount'] < 300000)] = 3
        X['Class Current Loan'].loc[(X['Current Loan Amount'] >= 400000) & (X['Current Loan Amount'] < 500000)] = 4
        X['Class Current Loan'].loc[(X['Current Loan Amount'] >= 500000) & (X['Current Loan Amount'] < 600000)] = 5
        X['Class Current Loan'].loc[X['Current Loan Amount'] >= 600000] = 6

        # Monthly Income and Debt Diff
        X['Monthly Income and Debt Diff '] = X['Monthly Income'] - X['Monthly Debt']

        # Current Loan and Monthly Income Diff
        X['Current Loan and Monthly Income Diff'] = X['Current Loan Amount'] - X['Monthly Income']

        # Low Credit Score
        X['Low Credit Score'] = (X['Credit Score'] < 675).astype('int64')

        # Big Loan Amount
        X['Big Loan Amount'] = (X['Current Loan Amount'] > 200000).astype('int64')
        return X


DATA_ROOT = Path('./')
MODELS_PATH = Path('./models/')

# input
TRAIN_DATASET_PATH = DATA_ROOT / 'course_project_train.csv'
TEST_DATASET_PATH = DATA_ROOT / 'course_project_test.csv'
# output
PREPARED_TRAIN_DATASET_PATH = DATA_ROOT / 'train_prepared.csv'
PREPARED_TEST_DATASET_PATH = DATA_ROOT / 'test_prepared.csv'

train = pd.read_csv(TRAIN_DATASET_PATH)
test = pd.read_csv(TEST_DATASET_PATH)

# Dataset processing
processing = DataPreprocessing()
processing.fit(train)
processing.transform(train)
processing.transform(test)

# Filling Nan Annual Income
fill_income = AnnualIncomeFilling()
fill_income.fit(train)
fill_income.predict(train)
fill_income.predict(test)

# Filling Nan Credit Score
fill_score = CreditScoreFilling()
fill_score.fit(train)
fill_score.predict(train)
fill_score.predict(test)

# Dataset features generating
feat_gen = FeatureGenerator()
feat_gen.fit(train)
feat_gen.transform(train)
feat_gen.transform(test)

# Save Dataset
save(train, PREPARED_TRAIN_DATASET_PATH)
save(test, PREPARED_TEST_DATASET_PATH)

"""Final model"""

TARGET = 'Credit Default'

BASE_FEATURE_NAMES = ['Annual Income', 'Years in current job', 'Tax Liens', 'Number of Open Accounts',
                      'Years of Credit History', 'Maximum Open Credit', 'Number of Credit Problems',
                      'Bankruptcies', 'Term', 'Current Loan Amount', 'Current Credit Balance', 'Monthly Debt',
                      'Credit Score']

NEW_FEATURE_NAMES = ['Is Credit Score too large', 'Home Ownership_Home Mortgage', 'Home Ownership_Own Home',
                     'Home Ownership_Rent', 'Purpose_business', 'Purpose_medical', 'Purpose_other',
                     'Purpose_personal', 'Nan Annual Income', 'Mean Income By Stage', 'Nan Credit Score',
                     'Class Credit Score', 'Monthly Income', 'Mean Credit Score In Class',
                     'Monthly Income and Debt Diff', 'Low Credit Score', 'Big Loan Amount',
                     'Current Loan and Monthly Income Diff', 'Class Current Loan']

disbalance = train[TARGET].value_counts()[0] / train[TARGET].value_counts()[1]

X_train = train[BASE_FEATURE_NAMES + NEW_FEATURE_NAMES]
y_train = train[TARGET]

X_test = test[BASE_FEATURE_NAMES + NEW_FEATURE_NAMES]

catboost = CatBoostClassifier(learning_rate=0.03,
                          max_depth=4,
                          n_estimators=400,
                          class_weights=[1, disbalance],
                          silent=True,
                          random_state=21)

catboost.fit(X_train, y_train)

y_test = catboost.predict(X_test)

# Threshold

y_test_probs = catboost.predict_proba(X_test)
y_test = np.where(y_test_probs[:, 1] > 0.43, 1, 0)

CreditDefault_pred = pd.DataFrame({'Id': np.arange(0, y_test.shape[0]), 'Credit Default': y_test})
CreditDefault_pred.to_csv('Sokolova_predictions_fin.csv', index=False, encoding='utf-8', sep=',')

# Cross Validation
cv_score = cross_val_score(
    catboost,
    X_train,
    y_train,
    scoring='f1',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=21)
)
print(cv_score.mean())
