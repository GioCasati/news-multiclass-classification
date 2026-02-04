from time import time
import random
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from utils import global_cleaner
from utils import deduplicate
from utils import timestamp_encoding
from utils import process_text_columns


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def main(make_csv=False):
    t_0 = time()
    print('Elaborating...')

    # --- best parameters ---
    cleaning_parameters = {
        'extract_html_metadata': True,
        'extract_url_text': True,
        'html_strip': True,
        'remove_urls': True,
        'remove_emails': True,
        'convert_currency': True,
        'normalize_unicode': True,
        'remove_non_alphanum': True,
        'remove_numbers': True,
        'lower': True,
        'remove_short': True
    }
    text_processor_parameters = {
        'text_length': True,
        'title_length': True,
        'article_length': True,
        'digit_density': True
    }
    timestamp_encoding_parameters = {
        'date_present': True,
        'year': True,
        'month': True,
        'day': True,
        'hour': True,
        'weekday': True
    }
    # -----------------------

    ## INITIALIZATION
    # 0. Load datasets
    df_train = pd.read_csv('../dat/development.csv', index_col='Id')
    df_eval = pd.read_csv('../data/evaluation.csv', index_col='Id')

    # 1. Text cleaning
    df_train = global_cleaner(df_train, cleaning_params=cleaning_parameters)
    df_eval = global_cleaner(df_eval, cleaning_params=cleaning_parameters)

    # 2. Duplicates removal
    duplicate_keys = ['source', 'title', 'article']
    df_train = deduplicate(df_train, duplicate_keys, 'label')


    ## FULL PIPELINE

    # 1. Preprocessor
    # 1.1 source
    source_transformer = Pipeline(steps=[
        ('source_imputer', SimpleImputer(strategy='constant',
                                         fill_value='Unknown_source')),
        ('source_onehot', OneHotEncoder(sparse_output=True,
                                        min_frequency=4,
                                        handle_unknown='infrequent_if_exist',
                                        dtype='int8'))
    ])

    # 1.2 title & article
    text_transformer = ColumnTransformer(transformers=[
        ('text_tfidf', TfidfVectorizer(stop_words='english',
                                       max_features=50000,
                                       ngram_range=(1, 2),
                                       min_df=5,
                                       max_df=0.9,
                                       sublinear_tf=True), 'full_text'),
        ('text_meta_extractor', Pipeline(steps=[
            ('meta_imputer', SimpleImputer(strategy='median')),
            ('meta_scaler', MinMaxScaler())
        ]),
        [key for key in list(text_processor_parameters.keys()) if text_processor_parameters[key] is True])
    ])
    title_article_transformer = Pipeline(steps=[
        ('title_article_combiner', FunctionTransformer(process_text_columns,
                                                       kw_args=text_processor_parameters,
                                                       validate=False)),
        ('text_feature_extractor', text_transformer)
    ])

    # 1.3 page_rank
    page_rank_transformer = Pipeline(steps=[
        ('pagerank_imputer', SimpleImputer(strategy='most_frequent')),
        ('pagerank_scaler', MinMaxScaler())
    ])

    # 1.4 timestamp
    timestamp_transformer = Pipeline(steps=[
        ('timestamp_encoder', FunctionTransformer(timestamp_encoding,
                                                  kw_args=timestamp_encoding_parameters,
                                                  validate=False)),
        ('timestamp_imputer', SimpleImputer(strategy='median')),
        ('timestamp_scaler', MinMaxScaler())
    ])

    # 1.5 Complete Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('source_preproc', source_transformer, ['source']),
        ('text_preproc', title_article_transformer, ['title', 'article']),
        ('pagerank_preproc', page_rank_transformer, ['page_rank']),
        ('timestamp_preproc', timestamp_transformer, ['timestamp'])
    ],
    remainder='drop')

    # 2. Classifier
    # 2.1 Logistic Regression
    lr = LogisticRegression(
        C=1.6,
        class_weight='balanced',
        solver='saga',
        max_iter=2000,
        random_state=SEED
    )

    # 2.2 Linear Support Vector Classifier
    lsvc = LinearSVC(
        class_weight='balanced',
        dual=False,
        C=0.1,
        random_state=SEED
    )

    # 2.3 Multinomial Naive Bayes
    mnb = MultinomialNB(
        alpha=0.5
    )

    # 2.4 Voting Classifier
    clf = VotingClassifier(
        estimators=[
            ('SVM', lsvc),
            ('LR', lr),
            ('NB', mnb)
        ],
        voting='hard'
    )

    # 3. Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    ## FITTING AND PREDICTION
    # 1. Fitting
    model_pipeline.fit(df_train, df_train['label'])

    # 2. Prediction
    y_pred = model_pipeline.predict(df_eval)

    # 3. Submission
    if make_csv:
        submission = pd.DataFrame({
            'Id': list(df_eval.index),
            'Predicted': y_pred
        })
        submission.to_csv('../results/submission.csv', index=False)

    print('Completed.')
    print(f'Time elapsed: {time() - t_0}s')

if __name__ == '__main__':
    main(True)