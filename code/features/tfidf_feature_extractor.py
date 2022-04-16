from .util import MIN_DF
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import copy


class TFIDFFeatureExtractor:
    def __init__(self):
        self.log_text_feature_prefix = 'log_text'

    def train_tv_models(self, text_df, _log_df=None):
        model_dict = {}
        _text_df = copy.deepcopy(text_df)
        for _col in _text_df.columns:
            if _col.startswith(self.log_text_feature_prefix):
                # print(f"tv_model:{tv_model}")
                from sklearn.feature_extraction.text import CountVectorizer
                _tv_model = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None,
                                            min_df=MIN_DF)
                _count_model = CountVectorizer(min_df=MIN_DF)
                _tf_model = TfidfVectorizer(use_idf=False, norm=None, min_df=MIN_DF)
                if _log_df is not None:
                    print(f"_log_df:{_log_df.columns}")
                    print(_log_df.head(5))
                    for _text_col in _log_df.columns:
                        if _text_col in _col:
                            _tv_model.fit(
                                list(_log_df[_text_col].apply(lambda val: '_'.join([c for c in val.split()]))))
                else:
                    # print(f"tfidf_words:")
                    # print(sorted(list(set(list(_text_df[_col].apply(lambda val: val.lower().strip()).values)))))
                    _tv_model.fit(list(_text_df[_col].apply(lambda val: val.lower().strip()).values))
                    print(sorted(list(_tv_model.vocabulary_.keys())))
                    _tf_model.fit(list(_text_df[_col].apply(lambda val: val.lower().strip()).values))
                    _count_model.fit(list(_text_df[_col].apply(lambda val: val.lower().strip()).values))
                model_dict[_col] = (_tv_model, _tf_model, _count_model)
        return model_dict

    def transform(self, _text_df, model_dict, train=True, vis_topic=False):

        feature_df = None
        for _col in _text_df.columns:
            if _col.startswith(self.log_text_feature_prefix):
                try:
                    tv_model = model_dict[_col][0]
                    tf_model = model_dict[_col][1]
                    count_model = model_dict[_col][2]
                    # print(f"_col:{_col}")
                    # print(f"tv_model:{tv_model.vocabulary_}")
                    sentences = _text_df[_col].apply(lambda val: val.lower()).values
                    tfidf_features = tv_model.transform(sentences).A
                    # print(tfidf_features[:5])
                    tfidf_columns = [f'{_col}_tfidf_feature_{i}' for i in range(tfidf_features.shape[1])]
                    curr_tfidf_feature_df = pd.DataFrame(tfidf_features, columns=tfidf_columns)

                    tf_features = tf_model.transform(sentences).A
                    tf_columns = [f'{_col}_tf_feature_{i}' for i in range(tf_features.shape[1])]
                    curr_tf_feature_df = pd.DataFrame(tf_features, columns=tf_columns)

                    count_features = count_model.transform(sentences).A
                    count_vec_columns = [f'{_col}_count_vec_feature_{i}' for i in range(count_features.shape[1])]
                    curr_count_feature_df = pd.DataFrame(count_features, columns=count_vec_columns)

                    if feature_df is None:
                        feature_df = pd.concat([curr_tfidf_feature_df, curr_tf_feature_df, curr_count_feature_df],
                                               axis=1)
                    else:
                        feature_df = pd.concat(
                            (feature_df, curr_tfidf_feature_df, curr_tf_feature_df, curr_count_feature_df), axis=1)

                except Exception as e:
                    print(f"{e}")

        fault_columns = ['sn', 'fault_time']
        if train is True:
            fault_columns += ['label']
        return pd.concat((_text_df[fault_columns], feature_df), axis=1)
