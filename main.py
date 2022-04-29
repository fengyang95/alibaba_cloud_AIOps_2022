import pandas as pd
import os
from code.features import TFIDFFeatureExtractor, BaseFeatureExtractor
from code.classifier import Classifer
from code.utils.metrics import macro_f1_val
import copy
import logging

logging.basicConfig(level=logging.INFO)


class WorkFlow:
    def __init__(self, log_df: pd.DataFrame,
                 fault_label_df: pd.DataFrame, fault_venus_df: pd.DataFrame, fault_crash_dump_df: pd.DataFrame,
                 submit_df: pd.DataFrame, submit_venus_df: pd.DataFrame, submit_crash_dump_df: pd.DataFrame):
        self.log_df = log_df
        self.fault_label_df = fault_label_df
        self.fault_venus_df = fault_venus_df
        self.fault_crash_dump_df = fault_crash_dump_df

        self.submit_df = submit_df
        self.submit_venus_df = submit_venus_df
        self.submit_crash_dump_df = submit_crash_dump_df

        self.tfidf_feature_extractor = TFIDFFeatureExtractor()

    def get_samples(self, train=True, tfidf_model_dict=None,
                    doc2vec_model=None):
        if train is True:
            _fault_df = self.fault_label_df.copy(deep=True)
            _venus_df = self.fault_venus_df.copy(deep=True)
            _crashdump_df = self.fault_crash_dump_df.copy(deep=True)

            fault_columns = ['sn', 'fault_time', 'label']
        else:
            _fault_df = self.submit_df.copy(deep=True)
            _venus_df = self.submit_venus_df.copy(deep=True)
            _crashdump_df = self.submit_crash_dump_df.copy(deep=True)
            fault_columns = ['sn', 'fault_time']

        _log_df = copy.deepcopy(self.log_df)
        base_feature_extractor = BaseFeatureExtractor(_fault_df, _log_df, _crashdump_df, _venus_df)

        _num_feature_df, _text_feature_df, _sentences_feature_df, _processed_log_df, _venus_feature_df, _crashdump_feature_df = base_feature_extractor.extract_features()

        _text_columns = fault_columns + [_col for _col in _text_feature_df.columns if
                                         _col.startswith(
                                             base_feature_extractor.log_text_feature_prefix)]

        _text_feature_df_for_tv = _text_feature_df[_text_columns]
        logging.info(f"basic features done!")

        if train is True:
            tfidf_model_dict = self.tfidf_feature_extractor.train_tv_models(_text_feature_df_for_tv, _log_df=None)

            _tfidf_features_df = self.tfidf_feature_extractor.transform(_text_feature_df_for_tv,
                                                                        model_dict=tfidf_model_dict,
                                                                        train=True).reset_index(drop=True)
        else:
            _tfidf_features_df = self.tfidf_feature_extractor.transform(_text_feature_df_for_tv,
                                                                        model_dict=tfidf_model_dict,
                                                                        train=False).reset_index(drop=True)

        text_df_head_columns = fault_columns + ['server_model']

        text_feature_df_overall = _text_feature_df[text_df_head_columns].reset_index(drop=True).merge(
            _tfidf_features_df,
            on=fault_columns,
            how='left').reset_index(drop=True)

        df_all = _num_feature_df.merge(text_feature_df_overall, on=fault_columns,
                                       how='left').reset_index(drop=True)

        df_all = df_all.merge(_venus_feature_df, on=fault_columns,
                              how='left', ).reset_index(drop=True)
        df_all = df_all.merge(_crashdump_feature_df, on=fault_columns,
                              how='left').reset_index(drop=True)

        if _sentences_feature_df is not None:
            tmp_log_df = _log_df.copy(deep=True)
            tmp_log_df['msgs'] = tmp_log_df['msg'].apply(lambda val: base_feature_extractor.process_log(val))

            df_all = df_all.merge(_sentences_feature_df, on=fault_columns,
                                  how='left').reset_index(drop=True)

        df_all['server_model'] = df_all['server_model'].apply(lambda val: int(val.split("M")[1]))

        feature_columns = [_col for _col in df_all.columns if
                           _col.startswith('num_feature')
                           or 'tfidf_feature' in _col
                           # or 'doc2vec_feature' in _col
                           # or 'count_vec_feature' in _col
                           ] + ['server_model']
                         # + [_col for _col in df_all.columns if _col.startswith('venus') or _col.startswith('crashdump')]

        # feature_columns = [_col for _col in df_all.columns if _col.startswith('num_feature')] + ['server_model']
        print(f"features:{feature_columns}")
        res_dict = {
            'tfidf_model_dict': tfidf_model_dict,
            'doc2vec_model': doc2vec_model
        }
        if train is True:
            df_all = df_all[df_all['label'] != -1].reset_index(drop=True)
            res_dict['y'] = df_all['label']

            res_dict['X'] = df_all[feature_columns]
            res_dict['head'] = df_all[['sn', 'fault_time']]
        else:
            res_dict['X'] = df_all[feature_columns]
            res_dict['head'] = df_all[['sn', 'fault_time']]
        logging.info(f"features done!")
        return res_dict

    def fit(self):
        global copy
        data_dict = self.get_samples(train=True)
        X = data_dict['X']
        y = data_dict['y']
        import numpy as np

        def get_noise_samples(_X, _y):
            noise_indices = set()
            _X_values = copy.deepcopy(X.values).astype(np.float)

            print(f"{np.mean(_X_values, axis=0).shape}")

            _X_values = (_X_values - np.mean(_X_values, axis=0)) / (np.std(_X_values, axis=0) + 1e-10)
            print(_X_values.shape)
            N = len(_X_values)
            A = np.zeros((N, N))
            for i in range(N - 1):
                delta = np.mean(np.square(_X_values[i] - _X_values[i + 1:]), axis=1)
                equal_sample_index = np.where(delta < 0.002)[0]
                for j in equal_sample_index:
                    sample_index = j + i + 1
                    if y[i] != y[sample_index]:
                        # print(f"noise label: diff:{delta[j]} i:{i} j:{sample_index}  yi:{y[i]} yj:{y[sample_index]}")
                        noise_indices.add(i)
                        noise_indices.add(sample_index)
            return noise_indices

        # noise_indices = get_noise_samples(X, y)
        # print(f"noise indices:{len(noise_indices)}")
        # from collections import Counter
        # print(F"NOISE:{Counter(y.iloc[list(noise_indices)].values)}")
        noise_indices={}

        valid_indices = [i for i in range(len(X)) if i not in noise_indices]
        X = X.iloc[valid_indices].reset_index(drop=True)
        y = y.iloc[valid_indices].reset_index(drop=True)

        tfidf_model_dict = data_dict['tfidf_model_dict']
        doc2vec_model = data_dict['doc2vec_model']

        # from lightgbm import LGBMClassifier
        #
        # from sklearn.model_selection import KFold
        # from code.metrics import macro_f1_val
        #
        # kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
        # models = []
        # macro_f1_list = []
        # auc_val_list = []
        # auc_train_list = []
        # print(f"len_features:{len(X.columns)}")
        # for fold_index, (train_index, val_index) in enumerate(kfold.split(X)):
        #     curr_cls = BaggingClassifer()
        #     X_train = X.iloc[train_index]
        #     y_train = y[train_index]
        #
        #     X_val = X.iloc[val_index]
        #     y_val = y[val_index]
        #
        #     curr_cls.fit(X_train, y_train,
        #                  cat_features=['server_model'])
        #
        #     # feature_importances = []
        #     # useless_words = []
        #     # for col, score in zip(X.columns, curr_catboost_cls.lgb.feature_importances_):
        #     #     if 'tfidf_feature' in col and score == 0:
        #     #         feature_col, index = col.split('_tfidf_feature_')
        #     #         for word, _index in tfidf_model_dict[feature_col][0].vocabulary_.items():
        #     #             if _index == int(index):
        #     #                 useless_word = word
        #     #                 useless_words.append(useless_word)
        #     #
        #     #     feature_importances.append((col, score))
        #     # feature_importances = sorted(feature_importances, key=lambda val: -val[1])
        #     # print(f"feature_importances:{feature_importances}")
        #     # print(f"useless_words:{useless_words}")
        #
        #     # useful_columns=[col for col,_ in feature_importances[:200]]
        #     # X_train=X_train[useful_columns]
        #     # X_val=X_val[useful_columns]
        #     # curr_catboost_cls.fit(X_train, y_train,
        #     #                       cat_features=['server_model'])
        #
        #     y_val_pred = curr_cls.predict(X_val)
        #
        #     print(f"confusion_metric:{confusion_matrix(y_val, y_val_pred)}")
        #
        #     curr_macro_f1 = macro_f1_val(y_val, y_val_pred)
        #
        #     y_train_pred = curr_cls.predict(X_train, y_train)
        #
        #     print(
        #         f"{fold_index}-Fold train_macro-F1:{macro_f1_val(y_train, y_train_pred)} val_macro-F1:{curr_macro_f1}  ")
        #     # print(f"second stage:auc_train:{auc_train:.4f} auc_val:{auc_val:.4f}")
        #     macro_f1_list.append(curr_macro_f1)
        #     models.append(curr_cls)
        #     # auc_train_list.append(auc_train)
        #     # auc_val_list.append(auc_val)
        #
        # print(f"macro-F1:{np.mean([val[0] for val in macro_f1_list])}")
        # print(f"first-F1:{np.mean([val[1] for val in macro_f1_list])}")
        # print(f"second-F1:{np.mean([val[2] for val in macro_f1_list])}")
        #
        # # print("second stage:")
        # # print(f"auc_train:{np.mean(auc_train_list):.4f}  auc_val:{np.mean(auc_val_list):.4f}")
        # probas = []
        #
        cls = Classifer(k_fold=9, tuna=False)
        cat_features = [col for col in X.columns if 'catfeature' in col]
        cls.fit(X, y, cat_features=['server_model'] + cat_features)
        logging.info(f"train done!")
        train_pred = cls.predict(X)
        logging.info(f"train macro-F1:{macro_f1_val(y, train_pred)}")
        # print(f"second_stage: train auc:{train_auc:.4f}")

        return {
            'tv_model_dict': tfidf_model_dict,
            'cls_model': cls,
            'doc2vec_model': doc2vec_model
        }

    def predict(self, tv_model_dict, cls_model, doc2vec_model):
        data_dict = self.get_samples(train=False, tfidf_model_dict=tv_model_dict,
                                     doc2vec_model=doc2vec_model)

        X = data_dict['X']
        head = data_dict['head']
        y_pred = pd.DataFrame({'label': cls_model.predict(X)})
        return pd.concat([head, y_pred], axis=1)

    def executor(self):
        model_dict = self.fit()
        cls_model = model_dict['cls_model']
        tv_model_dict = model_dict['tv_model_dict']
        doc2vec_model = model_dict['doc2vec_model']
        _result_df = self.predict(tv_model_dict, cls_model, doc2vec_model)
        return _result_df


if __name__ == '__main__':
    data_dir = 'data'

    submit_dir = 'tcdata'

    import time

    begin = time.time()

    log_df = pd.read_csv(os.path.join(data_dir, 'preliminary_sel_log_dataset.csv'))
    log_df2 = pd.read_csv(os.path.join(data_dir, 'preliminary_sel_log_dataset_a.csv'))
    log_df3 = pd.read_csv(os.path.join(submit_dir, 'final_sel_log_dataset_a.csv'))
    log_df = pd.concat((log_df, log_df2, log_df3), axis=0).reset_index(drop=True)

    fault_label_df = pd.read_csv(os.path.join(data_dir, 'preliminary_train_label_dataset.csv'))
    fault_label_df2 = pd.read_csv(os.path.join(data_dir, 'preliminary_train_label_dataset_s.csv'))
    fault_label_df = pd.concat((fault_label_df, fault_label_df2), axis=0).reset_index(drop=True)

    fault_venus_df = pd.read_csv(os.path.join(data_dir, 'preliminary_venus_dataset.csv'))
    fault_crashdump_df = pd.read_csv(os.path.join(data_dir, 'preliminary_crashdump_dataset.csv'))

    submit_df = pd.read_csv(os.path.join(submit_dir, 'final_submit_dataset_a.csv'))
    submit_venus_df = pd.read_csv(os.path.join(submit_dir, 'final_venus_dataset_a.csv'))
    submit_crashdump_df = pd.read_csv(os.path.join(submit_dir, 'final_crashdump_dataset_a.csv'))

    extra_fault_df = submit_df.copy(deep=True)
    extra_fault_df['label'] = -1

    fault_label_df = pd.concat((fault_label_df, extra_fault_df), axis=0).reset_index(drop=True)

    logging.info(f"load data done!")
    work_flow = WorkFlow(log_df, fault_label_df, fault_venus_df, fault_crashdump_df,
                         submit_df, submit_venus_df, submit_crashdump_df)

    result_df = work_flow.executor()
    result_df.to_csv('result.csv', index=False)

    logging.info(f"time:{time.time() - begin:.1f}s")
