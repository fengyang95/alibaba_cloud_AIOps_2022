import pandas as pd
import copy
import random
from collections import Counter
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import LGBMClassifier
from code.utils.metrics import macro_f1_val
from sklearn.model_selection import KFold


# 第二阶段改成AUC验证
class HierarchicalClassifier:
    def __init__(self):
        self.first_cls = LGBMClassifier(n_estimators=1000, num_leaves=63, colsample_bytree=0.5,
                                        subsample=0.5, min_child_samples=50)
        self.second_cls = LGBMClassifier(n_estimators=200, num_leaves=15, colsample_bytree=0.5,
                                         subsample=0.5, min_child_samples=50)
        # self.second_cls=RandomForestClassifier()
        # self.second_cloumns=['num_feature_min_log_time_diff_seconds', 'num_feature_logs_count_mean_86400', 'num_feature_cos_time', 'num_feature_sin_time', 'server_model', 'num_feature_std_num_logs_per_interval', 'num_feature_logs_count_std_86400', 'num_feature_mean_num_logs_per_interval', 'num_feature_logs_count_skew_86400', 'num_feature_logs_count_kurtosis_86400', 'num_feature_num_std_each_log_86400', 'num_feature_num_mean_each_log_86400', 'num_feature_log_time_intervals_count', 'num_feature_logs_count_count_86400', 'num_feature_num_unique_logs_86400', 'num_feature_num_max_each_log_86400', 'num_feature_sin_day', 'num_feature_cos_day', 'log_text_fault_phenomenon_86400_tfidf_feature_6', 'log_text_fault_phenomenon_86400_tfidf_feature_22', 'num_feature_is_work_day', 'log_text_fault_phenomenon_86400_tfidf_feature_23', 'log_text_component_86400_tfidf_feature_0', 'num_feature_is_day_time', 'log_text_fault_phenomenon_86400_tfidf_feature_38', 'log_text_component_86400_tfidf_feature_21', 'log_text_fault_phenomenon_86400_tfidf_feature_13', 'log_text_fault_phenomenon_86400_tf_feature_38', 'log_text_component_86400_tfidf_feature_9', 'log_text_fault_phenomenon_86400_tfidf_feature_34', 'log_text_fault_phenomenon_86400_tfidf_feature_41', 'log_text_fault_phenomenon_86400_tfidf_feature_39', 'log_text_fault_phenomenon_86400_tfidf_feature_51', 'log_text_fault_phenomenon_86400_tf_feature_6', 'log_text_fault_phenomenon_86400_tf_feature_22', 'log_text_component_86400_tfidf_feature_16', 'log_text_fault_phenomenon_86400_tfidf_feature_36', 'log_text_fault_phenomenon_86400_tfidf_feature_45', 'log_text_fault_phenomenon_86400_tf_feature_51', 'log_text_component_86400_tfidf_feature_8', 'log_text_component_86400_tfidf_feature_13', 'log_text_fault_phenomenon_86400_tfidf_feature_5', 'log_text_fault_phenomenon_86400_tfidf_feature_19', 'log_text_fault_phenomenon_86400_tf_feature_23', 'num_feature_num_min_each_log_86400', 'log_text_component_86400_tfidf_feature_7', 'log_text_component_86400_tfidf_feature_11', 'log_text_component_86400_tfidf_feature_17', 'log_text_component_86400_tfidf_feature_20', 'log_text_fault_phenomenon_86400_tfidf_feature_1']

    def fit(self, X, y, cat_features=None, aug=True, n_aug=3,
            aug_prob=0.5):

        if aug is True:
            aug_X = pd.concat([copy.deepcopy(X) for _ in range(n_aug)], axis=0)
            aug_y = pd.concat([copy.deepcopy(y) for _ in range(n_aug)], axis=0)
            for col in aug_X.columns:
                if cat_features is not None and col in cat_features:
                    continue
                if len(np.unique(aug_X[col].values)) < 3:
                    continue
                if 'sin' in col or 'cos' in col:
                    continue
                if random.random() > (1 - aug_prob):
                    diff = aug_X[col].values.max() - aug_X[col].values.min()
                    noise = diff * 0.002 * np.random.random((len(aug_X),))
                    aug_X[col] = aug_X[col].values + noise

            X = pd.concat([X, aug_X], axis=0).reset_index(drop=True)
            y = pd.concat([y, aug_y], axis=0).reset_index(drop=True)

        sample_weight = copy.deepcopy(y)
        sample_weight[y == 0] = 7
        sample_weight[y == 1] = 3
        sample_weight[y == 2] = 1
        sample_weight[y == 3] = 4

        first_level_y = copy.deepcopy(y)
        first_level_y[first_level_y == 1] = 0
        first_level_y[first_level_y == 2] = 1
        first_level_y[first_level_y == 3] = 2

        self.first_cls.fit(X, first_level_y, categorical_feature=cat_features, sample_weight=sample_weight)

        X_second_level = X[(y == 0) | (y == 1)].reset_index(drop=True)
        y_second_level = y[(y == 0) | (y == 1)]
        second_level_sample_weight = copy.deepcopy(y_second_level)
        second_level_sample_weight[y_second_level == 0] = 7
        second_level_sample_weight[y_second_level == 1] = 3

        # self.second_cls=SelfPacedEnsemble(base_estimator=LGBMClassifier(boosting_type='gbdt', num_leaves=31, n_estimators=100,
        #                                                       learning_rate=0.1,subsample=0.5,
        #                                                       colsample_bytree=0.5,min_data_per_group=50,
        #                                                       #max_cat_threshold=32,
        #                                                       ),
        #                         feature_columns=X.columns,
        #                         random_state=random.seed(2024),
        #                         k_bins=20, n_estimators=20,
        #                         hardness_func=lambda y_pred,y_label:(y_pred-y_label)**2,
        #
        #                         )

        self.second_cls.fit(X_second_level, pd.Series(y_second_level).astype(int), categorical_feature=cat_features,
                            sample_weight=second_level_sample_weight
                            )

        first_importances = sorted([(col, val) for col, val in zip(X.columns, self.first_cls.feature_importances_)],
                                   key=lambda val: -val[1])
        # print(F"first feature importances:{first_importances}")
        second_importances = sorted([(col, val) for col, val in zip(X.columns, self.second_cls.feature_importances_)],
                                    key=lambda val: -val[1])
        # print(F"second feature importances:{second_importances}")
        # useful_features=[col for col,val in second_importances[:50]]
        # print(f"{useful_features}")

    def predict(self, X, y_true=None):
        first_level_y_pred = self.first_cls.predict(X)
        second_level_y_pred = self.second_cls.predict(X)
        y = copy.deepcopy(second_level_y_pred)
        y[first_level_y_pred == 1] = 2
        y[first_level_y_pred == 2] = 3

        auc = None
        if y_true is not None:
            y_true_second = y_true[(y_true == 0) | (y_true == 1)]
            pred_proba = self.second_cls.predict_proba(X)
            pred_proba = pred_proba[(y_true == 0) | (y_true == 1)][:, 1]
            auc = roc_auc_score(y_true_second, pred_proba)
        return y.astype(int), auc


class Classifer:

    @staticmethod
    def aug(X, y, n_aug=3, cat_features=None, aug_prob=0.5):

        aug_X = pd.concat([copy.deepcopy(X) for _ in range(n_aug)], axis=0)
        aug_y = pd.concat([copy.deepcopy(y) for _ in range(n_aug)], axis=0)
        for col in aug_X.columns:
            if cat_features is not None and col in cat_features:
                continue
            if len(np.unique(aug_X[col].values)) < 3:
                continue
            if 'sin' in col or 'cos' in col:
                continue
            if random.random() > (1 - aug_prob):
                diff = aug_X[col].values.max() - aug_X[col].values.min()
                noise = diff * 0.005 * np.random.random((len(aug_X),))
                aug_X[col] = aug_X[col].values + noise

        new_X = pd.concat([X, aug_X], axis=0).reset_index(drop=True)
        new_y = pd.concat([y, aug_y], axis=0).reset_index(drop=True)

        return new_X, new_y

    class BaggingClassifier:
        def __init__(self):
            pass

        def fit(self, X, y, categorical_feature=None, class_weight=None):
            lgb_param = {
                "n_estimators": 200,
                "num_leaves": 31,
                "colsample_bytree": 0.5,
                "subsample": 0.8,
                "subsample_freq": 2,
                "min_child_samples": 20,
                "reg_alpha": 0.2,
                "reg_lambda": 8,
            }
            self.lgb_cls = LGBMClassifier()

    class LGBMClassifierKFold:
        def __init__(self, kfold=5):
            self.k_fold = kfold

        def fit(self, X, y, categorical_feature=None, class_weight=None):

            lgb_param1 = {'n_estimators': 187, 'reg_alpha': 8.91246375225518,
                          'reg_lambda': 2.7967663580318955, 'num_leaves': 37,
                          'colsample_bytree': 0.6050376610030631, 'subsample': 0.6043461631199231,
                          'subsample_freq': 3, 'min_child_samples': 81}
            lgb_param2 = {'n_estimators': 121, 'reg_alpha': 0.23214069098774093,
                          'reg_lambda': 0.6643772879850578, 'num_leaves': 44,
                          'colsample_bytree': 0.7079450590587428, 'subsample': 0.5131677103090537,
                          'subsample_freq': 1, 'min_child_samples': 28}

            lgb_param3 = {'n_estimators': 162, 'reg_alpha': 3.018333689844611,
                          'reg_lambda': 0.33920268654963737, 'num_leaves': 20,
                          'colsample_bytree': 0.5129610441927995, 'subsample': 0.9468283604918678,
                          'subsample_freq': 4, 'min_child_samples': 100}

            lgb_param4 = {'n_estimators': 122, 'reg_alpha': 1.5248965455976116,
                          'reg_lambda': 3.191610084922332, 'num_leaves': 34,
                          'colsample_bytree': 0.432366681321017, 'subsample': 0.8836395235516155,
                          'subsample_freq': 3, 'min_child_samples': 37}
            lgb_param5 = {'n_estimators': 113, 'reg_alpha': 0.24100077205018797,
                          'reg_lambda': 1.0685052865691247, 'num_leaves': 44,
                          'colsample_bytree': 0.43475105037752615, 'subsample': 0.8563684980698903,
                          'subsample_freq': 6, 'min_child_samples': 43}

            self.lgb_list1 = [LGBMClassifier(class_weight=class_weight, **lgb_param1) for _ in range(self.k_fold + 1)]
            # self.lgb_list2 = [CatBoostClassifier(class_weights=class_weight, verbose=False, n_estimators=1000) for
            # _ in range(self.k_fold + 1)]
            self.lgb_list2 = [LGBMClassifier(class_weight=class_weight, **lgb_param2) for _ in range(self.k_fold + 1)]
            self.lgb_list3 = [LGBMClassifier(class_weight=class_weight, **lgb_param3) for _ in range(self.k_fold + 1)]
            self.lgb_list4 = [LGBMClassifier(class_weight=class_weight, **lgb_param4) for _ in range(self.k_fold + 1)]
            self.lgb_list5 = [LGBMClassifier(class_weight=class_weight, **lgb_param5) for _ in range(self.k_fold + 1)]
            from catboost import CatBoostClassifier
            self.lgb_list6 = [CatBoostClassifier(class_weights=class_weight,verbose=False) for _ in range(self.k_fold+1)]

            print(f"len_features:{len(X.columns)}")
            kfold = KFold(n_splits=self.k_fold, shuffle=True, random_state=2022)
            f1_list = []
            for fold_index, (train_index, val_index) in enumerate(kfold.split(X)):
                X_train = X.iloc[train_index]
                y_train = y[train_index]
                X_val = X.iloc[val_index]
                y_val = y[val_index]

                X_train_aug, y_train_aug = Classifer.aug(X_train, y_train, n_aug=3,
                                                         cat_features=categorical_feature, aug_prob=0.5)
                # X_train_aug, y_train_aug = X_train, y_train
                self.lgb_list1[fold_index].fit(X_train_aug, y_train_aug, categorical_feature=categorical_feature)
                self.lgb_list2[fold_index].fit(X_train_aug, y_train_aug, categorical_feature=categorical_feature)
                self.lgb_list3[fold_index].fit(X_train_aug, y_train_aug, categorical_feature=categorical_feature)
                self.lgb_list4[fold_index].fit(X_train_aug, y_train_aug, categorical_feature=categorical_feature)
                self.lgb_list5[fold_index].fit(X_train_aug, y_train_aug, categorical_feature=categorical_feature)
                self.lgb_list6[fold_index].fit(X_train_aug, y_train_aug, cat_features=categorical_feature)
                feature_importances = sorted(
                    [(col, val) for col, val in zip(X.columns, self.lgb_list1[fold_index].feature_importances_)],
                    key=lambda val: -val[1])
                print(f"feature_importances:{feature_importances}")
                preds = np.argmax(
                    self.lgb_list1[fold_index].predict_proba(X_val) + self.lgb_list2[fold_index].predict_proba(X_val) +
                    self.lgb_list3[fold_index].predict_proba(X_val) +
                    self.lgb_list4[fold_index].predict_proba(X_val) + self.lgb_list5[fold_index].predict_proba(X_val) +
                    self.lgb_list6[fold_index].predict_proba(X_val),
                    axis=1)
                pred_labels = np.rint(preds)

                preds_train = np.argmax(
                    self.lgb_list1[fold_index].predict_proba(X_train) +
                    self.lgb_list2[fold_index].predict_proba(X_train) +
                    self.lgb_list3[fold_index].predict_proba(X_train) +
                    self.lgb_list4[fold_index].predict_proba(X_train) +
                    self.lgb_list5[fold_index].predict_proba(X_train) +
                    self.lgb_list6[fold_index].predict_proba(X_train),
                    axis=1)
                f1_train = macro_f1_val(y_train, np.rint(preds_train))[0]

                f1 = macro_f1_val(y_val, pred_labels)[0]
                print(f"fold_index:{fold_index} val_F1:{f1} train_F1:{f1_train}")
                f1_list.append(f1)
            print(f"overall F1:{np.mean(f1_list)}")

        def predict(self, X):
            y = None
            for i in range(self.k_fold):
                # 融合的时候取概率最大值
                # lgb_1_proba = self.lgb_list1[i].predict_proba(X)
                # lgb_2_proba = self.lgb_list2[i].predict_proba(X)
                # lgb_3_proba = self.lgb_list3[i].predict_proba(X)
                # lgb_proba = np.concatenate((lgb_1_proba[:, :, np.newaxis], lgb_2_proba[:, :, np.newaxis],
                #                             lgb_3_proba[:, :, np.newaxis]), axis=2)
                # indices = np.argmax(lgb_proba, axis=2)
                # curr_y = np.argmax(lgb_proba[:, :, indices], axis=1)

                curr_y = np.argmax(self.lgb_list1[i].predict_proba(X)
                                   + self.lgb_list2[i].predict_proba(X) +
                                   self.lgb_list3[i].predict_proba(X) +
                                   self.lgb_list4[i].predict_proba(X) +
                                   self.lgb_list5[i].predict_proba(X) +
                                   self.lgb_list6[i].predict_proba(X)
                                   , axis=1)
                if y is None:
                    y = curr_y[:, np.newaxis]
                else:
                    y = np.concatenate((y, curr_y[:, np.newaxis]), axis=1)
            ans = []
            for i in range(len(y)):
                ans.append(int(np.argmax(np.bincount(y[i]))))
            return np.array(ans)

        def predict_proba(self, X):
            raise NotImplementedError

    class LGBMClassifierTuna:
        def __init__(self, kfold=5):
            self.kfold = kfold

        def fit(self, X, y, categorical_feature=None, class_weight=None):
            import numpy as np
            import optuna
            print(f"len_features:{len(X.columns)}")
            import lightgbm as lgb
            import sklearn.datasets
            import sklearn.metrics
            from sklearn.model_selection import train_test_split

            # FYI: Objective functions can take additional arguments
            # (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
            def objective(trial):
                # data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)

                kfold = KFold(n_splits=self.kfold, shuffle=True, random_state=2022)

                param = {
                    # "objective": "binary",
                    # "metric": "binary_logloss",
                    # "class_weight": class_weight,
                    "n_estimators": trial.suggest_int('n_estimators', 100, 500),
                    "verbosity": -1,
                    "boosting_type": "gbdt",
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                    "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
                    "min_child_samples": trial.suggest_int("min_child_samples", 20, 100)
                }

                cls = LGBMClassifier(class_weight=class_weight, **param)

                f_scores = []
                for fold_index, (train_index, val_index) in enumerate(kfold.split(X)):
                    X_train = X.iloc[train_index]
                    y_train = y[train_index]

                    X_val = X.iloc[val_index]
                    y_val = y[val_index]

                    X_train_aug, y_train_aug = Classifer.aug(X_train, y_train, n_aug=3,
                                                             cat_features=categorical_feature, aug_prob=0.5)
                    # X_train_aug, y_train_aug = X_train, y_train
                    # self.lgb_list[fold_index].fit(X_train_aug, y_train_aug, categorical_feature=categorical_feature)

                    cls.fit(X_train_aug, y_train_aug, categorical_feature=categorical_feature)
                    preds = cls.predict(X_val)
                    pred_labels = np.rint(preds)
                    f1 = macro_f1_val(y_val, pred_labels)[0]
                    f_scores.append(f1)

                return np.mean(np.array(f_scores))  # +np.min(np.array(f_scores))

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=100)

            print("Number of finished trials: {}".format(len(study.trials)))

            print("Best trial:")
            trial = study.best_trial
            print("  Value: {}".format(trial.value))
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

            self.lgb_list = [lgb.LGBMClassifier(class_weight=class_weight, **trial.params) for _ in range(self.kfold)]
            kfold = KFold(n_splits=self.kfold, shuffle=True, random_state=2022)
            for fold_index, (train_index, val_index) in enumerate(kfold.split(X)):
                X_train = X.iloc[train_index]
                y_train = y[train_index]
                X_val = X.iloc[val_index]
                y_val = y[val_index]
                X_train_aug, y_train_aug = Classifer.aug(X_train, y_train, n_aug=3,
                                                         cat_features=categorical_feature, aug_prob=0.5)
                # X_train_aug, y_train_aug = X_train, y_train
                # self.lgb_list[fold_index].fit(X_train_aug, y_train_aug, categorical_feature=categorical_feature)
                self.lgb_list[fold_index].fit(X_train_aug, y_train_aug, categorical_feature=categorical_feature)

        def predict(self, X):
            y = None
            for i in range(self.kfold):
                curr_y = self.lgb_list[i].predict(X)
                if y is None:
                    y = curr_y[:, np.newaxis]
                else:
                    y = np.concatenate((y, curr_y[:, np.newaxis]), axis=1)
            ans = []
            for i in range(len(y)):
                ans.append(int(np.argmax(np.bincount(y[i]))))
            # ans = self.lgb.predict(X)
            return np.array(ans)

        def predict_proba(self, X):
            raise NotImplementedError

    def __init__(self, tuna=True, k_fold=5):
        self.tuna = tuna
        self.k_fold = k_fold

        self.lgbkfold = self.LGBMClassifierKFold(kfold=k_fold)
        self.lgb_tuna = self.LGBMClassifierTuna(kfold=k_fold)

    def fit(self, X, y, cat_features=None, aug=True, n_aug=3,
            aug_prob=0.5):
        # 数据增强
        print(f"0:{len(y[y == 0])} 1:{len(y[y == 1])} 2:{len(y[y == 2])} 3:{len(y[y == 3])}")
        n_samples = max(Counter(list(y)).values())
        class_weight = {0: 7, 1: 3, 2: 1, 3: 4}
        # self.xgboost.fit(new_X, y, sample_weight=sample_weight)
        # self.lgb.fit(X, y, categorical_feature=cat_features)
        if self.tuna is True:
            self.lgb_tuna.fit(X, y, categorical_feature=cat_features, class_weight=class_weight)
        else:
            self.lgbkfold.fit(X, y, categorical_feature=cat_features, class_weight=class_weight)
        #              eval_class_weight={0: 3, 1: 2, 2: 1, 3: 1})
        # # self.ngboost.fit(X,y,sample_weight=sample_weight)

    def predict_proba(self, X, cat_features=None):
        raise NotImplementedError

    def predict(self, X, cat_features=None):

        if self.tuna is True:
            return self.lgb_tuna.predict(X)
        else:
            return self.lgbkfold.predict(X)
