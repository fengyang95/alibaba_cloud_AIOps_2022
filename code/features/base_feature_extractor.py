import pandas as pd
from scipy.stats import skew, kurtosis
from collections import Counter
import re
from chinese_calendar import is_workday, is_in_lieu
import numpy as np

import math
from datetime import datetime

from .util import SPACE_STR, MIN_DF, TIME_FORMAT, STOP_WORDS


def module_to_cat_int(module):
    counter = {'module1': 83, 'module1,module7': 56, 'module1,module8': 41, 'module4': 31,
               'module9': 29, 'module7': 25, 'module0': 21, 'module10,module11,module3,module4,module9': 16,
               'module8': 14, 'module3': 14, 'module10': 13, 'module11': 13, 'module3,module4': 9,
               'module3,module7': 8, 'module2': 7, 'port a,module1,module7': 3,
               'module10,module11,module9': 3, 'module0,module8': 2, 'port a,module1': 2,
               'port a,module1,module8': 2, 'module10,module4': 2, 'module11,module4,module9': 2,
               'module19': 2, 'module4,module9': 2, 'port a,module11': 1, 'port a,module8': 1,
               'port c,module1,module8': 1, 'module10,module9': 1,
               'module10,module11,module14,module3,module4,module9': 1,
               'module11,module4': 1, 'module10,module11,module13,module3,module4,module9': 1,
               'module10,module11,module7,module9': 1,
               'in traffic control,irpp0,pcie rootport 0:0.0,module10,module12,module19,module5,module9': 1,
               'module12,module19,module5': 1, 'module12,module19,module5,module9': 1,
               'irpp0,pcie rootport 0:0.0,module4': 1, 'module5,module9': 1,
               'module0,module1': 1, 'module10,module3,module4': 1,
               'module10,module7': 1, 'module11,module3,module4,module9': 1,
               'module3,module9': 1, 'pcie rootport 2b:3.0,port a,module8': 1,
               'module17,module19,module3,module4': 1,
               'module10,module11,module12,module13,module14,module3,module4,module9': 1,
               'module10,module11,module12,module14,module3,module4,module9': 1, 'module11,module3,module4': 1,
               'module10,module11,module3,module4,module8,module9': 1,
               'module10,module11,module19,module3,module4,module5,module9': 1, 'module19,module3,module4': 1,
               'irpp1,pcie rootport a2:0.0,module10,module11,module12,module14,module19,module3,module4,module5,module7,module9': 1,
               'module10,module11': 1, 'module17': 1, 'module3,module7,module9': 1, 'module17,module18,module19': 1,
               'module13,module14': 1, 'module3,module4,module9': 1}
    if module in counter:
        return counter[module]
    else:
        return 0


def fault_code_to_cat_int(fault_code_str, i):
    if i == 0:
        if fault_code_str == 'cpu0':
            return 0
        elif fault_code_str == 'cpu1':
            return 1
        else:
            return 2
    elif i == 1:
        if fault_code_str.startswith('core'):
            return 0
        elif fault_code_str.startswith('cha'):
            return 1
        elif fault_code_str.startswith('m'):
            return 2
        elif fault_code_str == 'NA':
            return 3
        else:
            return 4
    elif i == 2:
        if fault_code_str == 'cod2':
            return 0
        elif fault_code_str == 'cod1':
            return 1
        else:
            return 2
    else:
        if fault_code_str == '0xc':
            return 0
        elif fault_code_str == '0x1':
            return 1
        elif fault_code_str == 'NA':
            return 2
        else:
            return 3


class BaseFeatureExtractor:

    def __init__(self, fault_df: pd.DataFrame, log_df: pd.DataFrame, crashdump_df: pd.DataFrame,
                 venus_df: pd.DataFrame):
        """

        :param fault_df: [sn,fault_time,Optional[label]]
        :param log_df: [sn,time,msg,server_model]
        """
        self.fault_df = fault_df
        self.log_df = log_df
        if 'label' in self.fault_df.columns:
            self.train = True
        else:
            self.train = False

        self.crashdump_df = crashdump_df
        self.venus_df = venus_df

        self.venus_feature_prefix = 'venus'
        self.crashdump_feature_prefix = 'crashdump'

        self.num_feature_prefix = 'num_feature'
        self.log_feature_prefix = 'log_feature'
        self.log_count_feature_prefix = 'logs_count'
        self.log_text_feature_prefix = 'log_text'
        self.log_sentences_feature_preifx = 'log_sentences'
        self.fault_df_columns = list(self.fault_df.columns)

        self.log_cols = ['component', 'fault_phenomenon', 'asserted_type', 'extend_description']

    @staticmethod
    def process_log(log: str):
        if log is None:
            return 'none'
        log = log.lower()

        log = log.replace('/', SPACE_STR)

        log = log.replace('_', SPACE_STR)
        log = re.sub(u"([^\u4e00-\u9fa5\u0041-\u005a\u0061-\u007a ])", " ", log)

        log = log.replace('memory', 'mem')
        log = log.replace('chnge', 'change')
        log = log.replace('chck', 'check')
        log = log.replace('redundancy', 'redundant')
        log = log.replace('machne', 'machine')
        log = log.replace('firmwares', 'firmware')
        log = log.replace('status', 'stat').replace('state', 'stat')
        log = log.replace('presence', 'present')
        log = log.replace('limiting', 'limit')
        log = log.replace('subsystem', 'subsys')
        log = log.replace('buttonmm', 'button')

        for alpha in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']:
            log = log.replace(f'dimm{alpha}', 'dimm')
            # log = log.replace(f'cpu{alpha}', 'cpu')
            log = log.replace(f"ch{alpha}", 'ch')

        log = log.replace('statustota', 'status')

        words = log.split()
        new_words = []
        for word in words:
            # origin_word = copy.deepcopy(word)
            # word = PS.stem(word)
            # if word != origin_word:
            #     print(f"ori:{origin_word}  after:{word}")
            if word == 'dim':
                word = 'dimm'
            if len(word) < 3 and word not in {'ps', 'os', 'fw',
                                              'cpu', 'mem', 'fan', 'oem',
                                              'ch', 'set', 'nmi', 'bay', 'bus',
                                              'hdd', 'hot', 'nic', 'sel',
                                              'low', 'off', 'fru',
                                              'met', 'ecc', 'rpm',
                                              'non', }:
                continue

            new_words.append(word)
        log = SPACE_STR.join(
            [word for word in new_words
             if word not in STOP_WORDS])
        return log.strip()

    @staticmethod
    def get_len_unique_logs(_logs):
        _logs = [BaseFeatureExtractor.process_log(log) for log in _logs]
        _new_logs = []
        for _log in _logs:
            _new_logs.append(
                ' '.join([c for c in _log.split() if len(c) > 2 or (len(c) == 2 and c not in {'ps', 'os', 'fw'})]))
        return len(set(_new_logs))

    @staticmethod
    def counter_logs(_logs):
        _logs = [BaseFeatureExtractor.process_log(log) for log in _logs]
        _new_logs = []
        for _log in _logs:
            _new_logs.append(
                ' '.join([c for c in _log.split() if len(c) > 2 or (len(c) == 2 and c not in {'ps', 'os', 'fw'})]))
        return Counter(_new_logs)

    @staticmethod
    def counter_mean_num_each_log(_logs):
        counter = BaseFeatureExtractor.counter_logs(_logs)
        if len(counter) > 0:
            return np.mean(list(counter.values()))
        return 0.

    @staticmethod
    def counter_std_num_each_log(_logs):
        counter = BaseFeatureExtractor.counter_logs(_logs)
        if len(counter) > 0:
            return np.std(list(counter.values()))
        return 0.

    @staticmethod
    def counter_min_num_each_log(_logs):

        counter = BaseFeatureExtractor.counter_logs(_logs)
        if len(counter) > 0:
            return np.min(list(counter.values()))
        return 0.

    @staticmethod
    def counter_max_num_each_log(_logs):

        counter = BaseFeatureExtractor.counter_logs(_logs)
        if len(counter) > 0:
            return np.max(list(counter.values()))
        return 0.

    @staticmethod
    def is_daytime(val):
        hours = int(val[11:13])
        if 9 <= hours <= 18:
            return True
        else:
            return False

    def cos_time(self, val):
        hours = int(val[11:13])
        minutes = int(val[14:16])
        seconds = int(val[17:19])
        _val = seconds + minutes * 60 + hours * 3600
        period = 3600 * 24 / (2 * np.pi)
        cos = math.cos(_val / period)
        return cos

    def sin_time(self, val):
        hours = int(val[11:13])
        minutes = int(val[14:16])
        seconds = int(val[17:19])
        _val = seconds + minutes * 60 + hours * 3600
        period = 3600 * 24 / (2 * np.pi)
        sin = math.sin(_val / period)
        return sin

    def sin_day(self, val):
        day = datetime.strptime(val, TIME_FORMAT).weekday() + 1
        period = 7 / (2 * np.pi)
        return math.sin(day / period)

    def cos_day(self, val):
        day = datetime.strptime(val, TIME_FORMAT).weekday() + 1
        period = 7 / (2 * np.pi)
        return math.cos(day / period)

    @staticmethod
    def extract_words_from_logs(_logs, remove_stop_words=False,
                                single_word=False,
                                most_freq=False,
                                least_freq=False):
        # 前提是按照时间降序排列
        last_time = 0
        word_set = set()
        log_list = []
        # used_logs=set()
        for _log in _logs[::-1]:
            time = int(_log.split()[0])
            # print(f"time:{time}")
            _log = ' '.join(
                [c for c in _log.split()[1:] if (len(c) > 2 or (len(c) == 2 and c not in {'ps', 'os', 'fw'}))])
            log_list.append(_log)
            if last_time == 0:
                interval = 0
            else:
                interval = time - last_time
            last_time = time

            # print(f"interval:{interval}")
            if interval <= 3600 * 24:
                # # 重复的log不需要
                # if _log not in used_logs:
                #     used_logs.add(_log)
                # else:
                #     continue
                log_list.append(_log)
            else:
                break

        if most_freq is True or least_freq is True:
            from collections import Counter
            counter = Counter(log_list)
            if most_freq is True:
                max_cnt = max(counter.values())
                log_list = [_log for _log in counter.keys() if counter[_log] == max_cnt]
            elif least_freq is True:
                min_cnt = min(counter.values())
                log_list = [_log for _log in counter.keys() if counter[_log] == min_cnt]

        for _log in log_list:

            if single_word is True:
                for word in _log.split():
                    if remove_stop_words is True:
                        if word not in STOP_WORDS:
                            word_set.add(word)
                    else:
                        word_set.add(word)
            else:
                _contents = list(_log.split())
                _new_contents = []
                for c in _contents:
                    if c not in _new_contents:
                        _new_contents.append(c)
                word = '_'.join(_new_contents)

                if 'add_card' in word:
                    word = 'add_card'
                if 'boot_' in word:
                    word = 'boot'
                if 'button' in word:
                    word = 'button'
                if 'critical_interrupt' in word:
                    word = 'critical_interrupt'
                if 'drive_slot' in word:
                    word = 'drive_slot'
                if 'management_subsys_health' in word:
                    word = 'management_subsys_health'
                if 'microcontroller' in word:
                    word = 'microcontroller'
                if 'oem_cpu' in word:
                    word = 'oem_cpu'
                if 'power_supply_cpu' in word:
                    word = 'power_supply_cpu'
                elif 'power_supply_psu' in word:
                    word = 'power_supply_psu'
                elif 'power_supply' in word:
                    word = 'power_supply'
                if 'power_unit' in word:
                    word = 'power_unit'
                if 'processor_cpu' in word:
                    word = 'processor_cpu'
                elif 'processor' in word:
                    word = 'processor'
                if 'slot_connector' in word:
                    word = 'slot_connector'
                if 'system_acpi_power' in word:
                    word = 'system_acpi_power'
                if 'system_boot_initiated' in word:
                    word = 'system_boot_initiated'

                if 'mem_cpu_dimm' in word:
                    word = 'mem_cpu_dimm'
                elif 'mem_dimm' in word:
                    word = 'mem_dimm'
                elif 'mem_cpu' in word:
                    word = 'mem_cpu'
                elif 'mem' in word:
                    word = 'mem'

                if 'cpu' in word and 'mem' not in word:
                    word = 'cpu'
                elif 'mem' in word:
                    word = 'mem'

                if 'system_event' in word:
                    word = 'system_event'
                if 'system_firmware' in word:
                    word = 'system_firmware'

                if 'temperature_cpu' in word:
                    word = 'temperature_cpu'
                elif 'temperature' in word and 'dimm' in word:
                    word = 'temperature_dimm'
                elif 'temperature' in word and 'moc' in word:
                    word = 'temperature_moc'
                elif 'temperature' in word:
                    word = 'temperature'

                if 'correctable_ecc' in word:
                    word = 'correctable_ecc'

                if 'ecc' in word:
                    word = 'ecc'

                word_set.add(word)

            # words.add(log)

        from collections import Counter
        # print(Counter(log_list))
        # print(f"words:{word_set}")
        return SPACE_STR.join(
            (list([c for c in word_set if len(c) > 2 or (len(c) == 2 and c not in {'ps', 'os', 'fw'})]))).strip()

    @staticmethod
    def extract_sentences_from_logs(_logs, remove_stop_words=False,
                                    single_word=False, n_limits=1000, sep='\n'):
        # 前提是按照时间降序排列
        first_time = 0
        _new_logs = []
        for _log in _logs[::-1]:
            time = int(_log.split()[0])
            # print(f"time:{time}")
            _log = ' '.join([c for c in _log.split()[1:]])
            if first_time == 0:
                first_time = time
            interval = time - first_time
            # print(f"interval:{interval}")
            if interval <= 3600 * 24:
                if _log not in _new_logs:
                    _new_logs.append(_log)
            else:
                break

        # print(f"words:{word_set}")
        return sep.join(_new_logs[::-1])

    @staticmethod
    def num_intervals(times):
        # 多少个区间
        last_time = 0
        intervals = 0
        times = sorted(list(times))

        for time in times:
            if last_time == 0:
                last_time = time
            interval = time - last_time
            # print(f"interval:{interval}")
            if interval > 60:
                last_time = time
                intervals += 1
        return intervals

    @staticmethod
    def num_logs_per_interval(times):
        # 每个区间的日志数
        last_time = 0
        intervals = 0
        times = sorted(list(times))
        num_logs = []
        curr_num = 0
        for time in times:
            if last_time == 0:
                last_time = time
            interval = time - last_time
            # print(f"interval:{interval}")
            if interval > 60:
                last_time = time
                intervals += 1
                num_logs.append(curr_num)
                curr_num = 0
            else:
                curr_num += 1
        if curr_num > 0:
            num_logs.append(curr_num)
        return num_logs

    @staticmethod
    def mean_num_logs_per_interval(times):
        # 每个区间的平均日志数
        num_logs = BaseFeatureExtractor.num_logs_per_interval(times)
        if len(num_logs) > 0:
            return np.mean(np.array(num_logs))
        return 0.

    @staticmethod
    def std_num_logs_per_interval(times):
        # 每个区间的日志数std
        num_logs = BaseFeatureExtractor.num_logs_per_interval(times)
        if len(num_logs) > 0:
            return np.std(np.array(num_logs))
        return 0.

    @staticmethod
    def num_logs_first_interval(times):
        # 第一个区间的日志数
        last_time = 0
        times = sorted(list(times))
        curr_num = 0
        for time in times:
            if last_time == 0:
                last_time = time
            interval = time - last_time
            # print(f"interval:{interval}")
            if interval > 60:
                return curr_num
        return curr_num

    def preprocess_log_df(self) -> pd.DataFrame:
        """
        对log数据做处理
        1. 时间列提取
        :return:
        """
        _log_df = self.log_df.copy(deep=True).drop_duplicates().sort_values(by=['sn', 'time']).reset_index(drop=True)
        _log_df['time_date'] = pd.to_datetime(_log_df['time'])

        _log_df[self.log_cols] = _log_df['msg'].str.split('|',
                                                          3,
                                                          expand=True,
                                                          )
        # _log_df[self.log_cols[0]] = _log_df['msg']
        for col in self.log_cols:
            _log_df[col] = _log_df[col].apply(lambda _log: self.process_log(_log))
        # 统计词频
        # for col in ['component', 'fault_phenomenon', 'asserted_type', 'extend_description']:
        #     log_df[col] = log_df[col].apply(lambda _log: self.process_log(_log))
        #     words=[]
        #     for log in log_df[col].values:
        #         for word in log.split():
        #             words.append(word)
        #     from collections import Counter
        #     counter=Counter(words)
        #     print(f"col:{col}")
        #     print(f"len_counter:{len(counter)}")
        #     print(counter)

        return _log_df

    def preprocess_fault_df(self) -> pd.DataFrame:
        fault_df = self.fault_df.copy(deep=True)
        fault_df = fault_df.drop_duplicates().reset_index(drop=True)
        fault_df['fault_time_date'] = pd.to_datetime(fault_df['fault_time'])
        #
        fault_df['_'.join([self.num_feature_prefix, 'sin_time'])] = fault_df['fault_time'].apply(
            lambda val: self.sin_time(val))
        fault_df['_'.join([self.num_feature_prefix, 'cos_time'])] = fault_df['fault_time'].apply(
            lambda val: self.cos_time(val))
        fault_df['_'.join([self.num_feature_prefix, 'sin_day'])] = fault_df['fault_time'].apply(
            lambda val: self.sin_day(val))
        fault_df['_'.join([self.num_feature_prefix, 'cos_day'])] = fault_df['fault_time'].apply(
            lambda val: self.cos_day(val))

        #
        fault_df['_'.join([self.num_feature_prefix, 'is_day_time'])] = fault_df['fault_time'].apply(
            lambda val: self.is_daytime(val))

        fault_df['_'.join([self.num_feature_prefix, 'is_work_day'])] = fault_df['fault_time_date'].apply(
            lambda val: int(is_workday(val)))
        fault_df['_'.join([self.num_feature_prefix, 'is_in_lieu'])] = fault_df['fault_time_date'].apply(
            lambda val: int(is_in_lieu(val)))

        return fault_df

    def _extract_logs_count_features(self, overall_df: pd.DataFrame, time_seconds_before_fault: int) -> pd.DataFrame:
        _overall_df = overall_df.copy(deep=True)
        df_logs_count_before_fault = overall_df[
            (overall_df['fault_time_diff_seconds'] >= 0) & (
                    overall_df['fault_time_diff_seconds'] < time_seconds_before_fault)].reset_index(
            drop=True).groupby(self.fault_df_columns, as_index=False)['fault_time_diff_seconds'].agg(
            {
                '_'.join([self.num_feature_prefix, self.log_count_feature_prefix, 'count',
                          str(time_seconds_before_fault)]): 'count',
                '_'.join([self.num_feature_prefix, self.log_count_feature_prefix, 'mean',
                          str(time_seconds_before_fault)]): lambda time_diffs: np.mean(time_diffs),
                '_'.join(
                    [self.num_feature_prefix, self.log_count_feature_prefix, 'std', str(time_seconds_before_fault)])
                : lambda time_diffs: np.std(time_diffs),
                '_'.join(
                    [self.num_feature_prefix, self.log_count_feature_prefix, 'skew', str(time_seconds_before_fault)])
                : lambda time_diffs: skew(time_diffs),
                '_'.join(
                    [self.num_feature_prefix, self.log_count_feature_prefix, 'kurtosis',
                     str(time_seconds_before_fault)])
                : lambda time_diffs: kurtosis(time_diffs),
            })

        df_num_unique_count_before_fault = overall_df[
            (overall_df['fault_time_diff_seconds'] >= 0) & (
                    overall_df['fault_time_diff_seconds'] < time_seconds_before_fault)].reset_index(
            drop=True).groupby(self.fault_df_columns, as_index=False)['msg'].agg({
            '_'.join([self.num_feature_prefix, 'num_unique_logs', str(time_seconds_before_fault)]):
                lambda vals: self.get_len_unique_logs(vals),
            # 分组里面每种日志的平均条数
            '_'.join([self.num_feature_prefix, 'num_mean_each_log', str(time_seconds_before_fault)]):
                lambda vals: self.counter_mean_num_each_log(vals),
            # 分组里面每种日志的平均条数的方差
            '_'.join([self.num_feature_prefix, 'num_std_each_log', str(time_seconds_before_fault)]):
                lambda vals: self.counter_std_num_each_log(vals),

            '_'.join([self.num_feature_prefix, 'num_max_each_log', str(time_seconds_before_fault)]):
                lambda vals: self.counter_max_num_each_log(vals),

            '_'.join([self.num_feature_prefix, 'num_min_each_log', str(time_seconds_before_fault)]):
                lambda vals: self.counter_min_num_each_log(vals),

        })

        df_count = df_logs_count_before_fault.merge(df_num_unique_count_before_fault, on=self.fault_df_columns,
                                                    how='left').reset_index(drop=True)

        useful_feature_columns = self.fault_df_columns + [col for col in df_count.columns if
                                                          col.startswith(self.num_feature_prefix)
                                                          ]
        return df_count[useful_feature_columns].reset_index(drop=True)

    def _extract_logs_text_origin_features(self, _overall_df: pd.DataFrame, time_seconds_before_fault: int,
                                           ) -> pd.DataFrame:

        df_with_text = None
        overall_df = _overall_df.copy(deep=True)
        overall_df['fault_time_diff_seconds'] = (overall_df['fault_time_date'] - overall_df['time_date']).dt.seconds

        for log_col in ['msg']:
            _log_feature_col = '_'.join([self.log_sentences_feature_preifx, log_col,
                                         str(time_seconds_before_fault)])

            overall_df[log_col] = overall_df['fault_time_diff_seconds'].astype(str).str.cat(overall_df[log_col],
                                                                                            sep=' ')

            curr_df_logs_before_fault = overall_df[
                (overall_df['fault_time_diff_seconds'] >= 0) & (
                        overall_df['fault_time_diff_seconds'] < time_seconds_before_fault)].reset_index(
                drop=True).sort_values(
                by=['sn', 'time'], ascending=True).groupby(
                self.fault_df_columns, as_index=False)[log_col].agg({_log_feature_col:
                                                                         lambda _logs: self.extract_sentences_from_logs(
                                                                             _logs, remove_stop_words=True)})
            useful_columns = self.fault_df_columns + [_log_feature_col]

            curr_df_logs_before_fault = curr_df_logs_before_fault[useful_columns].reset_index(drop=True)
            if df_with_text is None:
                df_with_text = curr_df_logs_before_fault
            else:
                df_with_text = df_with_text.merge(curr_df_logs_before_fault, on=self.fault_df_columns,
                                                  how='left').reset_index(drop=True)
        return df_with_text

    def _extract_logs_text_features(self, _overall_df: pd.DataFrame, time_seconds_before_fault: int,
                                    num_cols=3) -> pd.DataFrame:

        df_with_text = None
        overall_df = _overall_df.copy(deep=True)
        overall_df['fault_time_diff_seconds'] = (overall_df['fault_time_date'] - overall_df['time_date']).dt.seconds

        for log_col in self.log_cols[:num_cols]:
            _log_feature_col = '_'.join([self.log_text_feature_prefix, log_col,
                                         str(time_seconds_before_fault)])

            overall_df[log_col] = overall_df['fault_time_diff_seconds'].astype(str).str.cat(overall_df[log_col],
                                                                                            sep=' ')

            curr_df_logs_before_fault = overall_df[
                (overall_df['fault_time_diff_seconds'] >= 0) & (
                        overall_df['fault_time_diff_seconds'] < time_seconds_before_fault)].reset_index(
                drop=True).sort_values(
                by=['sn', 'time'], ascending=True).groupby(
                self.fault_df_columns, as_index=False)[log_col].agg({_log_feature_col:
                                                                         lambda _logs: self.extract_words_from_logs(
                                                                             _logs, remove_stop_words=True)})
            useful_columns = self.fault_df_columns + [_log_feature_col]
            curr_df_logs_before_fault = curr_df_logs_before_fault[useful_columns].reset_index(drop=True)
            curr_df = curr_df_logs_before_fault

            if df_with_text is None:
                df_with_text = curr_df
            else:
                df_with_text = df_with_text.merge(curr_df, on=self.fault_df_columns,
                                                  how='left').reset_index(drop=True)
        return df_with_text

    def _extract_server_model_features(self, overall_df: pd.DataFrame) -> pd.DataFrame:
        df_server_model = overall_df[
            (overall_df['fault_time_diff_seconds'] >= 0)].reset_index(
            drop=True).sort_values(by=['sn', 'time'], ascending=True).groupby(
            self.fault_df_columns, as_index=False)['server_model'].agg({'server_model':
                                                                            lambda vals: ''.join(
                                                                                vals[-1:])}).reset_index(drop=True)
        return df_server_model

    def _extract_latest_log_time_feature(self, overall_df: pd.DataFrame) -> pd.DataFrame:
        df_min_log_time_diff_seconds = overall_df[
            (overall_df['fault_time_diff_seconds'] >= 0) & (
                    overall_df['fault_time_diff_seconds'] < 3600 * 24)].reset_index(
            drop=True).groupby(self.fault_df_columns, as_index=False)['fault_time_diff_seconds'].agg(
            {'_'.join((self.num_feature_prefix, 'min_log_time_diff_seconds')):
                 lambda vals: min(vals),
             '_'.join((self.num_feature_prefix, 'log_time_intervals_count')):
                 lambda vals: self.num_intervals(vals),
             '_'.join((self.num_feature_prefix, 'mean_num_logs_per_interval')):
                 lambda vals: self.mean_num_logs_per_interval(vals),
             '_'.join((self.num_feature_prefix, 'std_num_logs_per_interval')):
                 lambda vals: self.std_num_logs_per_interval(vals),
             '_'.join((self.num_feature_prefix, 'num_logs_first_interval')):
                 lambda vals: self.num_logs_first_interval(vals),
             }).reset_index(drop=True)
        return df_min_log_time_diff_seconds

    def extract_features(self):
        _fault_df = self.preprocess_fault_df()
        _log_df = self.preprocess_log_df()
        _overall_df = _fault_df.merge(_log_df, how='left', on='sn').reset_index(drop=True)
        _overall_df['fault_time_diff_seconds'] = (_overall_df['fault_time_date'] - _overall_df['time_date']).dt.seconds

        num_feature_df = self.fault_df.copy(deep=True)
        for time_seconds_before_fault in [3600 * 24]:
            curr_logs_count_feature_df = self._extract_logs_count_features(_overall_df,
                                                                           time_seconds_before_fault=time_seconds_before_fault)
            num_feature_df = num_feature_df.merge(curr_logs_count_feature_df, on=self.fault_df_columns,
                                                  how='left').reset_index(drop=True)
        latest_log_time_feature_df = self._extract_latest_log_time_feature(_overall_df)
        num_feature_df = num_feature_df.merge(latest_log_time_feature_df, on=self.fault_df_columns,
                                              how='left').reset_index(drop=True)

        for _col in num_feature_df.columns:
            if _col.startswith(self.num_feature_prefix):
                if 'min_log_time_diff_seconds' in _col:
                    num_feature_df[_col] = num_feature_df[_col].fillna(3600 * 24)
                else:
                    num_feature_df[_col] = num_feature_df[_col].fillna(0)

                # if _col.endswith('count') or 'logs_count_count' in _col:
                #     _derived_col = _col + '_if_greater_than_zero'
                #     num_feature_df[_derived_col] = 0
                #     num_feature_df.loc[num_feature_df[_col] > 0, _derived_col] = 1.
        # 加入日期统计特征
        num_feature_df = num_feature_df.merge(_fault_df, on=self.fault_df_columns, how='left').reset_index(drop=True)

        text_feature_df = _fault_df.copy(deep=True)
        for time_seconds_before_fault in [3600 * 24]:
            curr_text_feature_df = self._extract_logs_text_features(_overall_df, time_seconds_before_fault)
            text_feature_df = text_feature_df.merge(curr_text_feature_df, on=self.fault_df_columns,
                                                    how='left').reset_index(drop=True)
        server_model_feature_df = self._extract_server_model_features(_overall_df)
        text_feature_df = text_feature_df.merge(server_model_feature_df, on=self.fault_df_columns,
                                                how='left').reset_index(drop=True)
        for _col in text_feature_df.columns:
            if _col.startswith(self.log_text_feature_prefix):
                text_feature_df[_col] = text_feature_df[_col].fillna("")
            elif _col == 'server_model':
                text_feature_df[_col] = text_feature_df[_col].fillna('SM-1')

        sentences_feature_df = None
        # sentences_feature_df = self.fault_df.copy(deep=True)
        # for time_seconds_before_fault in [3600 * 24]:
        #     curr_sentences_feature_df = self._extract_logs_text_origin_features(_overall_df, time_seconds_before_fault)
        #     sentences_feature_df = sentences_feature_df.merge(curr_sentences_feature_df, on=self.fault_df_columns,
        #                                                       how='left').reset_index(drop=True)
        #
        # for _col in sentences_feature_df.columns:
        #     if _col.startswith(self.log_sentences_feature_preifx):
        #         sentences_feature_df[_col] = sentences_feature_df[_col].fillna("")

        venus_feature_df = self.fault_df.copy(deep=True)
        venus_feature_df = venus_feature_df.merge(self.venus_df, how='left', on=['sn', 'fault_time']).reset_index(
            drop=True)
        venus_feature_df = venus_feature_df.fillna('NA')
        print(venus_feature_df.columns)

        venus_feature_df[f'{self.venus_feature_prefix}_has_venus_info'] = 0
        venus_feature_df.loc[venus_feature_df['module'] != 'NA', f'{self.venus_feature_prefix}_has_venus_info'] = 1

        def _get_len_module(_module_str):
            if _module_str == 'NA':
                return 0
            else:
                return len(_module_str.split(','))

        venus_feature_df[f"{self.venus_feature_prefix}_len_module"] = venus_feature_df['module'].apply(
            lambda val: _get_len_module(val))
        venus_feature_df[f"{self.venus_feature_prefix}_catfeature_module"]=venus_feature_df['module'].apply(
            lambda val:module_to_cat_int(val)
        )

        crashdump_feature_df = self.fault_df.copy(deep=True)
        crashdump_feature_df = crashdump_feature_df.merge(self.crashdump_df, how='left',
                                                          on=['sn', 'fault_time']).reset_index(drop=True)
        crashdump_feature_df = crashdump_feature_df.fillna('NA.NA.NA.NA')
        crashdump_feature_df[f"{self.crashdump_feature_prefix}_has_crashdump_info"] = 0
        crashdump_feature_df.loc[crashdump_feature_df[
                                     'fault_code'] != 'NA.NA.NA.NA', f"{self.crashdump_feature_prefix}_has_crashdump_info"] = 1

        crash_fault_code_cols = [f"{self.crashdump_feature_prefix}_catfeature_{i}" for i in range(4)]
        crashdump_feature_df[crash_fault_code_cols] = crashdump_feature_df['fault_code'].str.split('.',
                                                                                                   3,
                                                                                                   expand=True,
                                                                                                   )
        # _log_df[self.log_cols[0]] = _log_df['msg']
        for i, col in enumerate(crash_fault_code_cols):
            crashdump_feature_df[col] = crashdump_feature_df[col].apply(lambda val: fault_code_to_cat_int(val, i))
            crashdump_feature_df[col] = crashdump_feature_df[col].astype(int)


        return num_feature_df, text_feature_df, sentences_feature_df, _log_df, venus_feature_df, crashdump_feature_df
