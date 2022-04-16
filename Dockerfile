FROM registry.cn-shanghai.aliyuncs.com/fengyang95/tianchi:1.0
RUN pip install -U chinesecalendar
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U optuna
ADD . /
WORKDIR /

