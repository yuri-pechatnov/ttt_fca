
## Печатнов Юрий М05-895б

Домашнее задание по теории решеток

В данной работе взят датасет Tic-Tac-Toe

Разработана собственная модификация алгоритма классификации на основе генераторов

И сравнена с `RandomForestClassifier` и `GradientBoostingClassifier` из `sklearn.ensemble`


```python
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from IPython.display import display, clear_output
```


```python
import sklearn.ensemble
```

Для начала бинаризуем признаки, чтобы сделать задачу более подходящей для применения fca

Затем на одних и тех же данных применим три алгоритма классификации


```python
def preprocess(df):
    df = df.copy()
    last_col = list(df)[-1]
    for col in list(df)[:-1]:
        df[col + "_0"] = (df[col] == 'x') | (df[col] == 'o')
        df[col + "_1"] = (df[col] == 'o')
        df = df.drop([col], axis=1)
    df["y"] = (df[last_col] == 'positive')
    df = df.drop([last_col], axis=1)
    return df

def get_common_features(X):
    if len(X) == 0:
        return None, None
    else:
        mask = (X == X[0]).sum(axis=0) == len(X)
        return mask, X[0][mask]
```

Рассматриваю две симетричные задачи и считаю `score` для каждой. Если `score` больше для положительной задачи то ответ положительный. Иначе отрицательный.
`score` считаю так: при фиксированном тестовом запросе перебираю тренировочные положительные примеры, считаю пересечения с ними. Затем делаю замыкание каждого пересечения пользуясь отрицательными примерами. Если окажется так, что замыкание лежит в тестовом запросе, то в `score` прибавляю (`размер замыкания` - `размер пересечения`) * `поддержка`.

Так же фильтрую пересечения по размеру, поддержке и достоверности.


```python
class FCAPredictor:
    def __init__(self, min_support=0.001, min_confidence=0.001, min_intersect=3):
        for name in self.__init__.__code__.co_varnames:
            if name != 'self':
                setattr(self, name, locals()[name])
        
    def fit(self, X, y):
        X = np.array(X)  # L x F
        y = (np.array(y) != 0)  # L
        
        self.examples = [X[~y], X[y]]  # negative and positive examples 
        self.examples_X = X
        self.examples_y = y
        
        self.targets_sums = [(~y).sum(), y.sum()]
        
    def predict_one(self, x):
        x = np.array(x)  # F
        
        # masks of equal features
        intersections = (self.examples_X == x)
        
        results = []
        
        # score
        rating2 = [0, 0]
        rating = [0, 0]
        for elem, cand, t in zip(self.examples_X, intersections, self.examples_y):
            min_subset_cardinality = cand.sum()
            if min_subset_cardinality < self.min_intersect:
                continue
            including_mask = ((intersections & cand).sum(axis=1) >= min_subset_cardinality)
            
            corresponding_targets = (self.examples_y[including_mask] == t)
            corresponding_X = self.examples_X[including_mask]
            
            support = corresponding_targets.sum() / self.targets_sums[int(t)]
            confidence = (~corresponding_targets).sum() / self.targets_sums[int(~t)]
    
            common_features_mask, common_features_values = \
                get_common_features(corresponding_X[~corresponding_targets])
            
            if common_features_mask is not None and np.all(x[common_features_mask] == common_features_values) \
                and support > self.min_support and confidence > self.min_confidence:
                rating2[int(~t)] += (common_features_mask.sum() - min_subset_cardinality) * support
             
            #rating[int(t)] += (support > self.min_support) & (unconfidence < self.max_unconfidence)
            #rating[int(t)] += ((support > self.min_support) & (support > 3 * unconfidence)) * intersection_cardinality
            rating[int(t)] += support > 2 * confidence
        
        rating2[1] /= self.targets_sums[1]
        rating2[0] /= self.targets_sums[0]
        return rating2[1] >= rating2[0]
            
        rating[1] /= self.targets_sums[1]
        rating[0] /= self.targets_sums[0]
        return rating[1] >= rating[0]
        
    def predict(self, X):
        X = np.array(X)  # L' x F
        y = np.array([self.predict_one(x) for x in X], dtype=np.bool)  # L
        return y        
```

Подсчет метрик и запуски


```python
def calc_metrics(y_pred, y_real):
    metrics = dict(
        TP=(y_pred & y_real).sum(),
        FP=(y_pred & ~y_real).sum(),
        TN=(~y_pred & ~y_real).sum(),
        FN=(~y_pred & y_real).sum(),
        accuracy = (y_pred == y_real).sum() / len(y_pred),
    )
    metrics.update(
        precision=metrics["TP"] / (metrics["TP"] + metrics["FP"]),
        recall=metrics["TP"] / (metrics["TP"] + metrics["FN"]),
    )
    return metrics


def try_it(train, test, predictor=None):
    train = preprocess(train)
    test = preprocess(test)
    predictor = predictor if predictor is not None else FCAPredictor()
    predictor.fit(train.drop(["y"], axis=1), train["y"])
    predictions = predictor.predict(test.drop(["y"], axis=1))

    
    real_targets = np.array(test["y"])
    return calc_metrics(predictions, real_targets)

def show_on_set(i, predictor=None):
    predictor = predictor if predictor is not None else FCAPredictor(min_intersect=12)
    return try_it(pd.read_csv("../../dataset/train%d.csv" % i), pd.read_csv("../../dataset/test%d.csv" % i), predictor)
```


```python
rows = []
for i in range(1, 11):
    pref = "%d_" % i
    row = {
        pref + "FCA": show_on_set(i, FCAPredictor(min_intersect=12)), 
        pref + "RANDOMFOREST": show_on_set(i, sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_depth=10)),
        pref + "GRADIENT-BOOSTING": show_on_set(i, sklearn.ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=10))
    }
    display(row)
    for k, v in row.items():
        v["method"] = k.split('_')[1]
        v["data_set"] = k.split('_')[0]
    rows += row.values()
clear_output(wait=True)
df = pd.DataFrame(rows)
display(df)
display(df.groupby(['method'], as_index=False).mean())
```


    {'1_FCA': {'FN': 0,
      'FP': 0,
      'TN': 32,
      'TP': 61,
      'accuracy': 1.0,
      'precision': 1.0,
      'recall': 1.0},
     '1_GRADIENT-BOOSTING': {'FN': 6,
      'FP': 0,
      'TN': 32,
      'TP': 55,
      'accuracy': 0.93548387096774188,
      'precision': 1.0,
      'recall': 0.90163934426229508},
     '1_RANDOMFOREST': {'FN': 5,
      'FP': 1,
      'TN': 31,
      'TP': 56,
      'accuracy': 0.93548387096774188,
      'precision': 0.98245614035087714,
      'recall': 0.91803278688524592}}



    {'2_FCA': {'FN': 0,
      'FP': 5,
      'TN': 31,
      'TP': 51,
      'accuracy': 0.94252873563218387,
      'precision': 0.9107142857142857,
      'recall': 1.0},
     '2_GRADIENT-BOOSTING': {'FN': 2,
      'FP': 4,
      'TN': 32,
      'TP': 49,
      'accuracy': 0.93103448275862066,
      'precision': 0.92452830188679247,
      'recall': 0.96078431372549022},
     '2_RANDOMFOREST': {'FN': 0,
      'FP': 5,
      'TN': 31,
      'TP': 51,
      'accuracy': 0.94252873563218387,
      'precision': 0.9107142857142857,
      'recall': 1.0}}



    {'3_FCA': {'FN': 0,
      'FP': 2,
      'TN': 33,
      'TP': 65,
      'accuracy': 0.97999999999999998,
      'precision': 0.97014925373134331,
      'recall': 1.0},
     '3_GRADIENT-BOOSTING': {'FN': 1,
      'FP': 1,
      'TN': 34,
      'TP': 64,
      'accuracy': 0.97999999999999998,
      'precision': 0.98461538461538467,
      'recall': 0.98461538461538467},
     '3_RANDOMFOREST': {'FN': 0,
      'FP': 2,
      'TN': 33,
      'TP': 65,
      'accuracy': 0.97999999999999998,
      'precision': 0.97014925373134331,
      'recall': 1.0}}



    {'4_FCA': {'FN': 0,
      'FP': 3,
      'TN': 27,
      'TP': 59,
      'accuracy': 0.9662921348314607,
      'precision': 0.95161290322580649,
      'recall': 1.0},
     '4_GRADIENT-BOOSTING': {'FN': 3,
      'FP': 3,
      'TN': 27,
      'TP': 56,
      'accuracy': 0.93258426966292129,
      'precision': 0.94915254237288138,
      'recall': 0.94915254237288138},
     '4_RANDOMFOREST': {'FN': 0,
      'FP': 3,
      'TN': 27,
      'TP': 59,
      'accuracy': 0.9662921348314607,
      'precision': 0.95161290322580649,
      'recall': 1.0}}



    {'5_FCA': {'FN': 4,
      'FP': 2,
      'TN': 25,
      'TP': 58,
      'accuracy': 0.93258426966292129,
      'precision': 0.96666666666666667,
      'recall': 0.93548387096774188},
     '5_GRADIENT-BOOSTING': {'FN': 8,
      'FP': 1,
      'TN': 26,
      'TP': 54,
      'accuracy': 0.898876404494382,
      'precision': 0.98181818181818181,
      'recall': 0.87096774193548387},
     '5_RANDOMFOREST': {'FN': 5,
      'FP': 3,
      'TN': 24,
      'TP': 57,
      'accuracy': 0.9101123595505618,
      'precision': 0.94999999999999996,
      'recall': 0.91935483870967738}}



    {'6_FCA': {'FN': 0,
      'FP': 1,
      'TN': 28,
      'TP': 56,
      'accuracy': 0.9882352941176471,
      'precision': 0.98245614035087714,
      'recall': 1.0},
     '6_GRADIENT-BOOSTING': {'FN': 0,
      'FP': 0,
      'TN': 29,
      'TP': 56,
      'accuracy': 1.0,
      'precision': 1.0,
      'recall': 1.0},
     '6_RANDOMFOREST': {'FN': 0,
      'FP': 2,
      'TN': 27,
      'TP': 56,
      'accuracy': 0.97647058823529409,
      'precision': 0.96551724137931039,
      'recall': 1.0}}



    {'7_FCA': {'FN': 0,
      'FP': 4,
      'TN': 40,
      'TP': 70,
      'accuracy': 0.96491228070175439,
      'precision': 0.94594594594594594,
      'recall': 1.0},
     '7_GRADIENT-BOOSTING': {'FN': 1,
      'FP': 2,
      'TN': 42,
      'TP': 69,
      'accuracy': 0.97368421052631582,
      'precision': 0.971830985915493,
      'recall': 0.98571428571428577},
     '7_RANDOMFOREST': {'FN': 0,
      'FP': 5,
      'TN': 39,
      'TP': 70,
      'accuracy': 0.95614035087719296,
      'precision': 0.93333333333333335,
      'recall': 1.0}}


Видим, что на задаче адаптированной под FCA, этот метод выдает лучшие результаты.
Но нужно учитывать адаптированность и то, что для FCA подбирались лучшие параметры, а для других моделей - нет.

Далее код для подбора параметров


```python
log_file = "log_txt"
with open(log_file, 'w') as f:
    f.write("\n")
for min_support in [0.001, 0.003, 0.005]:
    for min_confidence in [0.001, 0.003, 0.005]:
        for min_intersect in [12]:
            params = {
                "min_support": min_support,
                "min_confidence": min_confidence,
                "min_intersect": min_intersect,
            }
            predictor = FCAPredictor(**params)
            params.update(
                try_it(pd.read_csv("../../dataset/train2.csv"), pd.read_csv("../../dataset/test2.csv"), predictor)
            )
            with open(log_file, 'a') as f:
                f.write(str(params) + "\n")
            print(params)
```

    {'min_support': 0.001, 'min_confidence': 0.001, 'min_intersect': 12, 'TP': 51, 'FP': 5, 'TN': 31, 'FN': 0, 'accuracy': 0.94252873563218387, 'precision': 0.9107142857142857, 'recall': 1.0}
    {'min_support': 0.001, 'min_confidence': 0.003, 'min_intersect': 12, 'TP': 51, 'FP': 5, 'TN': 31, 'FN': 0, 'accuracy': 0.94252873563218387, 'precision': 0.9107142857142857, 'recall': 1.0}
    {'min_support': 0.001, 'min_confidence': 0.005, 'min_intersect': 12, 'TP': 49, 'FP': 3, 'TN': 33, 'FN': 2, 'accuracy': 0.94252873563218387, 'precision': 0.94230769230769229, 'recall': 0.96078431372549022}
    {'min_support': 0.003, 'min_confidence': 0.001, 'min_intersect': 12, 'TP': 51, 'FP': 5, 'TN': 31, 'FN': 0, 'accuracy': 0.94252873563218387, 'precision': 0.9107142857142857, 'recall': 1.0}
    {'min_support': 0.003, 'min_confidence': 0.003, 'min_intersect': 12, 'TP': 51, 'FP': 5, 'TN': 31, 'FN': 0, 'accuracy': 0.94252873563218387, 'precision': 0.9107142857142857, 'recall': 1.0}
    {'min_support': 0.003, 'min_confidence': 0.005, 'min_intersect': 12, 'TP': 49, 'FP': 3, 'TN': 33, 'FN': 2, 'accuracy': 0.94252873563218387, 'precision': 0.94230769230769229, 'recall': 0.96078431372549022}
    {'min_support': 0.005, 'min_confidence': 0.001, 'min_intersect': 12, 'TP': 50, 'FP': 4, 'TN': 32, 'FN': 1, 'accuracy': 0.94252873563218387, 'precision': 0.92592592592592593, 'recall': 0.98039215686274506}



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-46-b0137f1d2b79> in <module>()
         12             predictor = FCAPredictor(**params)
         13             params.update(
    ---> 14                 try_it(pd.read_csv("../train2.csv"), pd.read_csv("../test2.csv"), predictor)
         15             )
         16             with open(log_file, 'a') as f:


    <ipython-input-25-14c5055c3b52> in try_it(train, test, predictor)
         19     predictor = predictor if predictor is not None else FCAPredictor()
         20     predictor.fit(train.drop(["y"], axis=1), train["y"])
    ---> 21     predictions = predictor.predict(test.drop(["y"], axis=1))
         22 
         23 


    <ipython-input-19-5f69457aa1df> in predict(self, X)
         57     def predict(self, X):
         58         X = np.array(X)  # L' x F
    ---> 59         y = np.array([self.predict_one(x) for x in X], dtype=np.bool)  # L
         60         return y


    <ipython-input-19-5f69457aa1df> in <listcomp>(.0)
         57     def predict(self, X):
         58         X = np.array(X)  # L' x F
    ---> 59         y = np.array([self.predict_one(x) for x in X], dtype=np.bool)  # L
         60         return y


    <ipython-input-19-5f69457aa1df> in predict_one(self, x)
         35             corresponding_X = self.examples_X[including_mask]
         36 
    ---> 37             support = corresponding_targets.sum() / self.targets_sums[int(t)]
         38             confidence = (~corresponding_targets).sum() / self.targets_sums[int(~t)]
         39 


    KeyboardInterrupt: 


## TODO

Далее идею можно развивать: например стоит сделать подбор идеальных параметров частью процесса обучения:

разделить учебную выборку на пре-учебную и пре-тестовую,

далее перебирать параметры, обучаться на пре-учебной, валидироваться на пре-тестовой,

и так выбрать лучшие параметры.

В итоговой модели использовать полученные параметры и всю учебную выборку.


```python
!jupyter nbconvert lattices.ipynb --to markdown
```

    [NbConvertApp] Converting notebook lattices.ipynb to markdown
    [NbConvertApp] Writing 18808 bytes to lattices.md



```python
!head ../test1.csv

```

    V1,V2,V3,V4,V5,V6,V7,V8,V9,V10
    x,x,x,x,o,o,o,x,o,positive
    x,x,x,x,o,b,o,b,o,positive
    x,x,x,o,o,x,o,x,o,positive
    x,x,x,o,o,b,x,o,b,positive
    x,x,x,b,o,b,o,o,x,positive
    x,x,x,b,b,o,b,o,b,positive
    x,x,o,o,x,o,o,x,x,positive
    x,x,o,o,x,b,o,x,b,positive
    x,x,b,x,o,o,x,o,b,positive

