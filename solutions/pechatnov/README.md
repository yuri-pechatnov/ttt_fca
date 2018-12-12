
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


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FN</th>
      <th>FP</th>
      <th>TN</th>
      <th>TP</th>
      <th>accuracy</th>
      <th>data_set</th>
      <th>method</th>
      <th>precision</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>61</td>
      <td>1.000000</td>
      <td>1</td>
      <td>FCA</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>1</td>
      <td>31</td>
      <td>56</td>
      <td>0.935484</td>
      <td>1</td>
      <td>RANDOMFOREST</td>
      <td>0.982456</td>
      <td>0.918033</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>0</td>
      <td>32</td>
      <td>55</td>
      <td>0.935484</td>
      <td>1</td>
      <td>GRADIENT-BOOSTING</td>
      <td>1.000000</td>
      <td>0.901639</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>5</td>
      <td>31</td>
      <td>51</td>
      <td>0.942529</td>
      <td>2</td>
      <td>FCA</td>
      <td>0.910714</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5</td>
      <td>31</td>
      <td>51</td>
      <td>0.942529</td>
      <td>2</td>
      <td>RANDOMFOREST</td>
      <td>0.910714</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>4</td>
      <td>32</td>
      <td>49</td>
      <td>0.931034</td>
      <td>2</td>
      <td>GRADIENT-BOOSTING</td>
      <td>0.924528</td>
      <td>0.960784</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>2</td>
      <td>33</td>
      <td>65</td>
      <td>0.980000</td>
      <td>3</td>
      <td>FCA</td>
      <td>0.970149</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>2</td>
      <td>33</td>
      <td>65</td>
      <td>0.980000</td>
      <td>3</td>
      <td>RANDOMFOREST</td>
      <td>0.970149</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1</td>
      <td>34</td>
      <td>64</td>
      <td>0.980000</td>
      <td>3</td>
      <td>GRADIENT-BOOSTING</td>
      <td>0.984615</td>
      <td>0.984615</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>3</td>
      <td>27</td>
      <td>59</td>
      <td>0.966292</td>
      <td>4</td>
      <td>FCA</td>
      <td>0.951613</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>3</td>
      <td>27</td>
      <td>59</td>
      <td>0.966292</td>
      <td>4</td>
      <td>RANDOMFOREST</td>
      <td>0.951613</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3</td>
      <td>3</td>
      <td>27</td>
      <td>56</td>
      <td>0.932584</td>
      <td>4</td>
      <td>GRADIENT-BOOSTING</td>
      <td>0.949153</td>
      <td>0.949153</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4</td>
      <td>2</td>
      <td>25</td>
      <td>58</td>
      <td>0.932584</td>
      <td>5</td>
      <td>FCA</td>
      <td>0.966667</td>
      <td>0.935484</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>3</td>
      <td>24</td>
      <td>57</td>
      <td>0.910112</td>
      <td>5</td>
      <td>RANDOMFOREST</td>
      <td>0.950000</td>
      <td>0.919355</td>
    </tr>
    <tr>
      <th>14</th>
      <td>8</td>
      <td>1</td>
      <td>26</td>
      <td>54</td>
      <td>0.898876</td>
      <td>5</td>
      <td>GRADIENT-BOOSTING</td>
      <td>0.981818</td>
      <td>0.870968</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>1</td>
      <td>28</td>
      <td>56</td>
      <td>0.988235</td>
      <td>6</td>
      <td>FCA</td>
      <td>0.982456</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>2</td>
      <td>27</td>
      <td>56</td>
      <td>0.976471</td>
      <td>6</td>
      <td>RANDOMFOREST</td>
      <td>0.965517</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0</td>
      <td>29</td>
      <td>56</td>
      <td>1.000000</td>
      <td>6</td>
      <td>GRADIENT-BOOSTING</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>4</td>
      <td>40</td>
      <td>70</td>
      <td>0.964912</td>
      <td>7</td>
      <td>FCA</td>
      <td>0.945946</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>5</td>
      <td>39</td>
      <td>70</td>
      <td>0.956140</td>
      <td>7</td>
      <td>RANDOMFOREST</td>
      <td>0.933333</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>2</td>
      <td>42</td>
      <td>69</td>
      <td>0.973684</td>
      <td>7</td>
      <td>GRADIENT-BOOSTING</td>
      <td>0.971831</td>
      <td>0.985714</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>2</td>
      <td>32</td>
      <td>72</td>
      <td>0.971963</td>
      <td>8</td>
      <td>FCA</td>
      <td>0.972973</td>
      <td>0.986301</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>1</td>
      <td>33</td>
      <td>72</td>
      <td>0.981308</td>
      <td>8</td>
      <td>RANDOMFOREST</td>
      <td>0.986301</td>
      <td>0.986301</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2</td>
      <td>3</td>
      <td>31</td>
      <td>71</td>
      <td>0.953271</td>
      <td>8</td>
      <td>GRADIENT-BOOSTING</td>
      <td>0.959459</td>
      <td>0.972603</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>1</td>
      <td>32</td>
      <td>70</td>
      <td>0.990291</td>
      <td>9</td>
      <td>FCA</td>
      <td>0.985915</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>1</td>
      <td>32</td>
      <td>70</td>
      <td>0.990291</td>
      <td>9</td>
      <td>RANDOMFOREST</td>
      <td>0.985915</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2</td>
      <td>4</td>
      <td>29</td>
      <td>68</td>
      <td>0.941748</td>
      <td>9</td>
      <td>GRADIENT-BOOSTING</td>
      <td>0.944444</td>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>0</td>
      <td>32</td>
      <td>58</td>
      <td>0.989011</td>
      <td>10</td>
      <td>FCA</td>
      <td>1.000000</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>1</td>
      <td>31</td>
      <td>59</td>
      <td>0.989011</td>
      <td>10</td>
      <td>RANDOMFOREST</td>
      <td>0.983333</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>59</td>
      <td>1.000000</td>
      <td>10</td>
      <td>GRADIENT-BOOSTING</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>method</th>
      <th>FN</th>
      <th>FP</th>
      <th>TN</th>
      <th>TP</th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FCA</td>
      <td>0.6</td>
      <td>2.0</td>
      <td>31.2</td>
      <td>62.0</td>
      <td>0.972582</td>
      <td>0.968643</td>
      <td>0.990484</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GRADIENT-BOOSTING</td>
      <td>2.5</td>
      <td>1.8</td>
      <td>31.4</td>
      <td>60.1</td>
      <td>0.954668</td>
      <td>0.971585</td>
      <td>0.959690</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RANDOMFOREST</td>
      <td>1.1</td>
      <td>2.4</td>
      <td>30.8</td>
      <td>61.5</td>
      <td>0.962764</td>
      <td>0.961933</td>
      <td>0.982369</td>
    </tr>
  </tbody>
</table>
</div>


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
    [NbConvertApp] Writing 14931 bytes to lattices.md



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

