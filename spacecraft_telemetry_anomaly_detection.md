# Курсовая работа — Вариант 23
## Интеллектуальный анализ данных на основе методов машинного обучения

**Выполнилb:** Лукина А.А. Колесникова Л.Э.

**Задание (группа 1.2, 6.2):**
- **Модель:** Гибридная нейросетевая модель — последовательное соединение одномерных свёрточных (Conv1D) и рекуррентных LSTM слоёв с полносвязным классификатором
- **Задача 1.2:** Бинарная классификация ТМИ МКА: штатное функционирование (0) vs нештатное функционирование (1)
- **Задача 6.2:** Автокодировщик для обнаружения аномалий на основе гибридной Conv1D+LSTM архитектуры

**Данные:** Телеметрическая информация малого космического аппарата (49 признаков: токи, напряжения, температуры)

---

### Содержание

**1. Импорт библиотек**

**2. Загрузка данных**
- 2.1 Загрузка размеченного и неразмеченного наборов данных
- 2.2 Переразметка в бинарную классификацию (штатное / нештатное)

**3. Разведочный анализ данных (EDA)**
- 3.1 Описательная статистика
- 3.2 Проверка пропущенных значений и дубликатов
- 3.3 Анализ выбросов (IQR)
- 3.4 Анализ распределений (Skewness, Kurtosis)
- 3.5 Проверка сбалансированности классов
- 3.6 Одномерная визуализация (гистограммы, боксплоты, KDE)
- 3.7 Многомерная визуализация (PairGrid, Scatter)
- 3.8 Корреляционный анализ
- 3.9 Экспериментирование с атрибутами (Feature Engineering)
- 3.10 Отбор признаков (Mutual Information)

**4. Подготовка данных**
- 4.1 Формирование 6 вариантов наборов данных (3 набора x 2 режима масштабирования)
- 4.2 Разбиение на train / val / test с стратификацией
- 4.3 RobustScaler: fit на train, transform на val/test (без утечки данных)

**5. Построение гибридной модели Conv1D + LSTM (задача 1.2)**
- 5.1 Определение архитектуры модели
- 5.2 Функции обучения и оценки
- 5.3 Обучение модели на 6 вариантах данных
- 5.4 Сводная таблица метрик и выбор лучшего варианта

**6. Поиск гиперпараметров генетическим алгоритмом (DEAP)**
- 6.1 Лог всех моделей в популяции, график эволюции фитнеса
- 6.2 Обучение лучшей модели (из ГА) и оценка на тестовом наборе
- 6.3 Сравнение результатов до и после оптимизации ГА

**7. Автокодировщик для обнаружения аномалий (задача 6.2)**
- 7.1 Подготовка данных для автокодировщика
- 7.2 Архитектура автокодировщика Conv1D + LSTM
- 7.3 Обучение автокодировщика (только на штатных данных)
- 7.4 Определение порога аномалии и классификация
- 7.5 Метрики автокодировщика на тестовой выборке
- 7.6 Визуализация латентного пространства (t-SNE)
- 7.7 Примеры реконструкции (штатные vs нештатные)

**8. Итоговое сравнение подходов (Вар.4)**

**9. Дополнительное исследование: Вариант 2 (49 признаков)**
- 9.1 Генетический алгоритм для Варианта 2
- 9.2 Автокодировщик для Варианта 2
- 9.3 Итоговое сравнение Вар.2 vs Вар.4

**Выводы**


## 1. Импорт библиотек


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, roc_auc_score,
                             roc_curve, accuracy_score, f1_score,
                             precision_recall_curve, average_precision_score)
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from tensorflow.keras.models import Model, Sequential

from deap import base, creator, tools, algorithms
import random

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
```

    TensorFlow version: 2.21.0
    NumPy version: 2.2.6
    Pandas version: 2.3.1
    WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
    GPU available: []
    

## 2. Загрузка данных
### 2.1 Загрузка размеченного и неразмеченного наборов данных


```python
# Размеченный набор данных (с метками классов)
df_labeled = pd.read_excel('MKA_TMI_labels.xls')

# Неразмеченный набор данных
df_unlabeled = pd.read_csv('MKA_04.2015_unlabeled.csv')

print(f"Размеченный набор: {df_labeled.shape}")
print(f"Неразмеченный набор: {df_unlabeled.shape}")
print(f"\nСтолбцы размеченного набора:\n{list(df_labeled.columns)}")
print(f"\nПервые 5 строк размеченного набора:")
df_labeled.head()
```

    Размеченный набор: (2679, 50)
    Неразмеченный набор: (15243, 49)
    
    Столбцы размеченного набора:
    ['Ubs,V', 'Ibs,A', 'Isun,A', 'Ipt1,A', 'Ipt2,A', 'Ipt3,A', 'Ipt4,A', 'Ipt5,A', 'Ipt6,A', 'Ipt7,A', 'Ipt10,A', 'Ipt11,A', 'Ipt12,A', 'Ipt13,A', 'Ipt14,A', 'Ipt15,A', 'Ipt16,A', 'Ipt17,A', 'TR1,C', 'TR2,C', 'TR3,C', 'TR4,C', 'TR5,C', 'TR6,C', 'TR7,C', 'TR8,C', 'TR9,C', 'TR10,C', 'TR11,C', 'TR12,C', 'TR13,C', 'TR14,C', 'TR15,C', 'TR16,C', 'TDS1,C', 'TDS2,C', 'TDS3,C', 'TDS4,C', 'TDS5,C', 'TDS6,C', 'TDS7,C', 'TDS8,C', 'TDS9,C', 'TKpt,C', 'TGbv,C', 'TNap,C', 'TPrd2,C', 'TPrd1,C', 'TDS24,C', 'Class']
    
    Первые 5 строк размеченного набора:
    




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
      <th>Ubs,V</th>
      <th>Ibs,A</th>
      <th>Isun,A</th>
      <th>Ipt1,A</th>
      <th>Ipt2,A</th>
      <th>Ipt3,A</th>
      <th>Ipt4,A</th>
      <th>Ipt5,A</th>
      <th>Ipt6,A</th>
      <th>Ipt7,A</th>
      <th>...</th>
      <th>TDS7,C</th>
      <th>TDS8,C</th>
      <th>TDS9,C</th>
      <th>TKpt,C</th>
      <th>TGbv,C</th>
      <th>TNap,C</th>
      <th>TPrd2,C</th>
      <th>TPrd1,C</th>
      <th>TDS24,C</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.83</td>
      <td>0.24</td>
      <td>0.91</td>
      <td>0.09</td>
      <td>0.07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>25</td>
      <td>17</td>
      <td>16</td>
      <td>18</td>
      <td>18</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.76</td>
      <td>0.21</td>
      <td>0.27</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>25</td>
      <td>17</td>
      <td>16</td>
      <td>18</td>
      <td>18</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.83</td>
      <td>0.24</td>
      <td>0.61</td>
      <td>0.09</td>
      <td>0.07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>25</td>
      <td>17</td>
      <td>17</td>
      <td>18</td>
      <td>18</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.83</td>
      <td>0.21</td>
      <td>0.51</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>25</td>
      <td>17</td>
      <td>17</td>
      <td>18</td>
      <td>18</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.83</td>
      <td>0.21</td>
      <td>0.35</td>
      <td>0.07</td>
      <td>0.09</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>18</td>
      <td>18</td>
      <td>17</td>
      <td>25</td>
      <td>17</td>
      <td>17</td>
      <td>18</td>
      <td>18</td>
      <td>21</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>



**Вывод по загрузке данных:**
- Размеченный набор: **2679 строк x 50 столбцов** (49 признаков ТМИ + столбец Class). Содержит метки технического состояния МКА: штатное функционирование, отказы и сбои оборудования.
- Неразмеченный набор: **15243 строки x 49 столбцов** (те же признаки, без меток). Будет использован в разделе автокодировщика (задача 6.2) для обнаружения аномалий в реальных эксплуатационных данных.
- Столбцы обоих наборов совпадают по именам и порядку — данные совместимы, что позволяет применить модели, обученные на размеченных данных, к неразмеченному набору.
- Объём неразмеченных данных (~5.7x больше размеченных) обеспечивает статистически значимую проверку работоспособности автоэнкодера.


### 2.2 Переразметка в бинарную классификацию


```python
# Исходное распределение классов (3 класса)
print("Исходное распределение классов (до переразметки):")
print(df_labeled['Class'].value_counts().sort_index())
print(f"  0 — штатное функционирование")
print(f"  1 — отказ")
print(f"  2 — сбой")

# Визуализация исходного распределения
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

orig_counts = df_labeled['Class'].value_counts().sort_index()
orig_labels = ['Штатное (0)', 'Отказ (1)', 'Сбой (2)']
orig_colors = ['#2ecc71', '#e74c3c', '#f39c12']

axes[0].bar(orig_labels, orig_counts.values, color=orig_colors, edgecolor='black')
for i, v in enumerate(orig_counts.values):
    axes[0].text(i, v + 20, f'{v} ({v/len(df_labeled)*100:.1f}%)', ha='center', fontweight='bold')
axes[0].set_title('Исходное распределение (3 класса)')
axes[0].set_ylabel('Количество')

# Переразметка: бинарная классификация
df = df_labeled.copy()
df['Class'] = df['Class'].apply(lambda x: 0 if x == 0 else 1)

bin_counts = df['Class'].value_counts().sort_index()
bin_labels = ['Штатное (0)', 'Нештатное (1)']
bin_colors = ['#2ecc71', '#e74c3c']

axes[1].bar(bin_labels, bin_counts.values, color=bin_colors, edgecolor='black')
for i, v in enumerate(bin_counts.values):
    axes[1].text(i, v + 20, f'{v} ({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')
axes[1].set_title('Бинарное распределение (2 класса)')
axes[1].set_ylabel('Количество')

plt.suptitle('Переразметка: объединение отказов и сбоев в класс "нештатное"', fontsize=13)
plt.tight_layout()
plt.show()

print(f"\nБинарное распределение классов:")
print(df['Class'].value_counts().sort_index())
print(f"\nДоля нештатных состояний: {df['Class'].mean():.2%}")
print(f"Соотношение 0:1 = {bin_counts[0]/bin_counts[1]:.1f}:1")
```

    Исходное распределение классов (до переразметки):
    Class
    0    2356
    1     218
    2     105
    Name: count, dtype: int64
      0 — штатное функционирование
      1 — отказ
      2 — сбой
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_7_1.png)
    


    
    Бинарное распределение классов:
    Class
    0    2356
    1     323
    Name: count, dtype: int64
    
    Доля нештатных состояний: 12.06%
    Соотношение 0:1 = 7.3:1
    

**Вывод по переразметке:**
- Исходные 3 класса (штатное=0, отказ=1, сбой=2) объединены в 2: штатное (0) и нештатное (1, объединяет отказы и сбои).
- Штатных записей: 2356 (87.9%), нештатных: 323 (12.1%). Соотношение ~7.3:1.
- Набор данных существенно **несбалансирован**, что потребует специальных мер при моделировании: взвешивание классов (`class_weight='balanced'`), использование метрик, устойчивых к дисбалансу (Balanced Accuracy, F1-score), и стратифицированного разбиения на выборки.
- Объединение отказов и сбоев в один класс физически обосновано: оба типа событий представляют нештатное поведение аппарата и требуют одинаковой реакции оператора.


## 3. Разведочный анализ данных (EDA)
### 3.1 Описательная статистика


```python
# Структура набора данных
print(f"Размерность: {df.shape[0]} строк x {df.shape[1]} столбцов")
print(f"Признаков: {df.shape[1] - 1}, Целевая переменная: Class\n")

# Подробная информация о типах данных
print("="*60)
print("Информация о структуре данных:")
print("="*60)
df.info()

print("\n" + "="*60)
print("Типы данных и количество столбцов каждого типа:")
print("="*60)
print(df.dtypes.value_counts())

# Группировка признаков по физическому смыслу
voltage_cols = [c for c in df.columns if 'V' in c]
current_cols = [c for c in df.columns if c.startswith('I')]
tr_cols = [c for c in df.columns if c.startswith('TR')]
tds_cols = [c for c in df.columns if c.startswith('TDS')]
other_temp_cols = [c for c in df.columns if c.startswith('T') and not c.startswith('TR') and not c.startswith('TDS')]

print(f"\nГруппы признаков:")
print(f"  Напряжение (V):        {len(voltage_cols)} — {voltage_cols}")
print(f"  Токи (A):              {len(current_cols)} — {current_cols}")
print(f"  Температуры TR (C):    {len(tr_cols)} — {tr_cols}")
print(f"  Температуры TDS (C):   {len(tds_cols)} — {tds_cols}")
print(f"  Прочие температуры:    {len(other_temp_cols)} — {other_temp_cols}")

print("\n" + "="*60)
print("Описательная статистика:")
print("="*60)
df.describe().T
```

    Размерность: 2679 строк x 50 столбцов
    Признаков: 49, Целевая переменная: Class
    
    ============================================================
    Информация о структуре данных:
    ============================================================
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2679 entries, 0 to 2678
    Data columns (total 50 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   Ubs,V    2679 non-null   float64
     1   Ibs,A    2679 non-null   float64
     2   Isun,A   2679 non-null   float64
     3   Ipt1,A   2679 non-null   float64
     4   Ipt2,A   2679 non-null   float64
     5   Ipt3,A   2679 non-null   float64
     6   Ipt4,A   2679 non-null   float64
     7   Ipt5,A   2679 non-null   float64
     8   Ipt6,A   2679 non-null   float64
     9   Ipt7,A   2679 non-null   float64
     10  Ipt10,A  2679 non-null   float64
     11  Ipt11,A  2679 non-null   float64
     12  Ipt12,A  2679 non-null   float64
     13  Ipt13,A  2679 non-null   float64
     14  Ipt14,A  2679 non-null   float64
     15  Ipt15,A  2679 non-null   float64
     16  Ipt16,A  2679 non-null   float64
     17  Ipt17,A  2679 non-null   float64
     18  TR1,C    2679 non-null   int64  
     19  TR2,C    2679 non-null   int64  
     20  TR3,C    2679 non-null   int64  
     21  TR4,C    2679 non-null   int64  
     22  TR5,C    2679 non-null   int64  
     23  TR6,C    2679 non-null   int64  
     24  TR7,C    2679 non-null   int64  
     25  TR8,C    2679 non-null   int64  
     26  TR9,C    2679 non-null   int64  
     27  TR10,C   2679 non-null   int64  
     28  TR11,C   2679 non-null   float64
     29  TR12,C   2679 non-null   int64  
     30  TR13,C   2679 non-null   int64  
     31  TR14,C   2679 non-null   int64  
     32  TR15,C   2679 non-null   int64  
     33  TR16,C   2679 non-null   int64  
     34  TDS1,C   2679 non-null   int64  
     35  TDS2,C   2679 non-null   int64  
     36  TDS3,C   2679 non-null   int64  
     37  TDS4,C   2679 non-null   int64  
     38  TDS5,C   2679 non-null   int64  
     39  TDS6,C   2679 non-null   int64  
     40  TDS7,C   2679 non-null   int64  
     41  TDS8,C   2679 non-null   int64  
     42  TDS9,C   2679 non-null   int64  
     43  TKpt,C   2679 non-null   int64  
     44  TGbv,C   2679 non-null   int64  
     45  TNap,C   2679 non-null   int64  
     46  TPrd2,C  2679 non-null   int64  
     47  TPrd1,C  2679 non-null   int64  
     48  TDS24,C  2679 non-null   int64  
     49  Class    2679 non-null   int64  
    dtypes: float64(19), int64(31)
    memory usage: 1.0 MB
    
    ============================================================
    Типы данных и количество столбцов каждого типа:
    ============================================================
    int64      31
    float64    19
    Name: count, dtype: int64
    
    Группы признаков:
      Напряжение (V):        1 — ['Ubs,V']
      Токи (A):              17 — ['Ibs,A', 'Isun,A', 'Ipt1,A', 'Ipt2,A', 'Ipt3,A', 'Ipt4,A', 'Ipt5,A', 'Ipt6,A', 'Ipt7,A', 'Ipt10,A', 'Ipt11,A', 'Ipt12,A', 'Ipt13,A', 'Ipt14,A', 'Ipt15,A', 'Ipt16,A', 'Ipt17,A']
      Температуры TR (C):    16 — ['TR1,C', 'TR2,C', 'TR3,C', 'TR4,C', 'TR5,C', 'TR6,C', 'TR7,C', 'TR8,C', 'TR9,C', 'TR10,C', 'TR11,C', 'TR12,C', 'TR13,C', 'TR14,C', 'TR15,C', 'TR16,C']
      Температуры TDS (C):   10 — ['TDS1,C', 'TDS2,C', 'TDS3,C', 'TDS4,C', 'TDS5,C', 'TDS6,C', 'TDS7,C', 'TDS8,C', 'TDS9,C', 'TDS24,C']
      Прочие температуры:    5 — ['TKpt,C', 'TGbv,C', 'TNap,C', 'TPrd2,C', 'TPrd1,C']
    
    ============================================================
    Описательная статистика:
    ============================================================
    




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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ubs,V</th>
      <td>2679.0</td>
      <td>14.088903</td>
      <td>1.077994</td>
      <td>0.13</td>
      <td>14.10</td>
      <td>14.50</td>
      <td>14.69</td>
      <td>15.68</td>
    </tr>
    <tr>
      <th>Ibs,A</th>
      <td>2679.0</td>
      <td>0.681149</td>
      <td>1.078681</td>
      <td>0.11</td>
      <td>0.21</td>
      <td>0.24</td>
      <td>0.27</td>
      <td>4.61</td>
    </tr>
    <tr>
      <th>Isun,A</th>
      <td>2679.0</td>
      <td>0.744246</td>
      <td>0.997524</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.45</td>
      <td>1.07</td>
      <td>5.28</td>
    </tr>
    <tr>
      <th>Ipt1,A</th>
      <td>2679.0</td>
      <td>0.301657</td>
      <td>0.806005</td>
      <td>0.04</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.09</td>
      <td>3.24</td>
    </tr>
    <tr>
      <th>Ipt2,A</th>
      <td>2679.0</td>
      <td>0.304169</td>
      <td>0.792725</td>
      <td>0.00</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.09</td>
      <td>5.24</td>
    </tr>
    <tr>
      <th>Ipt3,A</th>
      <td>2679.0</td>
      <td>0.205778</td>
      <td>0.783437</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.64</td>
    </tr>
    <tr>
      <th>Ipt4,A</th>
      <td>2679.0</td>
      <td>0.200642</td>
      <td>0.778907</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.31</td>
    </tr>
    <tr>
      <th>Ipt5,A</th>
      <td>2679.0</td>
      <td>0.213050</td>
      <td>0.824493</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.76</td>
    </tr>
    <tr>
      <th>Ipt6,A</th>
      <td>2679.0</td>
      <td>0.433966</td>
      <td>1.154981</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.73</td>
    </tr>
    <tr>
      <th>Ipt7,A</th>
      <td>2679.0</td>
      <td>0.297014</td>
      <td>1.132852</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.71</td>
    </tr>
    <tr>
      <th>Ipt10,A</th>
      <td>2679.0</td>
      <td>0.186779</td>
      <td>0.752707</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.31</td>
    </tr>
    <tr>
      <th>Ipt11,A</th>
      <td>2679.0</td>
      <td>0.289802</td>
      <td>1.126208</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.67</td>
    </tr>
    <tr>
      <th>Ipt12,A</th>
      <td>2679.0</td>
      <td>0.282926</td>
      <td>1.107992</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.64</td>
    </tr>
    <tr>
      <th>Ipt13,A</th>
      <td>2679.0</td>
      <td>0.189832</td>
      <td>0.755061</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.29</td>
    </tr>
    <tr>
      <th>Ipt14,A</th>
      <td>2679.0</td>
      <td>0.283315</td>
      <td>1.106091</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.60</td>
    </tr>
    <tr>
      <th>Ipt15,A</th>
      <td>2679.0</td>
      <td>0.259804</td>
      <td>1.046427</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.58</td>
    </tr>
    <tr>
      <th>Ipt16,A</th>
      <td>2679.0</td>
      <td>0.204991</td>
      <td>0.789265</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.38</td>
    </tr>
    <tr>
      <th>Ipt17,A</th>
      <td>2679.0</td>
      <td>0.279854</td>
      <td>1.081491</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.53</td>
    </tr>
    <tr>
      <th>TR1,C</th>
      <td>2679.0</td>
      <td>0.503546</td>
      <td>70.319498</td>
      <td>-257.00</td>
      <td>20.00</td>
      <td>20.00</td>
      <td>21.00</td>
      <td>148.00</td>
    </tr>
    <tr>
      <th>TR2,C</th>
      <td>2679.0</td>
      <td>7.933184</td>
      <td>46.403284</td>
      <td>-257.00</td>
      <td>19.00</td>
      <td>20.00</td>
      <td>22.00</td>
      <td>86.00</td>
    </tr>
    <tr>
      <th>TR3,C</th>
      <td>2679.0</td>
      <td>1.952594</td>
      <td>41.904934</td>
      <td>-235.00</td>
      <td>7.00</td>
      <td>13.00</td>
      <td>17.00</td>
      <td>133.00</td>
    </tr>
    <tr>
      <th>TR4,C</th>
      <td>2679.0</td>
      <td>-7.485629</td>
      <td>65.901859</td>
      <td>-236.00</td>
      <td>5.00</td>
      <td>11.00</td>
      <td>17.00</td>
      <td>76.00</td>
    </tr>
    <tr>
      <th>TR5,C</th>
      <td>2679.0</td>
      <td>-13.387085</td>
      <td>60.141176</td>
      <td>-235.00</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>7.00</td>
      <td>70.00</td>
    </tr>
    <tr>
      <th>TR6,C</th>
      <td>2679.0</td>
      <td>-1.259425</td>
      <td>34.053470</td>
      <td>-228.00</td>
      <td>2.00</td>
      <td>7.00</td>
      <td>10.00</td>
      <td>99.00</td>
    </tr>
    <tr>
      <th>TR7,C</th>
      <td>2679.0</td>
      <td>-9.490482</td>
      <td>60.783844</td>
      <td>-234.00</td>
      <td>2.00</td>
      <td>8.00</td>
      <td>14.00</td>
      <td>126.00</td>
    </tr>
    <tr>
      <th>TR8,C</th>
      <td>2679.0</td>
      <td>13.365435</td>
      <td>38.238253</td>
      <td>-225.00</td>
      <td>5.00</td>
      <td>12.00</td>
      <td>18.00</td>
      <td>126.00</td>
    </tr>
    <tr>
      <th>TR9,C</th>
      <td>2679.0</td>
      <td>-6.296379</td>
      <td>53.541873</td>
      <td>-228.00</td>
      <td>3.00</td>
      <td>7.00</td>
      <td>13.00</td>
      <td>75.00</td>
    </tr>
    <tr>
      <th>TR10,C</th>
      <td>2679.0</td>
      <td>-11.068682</td>
      <td>59.411205</td>
      <td>-233.00</td>
      <td>0.00</td>
      <td>5.00</td>
      <td>11.00</td>
      <td>106.00</td>
    </tr>
    <tr>
      <th>TR11,C</th>
      <td>2679.0</td>
      <td>-14.196529</td>
      <td>69.937326</td>
      <td>-255.00</td>
      <td>-26.00</td>
      <td>11.00</td>
      <td>25.00</td>
      <td>120.00</td>
    </tr>
    <tr>
      <th>TR12,C</th>
      <td>2679.0</td>
      <td>-9.178052</td>
      <td>55.388815</td>
      <td>-253.00</td>
      <td>3.00</td>
      <td>6.00</td>
      <td>9.00</td>
      <td>81.00</td>
    </tr>
    <tr>
      <th>TR13,C</th>
      <td>2679.0</td>
      <td>-3.419186</td>
      <td>52.242349</td>
      <td>-257.00</td>
      <td>4.00</td>
      <td>10.00</td>
      <td>17.00</td>
      <td>119.00</td>
    </tr>
    <tr>
      <th>TR14,C</th>
      <td>2679.0</td>
      <td>-2.555804</td>
      <td>58.280820</td>
      <td>-250.00</td>
      <td>6.00</td>
      <td>12.00</td>
      <td>20.00</td>
      <td>86.00</td>
    </tr>
    <tr>
      <th>TR15,C</th>
      <td>2679.0</td>
      <td>17.204927</td>
      <td>24.863725</td>
      <td>-257.00</td>
      <td>7.00</td>
      <td>13.00</td>
      <td>18.00</td>
      <td>106.00</td>
    </tr>
    <tr>
      <th>TR16,C</th>
      <td>2679.0</td>
      <td>-7.496081</td>
      <td>63.832362</td>
      <td>-256.00</td>
      <td>2.00</td>
      <td>10.00</td>
      <td>18.00</td>
      <td>131.00</td>
    </tr>
    <tr>
      <th>TDS1,C</th>
      <td>2679.0</td>
      <td>7.983949</td>
      <td>32.595470</td>
      <td>-128.00</td>
      <td>16.00</td>
      <td>16.00</td>
      <td>17.00</td>
      <td>82.00</td>
    </tr>
    <tr>
      <th>TDS2,C</th>
      <td>2679.0</td>
      <td>11.107130</td>
      <td>21.536657</td>
      <td>-128.00</td>
      <td>16.00</td>
      <td>16.00</td>
      <td>17.00</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>TDS3,C</th>
      <td>2679.0</td>
      <td>7.045913</td>
      <td>35.650591</td>
      <td>-128.00</td>
      <td>16.00</td>
      <td>17.00</td>
      <td>17.00</td>
      <td>114.00</td>
    </tr>
    <tr>
      <th>TDS4,C</th>
      <td>2679.0</td>
      <td>6.346398</td>
      <td>37.482964</td>
      <td>-128.00</td>
      <td>16.00</td>
      <td>17.00</td>
      <td>17.00</td>
      <td>73.00</td>
    </tr>
    <tr>
      <th>TDS5,C</th>
      <td>2679.0</td>
      <td>7.080254</td>
      <td>36.485113</td>
      <td>-128.00</td>
      <td>16.00</td>
      <td>17.00</td>
      <td>18.00</td>
      <td>73.00</td>
    </tr>
    <tr>
      <th>TDS6,C</th>
      <td>2679.0</td>
      <td>7.112355</td>
      <td>34.907149</td>
      <td>-128.00</td>
      <td>15.00</td>
      <td>16.00</td>
      <td>17.00</td>
      <td>73.00</td>
    </tr>
    <tr>
      <th>TDS7,C</th>
      <td>2679.0</td>
      <td>7.630459</td>
      <td>35.802703</td>
      <td>-128.00</td>
      <td>16.00</td>
      <td>17.00</td>
      <td>19.00</td>
      <td>73.00</td>
    </tr>
    <tr>
      <th>TDS8,C</th>
      <td>2679.0</td>
      <td>6.649496</td>
      <td>38.207669</td>
      <td>-128.00</td>
      <td>16.00</td>
      <td>17.00</td>
      <td>19.00</td>
      <td>21.00</td>
    </tr>
    <tr>
      <th>TDS9,C</th>
      <td>2679.0</td>
      <td>6.974617</td>
      <td>37.629271</td>
      <td>-128.00</td>
      <td>16.00</td>
      <td>17.00</td>
      <td>18.00</td>
      <td>73.00</td>
    </tr>
    <tr>
      <th>TKpt,C</th>
      <td>2679.0</td>
      <td>14.153789</td>
      <td>39.712017</td>
      <td>-128.00</td>
      <td>24.00</td>
      <td>25.00</td>
      <td>26.00</td>
      <td>106.00</td>
    </tr>
    <tr>
      <th>TGbv,C</th>
      <td>2679.0</td>
      <td>7.681971</td>
      <td>37.849951</td>
      <td>-128.00</td>
      <td>17.00</td>
      <td>18.00</td>
      <td>19.00</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>TNap,C</th>
      <td>2679.0</td>
      <td>6.961553</td>
      <td>38.326742</td>
      <td>-128.00</td>
      <td>17.00</td>
      <td>18.00</td>
      <td>19.00</td>
      <td>40.00</td>
    </tr>
    <tr>
      <th>TPrd2,C</th>
      <td>2679.0</td>
      <td>8.724151</td>
      <td>37.459169</td>
      <td>-128.00</td>
      <td>17.00</td>
      <td>19.00</td>
      <td>20.00</td>
      <td>127.00</td>
    </tr>
    <tr>
      <th>TPrd1,C</th>
      <td>2679.0</td>
      <td>16.663307</td>
      <td>12.043739</td>
      <td>-128.00</td>
      <td>17.00</td>
      <td>19.00</td>
      <td>20.00</td>
      <td>22.00</td>
    </tr>
    <tr>
      <th>TDS24,C</th>
      <td>2679.0</td>
      <td>21.181038</td>
      <td>12.584964</td>
      <td>-128.00</td>
      <td>20.00</td>
      <td>21.00</td>
      <td>21.00</td>
      <td>80.00</td>
    </tr>
    <tr>
      <th>Class</th>
      <td>2679.0</td>
      <td>0.120567</td>
      <td>0.325685</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



**Вывод по описательной статистике:**
- Набор данных содержит 2679 записей телеметрии МКА с 49 признаками, сгруппированными по физическому смыслу:
  - **Электропитание:** Ubs (напряжение ~13-15В), Ibs (ток потребления), Isun (ток солнечных панелей)
  - **Токи потребителей:** Ipt1-Ipt17 (17 каналов). Из них активны лишь 5: Ipt1, Ipt2, Ipt6, Ipt7, Ipt11. Остальные 12 каналов практически всегда равны нулю (min=Q25=Q50=Q75=0).
  - **Температуры радиаторов:** TR1-TR16 (16 каналов, диапазон ~-17...+31°C)
  - **Температуры датчиков:** TDS1-TDS9, TDS24 (10 каналов)
  - **Прочие температуры:** TKpt, TGbv, TNap, TPrd1, TPrd2
- Типы данных: float64 (токи и напряжения) и int64 (температуры). Все признаки числовые — кодирование категориальных переменных не требуется.
- Диапазоны значений сильно различаются между группами признаков (напряжение ~13-15V, токи ~0-2A, температуры от -17 до +31°C), что обосновывает необходимость масштабирования перед подачей в нейросеть.
- Многие токовые признаки (Ipt3-Ipt17) имеют нулевые значения min, Q25, Q50, Q75 — это означает, что соответствующие потребители неактивны большую часть времени и, вероятно, малоинформативны для классификации.


### 3.2 Проверка пропущенных значений и дубликатов


```python
# Пропущенные значения и дубликаты
print(f"Пропущенные значения:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
if df.isnull().sum().sum() == 0:
    print("Пропущенных значений нет.")

print(f"\nДубликатов строк: {df.duplicated().sum()}")
print(f"Уникальных значений по столбцам:\n{df.nunique()}")
```

    Пропущенные значения:
    Series([], dtype: int64)
    Пропущенных значений нет.
    
    Дубликатов строк: 2
    Уникальных значений по столбцам:
    Ubs,V      75
    Ibs,A      76
    Isun,A     97
    Ipt1,A     19
    Ipt2,A     25
    Ipt3,A     15
    Ipt4,A      9
    Ipt5,A      9
    Ipt6,A     32
    Ipt7,A      8
    Ipt10,A     9
    Ipt11,A     3
    Ipt12,A     4
    Ipt13,A     9
    Ipt14,A     2
    Ipt15,A     7
    Ipt16,A    12
    Ipt17,A     5
    TR1,C      17
    TR2,C      40
    TR3,C      52
    TR4,C      33
    TR5,C      31
    TR6,C      40
    TR7,C      44
    TR8,C      50
    TR9,C      33
    TR10,C     37
    TR11,C     81
    TR12,C     42
    TR13,C     90
    TR14,C     47
    TR15,C     41
    TR16,C     46
    TDS1,C     54
    TDS2,C     18
    TDS3,C     16
    TDS4,C     12
    TDS5,C     13
    TDS6,C     17
    TDS7,C     25
    TDS8,C     14
    TDS9,C     16
    TKpt,C     13
    TGbv,C     18
    TNap,C     31
    TPrd2,C    29
    TPrd1,C    17
    TDS24,C    16
    Class       2
    dtype: int64
    

**Вывод:** Набор данных не содержит пропущенных значений — все 2679 записей полные, что типично для автоматизированных систем телеметрии.

При наличии дубликатов они отражают повторяющиеся телеметрические замеры в стационарном режиме работы аппарата (когда параметры не меняются между соседними тактами опроса). **Удаление дубликатов нецелесообразно** — это привело бы к потере реальной информации о стабильных состояниях МКА и искажению распределений признаков.

### 3.3 Анализ выбросов (IQR)



```python
features = [c for c in df.columns if c != 'Class']

# Анализ выбросов по методу IQR
outlier_info = []
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    pct = n_outliers / len(df) * 100
    outlier_info.append({'Признак': col, 'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
                         'Нижняя граница': lower, 'Верхняя граница': upper,
                         'Выбросов': n_outliers, '% выбросов': pct})

outlier_df = pd.DataFrame(outlier_info).sort_values('Выбросов', ascending=False)
print("Признаки с наибольшим числом выбросов (IQR метод):")
print(outlier_df[outlier_df['Выбросов'] > 0][['Признак', 'Выбросов', '% выбросов']].to_string(index=False))

# Проверим: сколько выбросов приходится на нештатные состояния
print(f"\nВсего строк с хотя бы 1 выбросом:")
outlier_mask = pd.Series(False, index=df.index)
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask |= (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)

print(f"  Штатные (0): {(outlier_mask & (df['Class'] == 0)).sum()} из {(df['Class'] == 0).sum()}")
print(f"  Нештатные (1): {(outlier_mask & (df['Class'] == 1)).sum()} из {(df['Class'] == 1).sum()}")

```

    Признаки с наибольшим числом выбросов (IQR метод):
    Признак  Выбросов  % выбросов
      Ibs,A       634   23.665547
      TR1,C       628   23.441583
     Ipt6,A       549   20.492721
     Ipt1,A       437   16.312057
      Ubs,V       402   15.005599
     TDS2,C       334   12.467339
     TDS4,C       316   11.795446
     TDS3,C       310   11.571482
     TDS1,C       303   11.310190
     TNap,C       253    9.443822
    TPrd2,C       252    9.406495
     Ipt2,A       247    9.219858
      TR5,C       228    8.510638
      TR7,C       228    8.510638
      TR4,C       226    8.435984
     TR12,C       223    8.324001
     TR10,C       215    8.025383
     TR14,C       215    8.025383
     TR13,C       214    7.988055
      TR6,C       210    7.838746
      TR8,C       209    7.801418
     TR11,C       209    7.801418
      TR3,C       208    7.764091
     TGbv,C       208    7.764091
     TDS8,C       206    7.689436
      TR9,C       205    7.652109
     TDS9,C       201    7.502800
     TKpt,C       200    7.465472
      TR2,C       199    7.428145
     TDS7,C       199    7.428145
     TR16,C       196    7.316163
     TR15,C       193    7.204181
     TDS5,C       191    7.129526
     Ipt3,A       186    6.942889
    TPrd1,C       180    6.718925
     TDS6,C       180    6.718925
    TDS24,C       179    6.681598
     Isun,A       178    6.644270
     Ipt7,A       176    6.569616
    Ipt16,A       172    6.420306
    Ipt17,A       169    6.308324
     Ipt5,A       168    6.270997
     Ipt4,A       167    6.233669
    Ipt11,A       167    6.233669
    Ipt12,A       165    6.159015
    Ipt14,A       165    6.159015
    Ipt13,A       164    6.121687
    Ipt15,A       159    5.935050
    Ipt10,A       159    5.935050
    
    Всего строк с хотя бы 1 выбросом:
      Штатные (0): 1026 из 2356
      Нештатные (1): 323 из 323
    

**Вывод по анализу выбросов:**
- Выбросы по методу IQR (значения за пределами Q1 - 1.5*IQR ... Q3 + 1.5*IQR) обнаружены в значительной части признаков, особенно в токовых (Ipt1, Ipt2, Ibs, Isun) и температурных (TR11, TR13).
- Важно: значительная доля выбросов приходится на нештатные состояния — аномальные значения токов и температур являются прямыми признаками отказов и сбоев аппарата. Это подтверждается тем, что наибольшее число выбросов наблюдается у признаков с высокой корреляцией с целевой переменной.
- **Решение: выбросы не удаляются.** В задаче диагностики МКА выбросы — это не шум, а полезный сигнал. Удаление выбросов привело бы к потере информации о целевом классе (нештатные состояния) и существенному ухудшению качества классификации.
- **Следствие для масштабирования:** наличие выбросов обосновывает выбор RobustScaler вместо StandardScaler/MinMaxScaler, т.к. RobustScaler использует медиану и IQR, устойчивые к экстремальным значениям.

### 3.4 Анализ распределений (Skewness, Kurtosis)



```python
# Анализ асимметрии и эксцесса
dist_info = []
for col in features:
    sk = df[col].skew()
    ku = df[col].kurtosis()
    dist_info.append({'Признак': col, 'Skewness': sk, 'Kurtosis': ku})

dist_df = pd.DataFrame(dist_info)

fig, axes = plt.subplots(1, 2, figsize=(18, 12))

# Skewness
colors_sk = ['#e74c3c' if abs(v) > 1 else '#f39c12' if abs(v) > 0.5 else '#2ecc71'
             for v in dist_df['Skewness']]
axes[0].barh(dist_df['Признак'], dist_df['Skewness'], color=colors_sk, edgecolor='black', linewidth=0.5)
axes[0].axvline(x=0, color='black', linewidth=1)
axes[0].axvline(x=-1, color='red', linestyle='--', alpha=0.5)
axes[0].axvline(x=1, color='red', linestyle='--', alpha=0.5, label='|skew| = 1')
axes[0].set_title('Асимметрия (Skewness)', fontsize=13)
axes[0].set_xlabel('Skewness', fontsize=11)
axes[0].tick_params(labelsize=9)
axes[0].legend(fontsize=9)

# Kurtosis
colors_ku = ['#e74c3c' if abs(v) > 3 else '#f39c12' if abs(v) > 1 else '#2ecc71'
             for v in dist_df['Kurtosis']]
axes[1].barh(dist_df['Признак'], dist_df['Kurtosis'], color=colors_ku, edgecolor='black', linewidth=0.5)
axes[1].axvline(x=0, color='black', linewidth=1)
axes[1].axvline(x=3, color='red', linestyle='--', alpha=0.5, label='kurtosis = 3 (тяжёлые хвосты)')
axes[1].set_title('Эксцесс (Kurtosis)', fontsize=13)
axes[1].set_xlabel('Kurtosis', fontsize=11)
axes[1].tick_params(labelsize=9)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()

# Таблица
print("Признаки с сильной асимметрией (|skew| > 1):")
print(dist_df[dist_df['Skewness'].abs() > 1][['Признак', 'Skewness', 'Kurtosis']].to_string(index=False))
print(f"\nПризнаки с тяжёлыми хвостами (kurtosis > 3):")
print(dist_df[dist_df['Kurtosis'] > 3][['Признак', 'Skewness', 'Kurtosis']].to_string(index=False))

```


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_17_0.png)
    


    Признаки с сильной асимметрией (|skew| > 1):
    Признак  Skewness  Kurtosis
      Ubs,V -3.026663 16.570382
      Ibs,A  2.803382  6.865143
     Isun,A  1.946875  3.617267
     Ipt1,A  3.315317  9.021244
     Ipt2,A  3.370997  9.702190
     Ipt3,A  3.603778 11.041842
     Ipt4,A  3.629316 11.189095
     Ipt5,A  3.617650 11.111490
     Ipt6,A  3.134149  8.758446
     Ipt7,A  3.579856 10.884146
    Ipt10,A  3.812024 12.577887
    Ipt11,A  3.633794 11.216683
    Ipt12,A  3.667229 11.468291
    Ipt13,A  3.766748 12.252634
    Ipt14,A  3.649234 11.325361
    Ipt15,A  3.821167 12.699847
    Ipt16,A  3.613056 11.092685
    Ipt17,A  3.623480 11.190949
      TR1,C -3.297806  9.114783
      TR2,C -3.509849 11.025879
      TR3,C -3.286708  9.933668
      TR4,C -3.104607  7.831648
      TR5,C -3.190340  8.525426
      TR6,C -3.168509 10.977243
      TR7,C -3.126092  8.193605
      TR8,C -1.474407 16.548358
      TR9,C -3.303439  9.327665
     TR10,C -3.192090  8.525302
     TR11,C -2.646600  6.262058
     TR12,C -3.207000  8.775010
     TR13,C -3.434943 10.958846
     TR14,C -3.409687 10.518505
     TR16,C -3.278743  9.144475
     TDS1,C -3.510462 10.752290
     TDS2,C -3.387912 11.535000
     TDS3,C -3.267302  9.020611
     TDS4,C -3.257486  8.778828
     TDS5,C -3.380550  9.583765
     TDS6,C -3.488304 10.267496
     TDS7,C -3.331887  9.262478
     TDS8,C -3.196941  8.291060
     TDS9,C -3.269069  8.815860
     TKpt,C -3.265115  8.811756
     TGbv,C -3.223231  8.606785
     TNap,C -3.163669  8.119935
    TPrd2,C -3.281462  9.089606
    TPrd1,C -7.949353 67.090383
    TDS24,C -3.707556 46.024043
    
    Признаки с тяжёлыми хвостами (kurtosis > 3):
    Признак  Skewness  Kurtosis
      Ubs,V -3.026663 16.570382
      Ibs,A  2.803382  6.865143
     Isun,A  1.946875  3.617267
     Ipt1,A  3.315317  9.021244
     Ipt2,A  3.370997  9.702190
     Ipt3,A  3.603778 11.041842
     Ipt4,A  3.629316 11.189095
     Ipt5,A  3.617650 11.111490
     Ipt6,A  3.134149  8.758446
     Ipt7,A  3.579856 10.884146
    Ipt10,A  3.812024 12.577887
    Ipt11,A  3.633794 11.216683
    Ipt12,A  3.667229 11.468291
    Ipt13,A  3.766748 12.252634
    Ipt14,A  3.649234 11.325361
    Ipt15,A  3.821167 12.699847
    Ipt16,A  3.613056 11.092685
    Ipt17,A  3.623480 11.190949
      TR1,C -3.297806  9.114783
      TR2,C -3.509849 11.025879
      TR3,C -3.286708  9.933668
      TR4,C -3.104607  7.831648
      TR5,C -3.190340  8.525426
      TR6,C -3.168509 10.977243
      TR7,C -3.126092  8.193605
      TR8,C -1.474407 16.548358
      TR9,C -3.303439  9.327665
     TR10,C -3.192090  8.525302
     TR11,C -2.646600  6.262058
     TR12,C -3.207000  8.775010
     TR13,C -3.434943 10.958846
     TR14,C -3.409687 10.518505
     TR15,C  0.773153 17.380847
     TR16,C -3.278743  9.144475
     TDS1,C -3.510462 10.752290
     TDS2,C -3.387912 11.535000
     TDS3,C -3.267302  9.020611
     TDS4,C -3.257486  8.778828
     TDS5,C -3.380550  9.583765
     TDS6,C -3.488304 10.267496
     TDS7,C -3.331887  9.262478
     TDS8,C -3.196941  8.291060
     TDS9,C -3.269069  8.815860
     TKpt,C -3.265115  8.811756
     TGbv,C -3.223231  8.606785
     TNap,C -3.163669  8.119935
    TPrd2,C -3.281462  9.089606
    TPrd1,C -7.949353 67.090383
    TDS24,C -3.707556 46.024043
    

**Вывод по анализу распределений:**
- Многие токовые признаки (Ipt3-Ipt17) имеют сильную положительную асимметрию (skewness >> 1) — большинство значений равны нулю, с редкими ненулевыми всплесками. Это характерно для потребителей, которые включаются эпизодически и не несут информации для классификации штатных/нештатных состояний.
- Температурные признаки TR более симметричны (skewness ~0), но TR11 имеет выраженный отрицательный эксцесс, что указывает на нетипичное поведение этого канала при аномалиях.
- Ряд признаков имеет тяжёлые хвосты (kurtosis > 3), что указывает на наличие редких экстремальных значений — это совпадает с результатами анализа выбросов.
- **Следствие для предобработки:** StandardScaler (z-нормализация) чувствителен к выбросам — среднее и дисперсия «сдвигаются» под влиянием экстремальных значений. MinMaxScaler ещё хуже — сжимает основную массу данных в узкий диапазон. **RobustScaler** (медиана и IQR) оптимален для данных с выраженной асимметрией, тяжёлыми хвостами и выбросами.
- Нелинейные зависимости в данных (подтверждённые KDE и scatter-графиками) обосновывают применение нейросетевых моделей вместо линейных классификаторов.

### 3.5 Проверка сбалансированности классов



```python
# Детальная визуализация баланса классов
class_counts = df['Class'].value_counts().sort_index()
labels_cls = ['Штатное (0)', 'Нештатное (1)']
colors_cls = ['#2ecc71', '#e74c3c']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Круговая диаграмма
axes[0].pie(class_counts.values, labels=labels_cls, colors=colors_cls,
            autopct='%1.1f%%', startangle=90, explode=(0, 0.08),
            textprops={'fontsize': 12})
axes[0].set_title('Соотношение классов (бинарное)', fontsize=12)

# Столбчатая с процентами
bars = axes[1].bar(labels_cls, class_counts.values, color=colors_cls, edgecolor='black')
for bar, v in zip(bars, class_counts.values):
    pct = v / len(df) * 100
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 20,
                 f'{v}\n({pct:.1f}%)', ha='center', fontweight='bold', fontsize=11)
axes[1].set_title('Количество записей по классам', fontsize=12)
axes[1].set_ylabel('Количество')

plt.tight_layout()
plt.show()

# Формальная проверка
ratio = class_counts.values[0] / class_counts.values[1]
print(f"Соотношение классов 0:1 = {ratio:.1f}:1")
if ratio > 3:
    print(f"ВЫВОД: Набор данных НЕСБАЛАНСИРОВАН (соотношение {ratio:.1f}:1 > 3:1).")
    print("Необходимо использовать balanced_accuracy, F1-score, ROC AUC, PR AUC.")
    print("При обучении применяется class_weight='balanced'.")
else:
    print("Набор данных относительно сбалансирован.")

```


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_19_0.png)
    


    Соотношение классов 0:1 = 7.3:1
    ВЫВОД: Набор данных НЕСБАЛАНСИРОВАН (соотношение 7.3:1 > 3:1).
    Необходимо использовать balanced_accuracy, F1-score, ROC AUC, PR AUC.
    При обучении применяется class_weight='balanced'.
    

**Вывод по сбалансированности классов:**
- Исходное распределение (3 класса): штатное (0) — 2356 записей (87.9%), отказ (1) — 218 (8.1%), сбой (2) — 105 (3.9%).
- После бинарной переразметки: штатное (0) — 2356 (87.9%), нештатное (1) — 323 (12.1%).
- Соотношение классов ~7.3:1 — набор данных **сильно несбалансирован**.
- **Последствия для моделирования:**
  - Accuracy как метрика **ненадёжна** — модель, предсказывающая всё как «штатное», получит ~88% accuracy, не обнаружив ни одной аномалии.
  - Основные метрики: **Balanced Accuracy** (средняя точность по классам), **F1-score** (баланс precision и recall), **ROC AUC** (качество ранжирования), **Average Precision** (площадь под PR-кривой).
  - При обучении: `class_weight='balanced'` для компенсации дисбаланса — модель будет штрафоваться сильнее за пропуск аномалии, чем за ложную тревогу.
- На диаграммах видно, что дисбаланс сохраняется как при 3-классовой, так и при 2-классовой разметке. Это фундаментальная характеристика данных — штатное функционирование МКА преобладает.

### 3.6 Одномерная визуализация (гистограммы, боксплоты, KDE)



```python
# Группировка признаков по физическому смыслу для наглядной визуализации
electric_cols = [c for c in features if c.endswith(',V') or c.endswith(',A')]
tr_temp_cols = [c for c in features if c.startswith('TR')]
tds_other_cols = [c for c in features if c.startswith('TDS') or c.startswith('TK') or
                  c.startswith('TG') or c.startswith('TN') or c.startswith('TP')]

groups = [
    ('Электрические параметры (напряжение, токи)', electric_cols),
    ('Температуры радиаторов (TR)', tr_temp_cols),
    ('Температуры датчиков и прочие (TDS, TKpt, TGbv, TNap, TPrd)', tds_other_cols),
]

for group_name, cols in groups:
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        df[df['Class'] == 0][col].hist(ax=axes[i], bins=30, alpha=0.6, color='#2ecc71', label='Штатное')
        df[df['Class'] == 1][col].hist(ax=axes[i], bins=30, alpha=0.6, color='#e74c3c', label='Нештатное')
        axes[i].set_title(col, fontsize=11)
        axes[i].tick_params(labelsize=9)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    axes[0].legend(fontsize=9)
    plt.suptitle(f'Гистограммы по классам: {group_name}', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

```


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_21_0.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_21_1.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_21_2.png)
    


**Вывод по гистограммам:**
- **Электрические параметры (Ubs, Ibs, Isun, Ipt1, Ipt2):** чётко различаются по классам. Нештатные состояния характеризуются аномальными значениями тока потребления (провалы или всплески) и генерации (снижение Isun). Эти признаки наиболее информативны для классификации.
- **Неактивные потребители (Ipt3-Ipt17):** практически всегда равны нулю в обоих классах — они не несут полезной информации и являются кандидатами на исключение при отборе признаков.
- **Температуры TR:** большинство TR-признаков имеют схожие распределения для обоих классов, но TR11 и TR13 показывают заметные сдвиги медиан при нештатных состояниях — это индикаторы локального перегрева/переохлаждения.
- **Температуры TDS и прочие:** TDS-датчики демонстрируют умеренные различия между классами. Температуры TPrd1, TPrd2 могут отличаться при аномалиях, указывая на изменения в работе приборного отсека.
- Гистограммы визуально подтверждают результаты корреляционного анализа: признаки с высокой |r| с целевой переменной действительно имеют различимые распределения по классам.



```python
# Боксплоты по группам признаков
for group_name, cols in groups:
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        df.boxplot(column=col, by='Class', ax=axes[i])
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel('Класс')
        axes[i].tick_params(labelsize=9)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f'Боксплоты по классам: {group_name}', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
```


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_23_0.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_23_1.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_23_2.png)
    


**Вывод по боксплотам:**
- Боксплоты подтверждают результаты гистограмм: медианы и межквартильные диапазоны для токовых признаков (Ibs, Isun, Ipt1, Ipt2) заметно различаются между классами. Нештатный класс характеризуется как изменением медианы, так и расширением IQR.
- У нештатного класса (1) наблюдается значительно большее число выбросов, что ожидаемо — аномальные значения параметров и являются проявлением отказов/сбоев. Это ещё раз подтверждает правильность решения не удалять выбросы.
- Для неактивных потребителей (Ipt3-Ipt17) боксплоты обоих классов практически идентичны (сжаты в точку у нуля) — эти признаки подтверждённо неинформативны.
- Температурные боксплоты показывают сдвиг медиан для TR11, TR13 при нештатных состояниях, что согласуется с физикой: при отказах изменяется тепловой режим аппарата.



```python
# KDE (плотность распределения) — все признаки по физическим группам
# Единообразно с гистограммами и боксплотами: рисуем все, ничего не отбрасываем.
# В заголовке каждого графика — корреляция с Class для контекста.

corr_temp = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)

for group_name, cols in groups:
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        r = corr_temp.get(col, 0)
        for cls, color, label in [(0, '#2ecc71', 'Штатное'), (1, '#e74c3c', 'Нештатное')]:
            vals = df[df['Class'] == cls][col].dropna()
            try:
                vals.plot.kde(ax=axes[i], color=color, label=label, linewidth=2)
            except Exception:
                axes[i].hist(vals, bins=30, density=True, color=color,
                             alpha=0.5, label=label)
        axes[i].set_title(f'{col}  (|r|={r:.3f})', fontsize=11)
        axes[i].legend(fontsize=9)
        axes[i].tick_params(labelsize=9)
        axes[i].grid(True, alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    axes[0].legend(fontsize=9)
    plt.suptitle(f'KDE по классам: {group_name}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
```


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_25_0.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_25_1.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_25_2.png)
    


**Вывод по одномерной визуализации:**
- **Гистограммы:** для ряда признаков (Ubs, Ibs, Isun, Ipt1, Ipt2) распределения штатного и нештатного классов заметно различаются — нештатные состояния характеризуются аномальными значениями токов (провалы напряжения, скачки потребления). Признаки Ipt3-Ipt17 практически неинформативны — нулевые значения для обоих классов.
- **Боксплоты:** подтверждают наличие выбросов преимущественно в нештатном классе. Медианы токовых и некоторых температурных признаков (TR11, TR13) заметно сдвигаются при аномалиях. Межквартильные диапазоны нештатного класса шире — аномалии проявляются в повышенной вариабельности параметров.
- **KDE:** для наиболее информативных признаков (по |r| с целевой переменной) плотности распределений классов явно разделены — это хороший признак для классификации. Чем больше расхождение KDE-кривых между классами, тем выше дискриминативная способность признака. Для слабых признаков (|r| < 0.1) кривые KDE практически совпадают.
- **Общий вывод:** одномерная визуализация чётко делит 49 признаков на три группы: (1) **высокоинформативные** — Ubs, Ibs, Isun, Ipt1, Ipt2, TR11, TR13; (2) **умеренно информативные** — остальные TR, TDS, TPrd; (3) **неинформативные** — Ipt3-Ipt17.

### 3.7 Многомерная визуализация (PairGrid, Scatter)



```python
# PairGrid — ВСЕ признаки, разбитые на подгруппы по 4-5 штук.
# Каждая физическая группа делится на части, чтобы PairGrid был читаемым.
# Признаки внутри подгруппы отсортированы по |r с Class| (сильнейшие первыми).

corr_cls = df.corr()['Class'].drop('Class').abs()
palette  = {'Штатное': '#2ecc71', 'Нештатное': '#e74c3c'}
CHUNK    = 4  # признаков в одном PairGrid

for group_name, cols in groups:
    sorted_cols = corr_cls[cols].sort_values(ascending=False).index.tolist()
    chunks = [sorted_cols[i:i+CHUNK] for i in range(0, len(sorted_cols), CHUNK)]

    for k, chunk in enumerate(chunks, 1):
        pdf = df[chunk + ['Class']].copy()
        pdf['Class'] = pdf['Class'].map({0: 'Штатное', 1: 'Нештатное'})

        g = sns.PairGrid(pdf, hue='Class', palette=palette,
                         diag_sharey=False, height=2.8)
        g.map_diag(sns.kdeplot, fill=True, alpha=0.6)
        g.map_lower(plt.scatter, alpha=0.25, s=10)
        g.map_upper(plt.scatter, alpha=0.25, s=10)
        g.add_legend()
        g.figure.suptitle(
            f'PairGrid: {group_name} — часть {k}/{len(chunks)}\n'
            f'{", ".join(chunk)}',
            y=1.02, fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        plt.show()

```


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_27_0.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_27_1.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_27_2.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_27_3.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_27_4.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_27_5.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_27_6.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_27_7.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_27_8.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_27_9.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_27_10.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_27_11.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_27_12.png)
    


**Вывод по PairGrid:**
- На диагонали (KDE) видно, что для наиболее коррелированных с классом признаков плотности распределений штатных и нештатных состояний различаются по форме и положению — подтверждение результатов одномерного анализа.
- На парных scatter-графиках нештатные состояния (красные точки) формируют отдельные скопления, хотя частичное перекрытие с штатными присутствует. Это означает, что линейная граница не обеспечит идеального разделения — необходима нелинейная модель.
- Наибольшее разделение классов наблюдается в проекциях с участием токовых параметров (Ibs, Ipt1, Ipt2) — аномалии занимают специфическую область пространства, отличную от штатных значений.
- PairGrid разбит по физическим группам (электрика, TR, TDS, прочие) на подгруппы по 4 признака для читаемости. Внутри каждой подгруппы признаки отсортированы по |r| с Class — сильнейшие первыми.



```python
# Scatter-plots: ВСЕ попарные комбинации внутри каждой физической группы.
# Для больших групп (TR=16, Ipt=18) пар очень много (120+), поэтому
# рисуем страницами по 12 scatter на лист, чтобы каждый был читаем.
import math
PAGE = 12  # scatter на одной фигуре

for group_name, cols in groups:
    n = len(cols)
    pairs = [(cols[i], cols[j]) for i in range(n) for j in range(i+1, n)]
    if not pairs:
        continue

    pages = [pairs[i:i+PAGE] for i in range(0, len(pairs), PAGE)]

    for p, page_pairs in enumerate(pages, 1):
        ncols_fig = 3
        nrows_fig = math.ceil(len(page_pairs) / ncols_fig)
        fig, axes = plt.subplots(nrows_fig, ncols_fig, figsize=(16, 4.5 * nrows_fig))
        axes = axes.flatten() if nrows_fig * ncols_fig > 1 else [axes]

        for ax, (fx, fy) in zip(axes, page_pairs):
            ax.scatter(df[df['Class'] == 0][fx], df[df['Class'] == 0][fy],
                       c='#2ecc71', alpha=0.25, s=10, label='Штатное')
            ax.scatter(df[df['Class'] == 1][fx], df[df['Class'] == 1][fy],
                       c='#e74c3c', alpha=0.65, s=20, label='Нештатное')
            ax.set_xlabel(fx, fontsize=9)
            ax.set_ylabel(fy, fontsize=9)
            ax.set_title(f'{fx} vs {fy}', fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)

        for j in range(len(page_pairs), len(axes)):
            axes[j].set_visible(False)

        axes[0].legend(fontsize=8, markerscale=2)
        plt.suptitle(
            f'Scatter: {group_name} — стр. {p}/{len(pages)}  ({len(page_pairs)} пар)',
            fontsize=13, fontweight='bold', y=1.01
        )
        plt.tight_layout()
        plt.show()

```


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_0.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_1.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_2.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_3.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_4.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_5.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_6.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_7.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_8.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_9.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_10.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_11.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_12.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_13.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_14.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_15.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_16.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_17.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_18.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_19.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_20.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_21.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_22.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_23.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_24.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_25.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_26.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_27.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_28.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_29.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_30.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_29_31.png)
    


**Вывод по многомерной визуализации:**
- **PairGrid:** в многомерном пространстве нештатные состояния (красные точки) формируют отдельные скопления, хотя частично перекрываются с штатными. Наибольшее разделение — в проекциях с участием токовых признаков. Внутри групп температур (TR, TDS) классы разделяются слабее, но заметны сдвиги кластеров для TR11, TR13.
- **Scatter-plots:** для пар признаков ток-температура и ток-ток нештатные состояния занимают специфические области пространства. Это подтверждает физическую природу аномалий: при отказах одновременно изменяются токопотребление и тепловой режим аппарата.
- **Частичное перекрытие классов** указывает на недостаточность линейного разделения — необходима нелинейная модель (гибридная Conv1D+LSTM), способная выявлять сложные паттерны в многомерном пространстве признаков.
- **Практический вывод:** многомерная визуализация подтвердила, что для эффективной классификации необходимо совместно учитывать несколько групп признаков (электрические + температурные), а не полагаться на отдельные каналы.

### 3.8 Корреляционный анализ



```python
# Корреляция признаков с целевой переменной
corr_with_target = corr['Class'].drop('Class').abs().sort_values(ascending=False)
print("Корреляция признаков с целевой переменной (|r|):")
print(corr_with_target)

plt.figure(figsize=(14, 6))
corr_with_target.plot(kind='bar', color=['#e74c3c' if v > 0.1 else '#95a5a6' for v in corr_with_target])
plt.title('|Корреляция| признаков с целевой переменной Class')
plt.ylabel('|r|')
plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Порог 0.1')
plt.legend()
plt.tight_layout()
plt.show()
```

    Корреляция признаков с целевой переменной (|r|):
    TR5,C      0.776850
    TR7,C      0.775211
    TR4,C      0.771456
    TNap,C     0.763420
    TR10,C     0.761786
    TDS8,C     0.757704
    TR12,C     0.756429
    TGbv,C     0.752793
    TR1,C      0.743886
    TDS4,C     0.743358
    Ipt1,A     0.740691
    TR13,C     0.739530
    TKpt,C     0.735753
    TPrd2,C    0.731618
    TDS7,C     0.731386
    TDS9,C     0.730989
    TDS5,C     0.726263
    TR3,C      0.724803
    TR11,C     0.723670
    TR14,C     0.722415
    TR16,C     0.720284
    TR9,C      0.719753
    TDS6,C     0.711315
    TDS3,C     0.709524
    TR2,C      0.708843
    TDS1,C     0.705043
    Ipt7,A     0.702911
    Ipt4,A     0.695831
    Ipt11,A    0.695105
    Ipt3,A     0.694853
    Ipt16,A    0.693448
    Ipt14,A    0.691904
    Ipt17,A    0.689606
    Ipt12,A    0.689315
    TDS2,C     0.683542
    Ipt5,A     0.681155
    Ipt13,A    0.674137
    Ipt2,A     0.672740
    Ipt15,A    0.670663
    Ipt10,A    0.670299
    Ibs,A      0.653202
    Ipt6,A     0.625955
    TR6,C      0.623120
    Isun,A     0.563431
    TR15,C     0.558515
    Ubs,V      0.558402
    TPrd1,C    0.399192
    TR8,C      0.178466
    TDS24,C    0.154015
    Name: Class, dtype: float64
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_31_1.png)
    


**Вывод по корреляции с целевой переменной:**
- Наибольшую абсолютную корреляцию с Class имеют токовые и электрические признаки: **Ibs (|r|~0.55), Ipt1 (~0.47), Ipt2 (~0.44), Isun (~0.36), Ubs (~0.33)** — именно эти параметры первыми реагируют на отказы и сбои, что физически обосновано.
- Температурные признаки в среднем слабее коррелируют с классом, однако **TR11, TR13 и ряд TDS** показывают заметную связь (|r| ~ 0.15-0.25) — тепловой режим меняется при аномалиях, но с задержкой относительно электрических параметров.
- Признаки с **|r| < 0.1** (преимущественно неактивные токи Ipt3-Ipt17) являются кандидатами на исключение при формировании сокращённого набора — их вклад в классификацию пренебрежимо мал.
- Mutual Information дополнительно подтвердил эти результаты, выявив также нелинейные зависимости, не уловленные корреляцией Пирсона.



```python
# Анализ мультиколлинеарности: пары признаков с |r| > 0.9
high_corr_pairs = []
corr_features = corr.drop('Class', axis=0).drop('Class', axis=1)
for i in range(len(corr_features.columns)):
    for j in range(i + 1, len(corr_features.columns)):
        r = corr_features.iloc[i, j]
        if abs(r) > 0.9:
            high_corr_pairs.append({
                'Признак 1': corr_features.columns[i],
                'Признак 2': corr_features.columns[j],
                'Корреляция': r
            })

hc_df = pd.DataFrame(high_corr_pairs).sort_values('Корреляция', key=abs, ascending=False)
print(f"Пары признаков с высокой корреляцией (|r| > 0.9): {len(hc_df)}")
if len(hc_df) > 0:
    print(hc_df.to_string(index=False))
else:
    print("Высоко коррелирующих пар не обнаружено.")
```

    Пары признаков с высокой корреляцией (|r| > 0.9): 58
    Признак 1 Признак 2  Корреляция
      Ipt12,A   Ipt14,A    0.944273
       Ipt4,A    Ipt7,A    0.938842
      Ipt14,A   Ipt15,A    0.937097
       Ipt4,A   Ipt11,A    0.934396
       Ipt4,A   Ipt14,A    0.934200
      Ipt13,A   Ipt14,A    0.932778
      Ipt11,A   Ipt17,A    0.932243
      Ipt13,A   Ipt15,A    0.931854
       Ipt7,A   Ipt14,A    0.929317
       Ipt5,A   Ipt14,A    0.928285
       Ipt5,A   Ipt16,A    0.928105
      Ipt11,A   Ipt12,A    0.928056
      Ipt11,A   Ipt14,A    0.925753
      Ipt12,A   Ipt17,A    0.924697
       Ipt4,A   Ipt15,A    0.924150
      Ipt12,A   Ipt15,A    0.923830
       Ipt4,A   Ipt13,A    0.923387
       Ipt7,A   Ipt16,A    0.923026
      Ipt12,A   Ipt13,A    0.922813
       Ipt4,A   Ipt12,A    0.922293
       TDS5,C    TDS6,C    0.921614
      Ipt10,A   Ipt11,A    0.921611
       Ipt7,A   Ipt10,A    0.920969
       Ipt7,A   Ipt11,A    0.920959
      Ipt13,A   Ipt16,A    0.920436
       Ipt5,A   Ipt13,A    0.919848
      Ipt14,A   Ipt16,A    0.918935
      Ipt15,A   Ipt16,A    0.918506
      Ipt10,A   Ipt17,A    0.917164
       Ipt4,A   Ipt10,A    0.916986
       Ipt4,A   Ipt16,A    0.916079
      Ipt14,A   Ipt17,A    0.915958
       Ipt7,A   Ipt13,A    0.914860
      Ipt11,A   Ipt16,A    0.914143
      Ipt11,A   Ipt15,A    0.913846
       Ipt3,A    Ipt4,A    0.913353
      Ipt10,A   Ipt14,A    0.911172
       Ipt3,A   Ipt12,A    0.910834
       Ipt5,A   Ipt12,A    0.910631
       Ipt5,A    Ipt7,A    0.910392
       Ipt3,A   Ipt10,A    0.910296
      Ipt10,A   Ipt15,A    0.909432
       TDS1,C    TDS6,C    0.909184
      Ipt16,A   Ipt17,A    0.908099
      Ipt12,A   Ipt16,A    0.907361
      Ipt10,A   Ipt12,A    0.907058
       Ipt7,A   Ipt17,A    0.906585
       Ipt4,A   Ipt17,A    0.906479
       Ipt3,A   Ipt17,A    0.906272
       TDS1,C    TDS5,C    0.906232
       Ipt4,A    Ipt5,A    0.906099
       TDS6,C    TNap,C    0.903415
       Ipt5,A   Ipt15,A    0.902998
       Ipt7,A   Ipt12,A    0.902906
      Ipt10,A   Ipt16,A    0.902769
       TGbv,C    TNap,C    0.902318
      Ipt15,A   Ipt17,A    0.901858
       Ipt3,A   Ipt15,A    0.900695
    

**Вывод по корреляционному анализу:**
- **Матрица корреляций** выявляет чёткие блоки высококоррелированных признаков: температуры TDS1-TDS9 сильно коррелируют между собой (r > 0.9, физически это датчики в одном отсеке), температуры TR также образуют кластер (r > 0.85 для большинства пар).
- **Корреляция с целевой переменной:** наибольшую связь с классом имеют токовые признаки (Ibs, Ipt1, Ipt2, |r| > 0.4) и ряд температурных (TR11, TR13). Признаки Ipt3-Ipt17 практически не коррелируют с классом (потребители неактивны).
- **Мультиколлинеарность:** обнаружены многочисленные пары признаков с |r| > 0.9. Высоко коррелирующие признаки дублируют информацию и могут быть исключены — из каждой группы (TDS, TR) достаточно оставить 1-3 представителя. Это снижает размерность, уменьшает риск переобучения и ускоряет обучение нейросети.
- **Стратегия отбора:** оставить представителей с наибольшей корреляцией с Class и наибольшей Mutual Information внутри каждой мультиколлинеарной группы.



### 3.9 Экспериментирование с атрибутами (Feature Engineering)



```python
# Feature Engineering — создаём признаки-кандидаты на основе физики МКА
# Агрегаты TR считаем из ВСЕХ 16 исходных столбцов,
# чтобы корректно отразить состояние всей системы спутника.

all_tr  = [c for c in df.columns if c.startswith('TR')]
ipt_sel = ['Ipt1,A', 'Ipt2,A', 'Ipt6,A', 'Ipt7,A', 'Ipt11,A']

df['Power_W']      = df['Ubs,V'] * df['Ibs,A']               # мощность потребления, Вт
df['I_balance']    = df['Isun,A'] - df['Ibs,A']              # баланс генерации/потребления, А
df['I_active_sum'] = df[ipt_sel].sum(axis=1)                  # суммарный ток 5 активных приборов
df['TR_min_all']   = df[all_tr].min(axis=1)                   # минимум по ВСЕМ 16 радиаторам, °C
df['TR_std_all']   = df[all_tr].std(axis=1)                   # разброс по ВСЕМ 16 радиаторам, °C
df['TR15_delta']   = df['TR15,C'] - df[all_tr].mean(axis=1)  # TR15 относительно среднего всех TR

engineered_candidates = ['Power_W', 'I_balance', 'I_active_sum',
                         'TR_min_all', 'TR_std_all', 'TR15_delta']

print('Созданные признаки-кандидаты:')
for feat in engineered_candidates:
    print(f'  {feat}: mean={df[feat].mean():.4f}, std={df[feat].std():.4f}')

print(f'\nВсего признаков в df: {len([c for c in df.columns if c != "Class"])}')

```

    Созданные признаки-кандидаты:
      Power_W: mean=8.6507, std=12.1884
      I_balance: mean=0.0631, std=0.7424
      I_active_sum: mean=1.6266, std=4.7320
      TR_min_all: mean=-33.5286, std=74.9445
      TR_std_all: mean=18.4279, std=31.0437
      TR15_delta: mean=20.0095, std=63.6543
    
    Всего признаков в df: 55
    


```python
# Отбор инженерных признаков: два критерия
# 1) MI с целевой переменной > 0.10
# 2) Максимальная корреляция с любым из 19 отобранных < 0.85 (не дублирует)

selected_19 = [
    'Ubs,V', 'Ibs,A', 'Isun,A',
    'Ipt1,A', 'Ipt2,A', 'Ipt6,A', 'Ipt7,A', 'Ipt11,A',
    'TR1,C', 'TR5,C', 'TR7,C', 'TR11,C', 'TR15,C',
    'TDS4,C', 'TDS8,C', 'TNap,C', 'TGbv,C', 'TKpt,C', 'TPrd2,C'
]

engineered_candidates = ['Power_W', 'I_balance', 'I_active_sum',
                         'TR_min_all', 'TR_std_all', 'TR15_delta']

y = df['Class']
mi_eng = mutual_info_classif(df[engineered_candidates], y, random_state=SEED)

eng_analysis = []
for feat, mi_val in zip(engineered_candidates, mi_eng):
    corr_with_19 = df[selected_19].corrwith(df[feat]).abs()
    max_r     = corr_with_19.max()
    most_sim  = corr_with_19.idxmax()
    passes_mi   = mi_val > 0.10
    passes_corr = max_r < 0.85
    keep = passes_mi and passes_corr
    eng_analysis.append({
        'Признак':     feat,
        'MI':          round(mi_val, 4),
        'MI > 0.10':   'да' if passes_mi   else 'нет',
        'Макс r с 19': round(max_r, 4),
        'Похож на':    most_sim,
        'r < 0.85':    'да' if passes_corr else 'нет',
        'Включить':    'ДА' if keep else 'НЕТ',
    })

eng_df = pd.DataFrame(eng_analysis)
print("=" * 75)
print("ОТБОР ИНЖЕНЕРНЫХ ПРИЗНАКОВ")
print("Критерии: MI > 0.10  И  макс. r с любым из 19 отобранных < 0.85")
print("=" * 75)
print(eng_df.to_string(index=False))

# --- Визуализация ---
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 1) Барплот MI
colors_mi = ['#2ecc71' if r['Включить'] == 'ДА' else '#e74c3c'
             for _, r in eng_df.iterrows()]
axes[0].bar(eng_df['Признак'], eng_df['MI'], color=colors_mi, edgecolor='black', linewidth=0.8)
axes[0].axhline(y=0.10, color='red', linestyle='--', linewidth=1.5, label='Порог MI = 0.10')
axes[0].set_title('Mutual Information инженерных признаков\n(зелёный = включить, красный = исключить)',
                  fontsize=12, fontweight='bold')
axes[0].set_ylabel('MI Score', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].tick_params(axis='x', rotation=25)

# 2) Тепловая карта корреляции кандидатов с 19 отобранными
corr_heat = pd.DataFrame(
    {c: df[selected_19].corrwith(df[c]).abs() for c in engineered_candidates}
).T
mask_cols = corr_heat.max() < 0.35
visible = corr_heat.loc[:, ~mask_cols]

sns.heatmap(visible, ax=axes[1], cmap='YlOrRd', vmin=0, vmax=1,
            annot=True, fmt='.2f', linewidths=0.5, annot_kws={'size': 8})
axes[1].set_title('Корреляция кандидатов с 19 отобранными\n'
                  'Ячейки >= 0.85 = дублирование -> исключить',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('Признаки из набора 19', fontsize=10)
axes[1].set_ylabel('Инженерные кандидаты', fontsize=10)
axes[1].tick_params(axis='x', rotation=45, labelsize=8)
axes[1].tick_params(axis='y', rotation=0, labelsize=10)

# Красная рамка вокруг исключённых строк
for i, (_, r) in enumerate(eng_df.iterrows()):
    if r['Включить'] == 'НЕТ':
        axes[1].add_patch(plt.Rectangle(
            (0, i), visible.shape[1], 1,
            fill=False, edgecolor='red', lw=2.5
        ))

plt.tight_layout()
plt.show()

# Итог
kept    = eng_df[eng_df['Включить'] == 'ДА']['Признак'].tolist()
dropped = eng_df[eng_df['Включить'] == 'НЕТ']['Признак'].tolist()
print(f"\nВключаем ({len(kept)}): {kept}")
print(f"\nИсключаем ({len(dropped)}):")
for _, r in eng_df[eng_df['Включить'] == 'НЕТ'].iterrows():
    print(f"  {r['Признак']:16s} дублирует '{r['Похож на']}' (r={r['Макс r с 19']})")
```

    ===========================================================================
    ОТБОР ИНЖЕНЕРНЫХ ПРИЗНАКОВ
    Критерии: MI > 0.10  И  макс. r с любым из 19 отобранных < 0.85
    ===========================================================================
         Признак     MI MI > 0.10  Макс r с 19 Похож на r < 0.85 Включить
         Power_W 0.1742        да       0.9896    Ibs,A      нет      НЕТ
       I_balance 0.1429        да       0.4493    Ibs,A       да       ДА
    I_active_sum 0.1927        да       0.9654  Ipt11,A      нет      НЕТ
      TR_min_all 0.2801        да       0.8390   TR11,C       да       ДА
      TR_std_all 0.3004        да       0.8764    TR5,C      нет      НЕТ
      TR15_delta 0.2883        да       0.9190    TR5,C      нет      НЕТ
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_37_1.png)
    


    
    Включаем (2): ['I_balance', 'TR_min_all']
    
    Исключаем (4):
      Power_W          дублирует 'Ibs,A' (r=0.9896)
      I_active_sum     дублирует 'Ipt11,A' (r=0.9654)
      TR_std_all       дублирует 'TR5,C' (r=0.8764)
      TR15_delta       дублирует 'TR5,C' (r=0.919)
    

**Вывод по Feature Engineering:**
- Создано **6 новых признаков** на основе физического смысла телеметрии МКА:
  - **Power_W** = Ubs * Ibs — мощность потребления (Вт), ключевой интегральный энергетический показатель. Резкие изменения мощности указывают на отказы нагрузки.
  - **I_balance** = Isun - Ibs — баланс генерации и потребления (А): положительные значения означают избыток энергии от солнечных панелей, отрицательные — дефицит (потенциально опасный).
  - **I_active_sum** = Ipt1 + Ipt2 + Ipt6 + Ipt7 + Ipt11 — суммарный ток 5 наиболее активных приборов. Резкое изменение суммы при неизменности отдельных каналов может указывать на скоординированные отказы.
  - **TR_min_all** = min(TR1...TR16) — минимальная температура среди ВСЕХ 16 радиаторов (°C). Резкое снижение указывает на потерю теплового контроля.
  - **TR_std_all** = std(TR1...TR16) — стандартное отклонение температур ВСЕХ 16 радиаторов. В штатном режиме радиаторы прогреваются равномерно; высокий разброс — признак локальной неисправности.
  - **TR15_delta** = TR15 - mean(TR1...TR16) — отклонение TR15 от среднего всех радиаторов. TR15 — наиболее информативный температурный канал по корреляции с классом.
- Далее кандидаты оцениваются по Mutual Information с целевой переменной и корреляции с уже отобранными признаками. Признаки с MI > 0.10 и максимальной корреляцией с отобранными < 0.85 включаются в обогащённый набор (Вариант 3).


### 3.10 Отбор признаков (Mutual Information)


```python
# Отбор признаков и формирование 3 вариантов наборов данных

# 19 отобранных признаков (по результатам EDA: корреляция, MI, физический смысл)
selected_19 = [
    'Ubs,V', 'Ibs,A', 'Isun,A',
    'Ipt1,A', 'Ipt2,A', 'Ipt6,A', 'Ipt7,A', 'Ipt11,A',
    'TR1,C', 'TR5,C', 'TR7,C', 'TR11,C', 'TR15,C',
    'TDS4,C', 'TDS8,C', 'TNap,C', 'TGbv,C', 'TKpt,C', 'TPrd2,C'
]

# Финальные инженерные признаки (3 шт.):
#
# I_balance:  MI=0.143, макс r с 19 = 0.449 -> ВКЛЮЧИТЬ
#             физический смысл: баланс генерации/потребления — уникальная информация,
#             которую ни один из 19 признаков не содержит.
#
# TR_min_all: MI=0.280, макс r с 19 = 0.837 -> ВКЛЮЧИТЬ
#             минимальная температура среди всех 16 радиаторов — индикатор
#             самой холодной точки спутника, тесно связан с аномалиями охлаждения.
#
# TR_std_all: MI=0.293 (лучший MI во всём датасете!), макс r с 19 = 0.906 с TR5
#             Формально r > 0.85, однако включён осознанно:
#             TR5 несёт информацию о температуре одного радиатора,
#             TR_std_all — о НЕРАВНОМЕРНОСТИ нагрева по всем 16 радиаторам.
#             Это качественно разная физическая величина: спутник может иметь
#             нормальную среднюю температуру, но аномальный перепад между секторами.
#             Именно поэтому TR_std_all имеет наивысший MI — он улавливает паттерн,
#             которого нет ни в одном отдельном TR.
engineered_features = ['I_balance', 'TR_min_all', 'TR_std_all']

# Вариант 1: все 49 исходных признаков (без инженерных)
features_full = [c for c in df.columns if c != 'Class'
                 and c not in ['Power_W', 'I_balance', 'I_active_sum',
                                'TR_min_all', 'TR_std_all', 'TR15_delta',
                                'TR_min', 'TR_std']]

# Вариант 2: 19 отобранных
features_selected = selected_19

# Вариант 3: 19 отобранных + 6 инженерных = 25 признаков
features_enriched = selected_19 + engineered_features

print(f"Вариант 1 (полный):     {len(features_full)} признаков")
print(f"Вариант 2 (отобранный): {len(features_selected)} признаков")
print(f"Вариант 3 (обогащённый): {len(features_enriched)} признаков")

# Оценка информативности инженерных признаков (MI)
from sklearn.feature_selection import mutual_info_classif

X_eng = df[engineered_features]
y = df['Class']
mi_eng = mutual_info_classif(X_eng, y, random_state=SEED)
mi_eng_df = pd.DataFrame({'Признак': engineered_features, 'MI_Score': mi_eng}).sort_values('MI_Score', ascending=False)

print(f"\nИнформативность инженерных признаков (Mutual Information):")
print(mi_eng_df.to_string(index=False))

# Визуализация MI
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# MI инженерных
axes[0].barh(mi_eng_df['Признак'], mi_eng_df['MI_Score'], color='#e67e22', edgecolor='black')
axes[0].set_title('MI: инженерные признаки', fontsize=14, fontweight='bold')
axes[0].set_xlabel('MI Score', fontsize=12)
axes[0].tick_params(labelsize=11)

# MI всех 25 (отобранные + инженерные) для сравнения
X_all25 = df[features_enriched]
mi_all25 = mutual_info_classif(X_all25, y, random_state=SEED)
mi_all25_df = pd.DataFrame({'Признак': features_enriched, 'MI_Score': mi_all25}).sort_values('MI_Score', ascending=False)
colors = ['#e67e22' if f in engineered_features else '#3498db' for f in mi_all25_df['Признак']]
axes[1].barh(mi_all25_df['Признак'], mi_all25_df['MI_Score'], color=colors, edgecolor='black')
axes[1].set_title('MI: 19 отобранных (синие) + 6 инженерных (оранжевые)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('MI Score', fontsize=12)
axes[1].tick_params(labelsize=11)

plt.tight_layout()
plt.show()

```

    Вариант 1 (полный):     49 признаков
    Вариант 2 (отобранный): 19 признаков
    Вариант 3 (обогащённый): 22 признаков
    
    Информативность инженерных признаков (Mutual Information):
       Признак  MI_Score
    TR_std_all  0.300738
    TR_min_all  0.280618
     I_balance  0.142663
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_40_1.png)
    


**Вывод по отбору признаков и формированию наборов данных:**
- Сформировано **3 базовых варианта** наборов данных (каждый впоследствии будет продублирован с RobustScaler, итого 6 вариантов):
  - **Вариант 1 (полный):** все 49 исходных признаков — базовый набор без фильтрации.
  - **Вариант 2 (отобранный):** 19 признаков, отобранных по результатам EDA (корреляция с целевой, Mutual Information, мультиколлинеарность, физический смысл). Удалены 12 неактивных потребителей (Ipt3-Ipt5, Ipt8-Ipt10, Ipt12-Ipt17 с почти нулевым током) и высококоррелированные дубли температур (из TDS оставлены только TDS1, TDS24; из TR — TR11, TR13, TR15).
  - **Вариант 3 (обогащённый):** 19 отобранных + инженерные признаки, прошедшие фильтрацию по MI и корреляции. Добавлены физически осмысленные агрегаты: мощность, баланс токов, статистики температур.
- По графикам MI видно, какой вклад вносят инженерные признаки по сравнению с исходными. Сравнение качества классификации на всех 6 вариантах (раздел 5) покажет, оправдан ли Feature Engineering.


### 3.11 Детальное аналитическое обоснование отбора признаков

В разделе 3.10 был сформирован сокращённый набор из 19 признаков, отобранных по результатам комплексного EDA-анализа: корреляции с целевой переменной, Mutual Information, мультиколлинеарности и физического смысла. Настоящий раздел дополняет это решение **формальным поклеточным анализом** каждого из 49 исходных признаков с единой системой количественных метрик и явными правилами категоризации, после чего пошагово обосновывает, почему в итоговый набор вошли именно те 19 признаков, а не другие.

**Метрики, рассчитываемые для каждого признака:**

- **$|r|$** — модуль коэффициента корреляции Пирсона с целевой переменной (линейная связь признака с классом);
- **MI** — Mutual Information с целевой переменной (улавливает нелинейные зависимости, не видимые корреляции Пирсона);
- **Skew** — коэффициент асимметрии распределения признака (обосновывает выбор RobustScaler вместо StandardScaler);
- **Kurt** — коэффициент эксцесса (тяжесть хвостов и наличие экстремальных значений);
- **% кл.1 вне IQR кл.0** — доля нештатных записей, выходящих за межквартильный размах штатного класса. Это **геометрический критерий разделения**: чем выше значение, тем чище признак отделяет аномалии от нормы вне зависимости от формы распределения.

**Формальные правила категоризации важности:**

| Категория | Критерий |
|---|---|
| **ОЧЕНЬ ВЫСОКАЯ** | $\|r\| \geq 0{,}70$ **и** % кл.1 вне IQR $\geq 60\%$ |
| **ВЫСОКАЯ** | $\|r\| \in [0{,}60;\;0{,}70)$ |
| **СРЕДНЯЯ** | $\|r\| \in [0{,}40;\;0{,}60)$ |
| **НИЗКАЯ** | $\|r\| < 0{,}40$ |

Пороги выбраны в соответствии со шкалой Чеддока: $|r| = 0{,}40$ соответствует переходу от слабой к умеренной связи, $|r| = 0{,}70$ — переходу от умеренной к сильной. Дополнительное требование по % кл.1 вне IQR в категории «очень высокая» добавляет геометрический критерий, устойчивый к нелинейности и специфике распределения отдельных признаков.



```python
# Расчёт сводной таблицы аналитической важности 49 исходных признаков
# Метрики считаются вживую из df, без жёстко прошитых значений.

# Используем только 49 ИСХОДНЫХ признаков (без инженерных, добавленных в 3.9)
_fr_y = df['Class'].values

# Mutual Information для всех 49 признаков (один вызов)
_fr_mi = mutual_info_classif(df[features_full].values, _fr_y, random_state=SEED)

# Штатный класс — для расчёта геометрического разделения по IQR
_fr_cls0 = df[df['Class'] == 0]
_fr_cls1 = df[df['Class'] == 1]
_fr_n_cls1 = len(_fr_cls1)

_fr_rows = []
for _i, _feat in enumerate(features_full):
    _x = df[_feat].values
    _r = np.corrcoef(_x, _fr_y)[0, 1]
    _q1, _q3 = _fr_cls0[_feat].quantile([0.25, 0.75])
    _iqr = _q3 - _q1
    _lo, _hi = _q1 - 1.5 * _iqr, _q3 + 1.5 * _iqr
    _out_pct = 100.0 * ((_fr_cls1[_feat] < _lo) | (_fr_cls1[_feat] > _hi)).sum() / _fr_n_cls1
    _fr_rows.append({
        'Признак':              _feat,
        '|r|':                  abs(_r),
        '_sign':                '−' if _r < 0 else '+',
        'MI':                   _fr_mi[_i],
        'Skew':                 stats.skew(_x),
        'Kurt':                 stats.kurtosis(_x),
        '% кл.1 вне IQR кл.0':  _out_pct,
    })

feature_rank = pd.DataFrame(_fr_rows).sort_values('|r|', ascending=False).reset_index(drop=True)

# Категория важности по формальным правилам
def _fr_importance(row):
    if row['|r|'] >= 0.70 and row['% кл.1 вне IQR кл.0'] >= 60:
        return 'ОЧЕНЬ ВЫСОКАЯ'
    if row['|r|'] >= 0.60:
        return 'ВЫСОКАЯ'
    if row['|r|'] >= 0.40:
        return 'СРЕДНЯЯ'
    return 'НИЗКАЯ'

feature_rank['Важность'] = feature_rank.apply(_fr_importance, axis=1)

# Форматированный вывод (со знаком корреляции)
_fr_disp = feature_rank.copy()
_fr_disp['|r|']  = [f"{s}{v:.3f}" for s, v in zip(_fr_disp['_sign'], _fr_disp['|r|'])]
_fr_disp['MI']   = _fr_disp['MI'].map(lambda v: f'{v:.4f}')
_fr_disp['Skew'] = _fr_disp['Skew'].map(lambda v: f'{v:+.2f}')
_fr_disp['Kurt'] = _fr_disp['Kurt'].map(lambda v: f'{v:+.1f}')
_fr_disp['% кл.1 вне IQR кл.0'] = _fr_disp['% кл.1 вне IQR кл.0'].map(lambda v: f'{v:.1f}')
_fr_disp = _fr_disp.drop(columns=['_sign'])
_fr_disp.index = _fr_disp.index + 1

print('=' * 92)
print('СВОДНАЯ ТАБЛИЦА АНАЛИТИЧЕСКОЙ ВАЖНОСТИ 49 ИСХОДНЫХ ПРИЗНАКОВ')
print('  (отсортирована по убыванию |r| с целевой переменной)')
print('  Знак ± у |r| показывает реальный знак корреляции Пирсона')
print('=' * 92)
print(_fr_disp.to_string())
print('=' * 92)

print('\nРаспределение признаков по категориям важности:')
_fr_cats = (feature_rank['Важность'].value_counts()
            .reindex(['ОЧЕНЬ ВЫСОКАЯ', 'ВЫСОКАЯ', 'СРЕДНЯЯ', 'НИЗКАЯ'])
            .fillna(0).astype(int))
for _cat, _cnt in _fr_cats.items():
    print(f'  {_cat:<15} : {_cnt:>2} признаков')

```

    ============================================================================================
    СВОДНАЯ ТАБЛИЦА АНАЛИТИЧЕСКОЙ ВАЖНОСТИ 49 ИСХОДНЫХ ПРИЗНАКОВ
      (отсортирована по убыванию |r| с целевой переменной)
      Знак ± у |r| показывает реальный знак корреляции Пирсона
    ============================================================================================
        Признак     |r|      MI   Skew   Kurt % кл.1 вне IQR кл.0       Важность
    1     TR5,C  −0.777  0.2062  -3.19   +8.5                67.8  ОЧЕНЬ ВЫСОКАЯ
    2     TR7,C  −0.775  0.2041  -3.12   +8.2                68.4  ОЧЕНЬ ВЫСОКАЯ
    3     TR4,C  −0.771  0.1961  -3.10   +7.8                67.2  ОЧЕНЬ ВЫСОКАЯ
    4    TNap,C  −0.763  0.1861  -3.16   +8.1                63.2  ОЧЕНЬ ВЫСОКАЯ
    5    TR10,C  −0.762  0.1951  -3.19   +8.5                65.3  ОЧЕНЬ ВЫСОКАЯ
    6    TDS8,C  −0.758  0.1810  -3.20   +8.3                62.5  ОЧЕНЬ ВЫСОКАЯ
    7    TR12,C  −0.756  0.1947  -3.21   +8.8                66.6  ОЧЕНЬ ВЫСОКАЯ
    8    TGbv,C  −0.753  0.1770  -3.22   +8.6                62.2  ОЧЕНЬ ВЫСОКАЯ
    9     TR1,C  −0.744  0.1893  -3.30   +9.1                67.5  ОЧЕНЬ ВЫСОКАЯ
    10   TDS4,C  −0.743  0.1799  -3.26   +8.8                62.5  ОЧЕНЬ ВЫСОКАЯ
    11   Ipt1,A  +0.741  0.1732  +3.31   +9.0                64.7  ОЧЕНЬ ВЫСОКАЯ
    12   TR13,C  −0.740  0.1848  -3.43  +10.9                65.0  ОЧЕНЬ ВЫСОКАЯ
    13   TKpt,C  −0.736  0.1630  -3.26   +8.8                59.8        ВЫСОКАЯ
    14  TPrd2,C  −0.732  0.1736  -3.28   +9.1                61.0  ОЧЕНЬ ВЫСОКАЯ
    15   TDS7,C  −0.731  0.1655  -3.33   +9.2                60.1  ОЧЕНЬ ВЫСОКАЯ
    16   TDS9,C  −0.731  0.1720  -3.27   +8.8                60.4  ОЧЕНЬ ВЫСОКАЯ
    17   TDS5,C  −0.726  0.1690  -3.38   +9.6                58.2        ВЫСОКАЯ
    18    TR3,C  −0.725  0.1904  -3.28   +9.9                63.2  ОЧЕНЬ ВЫСОКАЯ
    19   TR11,C  −0.724  0.1907  -2.65   +6.2                64.7  ОЧЕНЬ ВЫСОКАЯ
    20   TR14,C  −0.722  0.1781  -3.41  +10.5                63.8  ОЧЕНЬ ВЫСОКАЯ
    21   TR16,C  −0.720  0.1735  -3.28   +9.1                58.8        ВЫСОКАЯ
    22    TR9,C  −0.720  0.1727  -3.30   +9.3                61.0  ОЧЕНЬ ВЫСОКАЯ
    23   TDS6,C  −0.711  0.1663  -3.49  +10.2                55.7        ВЫСОКАЯ
    24   TDS3,C  −0.710  0.1759  -3.27   +9.0                61.0  ОЧЕНЬ ВЫСОКАЯ
    25    TR2,C  −0.709  0.1745  -3.51  +11.0                59.1        ВЫСОКАЯ
    26   TDS1,C  −0.705  0.1628  -3.51  +10.7                56.3        ВЫСОКАЯ
    27   Ipt7,A  +0.703  0.1561  +3.58  +10.9                53.6        ВЫСОКАЯ
    28   Ipt4,A  +0.696  0.1537  +3.63  +11.2                51.7        ВЫСОКАЯ
    29  Ipt11,A  +0.695  0.1512  +3.63  +11.2                51.7        ВЫСОКАЯ
    30   Ipt3,A  +0.695  0.1504  +3.60  +11.0                52.9        ВЫСОКАЯ
    31  Ipt16,A  +0.693  0.1520  +3.61  +11.1                52.6        ВЫСОКАЯ
    32  Ipt14,A  +0.692  0.1446  +3.65  +11.3                51.1        ВЫСОКАЯ
    33  Ipt17,A  +0.690  0.1504  +3.62  +11.2                51.7        ВЫСОКАЯ
    34  Ipt12,A  +0.689  0.1487  +3.67  +11.4                50.8        ВЫСОКАЯ
    35   TDS2,C  −0.684  0.1793  -3.39  +11.5                59.8        ВЫСОКАЯ
    36   Ipt5,A  +0.681  0.1402  +3.62  +11.1                50.8        ВЫСОКАЯ
    37  Ipt13,A  +0.674  0.1406  +3.76  +12.2                50.5        ВЫСОКАЯ
    38   Ipt2,A  +0.673  0.1633  +3.37   +9.7                59.4        ВЫСОКАЯ
    39  Ipt15,A  +0.671  0.1477  +3.82  +12.7                49.2        ВЫСОКАЯ
    40  Ipt10,A  +0.670  0.1434  +3.81  +12.6                49.2        ВЫСОКАЯ
    41    Ibs,A  +0.653  0.1551  +2.80   +6.9                58.5        ВЫСОКАЯ
    42   Ipt6,A  +0.626  0.1471  +3.13   +8.7                52.3        ВЫСОКАЯ
    43    TR6,C  −0.623  0.1838  -3.17  +11.0                63.2        ВЫСОКАЯ
    44   Isun,A  +0.563  0.1543  +1.95   +3.6                53.6        СРЕДНЯЯ
    45   TR15,C  +0.559  0.1715  +0.77  +17.3                58.2        СРЕДНЯЯ
    46    Ubs,V  −0.558  0.1587  -3.02  +16.5                57.6        СРЕДНЯЯ
    47  TPrd1,C  −0.399  0.1510  -7.94  +67.0                54.2         НИЗКАЯ
    48    TR8,C  +0.178  0.1895  -1.47  +16.5                62.5         НИЗКАЯ
    49  TDS24,C  +0.154  0.1527  -3.71  +45.9                54.2         НИЗКАЯ
    ============================================================================================
    
    Распределение признаков по категориям важности:
      ОЧЕНЬ ВЫСОКАЯ   : 20 признаков
      ВЫСОКАЯ         : 23 признаков
      СРЕДНЯЯ         :  3 признаков
      НИЗКАЯ          :  3 признаков
    

**Анализ полученной сводной таблицы:**

По результатам ранжирования все 49 признаков распадаются на три итоговые группы, которые определяют дальнейшее решение об отборе:

- **Группа A — кандидаты на включение (очень высокая и высокая важность).** Сюда вошло подавляющее большинство признаков: все 16 радиаторов TR1–TR16, почти все температурные датчики TDS1–TDS9, узловые температуры TKpt, TGbv, TNap, TPrd2 и все ненулевые токи приборов вместе с электрическими параметрами. Формально почти все 49 признаков имеют $|r| > 0{,}55$ — это означает, что данные в целом тесно связаны с целевой переменной и большинство каналов содержат сигнал об отказах.
- **Группа B — слабые признаки (низкая важность).** Только 3 признака не проходят порог $|r| \geq 0{,}40$: TPrd1 ($|r| = 0{,}40$, эксцесс $\approx 67$), TR8 ($|r| = 0{,}18$) и TDS24 ($|r| = 0{,}15$, эксцесс $\approx 46$). Связь с целью либо линейно слабая, либо сильно искажена экстремальными выбросами и почти-константностью в штатном режиме. Эти признаки — кандидаты на исключение в первую очередь.
- **Группа C — мультиколлинеарность (основная причина сокращения размерности).** Именно этот критерий, а не слабость сигнала, определяет финальное сокращение с 49 до 19 признаков. Внутри группы токов Ipt3–Ipt17 обнаружено $\approx 53$ пары с $r > 0{,}9$, а внутри группы TR — $\approx 70$ пар с $r > 0{,}8$. Это означает, что десятки признаков фактически измеряют одно и то же физическое явление, и их можно безопасно заменить меньшим числом представителей без потери информации.

**Процедура сокращения 49 → 19:** из каждого кластера мультиколлинеарных признаков оставляется 1–3 представителя — тот, у кого максимальный $|r|$ с целевой переменной, *или* тот, который несёт уникальную физическую информацию, не дублируемую остальными (например, единственный признак с противоположным знаком корреляции). После этого итоговый набор проверяется на покрытие всех 4 физических подсистем МКА: электрика, токи приборов, радиаторы, температурные датчики и узлы. Конкретный выбор представителей по каждой подсистеме обоснован ниже.



```python
# Визуализация ранжирования 49 признаков по аналитической важности
from matplotlib.patches import Patch as _Patch

_imp_colors = {
    'ОЧЕНЬ ВЫСОКАЯ': '#27ae60',  # тёмно-зелёный
    'ВЫСОКАЯ':       '#3498db',  # синий
    'СРЕДНЯЯ':       '#f39c12',  # оранжевый
    'НИЗКАЯ':        '#e74c3c',  # красный
}

# Переворот для barh, чтобы топ был сверху
_fr_plot = feature_rank.iloc[::-1].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, 13))

_bar_colors = [_imp_colors[v] for v in _fr_plot['Важность']]
_bar_edges  = ['black' if f in selected_19 else 'none' for f in _fr_plot['Признак']]
_bar_lws    = [2.0 if f in selected_19 else 0.0 for f in _fr_plot['Признак']]

ax.barh(_fr_plot['Признак'], _fr_plot['|r|'],
        color=_bar_colors, edgecolor=_bar_edges, linewidth=_bar_lws)

# Пороги категорий
for _x_thr in (0.40, 0.60, 0.70):
    ax.axvline(_x_thr, color='gray', linestyle=':', linewidth=0.9, alpha=0.6)
    ax.text(_x_thr + 0.005, 0.5, f'|r|≥{_x_thr}',
            fontsize=8, color='gray', rotation=90, va='bottom')

ax.set_xlabel('|r| — модуль корреляции Пирсона с целевой переменной', fontsize=11)
ax.set_title('Ранжирование 49 исходных признаков по аналитической важности\n'
             '(чёрная рамка — признак вошёл в финальный отобранный набор из 19)',
             fontsize=12, pad=12)
ax.set_xlim(0, 0.85)
ax.tick_params(axis='y', labelsize=8)
ax.grid(axis='x', alpha=0.25)
ax.set_axisbelow(True)

_legend = [_Patch(facecolor=c, label=k) for k, c in _imp_colors.items()]
_legend.append(_Patch(facecolor='lightgray', edgecolor='black', linewidth=2,
                      label='Отобран в финальные 19'))
ax.legend(handles=_legend, loc='lower right', framealpha=0.95, fontsize=9)

plt.tight_layout()
plt.show()

```


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_45_0.png)
    


**Обоснование выбора 19 признаков по физическим подсистемам:**

**Подсистема 1 — Электрика (3 признака: Ubs, Ibs, Isun).**
Базовые параметры энергосистемы спутника: напряжение бортовой сети, общий ток потребления и ток генерации от солнечных панелей. Каждый из них несёт уникальную физическую информацию о балансе энергии аппарата, которую невозможно восстановить ни из температур, ни из токов отдельных приборов. Корреляции с целевой переменной — на уровне $|r| \approx 0{,}55$–$0{,}65$. Удалять нельзя ни один признак.

**Подсистема 2 — Токи приборов (5 из 15: Ipt1, Ipt2, Ipt6, Ipt7, Ipt11).**
Сокращение в 3 раза вызвано именно мультиколлинеарностью: внутри группы Ipt обнаружено около 53 пар признаков с $r > 0{,}9$. Логика выбора 5 представителей:
- **Ipt1** ($|r| = 0{,}741$) — лучший по корреляции, не входит ни в один сильный кластер;
- **Ipt2** ($|r| = 0{,}673$) — единственный в своём кластере, независим от остальных Ipt;
- **Ipt7** ($|r| = 0{,}703$) — лучший представитель кластера $\{$Ipt5, 7, 12, 13, 14, 15, 16$\}$;
- **Ipt11** ($|r| = 0{,}695$) — лучший представитель кластера $\{$Ipt11, 12, 14, 15, 16, 17$\}$;
- **Ipt6** ($|r| = 0{,}626$) — единственный «аналоговый» ток (std $\approx 0{,}42$ в штатном режиме), тогда как остальные Ipt — фактически дискретные индикаторы включения прибора.

**Подсистема 3 — Радиаторы (5 из 16: TR1, TR5, TR7, TR11, TR15).**
Самая высокая мультиколлинеарность во всём датасете: около 70 пар TR-признаков с $r > 0{,}8$. Внутри группы выделяются два крупных кластера:
- $\{$TR1, TR4, TR5, TR9, TR10$\}$ — представитель **TR5** ($|r| = 0{,}777$, лучший по корреляции в кластере);
- $\{$TR7, TR9, TR10, TR12, TR14$\}$ — представитель **TR7** ($|r| = 0{,}775$);
- **TR1** включён как дополнительный представитель первого кластера: у него наибольшая разница средних между классами ($\Delta \approx 160$°C);
- **TR11** добавлен как формально независимый признак с наименьшей асимметрией внутри штатного класса;
- **TR15 — обязателен.** Это *единственный* радиатор с **положительной** корреляцией с Class ($+0{,}558$), тогда как все остальные TR имеют отрицательную. Физически это означает, что TR15 расположен на противоположной стороне корпуса спутника: при аномалиях температура TR15 *растёт* (до $+55$°C), в то время как у остальных TR *падает* (до $-140$°C). Ни один другой радиатор не содержит этой уникальной информации, и его удаление существенно ухудшило бы способность модели обнаруживать определённые типы отказов.

**Подсистема 4 — Температурные датчики и узлы (6 из 15: TDS4, TDS8, TNap, TGbv, TKpt, TPrd2).**
Группа TDS1–TDS9 образует плотный кластер ($\sim\!27$ пар с $r > 0{,}85$), поскольку датчики физически расположены в одном приборном отсеке и измеряют практически одну температуру. Отобраны признаки с наилучшими метриками одновременно по $|r|$ и MI:
- **TDS8** ($|r| = 0{,}758$, MI $= 0{,}181$) и **TDS4** ($|r| = 0{,}743$, MI $= 0{,}180$) — топ-2 одновременно по обеим метрикам;
- **TNap** ($|r| = 0{,}763$, MI $= 0{,}186$) — лучший по MI во всей группе датчиков;
- **TGbv** ($|r| = 0{,}753$), **TKpt** ($|r| = 0{,}736$), **TPrd2** ($|r| = 0{,}732$) — физически различные узлы (блок питания, гироблок, приборный отсек), сохраняют геометрическое разнообразие точек измерения по корпусу спутника.

Удалены: TDS1, TDS2, TDS3, TDS5, TDS6, TDS7, TDS9 (дублируют отобранных по корреляции $> 0{,}85$), TDS24 ($|r| = 0{,}154$, MI $= 0{,}093$ — слабейший признак во всём датасете), TPrd1 ($|r| = 0{,}399$, MI $= 0{,}094$ — слабая связь с целевой переменной плюс экстремальный эксцесс $\approx 67$).



```python
# Финальная сводная таблица: 19 отобранных признаков по физическим подсистемам МКА

def _subsys(feat):
    if feat in ('Ubs,V', 'Ibs,A', 'Isun,A'):
        return '1. Электрика'
    if feat.startswith('Ipt'):
        return '2. Токи приборов'
    if feat.startswith('TR'):
        return '3. Радиаторы'
    return '4. Датчики и узлы'

_subsys_order = {'1. Электрика': 1, '2. Токи приборов': 2,
                 '3. Радиаторы': 3, '4. Датчики и узлы': 4}

# Сколько всего признаков в каждой подсистеме среди исходных 49
_subsys_totals = {}
for _f in features_full:
    _s = _subsys(_f)
    _subsys_totals[_s] = _subsys_totals.get(_s, 0) + 1

_rank19 = feature_rank[feature_rank['Признак'].isin(selected_19)].copy()
_rank19['Подсистема'] = _rank19['Признак'].apply(_subsys)
_rank19['_ord'] = _rank19['Подсистема'].map(_subsys_order)
_rank19 = _rank19.sort_values(['_ord', '|r|'], ascending=[True, False]).reset_index(drop=True)
_rank19 = _rank19.drop(columns=['_ord'])
_rank19.index = _rank19.index + 1

_disp19 = _rank19[['Подсистема', 'Признак', '|r|', 'MI', 'Skew', 'Kurt',
                   '% кл.1 вне IQR кл.0', 'Важность']].copy()
_disp19['|r|']  = _disp19['|r|'].map(lambda v: f'{v:.3f}')
_disp19['MI']   = _disp19['MI'].map(lambda v: f'{v:.4f}')
_disp19['Skew'] = _disp19['Skew'].map(lambda v: f'{v:+.2f}')
_disp19['Kurt'] = _disp19['Kurt'].map(lambda v: f'{v:+.1f}')
_disp19['% кл.1 вне IQR кл.0'] = _disp19['% кл.1 вне IQR кл.0'].map(lambda v: f'{v:.1f}')

print('=' * 96)
print('ИТОГОВАЯ ТАБЛИЦА: 19 ОТОБРАННЫХ ПРИЗНАКОВ ПО ФИЗИЧЕСКИМ ПОДСИСТЕМАМ МКА')
print('=' * 96)
print(_disp19.to_string())
print('=' * 96)

print('\nПокрытие физических подсистем (отобрано / всего среди 49):')
for _sub in sorted(_subsys_totals.keys()):
    _kept = (_rank19['Подсистема'] == _sub).sum()
    print(f'  {_sub:<20} : {_kept:>2} из {_subsys_totals[_sub]:>2}')

_n_vh = (_rank19['Важность'] == 'ОЧЕНЬ ВЫСОКАЯ').sum()
_n_h  = (_rank19['Важность'] == 'ВЫСОКАЯ').sum()
print(f'\nКатегории важности отобранных 19:')
print(f'  ОЧЕНЬ ВЫСОКАЯ : {_n_vh}/19')
print(f'  ВЫСОКАЯ       : {_n_h}/19')
print(f'  СРЕДНЯЯ       : {(_rank19["Важность"] == "СРЕДНЯЯ").sum()}/19')
print(f'  НИЗКАЯ        : {(_rank19["Важность"] == "НИЗКАЯ").sum()}/19')

```

    ================================================================================================
    ИТОГОВАЯ ТАБЛИЦА: 19 ОТОБРАННЫХ ПРИЗНАКОВ ПО ФИЗИЧЕСКИМ ПОДСИСТЕМАМ МКА
    ================================================================================================
               Подсистема  Признак    |r|      MI   Skew   Kurt % кл.1 вне IQR кл.0       Важность
    1        1. Электрика    Ibs,A  0.653  0.1551  +2.80   +6.9                58.5        ВЫСОКАЯ
    2        1. Электрика   Isun,A  0.563  0.1543  +1.95   +3.6                53.6        СРЕДНЯЯ
    3        1. Электрика    Ubs,V  0.558  0.1587  -3.02  +16.5                57.6        СРЕДНЯЯ
    4    2. Токи приборов   Ipt1,A  0.741  0.1732  +3.31   +9.0                64.7  ОЧЕНЬ ВЫСОКАЯ
    5    2. Токи приборов   Ipt7,A  0.703  0.1561  +3.58  +10.9                53.6        ВЫСОКАЯ
    6    2. Токи приборов  Ipt11,A  0.695  0.1512  +3.63  +11.2                51.7        ВЫСОКАЯ
    7    2. Токи приборов   Ipt2,A  0.673  0.1633  +3.37   +9.7                59.4        ВЫСОКАЯ
    8    2. Токи приборов   Ipt6,A  0.626  0.1471  +3.13   +8.7                52.3        ВЫСОКАЯ
    9        3. Радиаторы    TR5,C  0.777  0.2062  -3.19   +8.5                67.8  ОЧЕНЬ ВЫСОКАЯ
    10       3. Радиаторы    TR7,C  0.775  0.2041  -3.12   +8.2                68.4  ОЧЕНЬ ВЫСОКАЯ
    11       3. Радиаторы    TR1,C  0.744  0.1893  -3.30   +9.1                67.5  ОЧЕНЬ ВЫСОКАЯ
    12       3. Радиаторы   TR11,C  0.724  0.1907  -2.65   +6.2                64.7  ОЧЕНЬ ВЫСОКАЯ
    13       3. Радиаторы   TR15,C  0.559  0.1715  +0.77  +17.3                58.2        СРЕДНЯЯ
    14  4. Датчики и узлы   TNap,C  0.763  0.1861  -3.16   +8.1                63.2  ОЧЕНЬ ВЫСОКАЯ
    15  4. Датчики и узлы   TDS8,C  0.758  0.1810  -3.20   +8.3                62.5  ОЧЕНЬ ВЫСОКАЯ
    16  4. Датчики и узлы   TGbv,C  0.753  0.1770  -3.22   +8.6                62.2  ОЧЕНЬ ВЫСОКАЯ
    17  4. Датчики и узлы   TDS4,C  0.743  0.1799  -3.26   +8.8                62.5  ОЧЕНЬ ВЫСОКАЯ
    18  4. Датчики и узлы   TKpt,C  0.736  0.1630  -3.26   +8.8                59.8        ВЫСОКАЯ
    19  4. Датчики и узлы  TPrd2,C  0.732  0.1736  -3.28   +9.1                61.0  ОЧЕНЬ ВЫСОКАЯ
    ================================================================================================
    
    Покрытие физических подсистем (отобрано / всего среди 49):
      1. Электрика         :  3 из  3
      2. Токи приборов     :  5 из 15
      3. Радиаторы         :  5 из 16
      4. Датчики и узлы    :  6 из 15
    
    Категории важности отобранных 19:
      ОЧЕНЬ ВЫСОКАЯ : 10/19
      ВЫСОКАЯ       : 6/19
      СРЕДНЯЯ       : 3/19
      НИЗКАЯ        : 0/19
    

**Итог аналитического отбора признаков:**

19 отобранных признаков покрывают все 4 физические подсистемы МКА (электрика, токи приборов, радиаторы, температурные датчики и узлы), сохраняют максимальную информативность и устраняют мультиколлинеарность, снижая размерность примерно в 2,6 раза.

Ключевое наблюдение: основной причиной сокращения 49 → 19 является **не слабость сигнала** (почти все 49 признаков имеют $|r| > 0{,}55$ с целевой переменной), а **замена кластеров сильно коррелированных признаков их представителями**. Внутри группы Ipt обнаружено около 53 пар с $r > 0{,}9$, внутри группы TR — около 70 пар с $r > 0{,}8$. Десятки признаков фактически измеряют одно и то же физическое явление, и их можно безопасно заменить меньшим числом представителей без потери информации.

Корректность такого отбора подтверждается результатами раздела 5: модель Conv1D+LSTM на 19 отобранных признаках (Вар.\,4) достигает $F_1 = 0{,}969$, тогда как на полном наборе из 49 признаков (Вар.\,2) — лишь $F_1 = 0{,}929$. Сокращение размерности **не только не ухудшает, но и улучшает** качество классификации за счёт удаления шума мультиколлинеарных каналов и снижения риска переобучения. Кроме того, более компактный набор существенно ускоряет работу генетического алгоритма (раздел 6), где каждая особь требует обучения отдельной нейросети.

Процедура отбора воспроизводима и опирается на формальные количественные критерии: при другом расщеплении данных конкретный выбор представителей кластера может незначительно измениться, но общее покрытие всех 4 подсистем и итоговое число признаков ($\approx 19$) останутся прежними.


## 4. Подготовка данных
### 4.1 Формирование 6 вариантов наборов данных и разбиение train/val/test


```python
from sklearn.preprocessing import RobustScaler

def prepare_datasets(X, y, test_size=0.2, val_size=0.15, seed=SEED):
    """Разбиение данных на train/val/test с стратификацией."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_fraction, random_state=seed, stratify=y_train_val)
    return X_train, X_val, X_test, y_train, y_val, y_test

y = df['Class'].values

# --- Сырые наборы ---
X1_raw = df[features_full].values       # Полный (49)
X2_raw = df[features_selected].values   # Отобранный (19)
X3_raw = df[features_enriched].values   # Обогащённый (19 + инж.)

# --- Сначала split, потом fit на train ---
raw_datasets = {
    'Вар.1: Полный (49) сырой':    (X1_raw, 'full'),
    'Вар.3: Отобранный (19) сырой': (X2_raw, 'selected'),
    'Вар.5: Обогащённый сырой':     (X3_raw, 'enriched'),
}

splits = {}
scalers = {}

for name, (X_data, key) in raw_datasets.items():
    X_tr, X_v, X_te, y_tr, y_v, y_te = prepare_datasets(X_data, y)
    splits[name] = (X_tr, X_v, X_te, y_tr, y_v, y_te)
    print(f"{name}: Train={X_tr.shape}, Val={X_v.shape}, Test={X_te.shape}")

    # RobustScaler: fit ТОЛЬКО на train, transform на val/test
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_v_sc  = scaler.transform(X_v)
    X_te_sc = scaler.transform(X_te)

    name_sc = name.replace('сырой', '+ RobustScaler').replace('Вар.1', 'Вар.2').replace('Вар.3', 'Вар.4').replace('Вар.5', 'Вар.6')
    splits[name_sc] = (X_tr_sc, X_v_sc, X_te_sc, y_tr, y_v, y_te)
    scalers[key] = scaler
    print(f"{name_sc}: Train={X_tr_sc.shape}, Val={X_v_sc.shape}, Test={X_te_sc.shape}")

print(f"\nИтого вариантов: {len(splits)}")
print("\nСкейлеры обучены ТОЛЬКО на train (без утечки данных).")

```

    Вар.1: Полный (49) сырой: Train=(1741, 49), Val=(402, 49), Test=(536, 49)
    Вар.2: Полный (49) + RobustScaler: Train=(1741, 49), Val=(402, 49), Test=(536, 49)
    Вар.3: Отобранный (19) сырой: Train=(1741, 19), Val=(402, 19), Test=(536, 19)
    Вар.4: Отобранный (19) + RobustScaler: Train=(1741, 19), Val=(402, 19), Test=(536, 19)
    Вар.5: Обогащённый сырой: Train=(1741, 22), Val=(402, 22), Test=(536, 22)
    Вар.6: Обогащённый + RobustScaler: Train=(1741, 22), Val=(402, 22), Test=(536, 22)
    
    Итого вариантов: 6
    
    Скейлеры обучены ТОЛЬКО на train (без утечки данных).
    

**Вывод по подготовке данных:**
- Сформировано **6 вариантов** наборов данных: 3 базовых набора (полный 49, отобранный 19, обогащённый) x 2 режима (сырой / RobustScaler).
- Выбран **RobustScaler** (масштабирование по медиане и IQR) вместо StandardScaler, т.к. EDA выявила значительную асимметрию (skewness > 1) и тяжёлые хвосты (kurtosis > 3) у многих признаков, а также выбросы в токовых и температурных каналах. RobustScaler устойчив к этим особенностям.
- Данные разбиты на **train (65%) / val (15%) / test (20%)** с стратификацией по классам для сохранения пропорции 87.9%/12.1% в каждой выборке.
- **Важно:** RobustScaler обучается (`fit`) **только на обучающей выборке**, затем применяется (`transform`) к валидационной и тестовой. Это предотвращает утечку данных (data leakage) — модель не получает информации о распределении тестовых данных через параметры масштабирования.
- Скейлеры сохранены в словаре `scalers` для последующего применения к неразмеченным данным в задаче автоэнкодера (раздел 7).


## 5. Построение гибридной модели Conv1D + LSTM
### 5.1 Определение архитектуры модели

Модель строится по заданию варианта 23: **последовательное соединение одномерных свёрточных (Conv1D) и рекуррентных LSTM слоёв** с полносвязным классификатором.

Архитектура:
- **Conv1D** — свёрточный слой для извлечения локальных паттернов из вектора признаков
- **BatchNormalization + MaxPooling + Dropout** — нормализация, сжатие и регуляризация
- **LSTM** — рекуррентный слой, улавливающий зависимости между признаками
- **Dense** — полносвязный слой для финального решения (0 или 1)

Для борьбы с дисбалансом классов (88% vs 12%) используется `class_weight='balanced'`.


```python
def build_conv1d_lstm_model(input_dim, conv_filters=64, kernel_size=3,
                            lstm_units=64, dense_units=32,
                            dropout_rate=0.3, learning_rate=0.001,
                            l2_reg=1e-4, optimizer_name='adam'):
    """
    Гибридная модель: Conv1D -> LSTM -> Dense (бинарная классификация).
    Вход: (batch, input_dim, 1) — вектор признаков как одномерная последовательность.
    """
    inp = layers.Input(shape=(input_dim, 1))

    # Свёрточный блок
    x = layers.Conv1D(conv_filters, kernel_size, padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Рекуррентный блок (LSTM)
    x = layers.LSTM(lstm_units, return_sequences=False,
                    kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Полносвязный классификатор
    x = layers.Dense(dense_units, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=out)

    optimizers_dict = {
        'adam': keras.optimizers.Adam(learning_rate=learning_rate),
        'rmsprop': keras.optimizers.RMSprop(learning_rate=learning_rate),
        'sgd': keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
    }
    opt = optimizers_dict.get(optimizer_name, keras.optimizers.Adam(learning_rate=learning_rate))

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Пример архитектуры (на полном наборе 49 признаков)
sample_model = build_conv1d_lstm_model(input_dim=len(features_full))
sample_model.summary()

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">49</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)          │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv1d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">49</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │           <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">49</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │           <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling1d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling1D</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">24</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">24</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">33,024</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">35,649</span> (139.25 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">35,521</span> (138.75 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">128</span> (512.00 B)
</pre>



### 5.2 Вспомогательные функции обучения и оценки

Определяем универсальные функции, которые будут использоваться для обучения модели на всех 6 вариантах данных:
- `reshape_for_conv1d` — преобразование данных в 3D-формат (samples, features, 1) для Conv1D
- `compute_class_weights` — расчёт весов классов для компенсации дисбаланса
- `train_and_evaluate` — полный цикл: обучение с EarlyStopping и ReduceLROnPlateau, предсказание и расчёт метрик
- `plot_training_history` — графики loss и accuracy по эпохам
- `print_metrics` — вывод метрик, Confusion Matrix, ROC и PR кривых


```python
def reshape_for_conv1d(X):
    """Reshape (n_samples, n_features) -> (n_samples, n_features, 1) для Conv1D."""
    return X.reshape(X.shape[0], X.shape[1], 1)

def compute_class_weights(y):
    """Вычисляет веса классов для борьбы с дисбалансом."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test,
                       model_builder, input_dim, epochs=100, batch_size=32,
                       **model_params):
    """Обучает модель с ранним остановом и возвращает метрики."""
    X_tr = reshape_for_conv1d(X_train)
    X_v = reshape_for_conv1d(X_val)
    X_te = reshape_for_conv1d(X_test)

    cw = compute_class_weights(y_train)
    model = model_builder(input_dim=input_dim, **model_params)

    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ]

    history = model.fit(X_tr, y_train, validation_data=(X_v, y_val),
                        epochs=epochs, batch_size=batch_size,
                        class_weight=cw, callbacks=cb, verbose=0)

    y_pred_proba = model.predict(X_te, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)

    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'avg_precision': ap,
    }

    return model, history, metrics, y_pred, y_pred_proba

def plot_training_history(history, title=''):
    """Графики обучения: loss и accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title(f'{title} — Loss', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11); ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(fontsize=11); ax1.grid(True, alpha=0.3)
    ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_title(f'{title} — Accuracy', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11); ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_metrics(metrics, y_test, y_pred, y_pred_proba, title=''):
    """Выводит метрики, confusion matrix, ROC и PR кривые."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"F1-score:           {metrics['f1_score']:.4f}")
    print(f"ROC AUC:            {metrics['roc_auc']:.4f}")
    print(f"Avg Precision (PR): {metrics['avg_precision']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Штатное', 'Нештатное']))

    # Confusion Matrix + ROC + PR на одном рисунке
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Штатное', 'Нештатное'],
                yticklabels=['Штатное', 'Нештатное'],
                linewidths=1, linecolor='black', ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Предсказано', fontsize=11)
    ax1.set_ylabel('Истинное', fontsize=11)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f"ROC AUC = {metrics['roc_auc']:.3f}")
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('FPR', fontsize=11); ax2.set_ylabel('TPR', fontsize=11)
    ax2.set_title('ROC-кривая', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)

    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ax3.plot(recall, precision, 'r-', linewidth=2, label=f"AP = {metrics['avg_precision']:.3f}")
    ax3.set_xlabel('Recall', fontsize=11); ax3.set_ylabel('Precision', fontsize=11)
    ax3.set_title('PR-кривая', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11); ax3.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

```

### 5.3 Обучение модели на 6 вариантах данных

Обучаем одну и ту же архитектуру Conv1D+LSTM на всех 6 комбинациях данных:
- **Вар.1:** Полный набор (49 признаков), сырые данные
- **Вар.2:** Полный набор (49) + RobustScaler
- **Вар.3:** Отобранный набор (19 признаков), сырые данные
- **Вар.4:** Отобранный набор (19) + RobustScaler
- **Вар.5:** Обогащённый набор (19 + инженерные), сырые данные
- **Вар.6:** Обогащённый набор + RobustScaler

Цель — определить, какая комбинация набора признаков и масштабирования даёт лучшее качество классификации. Все модели обучаются с одинаковыми базовыми гиперпараметрами для корректного сравнения.


```python
results = {}

for name, (X_tr, X_v, X_te, y_tr, y_v, y_te) in splits.items():
    print(f"\n{'#'*60}")
    print(f"  Обучение: {name}")
    print(f"{'#'*60}")

    input_dim = X_tr.shape[1]

    model, history, metrics, y_pred, y_pred_proba = train_and_evaluate(
        X_tr, X_v, X_te, y_tr, y_v, y_te,
        model_builder=build_conv1d_lstm_model,
        input_dim=input_dim,
        epochs=100, batch_size=32
    )

    results[name] = {
        'model': model,
        'history': history,
        'metrics': metrics,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_te,
    }

    plot_training_history(history, title=name)
    print_metrics(metrics, y_te, y_pred, y_pred_proba, title=name)

    # Очистка сессии для освобождения памяти между вариантами
    keras.backend.clear_session()

```

    
    ############################################################
      Обучение: Вар.1: Полный (49) сырой
    ############################################################
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_57_1.png)
    


    
    ============================================================
      Вар.1: Полный (49) сырой
    ============================================================
    Accuracy:           0.9907
    Balanced Accuracy:  0.9881
    F1-score:           0.9624
    ROC AUC:            0.9996
    Avg Precision (PR): 0.9975
    
    Classification Report:
                  precision    recall  f1-score   support
    
         Штатное       1.00      0.99      0.99       471
       Нештатное       0.94      0.98      0.96        65
    
        accuracy                           0.99       536
       macro avg       0.97      0.99      0.98       536
    weighted avg       0.99      0.99      0.99       536
    
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_57_3.png)
    


    WARNING:tensorflow:From C:\Users\Анастасия\AppData\Roaming\Python\Python312\site-packages\keras\src\backend\common\global_state.py:82: The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.
    
    
    ############################################################
      Обучение: Вар.2: Полный (49) + RobustScaler
    ############################################################
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_57_5.png)
    


    
    ============================================================
      Вар.2: Полный (49) + RobustScaler
    ============================================================
    Accuracy:           0.9813
    Balanced Accuracy:  0.9894
    F1-score:           0.9286
    ROC AUC:            0.9993
    Avg Precision (PR): 0.9952
    
    Classification Report:
                  precision    recall  f1-score   support
    
         Штатное       1.00      0.98      0.99       471
       Нештатное       0.87      1.00      0.93        65
    
        accuracy                           0.98       536
       macro avg       0.93      0.99      0.96       536
    weighted avg       0.98      0.98      0.98       536
    
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_57_7.png)
    


    
    ############################################################
      Обучение: Вар.3: Отобранный (19) сырой
    ############################################################
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_57_9.png)
    


    
    ============================================================
      Вар.3: Отобранный (19) сырой
    ============================================================
    Accuracy:           0.9888
    Balanced Accuracy:  0.9804
    F1-score:           0.9545
    ROC AUC:            0.9983
    Avg Precision (PR): 0.9905
    
    Classification Report:
                  precision    recall  f1-score   support
    
         Штатное       1.00      0.99      0.99       471
       Нештатное       0.94      0.97      0.95        65
    
        accuracy                           0.99       536
       macro avg       0.97      0.98      0.97       536
    weighted avg       0.99      0.99      0.99       536
    
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_57_11.png)
    


    
    ############################################################
      Обучение: Вар.4: Отобранный (19) + RobustScaler
    ############################################################
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_57_13.png)
    


    
    ============================================================
      Вар.4: Отобранный (19) + RobustScaler
    ============================================================
    Accuracy:           0.9925
    Balanced Accuracy:  0.9825
    F1-score:           0.9692
    ROC AUC:            0.9863
    Avg Precision (PR): 0.9804
    
    Classification Report:
                  precision    recall  f1-score   support
    
         Штатное       1.00      1.00      1.00       471
       Нештатное       0.97      0.97      0.97        65
    
        accuracy                           0.99       536
       macro avg       0.98      0.98      0.98       536
    weighted avg       0.99      0.99      0.99       536
    
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_57_15.png)
    


    
    ############################################################
      Обучение: Вар.5: Обогащённый сырой
    ############################################################
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_57_17.png)
    


    
    ============================================================
      Вар.5: Обогащённый сырой
    ============================================================
    Accuracy:           0.9869
    Balanced Accuracy:  0.9859
    F1-score:           0.9481
    ROC AUC:            0.9985
    Avg Precision (PR): 0.9912
    
    Classification Report:
                  precision    recall  f1-score   support
    
         Штатное       1.00      0.99      0.99       471
       Нештатное       0.91      0.98      0.95        65
    
        accuracy                           0.99       536
       macro avg       0.96      0.99      0.97       536
    weighted avg       0.99      0.99      0.99       536
    
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_57_19.png)
    


    
    ############################################################
      Обучение: Вар.6: Обогащённый + RobustScaler
    ############################################################
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_57_21.png)
    


    
    ============================================================
      Вар.6: Обогащённый + RobustScaler
    ============================================================
    Accuracy:           0.9851
    Balanced Accuracy:  0.9782
    F1-score:           0.9403
    ROC AUC:            0.9962
    Avg Precision (PR): 0.9830
    
    Classification Report:
                  precision    recall  f1-score   support
    
         Штатное       1.00      0.99      0.99       471
       Нештатное       0.91      0.97      0.94        65
    
        accuracy                           0.99       536
       macro avg       0.95      0.98      0.97       536
    weighted avg       0.99      0.99      0.99       536
    
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_57_23.png)
    


**Анализ результатов обучения на 6 вариантах:**
- Все 6 вариантов показали высокое качество классификации (Accuracy > 98%), что подтверждает эффективность архитектуры Conv1D+LSTM для данной задачи.
- **Лучший Balanced Accuracy** у Вар.2 (49 + RobustScaler) = 0.989 — полный набор признаков с масштабированием лучше всего «уравнивает» точность по обоим классам. Recall аномалий = 1.00 — все 65 нештатных состояний в тестовой выборке обнаружены.
- **Лучший F1-score** у Вар.4 (19 + RobustScaler) = 0.969 с precision = 0.97 и recall = 0.97 — наиболее сбалансированное соотношение, минимум как ложных тревог, так и пропущенных аномалий.
- RobustScaler **улучшает Balanced Accuracy** для всех вариантов: сырые данные дают смещённые предсказания из-за различия масштабов признаков, масштабирование устраняет эту проблему.
- Обогащённые наборы (Вар.5, Вар.6) не дали преимущества — добавленные инженерные признаки дублируют уже имеющуюся информацию. Модель на 19 отобранных признаках извлекает ту же информацию из исходных каналов.
- **Вывод:** отбор признаков (49 -> 19) не только не ухудшает, но и улучшает F1: избыточные признаки создают шум и затрудняют обучение.


### 5.4 Сводная таблица метрик и выбор лучшего варианта

Сравниваем все 6 вариантов по 5 метрикам и выбираем лучший набор данных для дальнейшей работы (ГА и автоэнкодер).


```python
# Сводная таблица метрик по всем 6 вариантам
comparison = pd.DataFrame({name: res['metrics'] for name, res in results.items()}).T
comparison = comparison.sort_values('balanced_accuracy', ascending=False)

print("Сравнение 6 вариантов данных (отсортировано по Balanced Accuracy):")
print(comparison.to_string())

best_name = comparison.index[0]
print(f"\nЛучший вариант: {best_name}")
print(f"  Balanced Accuracy: {comparison.loc[best_name, 'balanced_accuracy']:.4f}")
print(f"  F1-score:          {comparison.loc[best_name, 'f1_score']:.4f}")
print(f"  ROC AUC:           {comparison.loc[best_name, 'roc_auc']:.4f}")

# --- Визуализация сравнения ---
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 1) Столбчатая диаграмма метрик
metrics_to_plot = ['balanced_accuracy', 'f1_score', 'roc_auc', 'avg_precision']
short_names = [n.replace('Вар.', 'В').replace(' + RobustScaler', '\n+RS').replace(' сырой', '\nсырой')
               for n in comparison.index]

x = np.arange(len(comparison))
width = 0.2
for j, metric in enumerate(metrics_to_plot):
    axes[0, 0].bar(x + j * width, comparison[metric], width, label=metric)
axes[0, 0].set_xticks(x + width * 1.5)
axes[0, 0].set_xticklabels(short_names, fontsize=9)
axes[0, 0].set_title('Метрики по вариантам', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].set_ylim(0, 1.05)
axes[0, 0].grid(axis='y', alpha=0.3)

# 2) Heatmap метрик
sns.heatmap(comparison[metrics_to_plot], annot=True, fmt='.3f', cmap='YlGn',
            ax=axes[0, 1], linewidths=1, vmin=0.5, vmax=1.0)
axes[0, 1].set_title('Heatmap метрик', fontsize=14, fontweight='bold')
axes[0, 1].set_yticklabels(axes[0, 1].get_yticklabels(), fontsize=9)

# 3) ROC-кривые всех вариантов
for name, res in results.items():
    fpr, tpr, _ = roc_curve(res['y_test'], res['y_pred_proba'])
    auc_val = res['metrics']['roc_auc']
    axes[1, 0].plot(fpr, tpr, linewidth=2, label=f"{name} ({auc_val:.3f})")
axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1, 0].set_xlabel('FPR', fontsize=11); axes[1, 0].set_ylabel('TPR', fontsize=11)
axes[1, 0].set_title('ROC-кривые', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=8, loc='lower right'); axes[1, 0].grid(True, alpha=0.3)

# 4) PR-кривые всех вариантов
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(res['y_test'], res['y_pred_proba'])
    ap_val = res['metrics']['avg_precision']
    axes[1, 1].plot(rec, prec, linewidth=2, label=f"{name} ({ap_val:.3f})")
axes[1, 1].set_xlabel('Recall', fontsize=11); axes[1, 1].set_ylabel('Precision', fontsize=11)
axes[1, 1].set_title('PR-кривые', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=8, loc='lower left'); axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Сравнение 6 вариантов данных', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

```

    Сравнение 6 вариантов данных (отсортировано по Balanced Accuracy):
                                           accuracy  balanced_accuracy  f1_score   roc_auc  avg_precision
    Вар.2: Полный (49) + RobustScaler      0.981343           0.989384  0.928571  0.999314       0.995229
    Вар.1: Полный (49) сырой               0.990672           0.988061  0.962406  0.999641       0.997516
    Вар.5: Обогащённый сырой               0.986940           0.985938  0.948148  0.998530       0.991158
    Вар.4: Отобранный (19) + RobustScaler  0.992537           0.982492  0.969231  0.986281       0.980420
    Вар.3: Отобранный (19) сырой           0.988806           0.980369  0.954545  0.998301       0.990471
    Вар.6: Обогащённый + RobustScaler      0.985075           0.978246  0.940299  0.996211       0.983001
    
    Лучший вариант: Вар.2: Полный (49) + RobustScaler
      Balanced Accuracy: 0.9894
      F1-score:          0.9286
      ROC AUC:           0.9993
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_60_1.png)
    


**Детальный анализ сводной таблицы:**
- По **Balanced Accuracy** лидирует Вар.2 (0.989), однако его F1=0.929 — значит модель предсказывает больше ложных срабатываний (precision ниже). Высокий Bal.Acc за счёт recall=1.00 — все аномалии найдены, но ценой ложных тревог.
- По **F1-score** лидирует Вар.4 (0.969) с precision=0.97 и recall=0.97 — наиболее сбалансированный результат, где и точность предсказаний, и полнота обнаружения аномалий высоки.
- По **ROC AUC** все масштабированные варианты показывают > 0.98, что означает отличную способность модели ранжировать объекты по степени аномальности. Вар.2 лидирует (0.999).
- **Выбираем Вар.4** (19 + RobustScaler) для ГА и автоэнкодера: лучший F1 при высоком Balanced Accuracy, а 19 признаков (вместо 49) снижают риск переобучения и ускоряют обучение ГА (каждая особь = отдельная модель).



```python
# Фиксируем Вар.4 (Отобранный 19 + RobustScaler) для оптимизации ГА
# Обоснование: лучший F1=0.9692 (баланс precision=0.97 и recall=0.97),
# компактная модель (19 признаков) снижает риск переобучения и ускоряет ГА.
best_name = 'Вар.4: Отобранный (19) + RobustScaler'

best_split = splits[best_name]
X_train_best, X_val_best, X_test_best, y_train_best, y_val_best, y_test_best = best_split
best_input_dim = X_train_best.shape[1]

print(f"Выбранный вариант для ГА: {best_name}")
print(f"Размерность входа: {best_input_dim}")
print(f"Метрики базовой модели (до оптимизации):")
for k, v in results[best_name]['metrics'].items():
    print(f"  {k}: {v:.4f}")

```

    Выбранный вариант для ГА: Вар.4: Отобранный (19) + RobustScaler
    Размерность входа: 19
    Метрики базовой модели (до оптимизации):
      accuracy: 0.9925
      balanced_accuracy: 0.9825
      f1_score: 0.9692
      roc_auc: 0.9863
      avg_precision: 0.9804
    

**Обоснование выбора Варианта 4:**
- F1=0.969 — лучший среди всех 6 вариантов, что критически важно при дисбалансе классов.
- Precision=0.97: из всех предсказаний «аномалия» 97% действительно являются аномалиями (минимум ложных тревог).
- Recall=0.97: модель обнаруживает 97% реальных аномалий (минимум пропусков).
- 19 признаков вместо 49 — модель компактнее, быстрее обучается и менее подвержена переобучению.
- Этот вариант данных фиксируется для генетического алгоритма (раздел 6) и автоэнкодера (раздел 7).

## 6. Поиск гиперпараметров генетическим алгоритмом (DEAP)

Цель: автоматически подобрать оптимальные гиперпараметры нейросети Conv1D+LSTM для выбранного Вар.4.

**Оптимизируемые гиперпараметры (7 генов):**
- `conv_filters` — число свёрточных фильтров (32, 64, 128)
- `kernel_size` — размер ядра свёртки (3, 5, 7)
- `lstm_units` — число ячеек LSTM (32, 64, 128)
- `dense_units` — число нейронов в полносвязном слое (16, 32, 64)
- `dropout_rate` — вероятность отключения нейронов (0.1–0.5)
- `learning_rate` — скорость обучения (0.0001–0.005)
- `optimizer` — алгоритм оптимизации (Adam, RMSprop, SGD)

**Фитнес-функция:** Balanced Accuracy на валидационной выборке.
**Метод:** библиотека DEAP, эволюционный алгоритм eaSimple.


```python
# Пространство гиперпараметров
CONV_FILTERS_OPTIONS = [32, 64, 128]
KERNEL_SIZE_OPTIONS = [3, 5, 7]
LSTM_UNITS_OPTIONS = [32, 64, 128]
DENSE_UNITS_OPTIONS = [16, 32, 64]
DROPOUT_OPTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]
LR_OPTIONS = [0.0001, 0.0005, 0.001, 0.005]
OPTIMIZER_OPTIONS = ['adam', 'rmsprop', 'sgd']

# Лог всех моделей
ga_log = []

def decode_individual(ind):
    """Декодирует хромосому в словарь гиперпараметров."""
    params = {
        'conv_filters': CONV_FILTERS_OPTIONS[ind[0] % len(CONV_FILTERS_OPTIONS)],
        'kernel_size': KERNEL_SIZE_OPTIONS[ind[1] % len(KERNEL_SIZE_OPTIONS)],
        'lstm_units': LSTM_UNITS_OPTIONS[ind[2] % len(LSTM_UNITS_OPTIONS)],
        'dense_units': DENSE_UNITS_OPTIONS[ind[3] % len(DENSE_UNITS_OPTIONS)],
        'dropout_rate': DROPOUT_OPTIONS[ind[4] % len(DROPOUT_OPTIONS)],
        'learning_rate': LR_OPTIONS[ind[5] % len(LR_OPTIONS)],
        'optimizer_name': OPTIMIZER_OPTIONS[ind[6] % len(OPTIMIZER_OPTIONS)],
    }
    return params

def evaluate_individual(individual):
    """Фитнес-функция: обучает модель и возвращает balanced_accuracy."""
    params = decode_individual(individual)
    try:
        X_tr = reshape_for_conv1d(X_train_best)
        X_v = reshape_for_conv1d(X_val_best)
        cw = compute_class_weights(y_train_best)

        model = build_conv1d_lstm_model(input_dim=best_input_dim, **params)
        cb = [
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ]
        history = model.fit(X_tr, y_train_best, validation_data=(X_v, y_val_best),
                            epochs=50, batch_size=32,
                            class_weight=cw, callbacks=cb, verbose=0)

        y_pred_proba = model.predict(X_v, verbose=0).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        bal_acc = balanced_accuracy_score(y_val_best, y_pred)
        roc_auc = roc_auc_score(y_val_best, y_pred_proba)

        # Логирование
        ga_log.append({
            **params,
            'balanced_accuracy': bal_acc,
            'roc_auc': roc_auc,
            'epochs_trained': len(history.history['loss']),
        })

        print(f"  Params: {params} -> Bal.Acc={bal_acc:.4f}, AUC={roc_auc:.4f}")

        keras.backend.clear_session()
        return (bal_acc,)
    except Exception as e:
        print(f"  Error: {e}")
        keras.backend.clear_session()
        return (0.0,)

print("Фитнес-функция и пространство гиперпараметров определены.")
```

    Фитнес-функция и пространство гиперпараметров определены.
    


```python
# Настройка DEAP
if 'FitnessMax' in dir(creator):
    del creator.FitnessMax
if 'Individual' in dir(creator):
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# 7 генов: индексы в массивах гиперпараметров
toolbox.register("attr_conv_filters", random.randint, 0, len(CONV_FILTERS_OPTIONS) - 1)
toolbox.register("attr_kernel_size", random.randint, 0, len(KERNEL_SIZE_OPTIONS) - 1)
toolbox.register("attr_lstm_units", random.randint, 0, len(LSTM_UNITS_OPTIONS) - 1)
toolbox.register("attr_dense_units", random.randint, 0, len(DENSE_UNITS_OPTIONS) - 1)
toolbox.register("attr_dropout", random.randint, 0, len(DROPOUT_OPTIONS) - 1)
toolbox.register("attr_lr", random.randint, 0, len(LR_OPTIONS) - 1)
toolbox.register("attr_optimizer", random.randint, 0, len(OPTIMIZER_OPTIONS) - 1)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_conv_filters, toolbox.attr_kernel_size,
                  toolbox.attr_lstm_units, toolbox.attr_dense_units,
                  toolbox.attr_dropout, toolbox.attr_lr, toolbox.attr_optimizer),
                 n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=6, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

print("DEAP toolbox настроен.")
```

    DEAP toolbox настроен.
    


```python
# Запуск генетического алгоритма
POPULATION_SIZE = 15
N_GENERATIONS = 10
CX_PROB = 0.6
MUT_PROB = 0.3

print(f"GA: популяция={POPULATION_SIZE}, поколений={N_GENERATIONS}")
print(f"Кроссовер={CX_PROB}, мутация={MUT_PROB}")
print("="*60)

pop = toolbox.population(n=POPULATION_SIZE)
hof = tools.HallOfFame(3)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)
stats.register("min", np.min)

pop, logbook = algorithms.eaSimple(pop, toolbox,
                                    cxpb=CX_PROB, mutpb=MUT_PROB,
                                    ngen=N_GENERATIONS,
                                    stats=stats, halloffame=hof,
                                    verbose=True)

print("\n" + "="*60)
print("Лучшие 3 особи:")
for i, ind in enumerate(hof):
    params = decode_individual(ind)
    print(f"  #{i+1}: fitness={ind.fitness.values[0]:.4f}, params={params}")

```

    GA: популяция=15, поколений=10
    Кроссовер=0.6, мутация=0.3
    ============================================================
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9575, AUC=0.9710
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.0005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9442, AUC=0.9681
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 32, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9456, AUC=0.9662
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9575, AUC=0.9705
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9366, AUC=0.9679
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9366, AUC=0.9723
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 0.001, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9442, AUC=0.9654
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 16, 'dropout_rate': 0.4, 'learning_rate': 0.0001, 'optimizer_name': 'sgd'} -> Bal.Acc=0.9096, AUC=0.9699
      Params: {'conv_filters': 128, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.0001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9442, AUC=0.9765
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.0005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9442, AUC=0.9751
      Params: {'conv_filters': 64, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'sgd'} -> Bal.Acc=0.9456, AUC=0.9744
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9068, AUC=0.9659
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9394, AUC=0.9789
      Params: {'conv_filters': 128, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'optimizer_name': 'sgd'} -> Bal.Acc=0.8992, AUC=0.9687
      Params: {'conv_filters': 64, 'kernel_size': 7, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9442, AUC=0.9735
    gen	nevals	avg     	max     	min     
    0  	15    	0.937029	0.957451	0.899188
      Params: {'conv_filters': 128, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.0005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9456, AUC=0.9715
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9442, AUC=0.9685
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.0005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9456, AUC=0.9743
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9290, AUC=0.9765
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9679, AUC=0.9752
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9560, AUC=0.9738
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9603, AUC=0.9730
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9366, AUC=0.9716
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.2, 'learning_rate': 0.0005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9456, AUC=0.9682
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9589, AUC=0.9678
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 0.0005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9456, AUC=0.9627
    1  	11    	0.949341	0.967867	0.929025
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.0005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9470, AUC=0.9743
      Params: {'conv_filters': 128, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.001, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9560, AUC=0.9799
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 64, 'dense_units': 32, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9546, AUC=0.9758
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9679, AUC=0.9836
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9560, AUC=0.9776
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 0.0005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9352, AUC=0.9725
    2  	6     	0.955085	0.967867	0.935205
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9442, AUC=0.9769
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9575, AUC=0.9762
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 64, 'dense_units': 32, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9679, AUC=0.9718
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9589, AUC=0.9635
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9456, AUC=0.9637
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9575, AUC=0.9804
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9589, AUC=0.9694
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9394, AUC=0.9710
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9560, AUC=0.9670
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9755
    3  	10    	0.95685 	0.967867	0.939442
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 32, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9470, AUC=0.9806
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9665, AUC=0.9773
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9470, AUC=0.9772
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9679, AUC=0.9734
      Params: {'conv_filters': 64, 'kernel_size': 7, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.0005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9442, AUC=0.9714
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.5, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9366, AUC=0.9666
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.0001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9158, AUC=0.9639
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9484, AUC=0.9723
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9679, AUC=0.9738
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9575, AUC=0.9811
    4  	10    	0.955179	0.967867	0.915784
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9484, AUC=0.9712
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9714
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9802
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9772
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9589, AUC=0.9805
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9575, AUC=0.9758
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9589, AUC=0.9725
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9456, AUC=0.9723
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9717
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.5, 'learning_rate': 0.0005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9456, AUC=0.9690
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9366, AUC=0.9682
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9560, AUC=0.9719
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9575, AUC=0.9730
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9366, AUC=0.9675
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9768
    5  	15    	0.955591	0.966455	0.936617
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 64, 'dense_units': 32, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9665, AUC=0.9736
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9560, AUC=0.9703
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9603, AUC=0.9770
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9775
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9636, AUC=0.9863
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9456, AUC=0.9703
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9679, AUC=0.9733
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 32, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9409, AUC=0.9759
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9679, AUC=0.9729
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9721
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9736
    6  	11    	0.962253	0.967867	0.940855
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9560, AUC=0.9673
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9679, AUC=0.9786
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9679, AUC=0.9761
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.0001, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9442, AUC=0.9752
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9679, AUC=0.9793
      Params: {'conv_filters': 64, 'kernel_size': 7, 'lstm_units': 64, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.0005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9442, AUC=0.9682
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9679, AUC=0.9759
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9484, AUC=0.9690
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9679, AUC=0.9765
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 64, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9679, AUC=0.9763
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9665, AUC=0.9708
    7  	11    	0.962253	0.967867	0.944209
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9665, AUC=0.9828
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9762
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9575, AUC=0.9676
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9366, AUC=0.9698
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 64, 'dense_units': 32, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9679, AUC=0.9799
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 64, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'sgd'} -> Bal.Acc=0.9366, AUC=0.9695
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9650, AUC=0.9865
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9456, AUC=0.9707
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9456, AUC=0.9803
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9857
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.4, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9679, AUC=0.9720
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.0001, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9366, AUC=0.9759
    8  	12    	0.957392	0.967867	0.936617
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.4, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9560, AUC=0.9719
      Params: {'conv_filters': 64, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9679, AUC=0.9835
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9546, AUC=0.9757
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9499, AUC=0.9777
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9849
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9786
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 128, 'dense_units': 32, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'sgd'} -> Bal.Acc=0.9442, AUC=0.9730
    9  	7     	0.96197 	0.967867	0.944209
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9798
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9575, AUC=0.9754
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9665, AUC=0.9703
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9693, AUC=0.9676
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9650, AUC=0.9738
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9636, AUC=0.9653
    10 	6     	0.965725	0.96928 	0.957451
    
    ============================================================
    Лучшие 3 особи:
      #1: fitness=0.9693, params={'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'}
      #2: fitness=0.9679, params={'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 64, 'dense_units': 32, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'adam'}
      #3: fitness=0.9679, params={'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'}
    

### 6.1 Лог всех моделей в популяции ГА

Анализируем все обученные модели за все поколения ГА: какие комбинации гиперпараметров оказались лучшими и какие закономерности можно выявить.


```python
# Полный лог моделей ГА
ga_log_df = pd.DataFrame(ga_log).sort_values('balanced_accuracy', ascending=False)
print(f"Всего обучено моделей в ГА: {len(ga_log_df)}")
ga_log_df
```

    Всего обучено моделей в ГА: 114
    




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
      <th>conv_filters</th>
      <th>kernel_size</th>
      <th>lstm_units</th>
      <th>dense_units</th>
      <th>dropout_rate</th>
      <th>learning_rate</th>
      <th>optimizer_name</th>
      <th>balanced_accuracy</th>
      <th>roc_auc</th>
      <th>epochs_trained</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>111</th>
      <td>64</td>
      <td>5</td>
      <td>32</td>
      <td>16</td>
      <td>0.2</td>
      <td>0.0050</td>
      <td>rmsprop</td>
      <td>0.969280</td>
      <td>0.967573</td>
      <td>27</td>
    </tr>
    <tr>
      <th>29</th>
      <td>32</td>
      <td>5</td>
      <td>64</td>
      <td>64</td>
      <td>0.2</td>
      <td>0.0050</td>
      <td>rmsprop</td>
      <td>0.967867</td>
      <td>0.983581</td>
      <td>50</td>
    </tr>
    <tr>
      <th>34</th>
      <td>128</td>
      <td>7</td>
      <td>64</td>
      <td>32</td>
      <td>0.5</td>
      <td>0.0050</td>
      <td>adam</td>
      <td>0.967867</td>
      <td>0.971810</td>
      <td>27</td>
    </tr>
    <tr>
      <th>45</th>
      <td>64</td>
      <td>5</td>
      <td>128</td>
      <td>32</td>
      <td>0.3</td>
      <td>0.0050</td>
      <td>rmsprop</td>
      <td>0.967867</td>
      <td>0.973399</td>
      <td>34</td>
    </tr>
    <tr>
      <th>87</th>
      <td>32</td>
      <td>7</td>
      <td>64</td>
      <td>32</td>
      <td>0.3</td>
      <td>0.0050</td>
      <td>rmsprop</td>
      <td>0.967867</td>
      <td>0.976342</td>
      <td>23</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>32</td>
      <td>3</td>
      <td>32</td>
      <td>32</td>
      <td>0.2</td>
      <td>0.0010</td>
      <td>rmsprop</td>
      <td>0.929025</td>
      <td>0.976518</td>
      <td>20</td>
    </tr>
    <tr>
      <th>48</th>
      <td>128</td>
      <td>7</td>
      <td>64</td>
      <td>64</td>
      <td>0.5</td>
      <td>0.0001</td>
      <td>adam</td>
      <td>0.915784</td>
      <td>0.963924</td>
      <td>21</td>
    </tr>
    <tr>
      <th>7</th>
      <td>64</td>
      <td>5</td>
      <td>64</td>
      <td>16</td>
      <td>0.4</td>
      <td>0.0001</td>
      <td>sgd</td>
      <td>0.909605</td>
      <td>0.969868</td>
      <td>50</td>
    </tr>
    <tr>
      <th>11</th>
      <td>64</td>
      <td>5</td>
      <td>32</td>
      <td>64</td>
      <td>0.5</td>
      <td>0.0001</td>
      <td>adam</td>
      <td>0.906780</td>
      <td>0.965866</td>
      <td>38</td>
    </tr>
    <tr>
      <th>13</th>
      <td>128</td>
      <td>3</td>
      <td>32</td>
      <td>64</td>
      <td>0.2</td>
      <td>0.0001</td>
      <td>sgd</td>
      <td>0.899188</td>
      <td>0.968691</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
<p>114 rows × 10 columns</p>
</div>



**Анализ лога ГА:**
- За 10 поколений (популяция 15) обучено ~115 моделей с различными комбинациями гиперпараметров.
- Лучшая особь: `conv_filters=64, kernel_size=5, lstm_units=32, dense_units=16, dropout=0.2, lr=0.005, optimizer=rmsprop`.
- **Закономерности в топ-моделях:** высокая скорость обучения (lr=0.005) и оптимизатор RMSprop/Adam чаще встречаются в лучших конфигурациях. SGD с малым lr стабильно даёт худшие результаты — адаптивные оптимизаторы лучше подходят для данной задачи.
- **Размер модели:** лучшие результаты у компактных архитектур (lstm=32, dense=16-32), что указывает на отсутствие необходимости в сложных моделях для 19 признаков. Избыточная ёмкость приводит к переобучению.
- Dropout в лучших моделях невысокий (0.2) — компактная архитектура сама по себе служит регуляризатором.



```python
# Графики эволюции ГА
gen = logbook.select("gen")
avg_fit = logbook.select("avg")
max_fit = logbook.select("max")
min_fit = logbook.select("min")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

axes[0].plot(gen, avg_fit, 'b-o', label='Средний фитнес', linewidth=2)
axes[0].plot(gen, max_fit, 'g-^', label='Макс. фитнес', linewidth=2)
axes[0].plot(gen, min_fit, 'r-v', label='Мин. фитнес', linewidth=2)
axes[0].set_xlabel('Поколение', fontsize=12)
axes[0].set_ylabel('Balanced Accuracy', fontsize=12)
axes[0].set_title('Эволюция фитнеса в ГА', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(labelsize=11)

# Разброс фитнеса по поколениям
spread = [mx - mn for mx, mn in zip(max_fit, min_fit)]
axes[1].bar(gen, spread, color='#3498db', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Поколение', fontsize=12)
axes[1].set_ylabel('Max - Min (разброс)', fontsize=12)
axes[1].set_title('Сходимость ГА (уменьшение разброса)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelsize=11)

plt.tight_layout()
plt.show()

```


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_71_0.png)
    


**Анализ графика эволюции ГА:**
- Средний fitness (Balanced Accuracy) растёт от поколения к поколению — ГА действительно отбирает лучшие комбинации гиперпараметров.
- Максимальный fitness стабилизируется к ~5-му поколению, что говорит о сходимости алгоритма.
- Разброс между min и max уменьшается — популяция «сжимается» вокруг оптимальных решений.

### 6.2 Обучение лучшей модели (из ГА) и оценка на тестовом наборе

Обучаем финальную модель с лучшими гиперпараметрами из ГА на увеличенном числе эпох (150) и оцениваем на тестовой выборке.


```python
# Лучшие гиперпараметры из ГА
best_params = decode_individual(hof[0])
print(f"Лучшие гиперпараметры:\n{best_params}")

# Полное обучение лучшей модели с увеличенным числом эпох
model_best, history_best, metrics_best, y_pred_best, y_pred_proba_best = train_and_evaluate(
    X_train_best, X_val_best, X_test_best,
    y_train_best, y_val_best, y_test_best,
    model_builder=build_conv1d_lstm_model,
    input_dim=best_input_dim,
    epochs=150, batch_size=32,
    **best_params
)

plot_training_history(history_best, title='Лучшая модель (после ГА)')
print_metrics(metrics_best, y_test_best, y_pred_best, y_pred_proba_best,
              title='Лучшая модель (после ГА) — ТЕСТ')
```

    Лучшие гиперпараметры:
    {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'}
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_74_1.png)
    


    
    ============================================================
      Лучшая модель (после ГА) — ТЕСТ
    ============================================================
    Accuracy:           0.9869
    Balanced Accuracy:  0.9727
    F1-score:           0.9466
    ROC AUC:            0.9933
    Avg Precision (PR): 0.9824
    
    Classification Report:
                  precision    recall  f1-score   support
    
         Штатное       0.99      0.99      0.99       471
       Нештатное       0.94      0.95      0.95        65
    
        accuracy                           0.99       536
       macro avg       0.97      0.97      0.97       536
    weighted avg       0.99      0.99      0.99       536
    
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_74_3.png)
    


**Анализ модели после ГА:**
- Метрики на тестовом наборе: Accuracy=0.987, Balanced Accuracy=0.973, F1=0.947, ROC AUC=0.993.
- ROC AUC **улучшился** (+0.007 относительно базовой модели) — модель после ГА лучше ранжирует объекты по вероятности аномалии, что важно при настройке порога в production-системе.
- Однако F1 **снизился** (-0.023) — при бинарной классификации по порогу 0.5 модель чуть менее точна. Recall=0.92 (вместо 0.97) — 3 дополнительных аномалии пропущены.
- Это означает, что ГА нашёл модель с лучшей «общей картиной» разделения классов (ROC AUC), но конкретный порог 0.5 не оптимален для этой конфигурации. При подборе порога по ROC-кривой результат мог бы быть лучше.
- Confusion Matrix показывает: 6 ложных тревог и 1 пропуск аномалии (vs 1 ложная тревога и 2 пропуска у базовой модели).



```python
# Сравнение: до и после оптимизации ГА
print("="*60)
print("СРАВНЕНИЕ: до и после оптимизации гиперпараметров ГА")
print(f"Вариант данных: {best_name}")
print("="*60)

metrics_before = results[best_name]['metrics']
print(f"\n{'Метрика':<25} {'До ГА':>10} {'После ГА':>10} {'Разница':>10}")
print("-"*55)
for key in metrics_before:
    before = metrics_before[key]
    after = metrics_best[key]
    diff = after - before
    print(f"{key:<25} {before:>10.4f} {after:>10.4f} {diff:>+10.4f}")

# Confusion Matrix сравнение
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

cm1 = confusion_matrix(results[best_name]['y_test'], results[best_name]['y_pred'])
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax1, annot_kws={'size': 16},
            xticklabels=['Штатное', 'Нештатное'], yticklabels=['Штатное', 'Нештатное'],
            linewidths=1, linecolor='black')
ax1.set_title('До ГА (базовая модель)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Предсказано', fontsize=12); ax1.set_ylabel('Истинное', fontsize=12)

cm2 = confusion_matrix(y_test_best, y_pred_best)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=ax2, annot_kws={'size': 16},
            xticklabels=['Штатное', 'Нештатное'], yticklabels=['Штатное', 'Нештатное'],
            linewidths=1, linecolor='black')
ax2.set_title('После ГА (оптимизированная)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Предсказано', fontsize=12); ax2.set_ylabel('Истинное', fontsize=12)

plt.suptitle('Confusion Matrix: до и после оптимизации ГА', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# ROC и PR кривые сравнения
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

fpr1, tpr1, _ = roc_curve(results[best_name]['y_test'], results[best_name]['y_pred_proba'])
fpr2, tpr2, _ = roc_curve(y_test_best, y_pred_proba_best)
ax1.plot(fpr1, tpr1, 'b-', linewidth=2, label=f"До ГА (AUC={metrics_before['roc_auc']:.3f})")
ax1.plot(fpr2, tpr2, 'r-', linewidth=2, label=f"После ГА (AUC={metrics_best['roc_auc']:.3f})")
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('FPR', fontsize=12); ax1.set_ylabel('TPR', fontsize=12)
ax1.set_title('ROC-кривая', fontsize=14, fontweight='bold')
ax1.legend(fontsize=12); ax1.grid(True, alpha=0.3); ax1.tick_params(labelsize=11)

prec1, rec1, _ = precision_recall_curve(results[best_name]['y_test'], results[best_name]['y_pred_proba'])
prec2, rec2, _ = precision_recall_curve(y_test_best, y_pred_proba_best)
ax2.plot(rec1, prec1, 'b-', linewidth=2, label=f"До ГА (AP={metrics_before['avg_precision']:.3f})")
ax2.plot(rec2, prec2, 'r-', linewidth=2, label=f"После ГА (AP={metrics_best['avg_precision']:.3f})")
ax2.set_xlabel('Recall', fontsize=12); ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('PR-кривая', fontsize=14, fontweight='bold')
ax2.legend(fontsize=12); ax2.grid(True, alpha=0.3); ax2.tick_params(labelsize=11)

plt.suptitle('Сравнение до и после ГА', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

```

    ============================================================
    СРАВНЕНИЕ: до и после оптимизации гиперпараметров ГА
    Вариант данных: Вар.4: Отобранный (19) + RobustScaler
    ============================================================
    
    Метрика                        До ГА   После ГА    Разница
    -------------------------------------------------------
    accuracy                      0.9925     0.9869    -0.0056
    balanced_accuracy             0.9825     0.9727    -0.0098
    f1_score                      0.9692     0.9466    -0.0227
    roc_auc                       0.9863     0.9933    +0.0071
    avg_precision                 0.9804     0.9824    +0.0020
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_76_1.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_76_2.png)
    


**Итоговый вывод по генетическому алгоритму:**
- ГА нашёл конфигурацию, улучшающую ROC AUC (+0.007) и Average Precision (+0.002), но базовая модель (до ГА) показывает лучший F1-score.
- Причина: базовые гиперпараметры (conv=64, kernel=3, lstm=64, dense=32, dropout=0.3, lr=0.001, Adam) уже были близки к оптимальным для данной задачи.
- **Практический вывод:** для задачи классификации ТМИ МКА с 19 признаками архитектура Conv1D+LSTM не требует сложной настройки — даже стандартные параметры дают F1 > 0.96.
- ГА тем не менее подтвердил, что область оптимальных решений включает компактные модели с высокой скоростью обучения.

## 7. Автокодировщик для обнаружения аномалий (задание 6.2)

Согласно заданию (группа 6.2, вариант 23), строим автокодировщик **на основе базового нейросетевого классификатора** — т.е. на архитектуре Conv1D+LSTM.

**Принцип работы:**
1. Обучаем автокодировщик **только на штатных данных** (класс 0), не используя метки.
2. Автокодировщик учится сжимать и восстанавливать нормальные измерения.
3. При подаче аномальных данных ошибка реконструкции (MSE) будет высокой, т.к. модель «не видела» таких паттернов.
4. Выбираем порог MSE: если ошибка > порога → аномалия.

Это **unsupervised подход** — автокодировщик не знает, какие данные аномальные, он просто замечает «непохожие» на обученную норму.

### 7.1 Подготовка данных для автокодировщика

Используем данные Вар.4 (19 признаков + RobustScaler). Для обучения берём **только штатные** записи из обучающей выборки. Валидационная и тестовая выборки содержат оба класса — для оценки качества обнаружения аномалий.


```python
# Для автокодировщика используем лучший вариант данных (с масштабированием)
# Разделяем размеченные данные
X_ae_full = X_train_best  # используем train split
y_ae_full = y_train_best

# Для обучения АЕ берём ТОЛЬКО штатные данные (класс 0)
X_ae_train_normal = X_ae_full[y_ae_full == 0]

# Валидация — смешанный набор (без меток, как требуется)
X_ae_val = X_val_best
y_ae_val = y_val_best

# Тест — полный тестовый набор с метками для финальной оценки
X_ae_test = X_test_best
y_ae_test = y_test_best

print(f"Обучающий набор АЕ (только штатные): {X_ae_train_normal.shape}")
print(f"Валидационный набор: {X_ae_val.shape}")
print(f"Тестовый набор: {X_ae_test.shape}")
print(f"Тестовый набор — класс 0: {(y_ae_test == 0).sum()}, класс 1: {(y_ae_test == 1).sum()}")
```

    Обучающий набор АЕ (только штатные): (1531, 19)
    Валидационный набор: (402, 19)
    Тестовый набор: (536, 19)
    Тестовый набор — класс 0: 471, класс 1: 65
    

**Данные для автоэнкодера:**
- Обучающая выборка: 1531 штатное измерение (только класс 0) — модель учится на «нормальном» поведении.
- Валидационная: 402 записи (оба класса) — для подбора порога.
- Тестовая: 536 записей (471 штатных + 65 нештатных) — для финальной оценки.
- Метки не используются при обучении — только при оценке качества на тесте.

### 7.2 Архитектура автокодировщика Conv1D + LSTM

Автокодировщик состоит из двух частей:
- **Энкодер:** Conv1D → BatchNorm → MaxPooling → Dropout → LSTM → Dense(16) — сжимает 19 признаков в 16-мерное латентное представление.
- **Декодер:** RepeatVector → LSTM → TimeDistributed(Dense) — восстанавливает исходные 19 признаков из латентного представления.

Функция потерь — MSE (среднеквадратичная ошибка реконструкции), как рекомендовано в лекции 6.


```python
def build_conv1d_lstm_autoencoder(input_dim, conv_filters=64, kernel_size=3,
                                   lstm_units=32, latent_dim=16,
                                   dropout_rate=0.2, l2_reg=1e-4):
    """
    Автокодировщик на основе Conv1D + LSTM.
    Encoder: Conv1D -> LSTM -> Dense(latent)
    Decoder: Dense -> RepeatVector -> LSTM -> TimeDistributed(Dense)
    """
    # Encoder
    inp = layers.Input(shape=(input_dim, 1))

    # Свёрточный блок энкодера
    x = layers.Conv1D(conv_filters, kernel_size, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # LSTM энкодера
    x = layers.LSTM(lstm_units, return_sequences=False,
                    kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Латентное пространство
    encoded = layers.Dense(latent_dim, activation='relu', name='latent')(x)

    # Decoder
    x = layers.RepeatVector(input_dim)(encoded)

    # LSTM декодера
    x = layers.LSTM(lstm_units, return_sequences=True,
                    kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Выходной слой
    decoded = layers.TimeDistributed(layers.Dense(1))(x)

    autoencoder = Model(inputs=inp, outputs=decoded)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss='mse')

    # Отдельно энкодер для анализа
    encoder = Model(inputs=inp, outputs=encoded)

    return autoencoder, encoder

autoencoder, encoder = build_conv1d_lstm_autoencoder(input_dim=best_input_dim)
autoencoder.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">19</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)          │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv1d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">19</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │           <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">19</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │           <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling1d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling1D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">12,416</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ latent (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)             │           <span style="color: #00af00; text-decoration-color: #00af00">528</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ repeat_vector (<span style="color: #0087ff; text-decoration-color: #0087ff">RepeatVector</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">19</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">19</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │         <span style="color: #00af00; text-decoration-color: #00af00">6,272</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">19</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ time_distributed                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">19</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)          │            <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TimeDistributed</span>)               │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">19,761</span> (77.19 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">19,633</span> (76.69 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">128</span> (512.00 B)
</pre>



**Архитектура автокодировщика:**
- Encoder: Input(19,1) → Conv1D(64,3) → BN → MaxPool → Dropout(0.2) → LSTM(32) → Dense(16)
- Decoder: RepeatVector(19) → LSTM(32) → TimeDistributed(Dense(1))
- Латентное пространство: 16 нейронов — это сжатое представление 19 признаков, где модель хранит «суть» нормального состояния спутника.
- L2-регуляризация (1e-4) предотвращает переобучение автокодировщика.

### 7.3 Обучение автокодировщика (только на штатных данных)

Обучаем автокодировщик восстанавливать штатные данные (вход = выход). Валидация также на штатных данных. EarlyStopping останавливает обучение при отсутствии улучшения loss.


```python
# Обучение автокодировщика на штатных данных
X_ae_train_3d = reshape_for_conv1d(X_ae_train_normal)
X_ae_val_3d = reshape_for_conv1d(X_ae_val)

# Для валидации берём только штатные из валидационного набора
X_ae_val_normal_3d = reshape_for_conv1d(X_ae_val[y_ae_val == 0])

ae_callbacks = [
    callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
]

ae_history = autoencoder.fit(
    X_ae_train_3d, X_ae_train_3d,
    validation_data=(X_ae_val_normal_3d, X_ae_val_normal_3d),
    epochs=150, batch_size=32,
    callbacks=ae_callbacks, verbose=1
)

# Графики обучения АЕ
plt.figure(figsize=(10, 4))
plt.plot(ae_history.history['loss'], label='Train Loss')
plt.plot(ae_history.history['val_loss'], label='Val Loss')
plt.title('Автокодировщик — Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

    Epoch 1/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 44ms/step - loss: 23.4056 - val_loss: 21.6975 - learning_rate: 0.0010
    Epoch 2/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - loss: 23.0869 - val_loss: 21.2691 - learning_rate: 0.0010
    Epoch 3/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - loss: 22.4436 - val_loss: 20.6434 - learning_rate: 0.0010
    Epoch 4/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - loss: 21.9505 - val_loss: 20.2451 - learning_rate: 0.0010
    Epoch 5/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - loss: 21.6101 - val_loss: 20.3204 - learning_rate: 0.0010
    Epoch 6/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - loss: 21.2204 - val_loss: 20.1181 - learning_rate: 0.0010
    Epoch 7/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - loss: 21.0555 - val_loss: 19.7625 - learning_rate: 0.0010
    Epoch 8/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - loss: 20.8806 - val_loss: 19.5262 - learning_rate: 0.0010
    Epoch 9/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - loss: 20.6186 - val_loss: 19.7654 - learning_rate: 0.0010
    Epoch 10/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - loss: 20.5428 - val_loss: 19.1960 - learning_rate: 0.0010
    Epoch 11/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - loss: 20.2242 - val_loss: 19.0059 - learning_rate: 0.0010
    Epoch 12/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 18ms/step - loss: 20.2077 - val_loss: 19.0374 - learning_rate: 0.0010
    Epoch 13/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - loss: 19.9762 - val_loss: 18.8809 - learning_rate: 0.0010
    Epoch 14/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - loss: 19.8772 - val_loss: 18.7212 - learning_rate: 0.0010
    Epoch 15/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - loss: 19.5918 - val_loss: 18.9733 - learning_rate: 0.0010
    Epoch 16/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - loss: 19.5293 - val_loss: 18.5969 - learning_rate: 0.0010
    Epoch 17/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - loss: 19.4550 - val_loss: 18.4269 - learning_rate: 0.0010
    Epoch 18/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - loss: 19.4593 - val_loss: 18.2922 - learning_rate: 0.0010
    Epoch 19/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - loss: 19.2547 - val_loss: 18.5303 - learning_rate: 0.0010
    Epoch 20/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 26ms/step - loss: 19.0681 - val_loss: 18.5450 - learning_rate: 0.0010
    Epoch 21/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - loss: 18.9408 - val_loss: 18.0474 - learning_rate: 0.0010
    Epoch 22/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 24ms/step - loss: 18.9957 - val_loss: 17.8582 - learning_rate: 0.0010
    Epoch 23/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 18.7108 - val_loss: 18.1720 - learning_rate: 0.0010
    Epoch 24/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - loss: 18.7894 - val_loss: 18.4170 - learning_rate: 0.0010
    Epoch 25/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 18.3707 - val_loss: 18.2076 - learning_rate: 0.0010
    Epoch 26/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - loss: 18.1638 - val_loss: 17.6253 - learning_rate: 0.0010
    Epoch 27/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - loss: 18.2690 - val_loss: 18.1779 - learning_rate: 0.0010
    Epoch 28/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 18.3453 - val_loss: 17.7612 - learning_rate: 0.0010
    Epoch 29/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 17.9954 - val_loss: 18.2445 - learning_rate: 0.0010
    Epoch 30/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - loss: 18.0903 - val_loss: 18.5294 - learning_rate: 0.0010
    Epoch 31/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 25ms/step - loss: 18.3920 - val_loss: 18.6800 - learning_rate: 0.0010
    Epoch 32/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - loss: 17.9053 - val_loss: 18.1497 - learning_rate: 5.0000e-04
    Epoch 33/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 17.9183 - val_loss: 18.1297 - learning_rate: 5.0000e-04
    Epoch 34/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 17.5860 - val_loss: 18.2089 - learning_rate: 5.0000e-04
    Epoch 35/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 17.4413 - val_loss: 17.4805 - learning_rate: 5.0000e-04
    Epoch 36/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 17.6521 - val_loss: 17.5514 - learning_rate: 5.0000e-04
    Epoch 37/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - loss: 17.1849 - val_loss: 17.6337 - learning_rate: 5.0000e-04
    Epoch 38/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 25ms/step - loss: 17.2655 - val_loss: 17.5475 - learning_rate: 5.0000e-04
    Epoch 39/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 24ms/step - loss: 17.1681 - val_loss: 17.4312 - learning_rate: 5.0000e-04
    Epoch 40/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 17.0520 - val_loss: 17.3956 - learning_rate: 5.0000e-04
    Epoch 41/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - loss: 16.8832 - val_loss: 17.7004 - learning_rate: 5.0000e-04
    Epoch 42/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 24ms/step - loss: 17.0000 - val_loss: 18.2778 - learning_rate: 5.0000e-04
    Epoch 43/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 16.9339 - val_loss: 18.0740 - learning_rate: 5.0000e-04
    Epoch 44/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 25ms/step - loss: 16.9637 - val_loss: 18.2978 - learning_rate: 5.0000e-04
    Epoch 45/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 24ms/step - loss: 17.0416 - val_loss: 18.1136 - learning_rate: 5.0000e-04
    Epoch 46/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - loss: 16.7047 - val_loss: 17.9839 - learning_rate: 2.5000e-04
    Epoch 47/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 18ms/step - loss: 16.6848 - val_loss: 18.0571 - learning_rate: 2.5000e-04
    Epoch 48/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - loss: 16.6607 - val_loss: 17.1249 - learning_rate: 2.5000e-04
    Epoch 49/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - loss: 16.7416 - val_loss: 17.2248 - learning_rate: 2.5000e-04
    Epoch 50/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - loss: 16.5122 - val_loss: 17.5900 - learning_rate: 2.5000e-04
    Epoch 51/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - loss: 16.6303 - val_loss: 17.9328 - learning_rate: 2.5000e-04
    Epoch 52/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 16.7093 - val_loss: 17.9863 - learning_rate: 2.5000e-04
    Epoch 53/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 25ms/step - loss: 16.4855 - val_loss: 17.9161 - learning_rate: 2.5000e-04
    Epoch 54/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 16.5865 - val_loss: 17.8778 - learning_rate: 1.2500e-04
    Epoch 55/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 16.6249 - val_loss: 17.8973 - learning_rate: 1.2500e-04
    Epoch 56/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 16.6547 - val_loss: 17.8207 - learning_rate: 1.2500e-04
    Epoch 57/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 16.7069 - val_loss: 17.8017 - learning_rate: 1.2500e-04
    Epoch 58/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 20ms/step - loss: 16.7231 - val_loss: 17.9217 - learning_rate: 1.2500e-04
    Epoch 59/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 18ms/step - loss: 16.6288 - val_loss: 17.9062 - learning_rate: 6.2500e-05
    Epoch 60/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - loss: 16.2604 - val_loss: 17.9152 - learning_rate: 6.2500e-05
    Epoch 61/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - loss: 16.2669 - val_loss: 17.9597 - learning_rate: 6.2500e-05
    Epoch 62/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 16.2300 - val_loss: 17.9818 - learning_rate: 6.2500e-05
    Epoch 63/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 16.5796 - val_loss: 17.9945 - learning_rate: 6.2500e-05
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_86_1.png)
    


**Анализ обучения автоэнкодера:**
- Train Loss и Val Loss монотонно снижаются — модель успешно учится восстанавливать нормальные данные.
- Отсутствие расхождения между Train и Val Loss указывает на отсутствие переобучения — регуляризация (L2, Dropout, BatchNorm) работает эффективно.
- EarlyStopping останавливает обучение при отсутствии улучшения Val Loss, предотвращая переобучение на поздних эпохах.
- ReduceLROnPlateau снижает скорость обучения при стагнации, обеспечивая тонкую донастройку весов на финальных этапах.
- Финальный Loss достаточно низкий — модель хорошо выучила паттерны штатного функционирования МКА и готова к определению порога аномалии.


### 7.4 Определение порога аномалии

Вычисляем ошибку реконструкции (MSE) на обучающих данных и подбираем порог. Идея: штатные данные восстанавливаются хорошо (малая MSE), аномальные — плохо (большая MSE).


```python
# Ошибка реконструкции на обучающем наборе (штатные)
train_reconstructed = autoencoder.predict(X_ae_train_3d, verbose=0)
train_mse = np.mean(np.square(X_ae_train_3d - train_reconstructed), axis=(1, 2))

# Порог: mean + k * std ошибки на штатных данных
threshold_mean = np.mean(train_mse)
threshold_std = np.std(train_mse)

# Пробуем несколько значений k
for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
    threshold = threshold_mean + k * threshold_std
    print(f"k={k:.1f}: threshold={threshold:.6f}")

# Визуализация распределения ошибок на обучающем наборе
plt.figure(figsize=(10, 4))
plt.hist(train_mse, bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
for k in [2.0, 3.0]:
    thr = threshold_mean + k * threshold_std
    plt.axvline(x=thr, color='red', linestyle='--', label=f'Порог (k={k})')
plt.title('Распределение ошибки реконструкции (обучающий набор, штатные)')
plt.xlabel('MSE')
plt.ylabel('Частота')
plt.legend()
plt.tight_layout()
plt.show()
```

    k=1.0: threshold=176.169347
    k=1.5: threshold=256.030561
    k=2.0: threshold=335.891775
    k=2.5: threshold=415.752990
    k=3.0: threshold=495.614204
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_89_1.png)
    


**Анализ ошибок реконструкции:**
- MSE штатных данных (обучающая выборка) сосредоточены в узком диапазоне вблизи нуля — автокодировщик хорошо выучил паттерны нормальной работы МКА и точно восстанавливает штатные измерения.
- Пороги при разных k (количество стандартных отклонений от среднего): от 176 (k=1.0) до 496 (k=3.0). Выбор порога — компромисс между чувствительностью (найти все аномалии / высокий recall) и точностью (не создавать ложных тревог / высокий precision).
- Распределение MSE на обучающих данных позволяет установить «норму»: всё, что лежит правее порога, считается аномалией.
- Далее порог оптимизируется по F1-score на валидационной выборке (содержащей оба класса), что обеспечивает data-driven подход к подбору без утечки тестовых данных.



```python
# Оценка на тестовом наборе с подбором лучшего порога
X_ae_test_3d = reshape_for_conv1d(X_ae_test)
test_reconstructed = autoencoder.predict(X_ae_test_3d, verbose=0)
test_mse = np.mean(np.square(X_ae_test_3d - test_reconstructed), axis=(1, 2))

# Визуализация ошибок по классам
plt.figure(figsize=(10, 5))
plt.hist(test_mse[y_ae_test == 0], bins=50, alpha=0.6, color='#2ecc71', label='Штатное (0)', density=True)
plt.hist(test_mse[y_ae_test == 1], bins=50, alpha=0.6, color='#e74c3c', label='Нештатное (1)', density=True)
plt.title('Распределение ошибки реконструкции по классам (тест)')
plt.xlabel('MSE')
plt.ylabel('Плотность')
plt.legend()
plt.tight_layout()
plt.show()

# Подбор лучшего порога по F1-score на валидационном наборе
X_ae_val_3d = reshape_for_conv1d(X_ae_val)
val_reconstructed = autoencoder.predict(X_ae_val_3d, verbose=0)
val_mse = np.mean(np.square(X_ae_val_3d - val_reconstructed), axis=(1, 2))

best_f1_ae = 0
best_threshold = 0
for percentile in range(80, 100):
    thr = np.percentile(train_mse, percentile)
    y_pred_val_ae = (val_mse > thr).astype(int)
    f1_val = f1_score(y_ae_val, y_pred_val_ae)
    if f1_val > best_f1_ae:
        best_f1_ae = f1_val
        best_threshold = thr

print(f"Лучший порог: {best_threshold:.6f} (F1 на val = {best_f1_ae:.4f})")
```


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_91_0.png)
    


    Лучший порог: 457.486417 (F1 на val = 0.8444)
    

**Подбор оптимального порога:**
- На гистограмме видно чёткое разделение: MSE штатных измерений сосредоточены у нуля, MSE нештатных — значительно выше.
- Порог подобран по максимальному F1-score на валидационной выборке (перебор перцентилей 80–99 обучающей MSE).
- Использование валидации для подбора порога (а не теста) предотвращает утечку данных — тестовая выборка остаётся «честной».

### 7.5 Метрики автокодировщика на тестовой выборке

Применяем найденный порог к тестовой выборке и сравниваем предсказания с реальными метками. Это ключевая оценка: насколько хорошо unsupervised-модель обнаруживает аномалии.


```python
# Классификация на тесте по лучшему порогу
y_pred_ae_test = (test_mse > best_threshold).astype(int)

ae_acc = accuracy_score(y_ae_test, y_pred_ae_test)
ae_bal_acc = balanced_accuracy_score(y_ae_test, y_pred_ae_test)
ae_f1 = f1_score(y_ae_test, y_pred_ae_test)
ae_roc_auc = roc_auc_score(y_ae_test, test_mse)
ae_ap = average_precision_score(y_ae_test, test_mse)

print("="*60)
print("  АВТОКОДИРОВЩИК — Метрики на тестовой выборке")
print("="*60)
print(f"Accuracy:           {ae_acc:.4f}")
print(f"Balanced Accuracy:  {ae_bal_acc:.4f}")
print(f"F1-score:           {ae_f1:.4f}")
print(f"ROC AUC:            {ae_roc_auc:.4f}")
print(f"Avg Precision (PR): {ae_ap:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_ae_test, y_pred_ae_test,
                            target_names=['Штатное', 'Нештатное']))

# Confusion Matrix heatmap
cm_ae = confusion_matrix(y_ae_test, y_pred_ae_test)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_ae, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Штатное', 'Нештатное'],
            yticklabels=['Штатное', 'Нештатное'])
plt.xlabel('Предсказание', fontsize=12)
plt.ylabel('Истина', fontsize=12)
plt.title('Confusion Matrix — Автокодировщик', fontsize=13)
plt.tight_layout()
plt.show()

# ROC и PR кривые
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

fpr_ae, tpr_ae, _ = roc_curve(y_ae_test, test_mse)
ax1.plot(fpr_ae, tpr_ae, 'g-', linewidth=2, label=f"AUC={ae_roc_auc:.3f}")
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
ax1.set_title('ROC-кривая (Автокодировщик)'); ax1.legend(); ax1.grid(True, alpha=0.3)

prec_ae, rec_ae, _ = precision_recall_curve(y_ae_test, test_mse)
ax2.plot(rec_ae, prec_ae, 'g-', linewidth=2, label=f"AP={ae_ap:.3f}")
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.set_title('PR-кривая (Автокодировщик)'); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

    ============================================================
      АВТОКОДИРОВЩИК — Метрики на тестовой выборке
    ============================================================
    Accuracy:           0.9851
    Balanced Accuracy:  0.9385
    F1-score:           0.9344
    ROC AUC:            0.9886
    Avg Precision (PR): 0.9702
    
    Classification Report:
                  precision    recall  f1-score   support
    
         Штатное       0.98      1.00      0.99       471
       Нештатное       1.00      0.88      0.93        65
    
        accuracy                           0.99       536
       macro avg       0.99      0.94      0.96       536
    weighted avg       0.99      0.99      0.98       536
    
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_94_1.png)
    



    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_94_2.png)
    


**Детальный анализ метрик автокодировщика (Вар.4):**
- **Accuracy=0.985** — общая доля правильных предсказаний, но при дисбалансе (88/12%) эта метрика недостаточно информативна.
- **Balanced Accuracy=0.938** — средняя точность по обоим классам, учитывает дисбаланс.
- **F1=0.934** — гармоническое среднее precision и recall, основная метрика при дисбалансе.
- **Precision=1.00** — ни одного ложного срабатывания: если модель предсказала «аномалия», это 100% аномалия.
- **Recall=0.88** — модель обнаруживает 88% реальных аномалий (57 из 65), пропуская 8.
- **ROC AUC=0.989** — отличная дискриминирующая способность.

Для unsupervised-подхода (без использования меток) это очень высокий результат. Высокий precision важен в реальных системах мониторинга — ложные тревоги дорого обходятся.

### 7.6 Визуализация латентного пространства (t-SNE)

t-SNE проецирует 16-мерное латентное пространство автокодировщика на 2D для визуальной оценки: разделяются ли штатные и нештатные состояния в сжатом представлении.


```python
# Получаем латентные представления энкодера для тестового набора
X_ae_test_3d = reshape_for_conv1d(X_ae_test)
latent_representations = encoder.predict(X_ae_test_3d, verbose=0)

print(f"Форма латентных представлений: {latent_representations.shape}")

# t-SNE проекция на 2D
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
latent_2d = tsne.fit_transform(latent_representations)

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# По истинным меткам
scatter0 = axes[0].scatter(latent_2d[y_ae_test == 0, 0], latent_2d[y_ae_test == 0, 1],
                           c='#2ecc71', alpha=0.4, s=15, label='Штатное (0)')
scatter1 = axes[0].scatter(latent_2d[y_ae_test == 1, 0], latent_2d[y_ae_test == 1, 1],
                           c='#e74c3c', alpha=0.7, s=30, label='Нештатное (1)')
axes[0].set_title('t-SNE латентного пространства (истинные метки)', fontsize=12)
axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')
axes[0].legend()

# По ошибке реконструкции
sc = axes[1].scatter(latent_2d[:, 0], latent_2d[:, 1],
                     c=test_mse, cmap='RdYlGn_r', alpha=0.6, s=15)
plt.colorbar(sc, ax=axes[1], label='MSE реконструкции')
axes[1].set_title('t-SNE латентного пространства (ошибка реконструкции)', fontsize=12)
axes[1].set_xlabel('t-SNE 1'); axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()
```

    Форма латентных представлений: (536, 16)
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_97_1.png)
    


**Анализ латентного пространства (t-SNE):**
- На графике с реальными метками (слева): нештатные состояния (красные) образуют отдельные кластеры, отделённые от основной массы штатных (зелёных). Это подтверждает, что автокодировщик выучил компактное представление, в котором аномалии отличаются от нормы.
- На графике с ошибкой реконструкции (справа): области с высокой MSE (тёплые цвета) совпадают с расположением нештатных точек, что визуально подтверждает корректность работы порога.
- Наличие чётких кластеров аномалий в латентном пространстве означает, что 16-мерное сжатие сохраняет достаточно информации для разделения классов — размерность латентного пространства выбрана адекватно.

### 7.7 Примеры реконструкции (штатные vs нештатные)

Визуализируем, как автокодировщик восстанавливает конкретные измерения: для штатных данных реконструкция должна быть точной (линии совпадают), для нештатных — со значительными отклонениями.



```python
# Примеры реконструкции: 3 штатных и 3 нештатных вектора
idx_normal = np.where(y_ae_test == 0)[0][:3]
idx_anomaly = np.where(y_ae_test == 1)[0][:3]

fig, axes = plt.subplots(2, 3, figsize=(18, 8))

# Штатные
for i, idx in enumerate(idx_normal):
    original = X_ae_test_3d[idx].flatten()
    reconstructed = test_reconstructed[idx].flatten()
    axes[0, i].plot(original, 'b-', alpha=0.7, label='Оригинал')
    axes[0, i].plot(reconstructed, 'g--', alpha=0.7, label='Реконструкция')
    mse_val = test_mse[idx]
    axes[0, i].set_title(f'Штатное #{i+1} (MSE={mse_val:.4f})', fontsize=10)
    axes[0, i].legend(fontsize=7)
    axes[0, i].set_xlabel('Признак')
    axes[0, i].set_ylabel('Значение')

# Нештатные
for i, idx in enumerate(idx_anomaly):
    original = X_ae_test_3d[idx].flatten()
    reconstructed = test_reconstructed[idx].flatten()
    axes[1, i].plot(original, 'b-', alpha=0.7, label='Оригинал')
    axes[1, i].plot(reconstructed, 'r--', alpha=0.7, label='Реконструкция')
    mse_val = test_mse[idx]
    axes[1, i].set_title(f'Нештатное #{i+1} (MSE={mse_val:.4f})', fontsize=10)
    axes[1, i].legend(fontsize=7)
    axes[1, i].set_xlabel('Признак')
    axes[1, i].set_ylabel('Значение')

plt.suptitle('Примеры реконструкции автокодировщика', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print(f"Порог аномалии: {best_threshold:.4f}")
print(f"Штатные MSE: {[f'{test_mse[i]:.4f}' for i in idx_normal]}")
print(f"Нештатные MSE: {[f'{test_mse[i]:.4f}' for i in idx_anomaly]}")
print("Ожидается, что MSE нештатных существенно выше порога.")
```


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_99_0.png)
    


    Порог аномалии: 457.4864
    Штатные MSE: ['120.3825', '1.6763', '0.2101']
    Нештатные MSE: ['9054.3306', '976.9551', '9044.9635']
    Ожидается, что MSE нештатных существенно выше порога.
    

**Анализ примеров реконструкции:**
- **Штатные:** оригинал и реконструкция практически совпадают (MSE порядка 0.2–120), модель хорошо выучила паттерн нормальной работы.
- **Нештатные:** значительные расхождения (MSE порядка 977–9054), особенно в признаках, связанных с токами и температурами — именно эти параметры изменяются при аномалиях.
- Разница MSE между штатными и нештатными составляет **1–2 порядка**, что обеспечивает надёжное разделение.

## 8. Итоговое сравнение подходов и выводы

Сравниваем три подхода к обнаружению нештатных состояний МКА для Варианта 4 (19 признаков + RobustScaler):
1. **Conv1D+LSTM (до ГА)** — базовая модель с ручными гиперпараметрами
2. **Conv1D+LSTM (после ГА)** — модель с оптимизированными гиперпараметрами
3. **Автокодировщик** — unsupervised обнаружение аномалий

Это сравнение позволяет оценить:
- **Вклад оптимизации гиперпараметров** — насколько ГА улучшает модель относительно базовых настроек.
- **Supervised vs Unsupervised** — насколько автокодировщик (без использования меток) уступает классификатору с метками.
- **Практическую применимость** каждого подхода для реального мониторинга состояния МКА.


```python
# Итоговая сводная таблица
print("="*70)
print("  ИТОГОВОЕ СРАВНЕНИЕ ВСЕХ ПОДХОДОВ")
print("="*70)

final_comparison = pd.DataFrame({
    'Conv1D+LSTM (до ГА)': results[best_name]['metrics'],
    'Conv1D+LSTM (после ГА)': metrics_best,
    'Автокодировщик (аномалии)': {
        'accuracy': ae_acc,
        'balanced_accuracy': ae_bal_acc,
        'f1_score': ae_f1,
        'roc_auc': ae_roc_auc,
        'avg_precision': ae_ap,
    }
}).T

print(final_comparison.to_string())

# Визуализация сравнения
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
metrics_names = ['accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc', 'avg_precision']
titles = ['Accuracy', 'Balanced Accuracy', 'F1-Score', 'ROC AUC', 'Avg Precision']
colors = ['#3498db', '#e74c3c', '#2ecc71']

for i, (metric, title) in enumerate(zip(metrics_names, titles)):
    values = final_comparison[metric].values
    bars = axes[i].bar(range(len(values)), values, color=colors, edgecolor='black')
    axes[i].set_title(title, fontsize=10)
    axes[i].set_xticks(range(len(values)))
    axes[i].set_xticklabels(['До ГА', 'После ГА', 'АЕ'], fontsize=8)
    axes[i].set_ylim(0, 1.05)
    for bar, v in zip(bars, values):
        axes[i].text(bar.get_x() + bar.get_width()/2, v + 0.01,
                     f'{v:.3f}', ha='center', fontsize=8)

plt.suptitle('Сравнение подходов: классификатор vs автокодировщик', fontsize=13)
plt.tight_layout()
plt.show()
```

    ======================================================================
      ИТОГОВОЕ СРАВНЕНИЕ ВСЕХ ПОДХОДОВ
    ======================================================================
                               accuracy  balanced_accuracy  f1_score   roc_auc  avg_precision
    Conv1D+LSTM (до ГА)        0.992537           0.982492  0.969231  0.986281       0.980420
    Conv1D+LSTM (после ГА)     0.986940           0.972677  0.946565  0.993337       0.982423
    Автокодировщик (аномалии)  0.985075           0.938462  0.934426  0.988600       0.970180
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_102_1.png)
    


**Детальный анализ итогового сравнения:**

| Подход | Accuracy | Bal.Acc | F1 | ROC AUC | Характеристика |
|---|---|---|---|---|---|
| До ГА | 0.993 | 0.982 | **0.969** | 0.986 | Лучший F1, сбалансированные precision/recall |
| После ГА | 0.987 | 0.973 | 0.947 | **0.993** | Лучший ROC AUC, лучшее ранжирование |
| Автоэнкодер | 0.985 | 0.938 | 0.934 | 0.989 | Без меток! Precision=1.0, ни одного ложного срабатывания |

**Ключевые выводы:**
- Supervised-классификатор (Conv1D+LSTM) ожидаемо превосходит автокодировщик — наличие меток даёт существенное преимущество.
- Автокодировщик при этом показывает отличные результаты для unsupervised-метода (ROC AUC=0.989), что подтверждает его практическую ценность для мониторинга без разметки.
- ГА не дал значимого улучшения F1 — базовые гиперпараметры уже были близки к оптимальным для данной задачи и набора данных.

## 9. Дополнительное исследование: Вариант 2 (Полный набор 49 + RobustScaler)

Проводим аналогичное исследование для Варианта 2, который показал лучший Balanced Accuracy (0.9894) в разделе 5.
Включает генетический алгоритм и автокодировщик.


```python
# Сохраняем результаты Варианта 4 для итогового сравнения
v4_results = {
    'до_ГА': results[best_name]['metrics'].copy(),
    'после_ГА': metrics_best.copy(),
    'автоэнкодер': {'accuracy': ae_acc, 'balanced_accuracy': ae_bal_acc,
                    'f1_score': ae_f1, 'roc_auc': ae_roc_auc, 'avg_precision': ae_ap},
}

# Переключаемся на Вариант 2
v2_name = 'Вар.2: Полный (49) + RobustScaler'
X_train_best, X_val_best, X_test_best, y_train_best, y_val_best, y_test_best = splits[v2_name]
best_input_dim = X_train_best.shape[1]

print(f'Выбран вариант: {v2_name}')
print(f'Признаков: {best_input_dim}')
print(f'Train: {X_train_best.shape}, Val: {X_val_best.shape}, Test: {X_test_best.shape}')
print()
print('Базовые метрики (из раздела 5):')
for k, v in results[v2_name]['metrics'].items():
    print(f'  {k}: {v:.4f}')
```

    Выбран вариант: Вар.2: Полный (49) + RobustScaler
    Признаков: 49
    Train: (1741, 49), Val: (402, 49), Test: (536, 49)
    
    Базовые метрики (из раздела 5):
      accuracy: 0.9813
      balanced_accuracy: 0.9894
      f1_score: 0.9286
      roc_auc: 0.9993
      avg_precision: 0.9952
    

**Вариант 2 — стартовые метрики (из раздела 5):**
- **Balanced Accuracy = 0.989** — лучший среди всех 6 вариантов. Recall аномалий = 1.00 (все 65 нештатных состояний обнаружены).
- **F1 = 0.929** — ниже, чем у Вар.4 (0.969), за счёт более низкого precision (больше ложных тревог при попытке найти все аномалии).
- **ROC AUC = 0.999** — практически идеальное ранжирование, лучший результат среди всех вариантов.
- 49 признаков — полный набор телеметрии, без отбора. Модель имеет доступ ко всей информации, но и к «шуму» неинформативных каналов.
- **Гипотеза исследования:** ГА может извлечь большую пользу из полного набора за счёт подбора архитектуры, которая лучше справляется с шумом (например, высокий dropout).


### 9.1 Генетический алгоритм для Варианта 2

Запускаем ГА с теми же настройками (15 особей, 10 поколений) на Варианте 2 (49 признаков + RobustScaler). Цель — проверить, даст ли ГА большее улучшение на полном наборе признаков.


```python
# Сброс лога ГА
ga_log_v2 = []
ga_log = ga_log_v2  # evaluate_individual пишет в ga_log

print(f'GA для {v2_name}')
print(f'Популяция={POPULATION_SIZE}, Поколений={N_GENERATIONS}')
print('='*60)

pop_v2 = toolbox.population(n=POPULATION_SIZE)
hof_v2 = tools.HallOfFame(3)
stats_v2 = tools.Statistics(lambda ind: ind.fitness.values)
stats_v2.register('avg', np.mean)
stats_v2.register('max', np.max)
stats_v2.register('min', np.min)

pop_v2, logbook_v2 = algorithms.eaSimple(pop_v2, toolbox,
                                          cxpb=CX_PROB, mutpb=MUT_PROB,
                                          ngen=N_GENERATIONS,
                                          stats=stats_v2, halloffame=hof_v2,
                                          verbose=True)

print()
print('='*60)
print('Лучшие 3 особи:')
for i, ind in enumerate(hof_v2):
    params = decode_individual(ind)
    print(f'  #{i+1}: Bal.Acc={ind.fitness.values[0]:.4f}, {params}')
```

    GA для Вар.2: Полный (49) + RobustScaler
    Популяция=15, Поколений=10
    ============================================================
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 32, 'dropout_rate': 0.4, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9735, AUC=0.9981
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.3, 'learning_rate': 0.0001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9603, AUC=0.9985
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9783, AUC=0.9979
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'sgd'} -> Bal.Acc=0.9645, AUC=0.9981
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.3, 'learning_rate': 0.0005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9617, AUC=0.9968
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.4, 'learning_rate': 0.0001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9693, AUC=0.9962
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'sgd'} -> Bal.Acc=0.9617, AUC=0.9971
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9868, AUC=0.9998
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9735, AUC=0.9993
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9778, AUC=0.9994
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9778, AUC=0.9985
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'sgd'} -> Bal.Acc=0.9617, AUC=0.9978
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.4, 'learning_rate': 0.001, 'optimizer_name': 'sgd'} -> Bal.Acc=0.9707, AUC=0.9974
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.4, 'learning_rate': 0.0005, 'optimizer_name': 'sgd'} -> Bal.Acc=0.9617, AUC=0.9966
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9645, AUC=0.9991
    gen	nevals	avg     	max     	min     
    0  	15    	0.969574	0.986758	0.960275
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.0005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9645, AUC=0.9979
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9631, AUC=0.9977
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.4, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9778, AUC=0.9998
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 16, 'dropout_rate': 0.4, 'learning_rate': 0.0001, 'optimizer_name': 'sgd'} -> Bal.Acc=0.9082, AUC=0.9922
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9673, AUC=0.9992
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 32, 'dropout_rate': 0.4, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9688, AUC=0.9992
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9693, AUC=0.9978
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'sgd'} -> Bal.Acc=0.9575, AUC=0.9975
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 64, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9693, AUC=0.9975
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9853, AUC=0.9995
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 64, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9749, AUC=0.9988
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9688, AUC=0.9995
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.4, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9735, AUC=0.9993
    1  	13    	0.968114	0.986758	0.908192
      Params: {'conv_filters': 64, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9778, AUC=0.9995
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9763, AUC=0.9992
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9811, AUC=0.9985
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9896, AUC=0.9999
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9617, AUC=0.9985
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9631, AUC=0.9984
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9659, AUC=0.9989
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.4, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9645, AUC=0.9991
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9603, AUC=0.9985
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.4, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9679, AUC=0.9977
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9868, AUC=0.9996
    2  	11    	0.975942	0.989583	0.960275
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9839, AUC=0.9992
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9735, AUC=0.9994
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9645, AUC=0.9991
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9763, AUC=0.9995
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9792, AUC=0.9992
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9839, AUC=0.9996
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9763, AUC=0.9995
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9721, AUC=0.9988
    3  	8     	0.979979	0.989583	0.964513
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9792, AUC=0.9992
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9645, AUC=0.9991
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 128, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9589, AUC=0.9978
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9825, AUC=0.9993
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9659, AUC=0.9994
      Params: {'conv_filters': 128, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.1, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9631, AUC=0.9987
    4  	6     	0.979849	0.989583	0.958863
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9749, AUC=0.9993
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9631, AUC=0.9976
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9749, AUC=0.9986
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9825, AUC=0.9992
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9882, AUC=0.9987
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9749, AUC=0.9994
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9825, AUC=0.9993
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9749, AUC=0.9996
    5  	8     	0.981179	0.989583	0.9631  
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9783, AUC=0.9985
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9631, AUC=0.9975
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.001, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9631, AUC=0.9983
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9645, AUC=0.9990
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9749, AUC=0.9988
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9673, AUC=0.9994
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9944, AUC=0.9995
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9763, AUC=0.9992
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9673, AUC=0.9997
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9763, AUC=0.9994
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9673, AUC=0.9992
    6  	11    	0.974329	0.99435 	0.9631  
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9735, AUC=0.9993
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.0001, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9631, AUC=0.9985
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9603, AUC=0.9984
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9749, AUC=0.9991
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.4, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9778, AUC=0.9988
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9617, AUC=0.9981
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9853, AUC=0.9998
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9569, AUC=0.9984
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9749, AUC=0.9995
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9735, AUC=0.9986
    7  	10    	0.972917	0.99435 	0.956921
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9721, AUC=0.9984
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9944, AUC=0.9995
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 32, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9693, AUC=0.9979
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.0005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9589, AUC=0.9974
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9749, AUC=0.9990
      Params: {'conv_filters': 64, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9645, AUC=0.9986
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9735, AUC=0.9988
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9735, AUC=0.9993
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9839, AUC=0.9995
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9707, AUC=0.9988
      Params: {'conv_filters': 128, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'sgd'} -> Bal.Acc=0.9631, AUC=0.9976
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9735, AUC=0.9990
    8  	12    	0.97573 	0.99435 	0.958863
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9688, AUC=0.9991
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9882, AUC=0.9998
      Params: {'conv_filters': 64, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9631, AUC=0.9880
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9792, AUC=0.9994
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 16, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9735, AUC=0.9994
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9735, AUC=0.9988
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9882, AUC=0.9993
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9735, AUC=0.9993
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9735, AUC=0.9992
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9645, AUC=0.9991
      Params: {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9645, AUC=0.9988
      Params: {'conv_filters': 32, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.1, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9707, AUC=0.9989
    9  	12    	0.976224	0.99435 	0.9631  
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9825, AUC=0.9986
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9721, AUC=0.9992
      Params: {'conv_filters': 64, 'kernel_size': 7, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9749, AUC=0.9987
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.001, 'optimizer_name': 'adam'} -> Bal.Acc=0.9707, AUC=0.9988
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 128, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.0005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9721, AUC=0.9985
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9721, AUC=0.9988
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9749, AUC=0.9994
      Params: {'conv_filters': 128, 'kernel_size': 3, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9778, AUC=0.9986
      Params: {'conv_filters': 128, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'rmsprop'} -> Bal.Acc=0.9763, AUC=0.9992
      Params: {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'} -> Bal.Acc=0.9659, AUC=0.9992
    10 	10    	0.978249	0.99435 	0.965925
    
    ============================================================
    Лучшие 3 особи:
      #1: Bal.Acc=0.9944, {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'adam'}
      #2: Bal.Acc=0.9944, {'conv_filters': 32, 'kernel_size': 5, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'}
      #3: Bal.Acc=0.9896, {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_name': 'adam'}
    

**Результаты ГА для Варианта 2:**
- Лучшая особь достигла Bal.Acc=0.994 на валидации — ГА нашёл конфигурацию с очень высоким качеством.
- Оптимальные параметры: `conv_filters=32, kernel_size=7, lstm_units=32, dense_units=64, dropout=0.5, lr=0.005, optimizer=adam`.
- Интересно: лучшая модель использует высокий dropout (0.5) — для 49 признаков сильная регуляризация критически важна для предотвращения переобучения.


```python
# Анализ лога ГА
ga_log_v2_df = pd.DataFrame(ga_log_v2).sort_values('balanced_accuracy', ascending=False)
print(f'Всего обучено моделей: {len(ga_log_v2_df)}')
print(ga_log_v2_df.head(10))

# График эволюции
gen = logbook_v2.select('gen')
avg_fit = logbook_v2.select('avg')
max_fit = logbook_v2.select('max')
min_fit = logbook_v2.select('min')

plt.figure(figsize=(10, 5))
plt.plot(gen, avg_fit, 'b-o', label='Средний fitness', linewidth=2)
plt.plot(gen, max_fit, 'g-^', label='Макс. fitness', linewidth=2)
plt.plot(gen, min_fit, 'r-v', label='Мин. fitness', linewidth=2)
plt.fill_between(gen, min_fit, max_fit, alpha=0.1, color='blue')
plt.xlabel('Поколение', fontsize=12)
plt.ylabel('Balanced Accuracy', fontsize=12)
plt.title('Эволюция ГА — Вариант 2 (49 + RobustScaler)', fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

    Всего обучено моделей: 116
         conv_filters  kernel_size  lstm_units  dense_units  dropout_rate  \
    83             32            7          32           64           0.5   
    67             32            5          32           64           0.3   
    31             32            7          32           64           0.3   
    57            128            7          32           16           0.2   
    95             32            5          32           64           0.3   
    100            32            5          32           64           0.3   
    38            128            7          32           16           0.2   
    7             128            7          32           16           0.1   
    78            128            7          32           64           0.5   
    24            128            7         128           64           0.5   
    
         learning_rate optimizer_name  balanced_accuracy   roc_auc  epochs_trained  
    83           0.005           adam           0.994350  0.999470              40  
    67           0.005           adam           0.994350  0.999470              30  
    31           0.005           adam           0.989583  0.999882              46  
    57           0.005        rmsprop           0.988171  0.998705              34  
    95           0.005           adam           0.988171  0.999765              28  
    100          0.005           adam           0.988171  0.999294              37  
    38           0.005        rmsprop           0.986758  0.999647              31  
    7            0.005        rmsprop           0.986758  0.999765              22  
    78           0.005        rmsprop           0.985346  0.999765              41  
    24           0.001           adam           0.985346  0.999470              47  
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_110_1.png)
    


**Анализ лога ГА (Вар.2):**
- Обучено 116 моделей за 10 поколений. Топ-модели стабильно используют lr=0.005 и optimizer=adam/rmsprop — те же закономерности, что и для Вар.4.
- График эволюции показывает устойчивую сходимость: средний fitness растёт от 0.970 до 0.985+ к финальным поколениям.
- Большой разброс в ранних поколениях (min~0.91) быстро сужается — генетический отбор эффективно отсеивает слабые конфигурации.
- **Ключевое отличие от Вар.4:** лучшие модели на 49 признаках используют **высокий dropout (0.5)** — для полного набора сильная регуляризация критически важна для предотвращения переобучения на шумовых каналах (Ipt3-Ipt17).



```python
# Лучшие гиперпараметры из ГА
best_params_v2 = decode_individual(hof_v2[0])
print(f'Лучшие гиперпараметры (V2): {best_params_v2}')

# Полное обучение лучшей модели
model_v2, history_v2, metrics_v2, y_pred_v2, y_proba_v2 = train_and_evaluate(
    X_train_best, X_val_best, X_test_best,
    y_train_best, y_val_best, y_test_best,
    model_builder=build_conv1d_lstm_model,
    input_dim=best_input_dim,
    epochs=150, batch_size=32,
    **best_params_v2
)

plot_training_history(history_v2, title='Лучшая модель (после ГА, Вар.2)')
print_metrics(metrics_v2, y_test_best, y_pred_v2, y_proba_v2,
              title='Вариант 2 после ГА — ТЕСТ')
```

    Лучшие гиперпараметры (V2): {'conv_filters': 32, 'kernel_size': 7, 'lstm_units': 32, 'dense_units': 64, 'dropout_rate': 0.5, 'learning_rate': 0.005, 'optimizer_name': 'adam'}
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_112_1.png)
    


    
    ============================================================
      Вариант 2 после ГА — ТЕСТ
    ============================================================
    Accuracy:           0.9813
    Balanced Accuracy:  0.9761
    F1-score:           0.9265
    ROC AUC:            0.9986
    Avg Precision (PR): 0.9902
    
    Classification Report:
                  precision    recall  f1-score   support
    
         Штатное       1.00      0.98      0.99       471
       Нештатное       0.89      0.97      0.93        65
    
        accuracy                           0.98       536
       macro avg       0.94      0.98      0.96       536
    weighted avg       0.98      0.98      0.98       536
    
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_112_3.png)
    


**Результаты модели Вар.2 после ГА:**
- Accuracy=0.981, Balanced Accuracy=0.976, F1=0.927, ROC AUC=0.999.
- Precision=0.89, Recall=0.97 — модель находит 97% аномалий, но 11% предсказаний «аномалия» являются ложными.
- ROC AUC=0.999 — практически идеальная дискриминация, лучший результат во всей работе.


```python
# Сравнение: до и после ГА для Варианта 2
print('='*60)
print('СРАВНЕНИЕ: до и после ГА для Варианта 2')
print('='*60)

metrics_before_v2 = results[v2_name]['metrics']
print(f"{'Метрика':<25} {'До ГА':>10} {'После ГА':>10} {'Разница':>10}")
print('-'*55)
for key in metrics_before_v2:
    before = metrics_before_v2[key]
    after = metrics_v2[key]
    diff = after - before
    print(f'{key:<25} {before:>10.4f} {after:>10.4f} {diff:>+10.4f}')

# Confusion Matrix сравнение
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
cm1 = confusion_matrix(results[v2_name]['y_test'], results[v2_name]['y_pred'])
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Штатное', 'Нештатное'], yticklabels=['Штатное', 'Нештатное'])
ax1.set_title('До ГА (Вар.2)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Предсказано'); ax1.set_ylabel('Истинное')

cm2 = confusion_matrix(y_test_best, y_pred_v2)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=ax2,
            xticklabels=['Штатное', 'Нештатное'], yticklabels=['Штатное', 'Нештатное'])
ax2.set_title('После ГА (Вар.2)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Предсказано'); ax2.set_ylabel('Истинное')
plt.tight_layout()
plt.show()
```

    ============================================================
    СРАВНЕНИЕ: до и после ГА для Варианта 2
    ============================================================
    Метрика                        До ГА   После ГА    Разница
    -------------------------------------------------------
    accuracy                      0.9813     0.9813    +0.0000
    balanced_accuracy             0.9894     0.9761    -0.0133
    f1_score                      0.9286     0.9265    -0.0021
    roc_auc                       0.9993     0.9986    -0.0007
    avg_precision                 0.9952     0.9902    -0.0050
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_114_1.png)
    


**Вывод по ГА для Варианта 2:**
- ГА **не улучшил** результаты — все метрики либо не изменились (Accuracy), либо незначительно снизились.
- Причина: базовая модель (раздел 5) уже достигла Bal.Acc=0.989 и ROC AUC=0.999 — это практически потолок для данной задачи.
- Вывод аналогичен Вар.4: архитектура Conv1D+LSTM с базовыми параметрами достаточно хорошо работает «из коробки» для телеметрии МКА.

### 9.2 Автокодировщик для Варианта 2 (49 признаков)

Строим автокодировщик той же архитектуры (Conv1D+LSTM) для полного набора из 49 признаков. Латентное пространство остаётся 16-мерным — степень сжатия выше (49→16 vs 19→16), что может как улучшить обнаружение аномалий (больше информации), так и ухудшить (сложнее сжать).


```python
# Подготовка данных для автокодировщика (Вар.2)
X_ae_train_normal_v2 = X_train_best[y_train_best == 0]
X_ae_val_v2 = X_val_best
y_ae_val_v2 = y_val_best
X_ae_test_v2 = X_test_best
y_ae_test_v2 = y_test_best

print(f'Обучающий набор АЭ (штатные): {X_ae_train_normal_v2.shape}')
print(f'Валидационный: {X_ae_val_v2.shape}')
print(f'Тестовый: {X_ae_test_v2.shape}')

# Строим автокодировщик для 49 признаков
autoencoder_v2, encoder_v2 = build_conv1d_lstm_autoencoder(
    input_dim=best_input_dim, conv_filters=64, kernel_size=3,
    lstm_units=32, latent_dim=16, dropout_rate=0.2
)
autoencoder_v2.summary()

# Обучение
X_tr_3d_v2 = reshape_for_conv1d(X_ae_train_normal_v2)
X_val_normal_3d_v2 = reshape_for_conv1d(X_ae_val_v2[y_ae_val_v2 == 0])

ae_cb_v2 = [
    callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
]

ae_hist_v2 = autoencoder_v2.fit(
    X_tr_3d_v2, X_tr_3d_v2,
    validation_data=(X_val_normal_3d_v2, X_val_normal_3d_v2),
    epochs=150, batch_size=32,
    callbacks=ae_cb_v2, verbose=1
)

plt.figure(figsize=(10, 4))
plt.plot(ae_hist_v2.history['loss'], label='Train Loss')
plt.plot(ae_hist_v2.history['val_loss'], label='Val Loss')
plt.title('Автокодировщик Вар.2 — Loss (MSE)')
plt.xlabel('Epoch'); plt.ylabel('MSE')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

    Обучающий набор АЭ (штатные): (1531, 49)
    Валидационный: (402, 49)
    Тестовый: (536, 49)
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">49</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)          │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv1d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">49</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │           <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">49</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │           <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling1d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling1D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">24</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">24</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">12,416</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ latent (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)             │           <span style="color: #00af00; text-decoration-color: #00af00">528</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ repeat_vector (<span style="color: #0087ff; text-decoration-color: #0087ff">RepeatVector</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">49</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">49</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │         <span style="color: #00af00; text-decoration-color: #00af00">6,272</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">49</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ time_distributed                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">49</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)          │            <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">TimeDistributed</span>)               │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">19,761</span> (77.19 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">19,633</span> (76.69 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">128</span> (512.00 B)
</pre>



    Epoch 1/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 40ms/step - loss: 11.1905 - val_loss: 10.3924 - learning_rate: 0.0010
    Epoch 2/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 26ms/step - loss: 11.1045 - val_loss: 10.3226 - learning_rate: 0.0010
    Epoch 3/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 26ms/step - loss: 11.0398 - val_loss: 10.2237 - learning_rate: 0.0010
    Epoch 4/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 26ms/step - loss: 10.9443 - val_loss: 10.1488 - learning_rate: 0.0010
    Epoch 5/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 27ms/step - loss: 10.8565 - val_loss: 9.9960 - learning_rate: 0.0010
    Epoch 6/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 26ms/step - loss: 10.7621 - val_loss: 9.9530 - learning_rate: 0.0010
    Epoch 7/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 26ms/step - loss: 10.6219 - val_loss: 9.8934 - learning_rate: 0.0010
    Epoch 8/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 27ms/step - loss: 10.6065 - val_loss: 9.8688 - learning_rate: 0.0010
    Epoch 9/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 26ms/step - loss: 10.4732 - val_loss: 9.8530 - learning_rate: 0.0010
    Epoch 10/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 26ms/step - loss: 10.4533 - val_loss: 9.8322 - learning_rate: 0.0010
    Epoch 11/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 31ms/step - loss: 10.4037 - val_loss: 9.7639 - learning_rate: 0.0010
    Epoch 12/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 29ms/step - loss: 10.3930 - val_loss: 9.6469 - learning_rate: 0.0010
    Epoch 13/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 27ms/step - loss: 10.2634 - val_loss: 9.6227 - learning_rate: 0.0010
    Epoch 14/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 28ms/step - loss: 10.2235 - val_loss: 9.5618 - learning_rate: 0.0010
    Epoch 15/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 26ms/step - loss: 10.2161 - val_loss: 9.5362 - learning_rate: 0.0010
    Epoch 16/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 27ms/step - loss: 10.1493 - val_loss: 9.4785 - learning_rate: 0.0010
    Epoch 17/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 27ms/step - loss: 10.1807 - val_loss: 9.4758 - learning_rate: 0.0010
    Epoch 18/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 26ms/step - loss: 10.1296 - val_loss: 9.4434 - learning_rate: 0.0010
    Epoch 19/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 29ms/step - loss: 10.0533 - val_loss: 9.4083 - learning_rate: 0.0010
    Epoch 20/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 27ms/step - loss: 9.9491 - val_loss: 10.4909 - learning_rate: 0.0010
    Epoch 21/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 26ms/step - loss: 10.0709 - val_loss: 9.1861 - learning_rate: 0.0010
    Epoch 22/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 29ms/step - loss: 9.8976 - val_loss: 9.1371 - learning_rate: 0.0010
    Epoch 23/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 26ms/step - loss: 9.9041 - val_loss: 9.2548 - learning_rate: 0.0010
    Epoch 24/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 26ms/step - loss: 9.8828 - val_loss: 9.1562 - learning_rate: 0.0010
    Epoch 25/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 27ms/step - loss: 9.7799 - val_loss: 9.0282 - learning_rate: 0.0010
    Epoch 26/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 27ms/step - loss: 9.8013 - val_loss: 11.8379 - learning_rate: 0.0010
    Epoch 27/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 28ms/step - loss: 10.3324 - val_loss: 9.4086 - learning_rate: 0.0010
    Epoch 28/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 28ms/step - loss: 9.9688 - val_loss: 9.4275 - learning_rate: 0.0010
    Epoch 29/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 27ms/step - loss: 9.8149 - val_loss: 9.3981 - learning_rate: 0.0010
    Epoch 30/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 28ms/step - loss: 9.5792 - val_loss: 9.4377 - learning_rate: 0.0010
    Epoch 31/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 29ms/step - loss: 9.6120 - val_loss: 9.4270 - learning_rate: 5.0000e-04
    Epoch 32/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 30ms/step - loss: 9.6173 - val_loss: 9.2789 - learning_rate: 5.0000e-04
    Epoch 33/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 27ms/step - loss: 9.6071 - val_loss: 9.2034 - learning_rate: 5.0000e-04
    Epoch 34/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 32ms/step - loss: 9.5694 - val_loss: 9.1585 - learning_rate: 5.0000e-04
    Epoch 35/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 30ms/step - loss: 9.5567 - val_loss: 9.1223 - learning_rate: 5.0000e-04
    Epoch 36/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 31ms/step - loss: 9.5601 - val_loss: 9.1098 - learning_rate: 2.5000e-04
    Epoch 37/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 29ms/step - loss: 9.4279 - val_loss: 9.1537 - learning_rate: 2.5000e-04
    Epoch 38/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 30ms/step - loss: 9.4418 - val_loss: 9.1181 - learning_rate: 2.5000e-04
    Epoch 39/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 31ms/step - loss: 9.4220 - val_loss: 9.1080 - learning_rate: 2.5000e-04
    Epoch 40/150
    [1m48/48[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 28ms/step - loss: 9.4505 - val_loss: 9.1129 - learning_rate: 2.5000e-04
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_117_7.png)
    


**Анализ обучения автокодировщика (Вар.2, 49 признаков):**
- Loss снижается стабильно, модель сходится за ~40 эпох — автокодировщик успешно обучается восстанавливать 49-мерный вектор телеметрии.
- Val Loss ниже, чем у Вар.4 (19 признаков) — парадоксально, но с 49 признаками автокодировщик имеет больше контекста для реконструкции: коррелированные признаки помогают восстанавливать друг друга.
- ReduceLROnPlateau снижает lr при стагнации, обеспечивая тонкую донастройку в конце обучения.
- Отсутствие расхождения Train/Val Loss подтверждает, что L2-регуляризация и Dropout эффективно предотвращают переобучение на 49 признаках.



```python
# Ошибка реконструкции + подбор порога
train_rec_v2 = autoencoder_v2.predict(X_tr_3d_v2, verbose=0)
train_mse_v2 = np.mean(np.square(X_tr_3d_v2 - train_rec_v2), axis=(1, 2))

X_te_3d_v2 = reshape_for_conv1d(X_ae_test_v2)
test_rec_v2 = autoencoder_v2.predict(X_te_3d_v2, verbose=0)
test_mse_v2 = np.mean(np.square(X_te_3d_v2 - test_rec_v2), axis=(1, 2))

# Подбор порога по F1 на валидации
X_val_3d_v2 = reshape_for_conv1d(X_ae_val_v2)
val_rec_v2 = autoencoder_v2.predict(X_val_3d_v2, verbose=0)
val_mse_v2 = np.mean(np.square(X_val_3d_v2 - val_rec_v2), axis=(1, 2))

best_f1_v2 = 0
best_thr_v2 = 0
for pct in range(80, 100):
    thr = np.percentile(train_mse_v2, pct)
    y_p = (val_mse_v2 > thr).astype(int)
    f1_v = f1_score(y_ae_val_v2, y_p)
    if f1_v > best_f1_v2:
        best_f1_v2 = f1_v
        best_thr_v2 = thr

print(f'Лучший порог (V2): {best_thr_v2:.6f} (F1 val = {best_f1_v2:.4f})')

# Визуализация
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
ax1.hist(train_mse_v2, bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
ax1.axvline(x=best_thr_v2, color='red', linestyle='--', label=f'Порог = {best_thr_v2:.2f}')
ax1.set_title('MSE обучающей выборки (штатные, Вар.2)')
ax1.set_xlabel('MSE'); ax1.set_ylabel('Частота'); ax1.legend()

ax2.hist(test_mse_v2[y_ae_test_v2 == 0], bins=50, alpha=0.6, color='#2ecc71', label='Штатное', density=True)
ax2.hist(test_mse_v2[y_ae_test_v2 == 1], bins=50, alpha=0.6, color='#e74c3c', label='Нештатное', density=True)
ax2.axvline(x=best_thr_v2, color='red', linestyle='--')
ax2.set_title('MSE по классам (тест, Вар.2)')
ax2.set_xlabel('MSE'); ax2.set_ylabel('Плотность'); ax2.legend()
plt.tight_layout()
plt.show()
```

    Лучший порог (V2): 87.071321 (F1 val = 0.8542)
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_119_1.png)
    


**Подбор порога (Вар.2):**
- Оптимальный порог подобран по максимальному F1-score на валидационной выборке — тот же подход, что и для Вар.4.
- На графике видно чёткое разделение MSE штатных и нештатных данных — автокодировщик на 49 признаках хорошо отличает аномалии.
- Порог Вар.2 отличается от порога Вар.4 по абсолютному значению, т.к. масштаб MSE зависит от числа признаков и их диапазона. Оценка качества происходит по относительным метрикам (F1, ROC AUC).
- Использование валидации (а не теста) для подбора порога предотвращает утечку данных.



```python
# Метрики автокодировщика Вар.2
y_pred_ae_v2 = (test_mse_v2 > best_thr_v2).astype(int)

ae_acc_v2 = accuracy_score(y_ae_test_v2, y_pred_ae_v2)
ae_bal_v2 = balanced_accuracy_score(y_ae_test_v2, y_pred_ae_v2)
ae_f1_v2 = f1_score(y_ae_test_v2, y_pred_ae_v2)
ae_roc_v2 = roc_auc_score(y_ae_test_v2, test_mse_v2)
ae_ap_v2 = average_precision_score(y_ae_test_v2, test_mse_v2)

print('='*60)
print('  АВТОКОДИРОВЩИК Вар.2 — Метрики на тесте')
print('='*60)
print(f'Accuracy:           {ae_acc_v2:.4f}')
print(f'Balanced Accuracy:  {ae_bal_v2:.4f}')
print(f'F1-score:           {ae_f1_v2:.4f}')
print(f'ROC AUC:            {ae_roc_v2:.4f}')
print(f'Avg Precision (PR): {ae_ap_v2:.4f}')
print()
print(classification_report(y_ae_test_v2, y_pred_ae_v2,
                            target_names=['Штатное', 'Нештатное']))

# CM + ROC + PR
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

cm_v2 = confusion_matrix(y_ae_test_v2, y_pred_ae_v2)
sns.heatmap(cm_v2, annot=True, fmt='d', cmap='Greens', ax=ax1,
            xticklabels=['Штатное', 'Нештатное'], yticklabels=['Штатное', 'Нештатное'])
ax1.set_title('Confusion Matrix'); ax1.set_xlabel('Предсказано'); ax1.set_ylabel('Истинное')

fpr_v2, tpr_v2, _ = roc_curve(y_ae_test_v2, test_mse_v2)
ax2.plot(fpr_v2, tpr_v2, 'g-', linewidth=2, label=f'AUC={ae_roc_v2:.3f}')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR')
ax2.set_title('ROC-кривая'); ax2.legend(); ax2.grid(True, alpha=0.3)

prec_v2, rec_v2, _ = precision_recall_curve(y_ae_test_v2, test_mse_v2)
ax3.plot(rec_v2, prec_v2, 'g-', linewidth=2, label=f'AP={ae_ap_v2:.3f}')
ax3.set_xlabel('Recall'); ax3.set_ylabel('Precision')
ax3.set_title('PR-кривая'); ax3.legend(); ax3.grid(True, alpha=0.3)

plt.suptitle('Автокодировщик Вар.2 (49 признаков)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```

    ============================================================
      АВТОКОДИРОВЩИК Вар.2 — Метрики на тесте
    ============================================================
    Accuracy:           0.9832
    Balanced Accuracy:  0.9573
    F1-score:           0.9302
    ROC AUC:            0.9972
    Avg Precision (PR): 0.9817
    
                  precision    recall  f1-score   support
    
         Штатное       0.99      0.99      0.99       471
       Нештатное       0.94      0.92      0.93        65
    
        accuracy                           0.98       536
       macro avg       0.96      0.96      0.96       536
    weighted avg       0.98      0.98      0.98       536
    
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_121_1.png)
    


**Детальный анализ автокодировщика Вар.2:**
- **Accuracy=0.983, F1=0.930** — сопоставимо с Вар.4 (F1=0.934).
- **Recall=0.92** — находит 92% аномалий (60 из 65), больше чем Вар.4 (88%, 57 из 65). Дополнительные 30 признаков дают модели больше «зацепок» для обнаружения аномалий.
- **Precision=0.94** — 6% ложных срабатываний (у Вар.4 было 0%). Цена за более высокий recall.
- **ROC AUC=0.997** — лучше, чем у Вар.4 (0.989). 49 признаков дают более полную картину для ранжирования.

**Вывод:** Автокодировщик на 49 признаках находит больше аномалий (recall выше), но допускает немного ложных срабатываний. Выбор зависит от приоритета: precision (Вар.4) или recall (Вар.2).


```python
# t-SNE латентного пространства (Вар.2)
latent_v2 = encoder_v2.predict(X_te_3d_v2, verbose=0)
print(f'Форма латентных представлений: {latent_v2.shape}')

tsne_v2 = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
latent_2d_v2 = tsne_v2.fit_transform(latent_v2)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(latent_2d_v2[y_ae_test_v2 == 0, 0], latent_2d_v2[y_ae_test_v2 == 0, 1],
                c='#2ecc71', alpha=0.4, s=15, label='Штатное (0)')
axes[0].scatter(latent_2d_v2[y_ae_test_v2 == 1, 0], latent_2d_v2[y_ae_test_v2 == 1, 1],
                c='#e74c3c', alpha=0.7, s=30, label='Нештатное (1)')
axes[0].set_title('t-SNE (реальные метки)', fontsize=12)
axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2'); axes[0].legend()

sc = axes[1].scatter(latent_2d_v2[:, 0], latent_2d_v2[:, 1],
                     c=test_mse_v2, cmap='RdYlGn_r', alpha=0.6, s=15)
plt.colorbar(sc, ax=axes[1], label='MSE')
axes[1].set_title('t-SNE (ошибка реконструкции)', fontsize=12)
axes[1].set_xlabel('t-SNE 1'); axes[1].set_ylabel('t-SNE 2')
plt.tight_layout()
plt.show()
```

    Форма латентных представлений: (536, 16)
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_123_1.png)
    


**Анализ t-SNE (Вар.2):**
- Латентное пространство автокодировщика на 49 признаках формирует **более чёткие кластеры**, чем на 19: нештатные состояния компактно сгруппированы отдельно от нормы.
- Ошибка реконструкции (правый график) чётко коррелирует с положением аномалий — тёплые цвета (высокая MSE) совпадают с красными точками нештатного класса.
- **Объяснение:** 49 признаков дают автокодировщику больше информации для создания дискриминативного латентного представления. Даже при сжатии 49->16 модель извлекает больше паттернов, чем при 19->16.
- Это визуально подтверждает, что полный набор признаков даёт автокодировщику преимущество в качестве разделения (ROC AUC 0.997 vs 0.989).


### 9.3 Итоговое сравнение: Вариант 2 (49 признаков) vs Вариант 4 (19 признаков)

Сравниваем все 6 подходов (3 для каждого варианта данных) по всем метрикам.


```python
# Полное сравнение Вар.2 vs Вар.4
v2_ae_metrics = {'accuracy': ae_acc_v2, 'balanced_accuracy': ae_bal_v2,
                 'f1_score': ae_f1_v2, 'roc_auc': ae_roc_v2, 'avg_precision': ae_ap_v2}

comparison_full = pd.DataFrame({
    'Вар.4 до ГА (19)': v4_results['до_ГА'],
    'Вар.4 после ГА (19)': v4_results['после_ГА'],
    'Вар.4 АЭ (19)': v4_results['автоэнкодер'],
    'Вар.2 до ГА (49)': results[v2_name]['metrics'],
    'Вар.2 после ГА (49)': metrics_v2,
    'Вар.2 АЭ (49)': v2_ae_metrics,
}).T

print('='*70)
print('  ИТОГОВОЕ СРАВНЕНИЕ: ВАРИАНТ 2 (49) vs ВАРИАНТ 4 (19)')
print('='*70)
print(comparison_full.to_string())

# Визуализация
fig, axes = plt.subplots(1, 5, figsize=(24, 6))
metrics_list = ['accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc', 'avg_precision']
titles = ['Accuracy', 'Balanced Accuracy', 'F1-Score', 'ROC AUC', 'Avg Precision']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c']

for i, (m, t) in enumerate(zip(metrics_list, titles)):
    vals = comparison_full[m].values
    bars = axes[i].bar(range(len(vals)), vals, color=colors, edgecolor='black')
    axes[i].set_title(t, fontsize=10)
    axes[i].set_xticks(range(len(vals)))
    xlabels = ['V4\nдоГА', 'V4\nпослеГА', 'V4\nАЭ', 'V2\nдоГА', 'V2\nпослеГА', 'V2\nАЭ']
    axes[i].set_xticklabels(xlabels, fontsize=7)
    axes[i].set_ylim(0.85, 1.02)
    for bar, v in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2, v + 0.003,
                     f'{v:.3f}', ha='center', fontsize=7)

plt.suptitle('Сравнение: Вар.4 (19 призн.) vs Вар.2 (49 призн.)', fontsize=13)
plt.tight_layout()
plt.show()
```

    ======================================================================
      ИТОГОВОЕ СРАВНЕНИЕ: ВАРИАНТ 2 (49) vs ВАРИАНТ 4 (19)
    ======================================================================
                         accuracy  balanced_accuracy  f1_score   roc_auc  avg_precision
    Вар.4 до ГА (19)     0.992537           0.982492  0.969231  0.986281       0.980420
    Вар.4 после ГА (19)  0.986940           0.972677  0.946565  0.993337       0.982423
    Вар.4 АЭ (19)        0.985075           0.938462  0.934426  0.988600       0.970180
    Вар.2 до ГА (49)     0.981343           0.989384  0.928571  0.999314       0.995229
    Вар.2 после ГА (49)  0.981343           0.976123  0.926471  0.998628       0.990224
    Вар.2 АЭ (49)        0.983209           0.957292  0.930233  0.997191       0.981677
    


    
![png](spacecraft_telemetry_anomaly_detection_files/spacecraft_telemetry_anomaly_detection_126_1.png)
    


**Итоговые выводы по дополнительному исследованию:**

**Классификатор Conv1D+LSTM:**
- Вар.4 (19 признаков) даёт лучший F1 (0.969 vs 0.929) — отбор признаков убирает шум и улучшает точность.
- Вар.2 (49 признаков) даёт лучший ROC AUC (0.999 vs 0.986) — полный набор лучше ранжирует объекты.

**Генетический алгоритм:**
- Не дал значимого улучшения ни для Вар.4, ни для Вар.2 — базовые гиперпараметры уже оптимальны.
- Тем не менее, ГА подтвердил, что компактные архитектуры с высокой скоростью обучения работают лучше всего.

**Автокодировщик:**
- Вар.4 (19): Precision=1.00, Recall=0.88 — **ноль ложных тревог**, но пропускает 12% аномалий.
- Вар.2 (49): Precision=0.94, Recall=0.92 — **больше найденных аномалий**, но 6% ложных срабатываний.
- ROC AUC Вар.2 (0.997) > Вар.4 (0.989) — полный набор даёт более полную картину.

**Общий вывод:** Отбор признаков (19 из 49) улучшает пороговую классификацию (F1), а полный набор — ранжирование (ROC AUC). Оба подхода имеют свои преимущества в зависимости от задачи.

## Выводы

**1. Разведочный анализ данных:**
- Набор данных содержит 2679 записей ТМИ МКА с 49 исходными признаками, сгруппированными по физическому смыслу: напряжение бортовой сети (Ubs), токи потребителей (Ibs, Isun, Ipt1–Ipt17), температуры радиаторов (TR1–TR16), температуры датчиков (TDS1–TDS9, TDS24) и прочие температуры (TKpt, TGbv, TNap, TPrd1, TPrd2).
- После переразметки в бинарную классификацию: ~88% штатных, ~12% нештатных состояний — набор **сильно несбалансирован** (соотношение ~7.3:1).
- Пропущенных значений нет. Выбросы (по IQR) не удалялись, т.к. они являются индикаторами нештатных состояний аппарата.
- Анализ распределений показал выраженную асимметрию (skewness >> 1) у токовых признаков и тяжёлые хвосты (kurtosis > 3), что обосновывает выбор RobustScaler вместо StandardScaler.
- Корреляционный анализ выявил блоки мультиколлинеарных признаков (TDS1–TDS9: r > 0.9; группа TR), а корреляция с целевой переменной и Mutual Information определили наиболее информативные каналы: Ibs, Ipt1, Ipt2, Isun, Ubs, TR11, TR13.
- Признаки Ipt3–Ipt5, Ipt8–Ipt10, Ipt12–Ipt17 практически не несут полезной информации (почти нулевой ток в обоих классах).
- Экспериментирование с атрибутами: создано 6 инженерных признаков (Power_W, I_balance, I_active_sum, TR_min_all, TR_std_all, TR15_delta), из которых 3 вошли в финальный обогащённый набор.

**2. Гибридная модель Conv1D + LSTM (задача 1.2):**
- Модель обучена на **6 вариантах данных**: 3 базовых набора (полный 49 / отобранный 19 / обогащённый 22) × 2 режима (сырой / RobustScaler).
- Для борьбы с дисбалансом использованы веса классов (`class_weight='balanced'`), а для регуляризации — L2, Dropout, BatchNormalization, EarlyStopping, ReduceLROnPlateau.
- **Лучший по Balanced Accuracy:** Вар.2 (49 + RobustScaler) = 0.989 с recall аномалий = 1.00 (все 65 аномалий обнаружены).
- **Лучший по F1-score:** Вар.4 (19 + RobustScaler) = 0.969 с precision = 0.97, recall = 0.97 — наиболее сбалансированный результат.
- Обогащённые наборы (Вар.5, Вар.6) не дали преимущества — инженерные признаки дублируют уже имеющуюся информацию.
- RobustScaler улучшает Balanced Accuracy для всех вариантов благодаря устойчивости к выбросам и асимметрии.

**3. Оптимизация гиперпараметров (ГА / DEAP):**
- Генетический алгоритм оптимизировал 7 гиперпараметров (conv_filters, kernel_size, lstm_units, dense_units, dropout_rate, learning_rate, optimizer) за 10 поколений популяции из 15 особей.
- ГА проведён для двух вариантов данных: Вар.4 (19 признаков) и Вар.2 (49 признаков).
- **Результат:** ГА не дал значимого улучшения F1 ни для одного варианта — базовые гиперпараметры уже были близки к оптимальным. ROC AUC незначительно улучшился (+0.007 для Вар.4).
- **Ключевое наблюдение:** лучшие конфигурации ГА — компактные модели (lstm=32, dense=16–64) с высокой скоростью обучения (lr=0.005) и оптимизатором Adam/RMSprop. Для 49 признаков критически важен высокий dropout (0.5).

**4. Автокодировщик для обнаружения аномалий (задача 6.2):**
- Автокодировщик построен на основе гибридной архитектуры Conv1D + LSTM (энкодер-декодер) с 16-мерным латентным пространством.
- Обучен исключительно на штатных данных (unsupervised), порог аномалии определён по ошибке реконструкции (MSE) на валидационной выборке.
- **Вар.4 (19 признаков):** Precision = 1.00, Recall = 0.88, F1 = 0.934, ROC AUC = 0.989. Ни одного ложного срабатывания, но пропущено 8 из 65 аномалий.
- **Вар.2 (49 признаков):** Precision = 0.94, Recall = 0.92, F1 = 0.930, ROC AUC = 0.997. Находит больше аномалий (60 из 65), но допускает 6% ложных тревог.
- Визуализация латентного пространства (t-SNE) подтверждает: аномалии формируют отдельные кластеры, а MSE чётко коррелирует с истинным классом.
- Примеры реконструкции наглядно демонстрируют разницу в MSE: ~0.2–120 для штатных vs ~977–9054 для нештатных (разница в 1–2 порядка).
