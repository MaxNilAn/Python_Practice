#!/usr/bin/env python
# coding: utf-8

# # 1. Введение

# ## 1.1. Цель и задачи работы

# **Цель** - построить методами машинного обучения на основе данных численного моделирования термогравитационного течения воздуха через горизонтальный пучок оребренных труб алгебраическое выражение для безразмерного коэффициента теплоотдачи в зависимости от физических и геометрических параметров с целью дальнейшего использования в инженерных расчетах.

# **Задачи:**
# - построение моделей машинного обучения на основе данных многопараметрических расчетов с использованием различных методов (линейная и полиномиальная регрессия, градиентный бустинг)
# - определение ключевых признаков (параметров задачи, оказывающих наибольшее влияние) с помощью метода SHAP для метода градиентного бустинга
# - получение выражения для безразмерного коэффициента теплоотдачи с использованием моделей символьной регрессии и нейросетевой архитектуры метода Колмогорова-Арнольда (Kolmogorov-Arnold Network, KAN), сопоставление использованных подходов с оценкой качества

# ## 1.2. Описание задачи о свободной конвекции вблизи однорядного пучка труб

# Постановка задачи для проведения численного моделирования и накопления данных для применения методов МО сформулирована в соответствии с условиями экспериментов, выполненных в Институте тепло- и массообмена имени А.В. Лыкова НАН Беларуси. Адаптированный для расчетов экспериментальный образец включал горизонтальный однорядный трубный пучок, который размещался в специальной камере и состоял из шести биметаллических оребренных труб. В экспериментах варьировался поперечный шаг труб и перепад температуры между поверхностью несущей трубы и окружающим воздухом.
# 
# Ламинарное свободноконвективное течение вязкой сжимаемой среды и теплообмен описывались в расчетах системой нестационарных уравнений Навье-Стокса, дополненной уравнением баланса энергии. Использовалась модель сжимаемого совершенного газа со свойствами, зависящими от температуры. Плотность рассчитывалась на основе уравнения Менделеева-Клапейрона. Задача решалась в сопряженной постановке, с учетом теплопередачи через трубные ребра.
# 
# Расчетные сетки имели размерность около 150 тыс. ячеек, в ходе методических исследований проверялась сеточная сходимость. Расчеты проводились в CFD-пакете ANSYS Fluent со вторым порядком точности при аппроксимации пространственных и временных производных. Во всех рассчитанных вариантах достигался статистически установившийся режим течения, а размер выборок превысил 500 характерных времен.
# 
# Набор данных для МО сформирован в результате проведения многопараметрических гидродинамических расчетов (в количестве более 3000) при варьировании параметров и вычислением во всех вариантах среднего числа Нуссельта.
# 
# На этапе разведывательного анализа данных из расширенного набора для записи в табличные файлы были выбраны следующие параметры: шаг труб, осредненный во времени перепад температуры между основанием трубы и окружающей средой и среднее (по трем трубам) число Нуссельта. Дополнительная предобработка данных при формировании датасета не производилась.

# ## 1.3. Описание задачи о термогравитационном течении через двухрядный пучок труб

# Выполнялось численное исследования влияния вытяжной шахты на течение и теплообмен в двухрядном горизонтальном пучке труб, расположенных в шахматном порядке, с относительно тесным поперечным кольцевым оребрением. Рассмотрены две конфигурации – без шахты и с шахтой, которая по высоте в девять раз превосходит диаметр оребрения и содержит внутренние вертикальные (разделительные) перегородки.
# 
# Постановка задачи основывалась на данных экспериментов, в которых наблюдался ламинарный режим течения воздух вблизи труб. Численно, на основе метода конечных объемов, решалась система нестационарных уравнений Навье-Стокса, дополненная уравнением энергии. Использовалась модель сжимаемого совершенного газа с термодинамическими и теплофизическими свойствами, зависящими от температуры. Задачи решались в сопряженной постановке, с учетом теплопередачи через ребра.
# 
# Для выполнения расчетов использовался пакет ANSYS Fluent 2019 R3. Расчеты на неструктурированных сетках с гексаэдральными элементами проводились со вторым порядком точности по пространству и времени; для аппроксимации конвективных потоков использовалась противопоточная схема второго порядка. Расчетные сетки включали около 350 тыс. ячеек, достаточность размерности данной сетки была установлена в результате исследования сеточной чувствительности решения. Шаг по времени принимался равным 0.02 с, что составляет около 0.1 характерных времен оцениваемых по скорости плавучести. Продолжительность выборок, соответствующих статистически установившемуся режиму течения, составляла около 200 с, что для всех вариантов превышало 1000 характерных времен.
# 
# Набор данных для МО сформирован в результате проведения многопараметрических гидродинамических расчетов (в количестве более 100) при варьировании параметров и вычислением во всех вариантах среднего числа Нуссельта.
# 
# Варьировались следующие параметры: шаг труб, осредненный во времени перепад температуры между основанием трубы и окружающей средой, высота шахты, шаг ребер. Дополнительная предобработка данных при формировании датасета не производилась.

# ## 1.4. Описание используемых моделей машинного обучения

# ### Линейная регрессия

# Линейная регрессия моделирует зависимую переменную как линейную комбинацию независимых признаков, подбирая коэффициенты методом наименьших квадратов для минимизации ошибки между предсказанными и реальными значениями.
# 
# Особенности метода:
# - Простая аналитическая форма
#   $$\hat y = \beta_0 + \sum_{i=1}^p \beta_i x_i$$  
#   где коэффициенты $\beta$ подбираются методом наименьших квадратов (OLS) для минимизации суммы квадратов отклонений между наблюдаемыми и предсказанными значениями
# - Высокая интерпретируемость
#   каждый коэффициент $\beta_i$ показывает влияние соответствующего признака на ответ при прочих равных.
# - Быстрое обучение
#   аналитическое решение OLS выполняется за полиномиальное время и масштабируется на большие объёмы данных.
# - Контроль переобучения
#   возможна регуляризация L2 (Ridge) и L1 (Lasso) для штрафа за большие коэффициенты и снижения дисперсии модели.

# ### Полиномиальная регрессия

# Полиномиальная регрессия расширяет линейную, добавляя в модель степени признаков до порядка $d$, при этом оставаясь линейной по параметрам и оцениваясь тем же методом наименьших квадратов.
# 
# Особенности метода:
# - Гибкость для криволинейных зависимостей
#   $$\hat y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d$$  
#   позволяющая аппроксимировать сложные тренды (Wikipedia).
# - Линейность по параметрам
#   параметры $\beta$ оцениваются аналитически через OLS, несмотря на нелинейность по $x$.
# - Управляемая сложность  
#   степень полинома $d$ контролирует баланс «гибкость — переобучение».
# - Простота реализации  
#   преобразование исходных признаков в полиномиальные степени и последующее применение OLS образуют понятный и детерминированный конвейер.

# ### QLattice

# Ссылка: https://docs.abzu.ai/docs/guides/getting_started/qlattice
# 
# Библиотека для символьной регрессии QLattice позволяет находить аналитические формулы, наилучшим образом описывающие данные. Это особенно полезно в инженерных задачах, где важна интерпретируемость модели и понимание физических закономерностей.
# 
# Особенности библиотеки:
# - поиск интерпретируемых формул, сложность которых можно контролировать, состоящих из базовых математических операций (+, -, *, /, tanh, exp, log)
# - применение эволюционного алгоритма и графовых моделей, представляющих связи между переменными через математические операции
# - автоматический отбор значимых признаков
# - устойчивость к зашумленным данным
# - удобная и поддерживаемая интеграция с Python
# 
# Таким образом, QLattice является мощным инструментом, который может быть использован в гидрогазодинамике для следующих целей: нахождение новых эмпирических зависимостей, упрощение сложных моделей, получение обоснованных и относительно простых инженерных формул. Последний вариант применения и рассматривается в данной работе.

# ### PySR

# Ссылка на библиотеку PySR: https://ai.damtp.cam.ac.uk/pysr/
# 
# Ссылка на GitHub: https://github.com/MilesCranmer/PySR
# 
# Библиотека для символьной регрессии PySR позволяет автоматически находить компактные и интерпретируемые аналитические выражения, наилучшим образом описывающие данные, сочетая гибкий эволюционный поиск с высокопроизводительным бэкендом на Julia.
# 
# Особенности библиотеки:
# 
# - Поиск формул из базовых операторов (+, -, *, /, sin, exp, log и др.) с контролем сложности через штрафы и ограничения глубины
# 
# - Многопопуляционный эволюционный алгоритм с циклом «эволюция – упрощение – оптимизация»
# 
# - Встроенная оптимизация численных констант (Nelder–Mead, BFGS)
# 
# - Автоматический отбор признаков и опциональное подавление шума
# 
# - Масштабирование на многопроцессорные и распределённые кластеры
# 
# - Удобная интеграция с Python: PySRRegressor, вывод в SymPy, LaTeX, а также экспорт в PyTorch и JAX
# 
# Принцип эволюционного алгоритма PySR:
# 
# PySR использует многопопуляционный эволюционный алгоритм с уникальным циклом «эволюция → упрощение → оптимизация», распараллеливая поиск по нескольким «островам» моделей одновременно
# 
# 1. Инициализация: создаётся несколько популяций случайных выражений
# 
# 2. Генетические операторы: в потомков вносят мутации (замена операторов, изменение констант, добавление/удаление поддеревьев) и кроссовер (обмен поддеревьями между выражениями), после чего проводится турни́рный отбор наиболее приспособленных
# 
# 3. Упрощение и оптимизация: каждый отобранный индивид упрощают (удаляют избыточные узлы), затем численно настраивают его константы методами Nelder–Mead или BFGS для точной подгонки
# 
# 4. Миграция: лучшие решения периодически переселяются между популяциями, что повышает разнообразие и ускоряет поиск глобальных оптимумов
# 
# После заданного числа итераций возвращается набор выражений на «Парето-границе» точность–сложность.
#     
# Таким образом, PySR отлично подходит для гидрогазодинамики: он помогает выявлять новые эмпирические зависимости, упрощать сложные численные модели и получать обоснованные инженерные формулы.

# ### KAN (библиотека PyKAN)

# Ссылка на статью с подробным описанием архитектуры: https://arxiv.org/pdf/2404.19756
# 
# Ссылка на библиотеку Pykan: https://github.com/KindXiaoming/pykan
# 
# Архитектура нейронных сетей Колмогорова-Арнольда (Kolmogorov-Arnold Networks, KAN) основана на теореме представления Колмогорова-Арнольда, которая утверждает, что любую непрерывную функцию нескольких переменных можно точно выразить через композицию конечного числа непрерывных функций одной переменной и операций сложения.
# 
# Особенности архитектуры KAN:
# 
# 1) Отказ от линейных преобразований
# - в отличие от традиционных MLP (Multilayer Perceptrons), где каждый нейрон выполняет линейное преобразование с последующей нелинейной активацией, KAN использует нелинейные функции на ребрах (а не в узлах)
# - узлы суммируют входящие сигналы, а нелинейные преобразования происходят на связях между нейронами
# 
# 2) Адаптивные функции активации
# - вместо фиксированных функций (ReLU, sigmoid) KAN использует сплайны или другие параметризованные функции, которые обучаются в процессе тренировки
# - позволяют сети автоматически подстраивать свою структуру под данные
# 
# 3) Иерархическая декомпозиция
# - KAN стремится разложить сложную многомерную функцию на цепочку более простых одномерных функций в соответствии с теоремой Колмогорова-Арнольда
# 
# 4) Интерпретируемость
# - поскольку каждая связь представляет собой параметризованную одномерную функцию, KAN потенциально более интерпретируем чем MLP
# - позволяет выводить символьные формулы, эффективно представляющие выученные закономерности
# 
# Таким образом, предполагается, что KAN может выполнить символьную регрессию не только эффективно, но и обоснованно, что имеет критическое значение для физических задач.

# # 2. Применение базовых моделей машинного обучения

# In[3]:


# Импорт необходимых Python-библиотек
import operator
import random

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

get_ipython().system('pip install pysr')
from pysr import PySRRegressor

import torch
import torch.nn as nn
import torch.optim as optim

get_ipython().system('pip install feyn')
import feyn

import shap

from google.colab import drive

get_ipython().system('pip install pykan')
from kan import *
get_ipython().system('pip install gplearn')
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function






# In[7]:


# Стиль рисунков
sns.set(style="white")
sns.set_palette("gray")


# In[8]:


drive.mount('/content/drive')
data1 = pd.read_excel('/content/drive/My Drive/Python ML/dataset_tube_bundle.xlsx', sheet_name='dataset_epsilon', index_col=None, usecols=['εT','Nu_sum'])
data1_Nu_pd = pd.DataFrame(data1['Nu_sum'])
data1_dT_pd = pd.DataFrame(data1['εT'])
drive.flush_and_unmount()


# In[9]:


# Примеры случайных значений переменных
print(f'Входная переменная dT:')
data1_dT_pd.sample(10)


# In[10]:


print(f'Целевая переменная Nu:')
data1_Nu_pd.sample(10)


# In[11]:


drive.mount('/content/drive')
data2 = pd.read_excel('/content/drive/My Drive/Python ML/dataset_tube_bundle.xlsx', sheet_name='dataset_sigma', index_col=None, usecols=['σ','Nu_sum'])
data2_Nu_pd = pd.DataFrame(data2['Nu_sum'])
data2_S_pd = pd.DataFrame(data2['σ'])
drive.flush_and_unmount()


# In[12]:


# Примеры случайных значений переменных
print(f'Входная переменная σ:')
data2_S_pd.sample(10)


# In[13]:


print(f'Целевая переменная Nu:')
data2_Nu_pd.sample(10)


# In[64]:


drive.mount('/content/drive')
data3 = pd.read_excel('/content/drive/My Drive/Python ML/dataset_tube_bundle_shaft.xlsx', sheet_name='bundle_shaft', index_col=None, usecols=['σ','s/d','H/d','εT','Nu_sum'])
data3_Nu_pd = pd.DataFrame(data3['Nu_sum'])
data3_Feat_pd = data3[['σ','s/d','H/d','εT']]
drive.flush_and_unmount()


# In[65]:


# Примеры случайных значений переменных
print(f'Входные переменные:')
data3_Feat_pd.sample(10)


# In[66]:


print(f'Целевая переменная Nu:')
data3_Nu_pd.sample(10)


# ## 2.1. Линейная регрессия

# ### 2.1.1 Зависимость числа Нуссельта от перепада температур

# In[ ]:


X_T  = data1_dT_pd.values
y_T = data1_Nu_pd.values


# In[ ]:


# разделяем на обучающую и тестовую
X_T_train, X_T_test, y_T_train, y_T_test = train_test_split(X_T, y_T, test_size=0.2, random_state=42)


# In[ ]:


# Обучение линейной регрессии
model_LR = LinearRegression()
model_LR.fit(X_T_train, y_T_train)

# Прогнозирование сна тестовой выборке
y_LR_test = model_LR.predict(X_T_test)

# Оценка качества модели тестовой
print(f'MSE_test: {mean_squared_error(y_T_test, y_LR_test)}')

# Прогнозирование на всей выборке
y_LR_all = model_LR.predict(X_T)

# Оценка качества модели на всех данных
print(f'MSE_all: {mean_squared_error(y_T, y_LR_all)}')


# In[ ]:


# Печать предсказанных значений Nu, весовых коэффициентов и свободного члена (для составления формулы)
print(y_LR_all)
print(model_LR.coef_)
print(model_LR.intercept_)


# In[ ]:


# Отрисовка графиков Nu от eT
plt.plot(data1_dT_pd, data1_Nu_pd, label='CFD')
plt.plot(data1_dT_pd, y_LR_all, label='Linear Reggression', c='r')
plt.ylabel('Nu')
plt.xlabel('εT')
plt.legend()
plt.grid(True)
plt.show()


# ### 2.1.2 Зависимость числа Нуссельта от расстояния между трубами

# In[ ]:


X_sigma = data2_S_pd.values
y_sigma = data2_Nu_pd.values


# In[ ]:


# разделяем на обучающую и тестовую
X_sigma_train, X_sigma_test, y_sigma_train, y_sigma_test = train_test_split(X_sigma, y_sigma, test_size=0.2, random_state=42)


# In[ ]:


# Обучение линейной регрессии
model_LR = LinearRegression()
model_LR.fit(X_sigma_train, y_sigma_train)

# Прогнозирование сна тестовой выборке
y_LR_test = model_LR.predict(X_sigma_test)

# Оценка качества модели тестовой
print(f'MSE_test: {mean_squared_error(y_sigma_test, y_LR_test)}')

# Прогнозирование на всей выборке
y_LR_all = model_LR.predict(X_sigma)

# Оценка качества модели на всех данных
print(f'MSE_all: {mean_squared_error(y_sigma, y_LR_all)}')


# In[ ]:


# Печать предсказанных значений Nu, весовых коэффициентов и свободного члена (для составления формулы)
print(y_LR_all)
print(model_LR.coef_)
print(model_LR.intercept_)


# In[ ]:


# Отрисовка графиков зависимости Nu от σ
plt.plot(data2_S_pd, data2_Nu_pd, label='CFD')
plt.plot(data2_S_pd, y_LR_all, label='Linear Reggression', c='r')
plt.ylabel('Nu')
plt.xlabel('σ')
plt.legend()
plt.grid(True)
plt.show()


# Линейная регрессия плохо подходит для подобного типа задач, поэтому дальше рассматривать данную модель нет смысла.

# ## 2.2 Полиномиальная регрессия

# ### 2.2.1 Зависимость числа Нуссельта от перепада температур

# In[ ]:


# Функция регрессии со степенью полинома
def regression(parameters, samples_train, targets_train) -> object:
    '''
    Функция выполняет регрессию со степенью полинома.

    Параметры вызова:
    parameters -- параметры модели
    samples_train -- образцы для обучения
    targets_train -- целевые данные для обучения

    Возвращаемые параметры:
    model -- обученная модель
    best_degree -- лучшая степень полинома
    '''

    # Использование функции GridSearchCV ("поиск по сетке") для поиска лучших параметров модели
    search = GridSearchCV(make_pipeline(PolynomialFeatures(), LinearRegression()),
                          param_grid=parameters, n_jobs=-1, cv=5)
    search.fit(samples_train, targets_train)

    # Получение лучшей степени полинома на основе результатов выполнения функции GridSearchCV
    best_degree = search.best_params_['polynomialfeatures__degree']

    # Создание модели с лучшей степенью полинома и обучение модели
    model = make_pipeline(PolynomialFeatures(degree=best_degree), LinearRegression(n_jobs=-1))
    return model.fit(samples_train, targets_train)


# In[ ]:


# Вызов функции, выполняющей регрессию со степенью полинома
model_PR = regression({'polynomialfeatures__degree': np.arange(1, 5)}, X_T, y_T)

# Прогнозирование с использованием обученной модели
y_PR_all = model_PR.predict(X_T)

# Оценка качества модели
print(f'MSE: {mean_squared_error(y_T, y_PR_all)}')


# In[ ]:


# Печать предсказанных значений Nu, весовых коэффициентов и свободного члена (для составления формулы)
print(y_PR_all)
print(model_PR[1].coef_)
print(model_PR[1].intercept_)


# In[ ]:


# Отрисовка графиков Nu от eT
plt.plot(data1_dT_pd, data1_Nu_pd, label='CFD')
plt.plot(data1_dT_pd, y_PR_all, label='Polynomial Reggression', c='r')
plt.ylabel('Nu')
plt.xlabel('εT')
plt.legend()
plt.grid(True)
plt.show()


# ### 2.2.2 Зависимость числа Нуссельта от расстояния между трубами

# In[ ]:


# Вызов функции, выполняющей регрессию со степенью полинома
model_PR = regression({'polynomialfeatures__degree': np.arange(1, 5)}, X_sigma, y_sigma)

# Прогнозирование с использованием обученной модели
y_PR_all = model_PR.predict(X_sigma)

# Оценка качества модели
print(f'MSE: {mean_squared_error(y_sigma, y_PR_all)}')


# In[ ]:


# Печать предсказанных значений Nu, весовых коэффициентов и свободного члена (для составления формулы)
print(y_PR_all)
print(model_PR[1].coef_)
print(model_PR[1].intercept_)


# In[ ]:


# Отрисовка графиков зависимости Nu от σ

plt.plot(data2_S_pd, data2_Nu_pd, label='CFD')
plt.plot(data2_S_pd, y_PR_all, label='Polynomial Regression', c='r')
plt.ylabel('Nu')
plt.xlabel('σ')
plt.legend()
plt.grid(True)
plt.show()


# # 3. Символьная регрессия

# ## 3.1. Зависимость числа Нуссельта от перепада температур

# ### QLattice

# In[ ]:


data1_Nu = data1_Nu_pd.values[:,0]
data1_dT = data1_dT_pd.values[:,0]


# **Исходные данные**

# In[ ]:


# Создание экземпляра класса
ql = feyn.QLattice(random_seed=21)

# Определение списков для моделей и потерь
models = []
loss_history1_ql = []

# Задание количества эпох
n_epochs = 10

# Вычисление исходной вероятности на основе входных данных
# (данные, целевая переменная)
priors = feyn.tools.estimate_priors(data1, 'Nu_sum')

# Обновление вероятностей
ql.update_priors(priors)

for epoch in range(n_epochs):
    # Отбор новых образцов моделей и добавление их в список
    # (названия столбцов,
    #  целевая переменная,
    #  тип (classification, regression, auto),
    #  сложность модели (глубина графа))
    models += ql.sample_models(data1.columns,
                               'Nu_sum',
                               'regression',
                               max_complexity=4
    )

    # Обучение отобранных моделей на данных
    # Возвращается список, отсортированный по потерям
    # (список моделей,
    #  обучающие данные,
    #  вид функции потерь (binary_cross_entropy, squared_error))
    models = feyn.fit_models(models, data1, 'squared_error')

    # Удаление плохих моделей
    # (список моделей,
    #  максимальное количество оставляемых)
    models = feyn.prune_models(models)

    # Предсказание лучшей модели
    pred = models[0].predict(data1_dT_pd)

    # Потери для лучшей модели
    loss_history1_ql.append(mean_squared_error(data1_Nu, pred))
    print(f'Epoch = {epoch+1}, MSE: {loss_history1_ql[-1]}')

    # Обновление моделей в соответствии с новыми отсортированными образцами
    ql.update(models)

# 10 лучших и достаточно разных моделей
# (список моделей, количество отбираемых)
best_models1 = feyn.get_diverse_models(models, n=10)


# In[ ]:


plt.figure(figsize=(10, 6))
plt.plot(range(1,len(loss_history1_ql)+1), loss_history1_ql, color="blue")
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.grid(True)
plt.yscale('log')
plt.show()


# In[ ]:


for i, model in enumerate(best_models1):
  sympy_model = model.sympify(signif=3)
  print('Model',i,':', sympy_model)


# In[ ]:


best1_ql = best_models1[3]
sympy_model = best1_ql.sympify(signif=3)
sympy_model.as_expr()


# In[ ]:


# Отрисовка графиков зависимости Nu от dT
plt.plot(data1_dT, data1_Nu, label='CFD')
plt.plot(data1_dT, best1_ql.predict(data1_dT_pd), label='QLattice', c='r')
plt.ylabel('Nu')
plt.xlabel('εT')
plt.legend()
plt.grid(True)
plt.show()


# ### PySR

# In[ ]:


X_T  = data1_dT_pd.values.reshape(-1, 1)
y_T = data1_Nu_pd.values


# **Исходные данные**

# По умолчанию PySR задействует все ядра на компьютере, при этом каждый запуск обучения может приводить к различным конечным результатам. Для воспроизведения результатов необходимо проводить обучение на одном ядре.

# In[ ]:


model = PySRRegressor(
    random_state=42, # для воспроизведения
    deterministic=True,
    parallelism='serial',
    maxsize=15, # максимальный размер формул
    niterations=400,               # число итераций эволюции: формулы - точность - лучшие
    population_size=50,           # размер популяции - в каждом поколении N функций
    parsimony=0.05, # штраф за сложность: MSE + parsimony * сложность
    binary_operators=["+", "-", "*", "/"], # бинарные операции
    unary_operators=[             # разрешенные операторы
        "sqrt", # хорошо описывает замедляющий рост
        "exp", #  сигмоидальные кривые
        "square", # если есть квадратичная
        "inv(x) = 1/x", # полезен для дробно-линейных
    ],
    constraints={
        "sqrt": 4,      # Не более 4 sqrt
        "exp": 1,       # Не более 1 exp в формуле
    },
    nested_constraints={
        "sqrt": {"sqrt": 0}
    },  # Запретить sqrt внутри sqrt
    extra_sympy_mappings={"inv": lambda x: 1 / x}, # определяем оператор для sympy
)

model.fit(X_T, y_T, variable_names=["T"])

print(model)


# Так как Google Colab, возможно, имеет другой случайный генератор, было принято загрузить имеющееся решение (обученную модель), полученное на локальной машине. Настройки модели полностью совпадают.

# In[ ]:


drive.mount('/content/drive')
model = PySRRegressor.from_file(
    run_directory="/content/drive/My Drive/Python ML/Models_PySR/model_T_1/",
)
drive.flush_and_unmount()


# In[ ]:


# Лучшая функция
rounded_expr = model.sympy().evalf(n=5)
rounded_expr.as_expr()


# In[ ]:


# Отрисовка графиков зависимости loss и score от complexity
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

index_value = model.get_best().name

### complexity - мера сложности формулы, учитывающая количество операторов, функций и констант в уравнении
### score - комбинированная метрика, которая учитывает точность формулы и её сложность

# График зависимости loss от complexity
axs[0].plot(model.equations_['complexity'], model.equations_['loss'], marker='x')
axs[0].set_xlabel('Complexity')
axs[0].set_ylabel('MSE')
axs[0].set_yscale('log')
axs[0].grid(True)

axs[0].scatter(model.equations_['complexity'].iloc[index_value],
                model.equations_['loss'].iloc[index_value],
                color='red', s=50, linewidth=2, label='Best Formula')

# График зависимости score от complexity
axs[1].plot(model.equations_['complexity'], model.equations_['score'], marker='x', color='orange')
axs[1].set_xlabel('Complexity')
axs[1].set_ylabel('Score')
axs[1].set_yscale('linear')
axs[1].grid(True)

axs[1].scatter(model.equations_['complexity'].iloc[index_value],
                model.equations_['score'].iloc[index_value],
                color='red', s=50, linewidth=2, label='Best Formula')

axs[0].legend()
axs[1].legend()

plt.tight_layout()  # Автоматическая настройка отступов
plt.show()


# In[ ]:


# Отрисовка графиков зависимости Nu от dT
# Преобразование данных в подходящий формат для предсказания:
# X должен быть 2D-массивом (n_samples, n_features)
X_plot = data1_dT_pd.values.reshape(-1, 1)

# Получение предсказаний модели:
y_pred = model.predict(X_plot)

plt.plot(data1_dT_pd, data1_Nu_pd, label='CFD')
plt.plot(data1_dT_pd, y_pred, label='PySR', c='r')
plt.ylabel('Nu')
plt.xlabel('εT')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# **Логарифм от исходных данных**

# In[ ]:


data1_Nu_log = np.log(data1_Nu)

plt.plot(data1_dT, data1_Nu, color='blue', label='Nu')
plt.plot(data1_dT, data1_Nu_log, color='red', label='log(Nu)')
plt.legend()

plt.figure()
plt.hist(data1_Nu, color='blue', bins=20, alpha=0.5, label='Nu')
plt.hist(data1_Nu_log, color='red', bins=20, alpha=0.5, label='log(Nu)')
plt.legend();


# Нет смысла искать формулу для log(Nu): значения по форме распределены так же, как и для Nu (зависимость не степенная).

# ###GPLearn

# In[4]:


def dtanh_fun(x):
    return np.tanh(x)

def exp_fun(x):
    return np.exp(np.longdouble(x))

dtanh = make_function(function=dtanh_fun, name='dtanh', arity=1)
exp = make_function(function=exp_fun, name='exp', arity=1)


# In[16]:


data_Nu_pd = pd.DataFrame(data1['Nu_sum'])
data_dT_pd = pd.DataFrame(data1['εT'])


# In[17]:


function_set = ['add', 'sub', 'mul', 'div']

model_gp = SymbolicRegressor(population_size=2000, tournament_size=50, init_depth=(4,10),
                           generations=50, stopping_criteria=5e-4,
                           p_crossover=0.7, p_subtree_mutation=0.2,
                           p_hoist_mutation=0.01, p_point_mutation=0.09,
                           max_samples=0.9, verbose=1, function_set=function_set,
                           parsimony_coefficient=1e-3, metric='rmse', random_state=94194,
                           const_range = (-10, 10), )



model_gp.fit(data_dT_pd.values, np.ravel(data_Nu_pd.values))


# In[18]:


plt.plot(data_dT_pd, data_Nu_pd, label='CFD', c='k')

X_train, X_test, y_train, y_test = train_test_split(data_dT_pd.values, data_Nu_pd.values, test_size=0.1, random_state=1)


X = np.linspace(data_dT_pd.values.min(), data_dT_pd.values.max(), 100)
Y = model_gp.predict(X[:, np.newaxis])
plt.plot(X, Y, label='GPLearn', c='r')

print(np.mean(np.abs(model_gp.predict(data_dT_pd.values)-data_Nu_pd.values)/data_Nu_pd.values))
plt.legend()
plt.ylabel('Nu', fontsize=12)
plt.xlabel('εT', fontsize=12)
plt.grid(True)
plt.show()


# In[ ]:





# ## 3.2. Зависимость числа Нуссельта от расстояния между трубами

# ### QLattice

# In[ ]:


data2_Nu = data2_Nu_pd.values[:,0]
data2_S = data2_S_pd.values[:,0]


# **Исходные данные**

# In[ ]:


# Создание экземпляра класса
ql = feyn.QLattice(random_seed=20)

# Определение списков для моделей и потерь
models = []
loss_history2_ql = []

# Задание количества эпох
n_epochs = 10

# Вычисление исходной вероятности на основе входных данных
priors = feyn.tools.estimate_priors(data2, 'Nu_sum')

# Обновление вероятностей
ql.update_priors(priors)

for epoch in range(n_epochs):
    # Отбор новых образцов моделей и добавление их в список
    models += ql.sample_models(data2.columns, 'Nu_sum', 'regression')

    # Обучение отобранных моделей на данных
    # Возвращается список, отсортированный по потерям
    models = feyn.fit_models(models, data2, 'squared_error')

    # Удаление плохих моделей
    models = feyn.prune_models(models)

    # Предсказание лучшей модели
    pred = models[0].predict(data2_S_pd)

    # Потери для лучшей модели
    loss_history2_ql.append(mean_squared_error(data2_Nu, pred))
    print(f'Epoch = {epoch+1}, MSE: {loss_history2_ql[-1]}')

    # Обновление моделей в соответствии с новыми отсортированными образцами
    ql.update(models)

# 10 лучших и достаточно разных моделей
best_models2 = feyn.get_diverse_models(models, n=10)


# In[ ]:


plt.figure(figsize=(10, 6))
plt.plot(range(1,len(loss_history2_ql)+1), loss_history2_ql, color="blue")
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.grid(True)
plt.yscale('log')
plt.show()


# In[ ]:


for i, model in enumerate(best_models2):
  sympy_model = model.sympify(signif=3)
  print('Model',i,':', sympy_model)


# In[ ]:


best2_ql = best_models2[7]
sympy_model = best2_ql.sympify(signif=3)
sympy_model.as_expr()


# In[ ]:


# Отрисовка графиков зависимости Nu от σ
plt.plot(data2_S, data2_Nu, label='CFD')
plt.plot(data2_S, best2_ql.predict(data2_S_pd), label='QLattice', c='r')
plt.ylabel('Nu')
plt.xlabel('σ')
plt.legend()
plt.grid(True)
plt.show()


# ### PySR

# In[ ]:


X_sigma  = data2_S_pd.values.reshape(-1, 1)
y_sigma = data2_Nu_pd.values


# **Исходные данные**

# In[ ]:


model_sigma = PySRRegressor(
    random_state=42,      # для воспроизведения результатов
    deterministic=True,
    parallelism='serial',
    maxsize=30,          # максимальный размер формул
    niterations=400,     # число итераций эволюции
    population_size=20,  # размер популяции
    parsimony=0.000001,  # штраф за сложность
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[
        "exp",
        "square",
    ],
    nested_constraints={
        "exp": {"exp": 0},
        "square": {"square": 0},
    },
)

model_sigma.fit(X_sigma, y_sigma, variable_names=["s"])

print(model_sigma)


# In[ ]:


drive.mount('/content/drive')
model_sigma = PySRRegressor.from_file(
    run_directory="/content/drive/My Drive/Python ML/Models_PySR/model_sigma_1/",
)
drive.flush_and_unmount()


# In[ ]:


# Лучшая функция
rounded_expr = model_sigma.sympy().evalf(n=5)
#print(rounded_expr)
rounded_expr.as_expr()


# In[ ]:


# Отрисовка графиков зависимости loss и score от complexity
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

index_value = model_sigma.get_best().name

### complexity - мера сложности формулы, учитывающая количество операторов, функций и констант в уравнении
### score - комбинированная метрика, которая учитывает точность формулы и её сложность

# График зависимости loss от complexity
axs[0].plot(model_sigma.equations_['complexity'], model_sigma.equations_['loss'], marker='x')
axs[0].set_xlabel('Complexity')
axs[0].set_ylabel('MSE')
axs[0].set_yscale('log')
axs[0].grid(True)

axs[0].scatter(model_sigma.equations_['complexity'].iloc[index_value],
                model_sigma.equations_['loss'].iloc[index_value],
                color='red', s=50, linewidth=2, label='Best Formula')

# График зависимости score от complexity
axs[1].plot(model_sigma.equations_['complexity'], model_sigma.equations_['score'], marker='x', color='orange')
axs[1].set_xlabel('Complexity')
axs[1].set_ylabel('Score')
axs[1].set_yscale('linear')
axs[1].grid(True)

axs[1].scatter(model_sigma.equations_['complexity'].iloc[index_value],
                model_sigma.equations_['score'].iloc[index_value],
                color='red', s=50, linewidth=2, label='Best Formula')

axs[0].legend()
axs[1].legend()

plt.tight_layout()  # Автоматическая настройка отступов
plt.show()


# In[ ]:


# Отрисовка графиков зависимости Nu от σ
#print(model_sigma.equations_)
#print(model_sigma.equations_.index)

X_plot = data2_S_pd.values.reshape(-1, 1)

# Получение предсказаний модели:
y_pred = model_sigma.predict(X_plot)

plt.plot(data2_S_pd, data2_Nu_pd, label='CFD')
plt.plot(data2_S_pd, y_pred, label='PySR', c='r')
plt.ylabel('Nu')
plt.xlabel('σ')
plt.legend()
plt.grid(True)
plt.show()


# Оптимальная конфигурация и формула с учетом вычислительных затрат.

# **Логарифм от исходных данных**

# In[ ]:


data2_Nu_log = np.log(data2_Nu)

plt.plot(data2_S, data2_Nu, c='b', label='Nu')
plt.plot(data2_S, data2_Nu_log, c='r', label='log(Nu)')
plt.legend()

plt.figure()
plt.hist(data2_Nu, color='blue', bins=20, alpha=0.5, label='Nu')
plt.hist(data2_Nu_log, color='red', bins=20, alpha=0.5, label='log(Nu)')
plt.legend();


# ###GPLearn

# In[19]:


data_Nu_pd = pd.DataFrame(data2['Nu_sum'])
data_S_pd = pd.DataFrame(data2['σ'])


# In[20]:


function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'sin', 'cos']

model_gp = SymbolicRegressor(population_size=4000, tournament_size=50, init_depth=(2,10),
                           generations=100, stopping_criteria=5e-4,
                           p_crossover=0.5, p_subtree_mutation=0.3,
                           p_hoist_mutation=0.01, p_point_mutation=0.19,
                           max_samples=0.9, verbose=1, function_set=function_set,
                           parsimony_coefficient=7e-5, metric='rmse', random_state=994,
                           const_range = (-10, 10), )


model_gp.fit(data_S_pd.values, np.ravel(data_Nu_pd.values))


# In[21]:


plt.plot(data_S_pd, data_Nu_pd, label='CFD', c='k')

X_train, X_test, y_train, y_test = train_test_split(data_S_pd.values, data_Nu_pd.values, test_size=0.1, random_state=1)


X = np.linspace(data_S_pd.values.min(), data_S_pd.values.max(), 100)
Y = model_gp.predict(X[:, np.newaxis])
plt.plot(X, Y, label='GPLearn', c='r')

plt.legend()
plt.ylabel('Nu', fontsize=12)
plt.xlabel('σ', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# Нет смысла искать формулу для log(Nu): значения по форме распределены так же, как и для Nu (зависимость не степенная).

# ## 3.3. Термогравитационное течение воздуха через двухрядный пучок труб

# ## **Исходные данные**

# ### QLattice

# In[46]:


data3_Nu = data3_Nu_pd.values[:,0]
data3_Feat = data3_Feat_pd.values


# In[ ]:


# Создание экземпляра класса
ql = feyn.QLattice(random_seed=14)

# Определение списков для моделей и потерь
models = []
loss_history3_ql = []

# Задание количества эпох
n_epochs = 20

# Вычисление исходной вероятности на основе входных данных
priors = feyn.tools.estimate_priors(data3, 'Nu_sum')

# Обновление вероятностей
ql.update_priors(priors)

for epoch in range(n_epochs):
    # Отбор новых образцов моделей и добавление их в список
    models += ql.sample_models(data3.columns, 'Nu_sum', 'regression')

    # Обучение отобранных моделей на данных
    # Возвращается список, отсортированный по потерям
    models = feyn.fit_models(models, data3, 'squared_error')

    # Удаление плохих моделей
    models = feyn.prune_models(models)

    # Предсказание лучшей модели
    pred = models[0].predict(data3_Feat_pd)

    # Потери для лучшей модели
    loss_history3_ql.append(mean_squared_error(data3_Nu, pred))
    print(f'Epoch = {epoch+1}, MSE: {loss_history3_ql[-1]}')

    # Обновление моделей в соответствии с новыми отсортированными образцами
    ql.update(models)

# 10 лучших и достаточно разных моделей
best_models3 = feyn.get_diverse_models(models, n=10)


# In[ ]:


plt.figure(figsize=(10, 6))
plt.plot(range(1,len(loss_history3_ql)+1), loss_history3_ql, color="blue")
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.grid(True)
plt.yscale('log')
plt.show()


# In[ ]:


for i, model in enumerate(best_models3):
  sympy_model = model.sympify(signif=3)
  print('Model',i,':', sympy_model)


# In[ ]:


best3_ql = best_models3[0]
sympy_model = best3_ql.sympify(signif=3)
sympy_model.as_expr()


# In[ ]:


plt.hist(data3_Nu, label='CFD', color='blue',bins=20, alpha=0.5)
plt.hist(best3_ql.predict(data3_Feat_pd), label='QLattice', color='red', bins=20, alpha=0.5)
plt.xlabel('Nu')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


rel_err = (data3_Nu - best3_ql.predict(data3_Feat_pd))/data3_Nu
plt.scatter(data3_Nu,rel_err,s=25)
plt.title('QLattice 1')
plt.xlabel('Nu')
plt.ylabel('(Nu - Nu_pred)/Nu')
plt.grid(True);


# In[ ]:


best3_ql2 = best_models3[2]
sympy_model = best3_ql2.sympify(signif=3)
sympy_model.as_expr()


# In[ ]:


plt.hist(data3_Nu, label='CFD', color='blue',bins=20, alpha=0.5)
plt.hist(best3_ql2.predict(data3_Feat_pd), label='QLattice', color='red', bins=20, alpha=0.5)
plt.xlabel('Nu')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


rel_err = (data3_Nu - best3_ql2.predict(data3_Feat_pd))/data3_Nu
plt.scatter(data3_Nu,rel_err,s=25)
plt.title('QLattice 2')
plt.xlabel('Nu')
plt.ylabel('(Nu - Nu_pred)/Nu')
plt.grid(True);


# **SHAP (для Nu)**

# In[ ]:


explainer_ebm = shap.Explainer(best3_ql.predict, data3_Feat_pd)
shap_values_ebm = explainer_ebm(data3_Feat_pd)
shap.plots.beeswarm(shap_values_ebm)


# In[ ]:


best3_ql.plot_response_2d(
    data=data3,
    fixed={
        'H/d': data3['H/d'].median(),
        'σ': data3['σ'].median()
    }
)


# In[ ]:


best3_ql.plot_response_2d(
    data=data3,
    fixed={
        'εT': data3['εT'].median(),
        'σ': data3['σ'].median()
    }
)


# ### PySR

# In[67]:


X_shaft = data3_Feat_pd.values
y_shaft = data3_Nu_pd.values


# PySR не поддерживает такие символы, как σ, а также в явном виде не дает задавать в названиях переменных / * и другие символы, поэтому решено было использовать обозначение s == σ, k == s/d, H == h/d, eT == T.

# In[ ]:


model_shaft = PySRRegressor(
    random_state=42,     # для воспроизведения результатов
    deterministic=True,
    parallelism='serial',
    maxsize=30,          # максимальный размер формул
    niterations=150,     # число итераций эволюции
    population_size=35,  # размер популяции
    parsimony=0.000001,  # штраф за сложность
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[
        "exp",
        "square",
        "sqrt",
        "tanh",
        "cos",
    ],
    constraints={
        "exp": 1,
        "cos": 1,
        "tanh": 1,
    },
    nested_constraints={
        "exp": {"exp": 0},
        "square": {"square": 0},
    },
)

model_shaft.fit(X_shaft, y_shaft, variable_names=["s", "k", "h", "T"])

print(model_shaft)


# In[68]:


drive.mount('/content/drive')
model_shaft = PySRRegressor.from_file(
    run_directory="/content/drive/My Drive/Python ML/Models_PySR/model_termo_1/",
)
drive.flush_and_unmount()


# In[69]:


# Лучшая функция
rounded_expr = model_shaft.sympy().evalf(n=5)
#print(rounded_expr)
rounded_expr.as_expr()


# In[50]:


# Отрисовка графиков зависимости loss и score от complexity
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

index_value = model_shaft.get_best().name

### complexity - мера сложности формулы, учитывающая количество операторов, функций и констант в уравнении
### score - комбинированная метрика, которая учитывает точность формулы и её сложность

# График зависимости loss от complexity
axs[0].plot(model_shaft.equations_['complexity'], model_shaft.equations_['loss'], marker='x')
axs[0].set_xlabel('Complexity')
axs[0].set_ylabel('MSE')
axs[0].set_yscale('log')
axs[0].grid(True)

axs[0].scatter(model_shaft.equations_['complexity'].iloc[index_value],
                model_shaft.equations_['loss'].iloc[index_value],
                color='red', s=50, linewidth=2, label='Best Formula')

# График зависимости score от complexity
axs[1].plot(model_shaft.equations_['complexity'], model_shaft.equations_['score'], marker='x', color='orange')
axs[1].set_xlabel('Complexity')
axs[1].set_ylabel('Score')
axs[1].set_yscale('linear')
axs[1].grid(True)

axs[1].scatter(model_shaft.equations_['complexity'].iloc[index_value],
                model_shaft.equations_['score'].iloc[index_value],
                color='red', s=50, linewidth=2, label='Best Formula')

axs[0].legend()
axs[1].legend()

plt.tight_layout()  # Автоматическая настройка отступов
plt.show()


# In[52]:


# Отрисовка гистограммы Nu
plt.hist(y_shaft, label='CFD', color='blue',bins=20, alpha=0.5)
plt.hist(model_shaft.predict(X_shaft), label='PySR', color='red', bins=20, alpha=0.5)
plt.xlabel('Nu')
plt.legend()
plt.grid(True)
plt.show()


# In[53]:


# Построение графиков зависимости ошибок от Nu
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

y_pred = model_shaft.predict(X_shaft)

# Истинные значения
y_true = y_shaft.flatten() # преобр в форму (N,)

# Вычисляем ошибки
abs_error = y_pred - y_true
rel_error = (abs_error / y_true)

# График абсолютных ошибок
axs[0].scatter(y_true, abs_error, alpha=0.7)
axs[0].set_xlabel('Nu')
axs[0].set_ylabel('Abs. error')
axs[0].set_title('Absolute Error')
axs[0].grid(True)

# График относительных ошибок
axs[1].scatter(y_true, rel_error, alpha=0.7)
axs[1].set_xlabel('Nu')
axs[1].set_ylabel('(Nu - Nu_pred)/Nu')
axs[1].set_title('Relative Error')
axs[1].grid(True)

plt.show()


# ### SHAP (для Nu) при использовании PySR

# In[72]:


data3_Feat_renamed = data3_Feat_pd.copy()
data3_Feat_renamed.columns = ['s','k','h','T']
explainer_ebm = shap.Explainer(model_shaft.predict, data3_Feat_renamed)
shap_values_ebm = explainer_ebm(data3_Feat_renamed)
shap.plots.beeswarm(shap_values_ebm)


# ###GPLearn
# 

# In[22]:


data_Nu_pd = pd.DataFrame(data3['Nu_sum'])
data_Feat_pd = pd.DataFrame(data3[['σ','s/d','H/d','εT']])


# In[23]:


function_set = ['add', 'sub', 'mul', 'div', 'sqrt', dtanh]

model_gp = SymbolicRegressor(population_size=500, tournament_size=50, init_depth=(5,8),
                           generations=100, stopping_criteria=5e-4,
                           p_crossover=0.5, p_subtree_mutation=0.3,
                           p_hoist_mutation=0.01, p_point_mutation=0.19,
                           max_samples=0.9, verbose=1, function_set=function_set,
                           parsimony_coefficient=1e-2, metric='rmse', random_state=2904,
                           const_range = (-100, 100), )


model_gp.fit(data_Feat_pd.values, np.ravel(data_Nu_pd.values))


# In[24]:


plt.hist(data_Nu_pd, label='CFD', color='blue',bins=20, alpha=0.5)
plt.hist(model_gp.predict(data_Feat_pd), label='GPLearn', color='red', bins=20, alpha=0.5, rwidth=0.9)
plt.xlabel('Nu', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# In[25]:


explainer = shap.Explainer(model_gp.predict, data_Feat_pd)

shap_values = explainer(data_Feat_pd)

shap.plots.beeswarm(shap_values)


# In[26]:


rel_err = (np.ravel(data_Nu_pd.values) - model_gp.predict(data_Feat_pd))/data_Nu_pd.values[:, 0]
plt.scatter(np.ravel(data_Nu_pd.values),np.ravel(rel_err),s=25, c='k')
plt.title('GPLearn')
plt.xlabel('Nu', fontsize=12)
plt.ylabel('(Nu - Nu_pred)/Nu', fontsize=12)
plt.grid(True)


# ### PyKAN

# In[ ]:


torch.set_default_dtype(torch.float64)

# width - задает количество слоев (количество элементов списка) и число нейронов
# в каждом слое (первый слой содержит столько же нейронов, скольо имеется
# признаков, последний столько, сколько имеется выходных переменных);
# grid - размер сетки B-сплайнов, аппроксимирующих функцию активации;
# k - степень сплайна;
# seed - число для повторяемости результатов
data3_kan = KAN(width=[4,3,1], grid=3, k=3, seed=38)


# In[ ]:


dataset = {}
dataset['train_input'] = torch.tensor(data3_Feat)
dataset['train_label'] = torch.tensor(data3_Nu.reshape(-1,1))
dataset['test_input'] = torch.tensor(data3_Feat)
dataset['test_label'] = torch.tensor(data3_Nu.reshape(-1,1))
print(dataset['train_input'].shape, dataset['train_label'].shape)


# In[ ]:


# Вид инициализированной сети
data3_kan(dataset['train_input'])
data3_kan.plot()


# In[ ]:


# Обучение модели:
# dataset - обучающие данные;
# opt - алгоритм оптимизации ('LBFGS'/'Adam');
# steps - количество итераций;
# lamb - параметр регуляризации (если 0, то регуляризация не осуществляется)
data3_kan.fit(dataset, opt="LBFGS", steps=50, lamb=0.001);


# In[ ]:


# Вид сети после первого этапа обучения
data3_kan.plot()


# In[ ]:


# prune - удаление слабых связей (нейроны без связей пропадают),
# что позволяет упростить модель
data3_kan = data3_kan.prune()
data3_kan.plot()


# In[ ]:


data3_kan.fit(dataset, opt="LBFGS", steps=50);


# In[ ]:


# refine(n) - переход от исходной сетки к сетке с
# размером n для улучшения модели
data3_kan = data3_kan.refine(10)


# In[ ]:


data3_kan.fit(dataset, opt="LBFGS", steps=50);


# In[ ]:


lib = ['x','x^2','x^3','x^4','exp','sqrt','sin','abs']
# Автоматическая замена сплайнов на аналитические (символьные)
# функции из списка lib
data3_kan.auto_symbolic(lib=lib)


# In[ ]:


pykan_loss = data3_kan.fit(dataset, opt="LBFGS", steps=50);


# In[ ]:


from kan.utils import ex_round
# Отображение формулы с количеством десятичных знаков - 4
ex_round(data3_kan.symbolic_formula()[0][0],4)


# In[ ]:


kan_predicted = data3_kan(dataset['test_input'])
print(kan_predicted[::10])


# In[ ]:


plt.hist(data3_Nu, label='CFD', color='blue',bins=20, alpha=0.5)
plt.hist(kan_predicted.detach().numpy()[:,0], label='KAN', color='red', bins=20, alpha=0.5)
plt.xlabel('Nu')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


rel_err = (data3_Nu - kan_predicted.detach().numpy()[:,0])/data3_Nu
plt.scatter(data3_Nu,rel_err,s=25)
plt.title('KAN')
plt.xlabel('Nu')
plt.ylabel('(Nu - Nu_pred)/Nu')
plt.grid(True);


# ###GBR (gradient boosted regressor)

# In[38]:


X_train = data3[['σ','s/d','H/d','εT']]

Y_train = data3['Nu_sum']


# In[39]:


gbr = GradientBoostingRegressor()


loss_history = []

gbr.fit(X_train, Y_train)


for i, y_pred in enumerate(gbr.staged_predict(X_train)):
    loss = mean_squared_error(Y_train, y_pred)
    loss_history.append(loss)

predictions_gbr = gbr.predict(X_train)


# In[40]:


plt.figure(figsize=(10, 6))
plt.plot(range(1,len(loss_history)+1), loss_history, color="blue")
plt.ylabel('MSE')
plt.xlabel('Base estimators')
plt.grid(True)
plt.yscale('log')
plt.show()


# In[41]:


plt.hist(Y_train, label='CFD', color='blue',bins=15, alpha=0.5)
plt.hist(predictions_gbr, label='GBR', color='red', bins=15, alpha=0.5, rwidth=0.9)
plt.xlabel('Nu', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# In[44]:


rel_err = (-predictions_gbr+Y_train)/predictions_gbr
plt.scatter(Y_train,np.ravel(rel_err),s=25, c='k')
plt.title('GBR')
plt.xlabel('Nu', fontsize=12)
plt.ylim(-0.5, 0.5)
plt.ylabel('(Nu - Nu_pred)/Nu', fontsize=12)
plt.grid(True)


# In[43]:


explainer = shap.Explainer(gbr.predict, X_train)

shap_values = explainer(X_train)

shap.plots.beeswarm(shap_values)


# ## **Логарифм от исходных данных**

# ### QLattice

# In[ ]:


data3_Nu_log = np.log(data3_Nu)
plt.hist(data3_Nu, color='blue', bins=20, alpha=0.5, label='Nu')
plt.hist(data3_Nu_log, color='red', bins=20, alpha=0.5, label='log(Nu)')
plt.legend();


# Распределения Nu и log(Nu) заметно отличаются по форме. Можно попробовать построить модель для логарифма с целью возможного получения более простой формулы.

# In[ ]:


data3_log = pd.DataFrame(np.hstack([data3_Feat,data3_Nu_log[:,np.newaxis]]), columns=['σ','s/d','H/d','εT','log(Nu)'])
data3_log.sample(10)


# In[ ]:


# Создание экземпляра класса
ql = feyn.QLattice(random_seed=19)

# Определение списков для моделей и потерь
models = []
loss_history3_ql_log = []

# Задание количества эпох
n_epochs = 50

# Вычисление исходной вероятности на основе входных данных
priors = feyn.tools.estimate_priors(data3_log, 'log(Nu)')

# Обновление вероятностей
ql.update_priors(priors)

for epoch in range(n_epochs):
    # Отбор новых образцов моделей и добавление их в список
    models += ql.sample_models(data3_log.columns,
                               'log(Nu)',
                               'regression'
    )

    # Обучение отобранных моделей на данных
    # Возвращается список, отсортированный по потерям
    models = feyn.fit_models(models, data3_log, 'squared_error')

    # Удаление плохих моделей
    models = feyn.prune_models(models)

    # Предсказание лучшей модели
    pred = models[0].predict(data3_Feat_pd)

    # Потери для лучшей модели
    loss_history3_ql_log.append(mean_squared_error(data3_Nu_log, pred))
    print(f'Epoch = {epoch+1}, MSE: {loss_history3_ql_log[-1]}')

    # Обновление моделей в соответствии с новыми отсортированными образцами
    ql.update(models)

# 10 лучших и достаточно разных моделей
best_models3_log = feyn.get_diverse_models(models, n=10)


# In[ ]:


plt.figure(figsize=(10, 6))
plt.plot(range(1,len(loss_history3_ql_log)+1), loss_history3_ql_log, color="blue")
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.grid(True)
plt.yscale('log')
plt.show()


# In[ ]:


for i, model in enumerate(best_models3_log):
  sympy_model = model.sympify(signif=3)
  print('Model',i,':', sympy_model)


# In[ ]:


best3log_ql = best_models3_log[4]
sympy_model = best3log_ql.sympify(signif=3)
sympy_model.as_expr()


# In[ ]:


plt.hist(data3_Nu_log, label='CFD', color='blue',bins=20, alpha=0.5)
plt.hist(best3log_ql.predict(data3_Feat_pd), label='QLattice', color='red', bins=20, alpha=0.5)
plt.xlabel('log(Nu)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


plt.hist(data3_Nu, label='CFD', color='blue',bins=20, alpha=0.5)
plt.hist(np.exp(best3log_ql.predict(data3_Feat_pd)), label='QLattice', color='red', bins=20, alpha=0.5)
plt.xlabel('Nu')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


rel_err = (data3_Nu - np.exp(best3log_ql.predict(data3_Feat_pd)))/data3_Nu
plt.scatter(data3_Nu,rel_err,s=25)
plt.title('QLattice log')
plt.xlabel('Nu')
plt.ylabel('(Nu - Nu_pred)/Nu')
plt.grid(True);


# **SHAP (для log(Nu))**

# In[ ]:


explainer_ebm = shap.Explainer(best3log_ql.predict, data3_Feat_pd)
shap_values_ebm = explainer_ebm(data3_Feat_pd)
shap.plots.beeswarm(shap_values_ebm)


# In[ ]:


best3log_ql.plot_response_2d(
    data=data3_log,
    fixed={
        'H/d': data3_log['H/d'].median(),
        'σ': data3_log['σ'].median()
    }
)


# In[ ]:


best3log_ql.plot_response_2d(
    data=data3_log,
    fixed={
        'εT': data3_log['εT'].median(),
        'σ': data3_log['σ'].median()
    }
)


# ### PySR

# In[ ]:


y_shaft_log = np.log(y_shaft)


# In[ ]:


model_shaft_log = PySRRegressor(
    random_state=42,     # для воспроизведения результатов
    deterministic=True,
    parallelism='serial',
    maxsize=30,          # максимальный размер формул
    niterations=200,     # число итераций эволюции
    population_size=35,  # размер популяции
    parsimony=0.000001,  # штраф за сложность
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[
        "exp",
        "square",
        "sqrt",
        "tanh",
        "cos",
    ],
    constraints={
        "exp": 1,
        "cos": 1,
        "tanh": 1,
    },
    nested_constraints={
        "exp": {"exp": 0},
        "square": {"square": 0},
    },
)

model_shaft_log.fit(X_shaft, y_shaft_log, variable_names=["s", "k", "h", "T"])

print(model_shaft_log)


# In[ ]:


drive.mount('/content/drive')
model_shaft_log = PySRRegressor.from_file(
    run_directory="/content/drive/My Drive/Python ML/Models_PySR/model_termo_log_1/",
)
drive.flush_and_unmount()


# In[ ]:


# Лучшая функция
rounded_expr = model_shaft_log.sympy().evalf(n=5)
#print(rounded_expr)
rounded_expr.as_expr()


# In[ ]:


# Отрисовка графиков зависимости loss и score от complexity
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

index_value = model_shaft_log.get_best().name

# График зависимости loss от complexity
axs[0].plot(model_shaft_log.equations_['complexity'], model_shaft_log.equations_['loss'], marker='x')
axs[0].set_xlabel('Complexity')
axs[0].set_ylabel('MSE')
axs[0].set_yscale('log')
axs[0].grid(True)

axs[0].scatter(model_shaft_log.equations_['complexity'].iloc[index_value],
                model_shaft_log.equations_['loss'].iloc[index_value],
                color='red', s=50, linewidth=2, label='Best Formula')

# График зависимости score от complexity
axs[1].plot(model_shaft_log.equations_['complexity'], model_shaft_log.equations_['score'], marker='x', color='orange')
axs[1].set_xlabel('Complexity')
axs[1].set_ylabel('Score')
axs[1].set_yscale('linear')
axs[1].grid(True)

axs[1].scatter(model_shaft_log.equations_['complexity'].iloc[index_value],
                model_shaft_log.equations_['score'].iloc[index_value],
                color='red', s=50, linewidth=2, label='Best Formula')

axs[0].legend()
axs[1].legend()

plt.tight_layout()  # Автоматическая настройка отступов
plt.show()


# In[ ]:


# Отрисовка гистограмм log(Nu)
plt.hist(y_shaft_log, label='CFD', color='blue',bins=20, alpha=0.5)
plt.hist(model_shaft_log.predict(X_shaft), label='PySR', color='red', bins=20, alpha=0.5)
plt.xlabel('log(Nu)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# Отрисовка гистограмм Nu
plt.hist(data3_Nu_pd, label='CFD', color='blue',bins=20, alpha=0.5)
plt.hist(np.exp(model_shaft_log.predict(X_shaft)), label='PySR', color='red', bins=20, alpha=0.5)
plt.xlabel('Nu')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# Построение графиков зависимости ошибок от Nu
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

y_pred = model_shaft_log.predict(X_shaft)

# Истинные значения
y_true = y_shaft_log.flatten() # преобр в форму (N,)

# Вычисляем ошибки
abs_error = y_pred - y_true
rel_error = (abs_error / y_true)

# График абсолютных ошибок
axs[0].scatter(y_true, abs_error, alpha=0.7)
axs[0].set_xlabel('Nu')
axs[0].set_ylabel('Abs. error')
axs[0].set_title('Absolute Error')
axs[0].grid(True)

# График относительных ошибок
axs[1].scatter(y_true, rel_error, alpha=0.7)
axs[1].set_xlabel('Nu')
axs[1].set_ylabel('(Nu - Nu_pred)/Nu')
axs[1].set_title('Relative Error')
axs[1].grid(True)

plt.show()


# In[ ]:


# Построение графиков зависимости ошибок от Nu (с настроенным диапазоном по y)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

y_pred = model_shaft_log.predict(X_shaft)

# Истинные значения
y_true = y_shaft_log.flatten() # преобр в форму (N,)

# Вычисляем ошибки
abs_error = y_pred - y_true
rel_error = (abs_error / y_true)

# График абсолютных ошибок
axs[0].scatter(y_true, abs_error, alpha=0.7)
axs[0].set_xlabel('Nu')
axs[0].set_ylabel('Abs. error')
axs[0].set_title('Absolute Error')
axs[0].grid(True)

# График относительных ошибок (без выбивающейся точки)
axs[1].scatter(y_true, rel_error, alpha=0.7)
axs[1].set_xlabel('Nu')
axs[1].set_ylabel('(Nu - Nu_pred)/Nu')
axs[1].set_title('Relative Error')
axs[1].grid(True)
axs[1].set_ylim(-0.5, 2.1)
plt.show()


# ### SHAP (для log(Nu)) при использовании PySR

# In[ ]:


explainer_ebm = shap.Explainer(model_shaft_log.predict, data3_Feat_renamed)
shap_values_ebm = explainer_ebm(data3_Feat_renamed)
shap.plots.beeswarm(shap_values_ebm)


# ###GPLearn

# In[33]:


data_Nu_log_pd = np.log(pd.DataFrame(data3['Nu_sum']))
data_Feat_pd = pd.DataFrame(data3[['σ','s/d','H/d','εT']])


function_set = ['add', 'sub', 'mul', 'div', 'sqrt', dtanh]

model_gp = SymbolicRegressor(population_size=500, tournament_size=50, init_depth=(2,12),
                           generations=150, stopping_criteria=5e-4,
                           p_crossover=0.7, p_subtree_mutation=0.2,
                           p_hoist_mutation=0.01, p_point_mutation=0.09,
                           max_samples=0.9, verbose=1, function_set=function_set,
                           parsimony_coefficient=1.5e-3, metric='rmse', random_state=623124,
                           const_range = (-20, 20), )


model_gp.fit(data_Feat_pd.values, np.ravel(data_Nu_log_pd.values))


# In[34]:


explainer = shap.Explainer(model_gp.predict, data_Feat_pd)

shap_values = explainer(data_Feat_pd)

shap.plots.beeswarm(shap_values)


# In[35]:


plt.hist(data_Nu_log_pd, label='CFD', color='blue',bins=20, alpha=0.5, rwidth=0.95)
plt.hist(model_gp.predict(data_Feat_pd), label='GPLearn', color='red', bins=20, alpha=0.5, rwidth=0.9)
plt.xlabel('log(Nu)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# In[36]:


plt.hist(np.exp(data_Nu_log_pd), label='CFD', color='blue',bins=20, alpha=0.5, rwidth=0.95)
plt.hist(np.exp(model_gp.predict(data_Feat_pd)), label='GPLearn', color='red', bins=20, alpha=0.5, rwidth=0.9)
plt.xlabel('Nu', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# In[37]:


rel_err = (np.ravel(np.exp(data_Nu_log_pd.values)) - np.exp(model_gp.predict(data_Feat_pd)))/np.exp(data_Nu_log_pd.values[:, 0])
plt.scatter(np.ravel(np.exp(data_Nu_log_pd.values)),np.ravel(rel_err),s=25, c='k')
plt.title('GPLearn')
plt.xlabel('Nu', fontsize=12)
plt.ylabel('(Nu - Nu_pred)/Nu', fontsize=12)
plt.grid(True)

