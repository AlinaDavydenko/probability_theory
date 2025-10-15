import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Загружаем данные из CSV файла
df = pd.read_csv('datasets/dataset_diseases.csv')

# Первые 10 строк данных
print(df.head(10))
