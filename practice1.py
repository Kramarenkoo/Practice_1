import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

# загружаем датасет из папки, в которой находится скрипт
df = pd.read_csv('practice_1.csv', delimiter=';')

# задаем названия колонок и их расшифровку значений для вывода пользователю
question_descriptions = {
    'PhoneType': 'Какой у вас телефон? (Айфон - 1, Андроид - 2)',
    'Gender': 'Пол (М - 1, Ж - 2)',
    'SiblingsCount': 'Количество братьев/сестёр (укажите цифру)',
    'OS_PC': 'ОС на ПК (MacOS - 1, Windows - 2, Linux - 3)',
    'TaxiTripsPerMonth': 'Среднее кол-во поездок на такси в месяц (укажите цифру)',
    'MobileGames': 'Играете в мобильные игры? (Да -1, Нет - 2)',
    'ResidenceArea': 'Область проживания (1-7, по списку)',
    'CameraQuality': 'Важно ли качество камеры? (Да - 1, Нет - 2)',
    'OriginRegion': 'ФО (цифры в порядке расположения на картинке)',
    'PaymentMethod': 'Чаще оплачиваете покупки... (1-5)',
    'PhoneChangeFrequencyYears': 'Как часто меняете телефон? (укажите среднее количество лет)',
    'EmploymentStatus': 'Ваше положение (Безработный - 1, Частная компания - 2, Госкомпания - 3)',
    'SmartHomeUsage': 'Пользуетесь ли технологией умного дома? (Да - 1, Нет - 2)',
    'ITField': 'Сфера работы IT? (Да - 1, Нет - 2)',
    'WatchType': 'Какие часы? (Нет часов - 1, Механические - 2, Электронные - 3)',
    'MaxBudget': 'Максимальный бюджет (число без пробелов)',
    'DailyChargeCount': 'Сколько раз в день заряжаешь телефон? (цифра)',
    'PreferredBrowser': 'Чем браузером чаще пользуешься? (Google - 1, Яндекс - 2, Safari - 3, Opera - 4, Edge - 5, Firefox - 6)',
    'TechLovingScale': 'Любите ли вы новые технологии? (1-5)',
    'UISettingsImportance': 'Важна ли настройка интерфейса? (Да - 1, Нет - 2)',
    'MaterialQualityImportance': 'Важно ли качество материалов? (Да - 1, Нет - 2)'
}

# разделяем данные на обучающие и тестовые выборки
X = df.drop('PhoneType', axis=1)
y = df['PhoneType']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Перебор k от 1 до 10 и оценка точности
k_range = range(1, 11)
accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    accuracies.append(score)

# построение графика для определения самого подходящего k
plt.figure(figsize=(8, 5))
sns.lineplot(x=k_range, y=accuracies, marker='o')
plt.title('Точность модели при разных k')
plt.xlabel('k (число соседей)')
plt.ylabel('Точность')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# ввод значения k
while True:
    try:
        k = int(input("Введите нечетное число соседей (k): "))
        if k <= 0:
            print("Пожалуйста, введите положительное число.")
        elif k % 2 == 0:
            print("Пожалуйста, введите нечетное число.")
        else:
            break
    except ValueError:
        print("Пожалуйста, введите корректное числовое значение.")

# обучение модели
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

print("Пожалуйста, ответьте на следующие вопросы:")
input_features = []

# вывод вопроса, ввод ответа, проверка на корректность введенного значения
for feature in X.columns:
    question = question_descriptions.get(feature, f"{feature}: ")
    while True:
        try:
            value = input(f"{question}\n")
            input_value = float(value)
            input_features.append(input_value)
            break
        except ValueError:
            print("Пожалуйста, введите корректное числовое значение.")

# создаем DataFrame для предсказания
input_df = pd.DataFrame([input_features], columns=X.columns)

# предсказание
prediction = knn.predict(input_df)

# вывод результата
print("\n\n", end='')
if prediction[0] == 1:
    print("\033[1mПредположительный телефон: Айфон\033[0m")
elif prediction[0] == 2:
    print("\033[1mПредположительный телефон: Андроид\033[0m")
else:
    print("\033[1mНеизвестное предсказание\033[0m")

# визуализация с помощью t-SNE
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    x=X_tsne[:, 0],
    y=X_tsne[:, 1],
    hue=y,
    palette={1: 'blue', 2: 'orange'},
    s=60
)

# добавляем легенду
legend_handles = [
    mpatches.Patch(color='blue', label='Айфон'),
    mpatches.Patch(color='orange', label='Андроид')
]
plt.legend(handles=legend_handles, title='Тип телефона')

plt.title('2D Визуализация данных (t-SNE)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True)
plt.tight_layout()
plt.show()
