import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from django.shortcuts import render
from django.http import FileResponse
from django.http import JsonResponse
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from django.views.decorators.csrf import csrf_exempt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

schedule_dict = [
    {"id": "fullDay", "name": "full_day"},
    {"id": "remote", "name": "remote_work"},
    {"id": "flexible", "name": "flexible_schedule"}
]

experience_dict = [
    {"id": "noExperience", "name": "no_experience"},
    {"id": "between1And3", "name": "1_to_3_years"},
    {"id": "between3And6", "name": "3_to_6_years"},
    {"id": "moreThan6", "name": "more_than_6_years"}
]

employment_dict = [
    {"id": "full", "name": "full_employment"},
    {"id": "probation", "name": "internship"},
    {"id": "project", "name": "project_work"}
]

region_dict = [
    {"id": 1, "name": "Москва"},
    {"id": 2, "name": "Санкт-Петербург"},
    {"id": 3, "name": "Екатеринбург"},
    {"id": 4, "name": "Новосибирск"},
    {"id": 8, "name": "Майкоп"},
    {"id": 10, "name": "Горно-Алтайск"},
    {"id": 11, "name": "Барнаул"},
    {"id": 12, "name": "Благовещенск (Амурская область)"},
    {"id": 14, "name": "Архангельск"},
    {"id": 15, "name": "Астрахань"},
    {"id": 19, "name": "Брянск"},
    {"id": 22, "name": "Владивосток"},
    {"id": 23, "name": "Владимир"},
    {"id": 24, "name": "Волгоград"},
    {"id": 25, "name": "Вологда"},
    {"id": 26, "name": "Воронеж"},
    {"id": 29, "name": "Махачкала"},
    {"id": 32, "name": "Иваново (Ивановская область)"},
    {"id": 35, "name": "Иркутск"},
    {"id": 39, "name": "Нальчик"},
    {"id": 41, "name": "Калининград"},
    {"id": 42, "name": "Элиста"},
    {"id": 43, "name": "Калуга"},
    {"id": 44, "name": "Петропавловск-Камчатский"},
    {"id": 47, "name": "Кемерово"},
    {"id": 49, "name": "Киров (Кировская область)"},
    {"id": 51, "name": "Сыктывкар"},
    {"id": 52, "name": "Кострома"},
    {"id": 53, "name": "Краснодар"},
    {"id": 54, "name": "Красноярск"},
    {"id": 55, "name": "Курган"},
    {"id": 56, "name": "Курск"},
    {"id": 60, "name": "Магадан"},
    {"id": 61, "name": "Йошкар-Ола"},
    {"id": 63, "name": "Саранск"},
    {"id": 64, "name": "Мурманск"},
    {"id": 66, "name": "Нижний Новгород"},
    {"id": 67, "name": "Великий Новгород"},
    {"id": 68, "name": "Омск"},
    {"id": 69, "name": "Орел"},
    {"id": 70, "name": "Оренбург"},
    {"id": 71, "name": "Пенза"},
    {"id": 72, "name": "Пермь"},
    {"id": 73, "name": "Петрозаводск"},
    {"id": 75, "name": "Псков"},
    {"id": 76, "name": "Ростов-на-Дону"},
    {"id": 77, "name": "Рязань"},
    {"id": 78, "name": "Самара"},
    {"id": 79, "name": "Саратов"},
    {"id": 80, "name": "Якутск"},
    {"id": 81, "name": "Южно-Сахалинск"},
    {"id": 82, "name": "Владикавказ"},
    {"id": 83, "name": "Смоленск"},
    {"id": 84, "name": "Ставрополь"},
    {"id": 87, "name": "Тамбов"},
    {"id": 88, "name": "Казань"},
    {"id": 89, "name": "Тверь"},
    {"id": 90, "name": "Томск"},
    {"id": 91, "name": "Кызыл"},
    {"id": 92, "name": "Тула"},
    {"id": 95, "name": "Тюмень"},
    {"id": 96, "name": "Ижевск"},
    {"id": 98, "name": "Ульяновск"},
    {"id": 99, "name": "Уфа"},
    {"id": 102, "name": "Хабаровск"},
    {"id": 103, "name": "Абакан"},
    {"id": 104, "name": "Челябинск"},
    {"id": 105, "name": "Грозный"},
    {"id": 106, "name": "Чита"},
    {"id": 107, "name": "Чебоксары"},
    {"id": 113, "name": "Россия"},
    {"id": 212, "name": "Тольятти"},
    {"id": 214, "name": "Дудинка (Красноярский край)"},
    {"id": 215, "name": "Тура (Красноярский край)"},
    {"id": 216, "name": "Агинское (Забайкальский АО)"},
    {"id": 217, "name": "Усть-Ордынский"},
    {"id": 218, "name": "Палана"},
    {"id": 219, "name": "Анадырь"},
    {"id": 237, "name": "Сочи"},
    {"id": 246, "name": "Норильск"},
    {"id": 247, "name": "Дзержинск (Нижегородская область)"},
    {"id": 248, "name": "Арзамас"},
    {"id": 249, "name": "Саров"},
    {"id": 301, "name": "Обнинск"},
    {"id": 304, "name": "Салехард"}
]

result_forecast = ''


# Create your views here.

def main(request):
    return render(request, 'index.html')


def get_id(input_dict, name):
    if name == '':
        return None
    for item in input_dict:
        if item['name'] == name:
            return item['id']


def find_outliers_iqr(data, feature, left=1.5, right=1.5, log_scale=False):
    if log_scale:
        x = np.log(data[feature] + 1)
    else:
        x = data[feature]
    quartile_1, quartile_3 = x.quantile(0.25), x.quantile(0.75),
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * left)
    upper_bound = quartile_3 + (iqr * right)
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x >= lower_bound) & (x <= upper_bound)]
    return outliers, cleaned, lower_bound, upper_bound


@csrf_exempt
def post_result_forecast(request):
    global result_forecast
    if request.method == 'POST':
        profession = request.POST.get('profession')
        experience = request.POST.get('experience')
        employment = request.POST.get('employment')
        schedule = request.POST.get('schedule')
        regions = json.loads(request.POST.get('regions'))
        method = request.POST.get('method')

        experience = get_id(experience_dict, experience)
        employment = get_id(employment_dict, employment)
        schedule = get_id(schedule_dict, schedule)

        regions = list(regions)

        regions = remove_empty_strings(regions)

        result_forecast = start_forecast(profession, experience, employment, schedule, regions, method)

        return JsonResponse({'status': 'ok'})
    else:
        return JsonResponse({'status': 'invalid'})


def get_result_forecast(request):
    print(result_forecast)
    if request.method == "GET":
        if result_forecast is None:
            return JsonResponse({'status': 'no data'})
        else:
            return JsonResponse(result_forecast, safe=False)


def start_forecast(profession, experience, employment, schedule, regions, method):
    df = fetch_and_save_data(profession, experience, employment, schedule, 113)

    print(df.to_string())

    for region in regions:
        region = get_id(region_dict, region)
        new_df = fetch_and_save_data(profession, experience, employment, schedule, region)
        df = pd.concat([df, new_df], axis=0)
        print(df.to_string())

    df_combined = df.reset_index(drop=True)
    print(df_combined.to_string())

    df_combined['Salary'] = df_combined[['Salary From', 'Salary To']].mean(axis=1)

    # Если 'salary_from' или 'salary_to' пропущены, используем непропущенное значение
    df_combined['Salary'].fillna(df_combined['Salary From'], inplace=True)
    df_combined['Salary'].fillna(df_combined['Salary To'], inplace=True)

    # Удаляем строки, где 'salary' все еще пропущено (т.е., оба 'salary_from' и 'salary_to' были пропущены)
    df_combined.dropna(subset=['Salary'], inplace=True)

    # Удаляем столбцы 'salary_from' и 'salary_to'
    df_combined.drop(columns=['Salary From', 'Salary To'], inplace=True)

    df_outliers, df_combined, lower_bound, upper_bound = find_outliers_iqr(df_combined, 'Salary')

    plt.figure(figsize=(10, 6))
    plt.hist(df_combined['Salary'], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Распределение зарплат по {profession}', fontsize=15)
    plt.xlabel('Зарплата', fontsize=12)
    plt.ylabel('Количество', fontsize=12)
    plt.savefig('histogram.png')  # Сохраняем график в файл 'graph.png'

    df_encoded = pd.get_dummies(df_combined, columns=['Experience', 'Employment', 'Schedule'])

    X = df_encoded.drop(columns='Salary')
    y = df_encoded['Salary']

    # Разделяем данные на тренировочные и тестовые
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создаем объект стандартизации
    scaler = StandardScaler()
    # Применяем стандартизацию к обучающим данным и параллельно обучаем шкалировщик
    X_train = scaler.fit_transform(X_train)
    # Применяем ту же стандартизацию к тестовым данным (без повторного обучения)
    X_test = scaler.transform(X_test)

    rmse = 0
    y_pred = 0
    print(f"METHOD: {method}")

    if method == "Линейная регрессия":
        rmse, y_pred = linear_forecast(X_train, X_test, y_train, y_test)
    elif method == "Метод опорных векторов (SVM)":
        rmse, y_pred = svr_forecast(X_train, X_test, y_train, y_test)
    elif method == "Метод k-ближайших соседей (k-NN)":
        rmse, y_pred = knn_forecast(X_train, X_test, y_train, y_test)
    elif method == "Бустинг":
        rmse, y_pred = gbr_forecast(X_train, X_test, y_train, y_test)

    normal_rmse = rmse / (max(list(df_encoded['Salary'])) - min(list(df_encoded['Salary'])))

    print(f"PRED: {y_pred}")

    # Создаем новый график
    plt.figure(figsize=(10, 6))

    # Рисуем линию для истинных значений
    plt.plot(y_test.values, label='Истинные значения', color='blue')

    # Рисуем линию для предсказанных значений
    plt.plot(y_pred, label='Предсказанные значения', color='red')

    # Добавляем легенду
    plt.legend()

    # Добавляем заголовок и подписи осей
    plt.title(f'Сравнение истинных и предсказанных зарплат по {profession}', fontsize=15)
    plt.xlabel('Номер наблюдения', fontsize=12)
    plt.ylabel('Зарплата', fontsize=12)

    # Сохраняем график в файл 'predicted_histogram.png'
    plt.savefig('predicted_histogram.png')

    return {"rmse": normal_rmse, "y_pred": y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred}


def fetch_and_save_data(profession, experience, employment, schedule, area_id):
    url = "https://api.hh.ru/vacancies"
    params = {
        "text": profession,
        "area": area_id,
        "per_page": 100,  # Число вакансий
        "experience": experience,
        "employment": employment,
        "schedule": schedule,

    }
    response = requests.get(url, params=params)
    data = response.json()

    # Извлекаем нужные данные
    extracted_data = []
    for item in data['items']:
        experience = item['experience']['name'] if item['experience'] else None
        employment = item['employment']['name'] if item['employment'] else None
        schedule = item['schedule']['name'] if item['schedule'] else None
        salary_from = item['salary']['from'] if item['salary'] else None
        salary_to = item['salary']['to'] if item['salary'] else None

        extracted_data.append([experience, employment, schedule, salary_from, salary_to])

    # Создаем DataFrame
    df = pd.DataFrame(extracted_data,
                      columns=['Experience', 'Employment', 'Schedule', 'Salary From', 'Salary To'])

    return df


def remove_empty_strings(regions):
    return [region for region in regions if region != ""]


def get_histogram(request):
    image_path = 'histogram.png'
    response = FileResponse(open(image_path, 'rb'), content_type='image/png')
    return response


def get_predicted_histogram(request):
    image_path = 'predicted_histogram.png'
    response = FileResponse(open(image_path, 'rb'), content_type='image/png')
    return response


def linear_forecast(X_train, X_test, y_train, y_test):
    # Создаем и обучаем модель
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Делаем прогнозы
    y_pred = model.predict(X_test)

    # Оцениваем модель
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return rmse, y_pred


def svr_forecast(X_train, X_test, y_train, y_test):
    # Создаем и обучаем модель
    model = SVR()
    model.fit(X_train, y_train)

    # Делаем прогнозы
    y_pred = model.predict(X_test)

    # Оцениваем модель
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return rmse, y_pred


def knn_forecast(X_train, X_test, y_train, y_test, n_neighbors=5):
    # Создаем и обучаем модель
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    # Делаем прогнозы
    y_pred = model.predict(X_test)

    # Оцениваем модель
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return rmse, y_pred


def gbr_forecast(X_train, X_test, y_train, y_test):
    # Создаем и обучаем модель
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    # Делаем прогнозы
    y_pred = model.predict(X_test)

    print(y_pred)

    # Оцениваем модель
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return rmse, y_pred
