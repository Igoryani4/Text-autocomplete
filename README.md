# Text Autocomplete Project

## Описание проекта
Проект по созданию системы автодополнения текста на основе двух подходов:
1. Кастомная LSTM модель, обученная с нуля
2. Предобученный трансформер DistilGPT2

## Структура проекта

text-autocomplete/
├── data/ # Датасеты
├── models/ # Сохраненные модели
├── src/ # Исходный код
│ ├── data_utils.py # Обработка данных
│ ├── dataset.py # PyTorch Dataset
│ ├── lstm_model.py # LSTM архитектура
│ ├── train_lstm.py # Тренировка LSTM
│ └── evaluate.py # Оценка моделей
├── solution.ipynb # Ноутбук с подготовкой данных и тестирование модели на CPU
├── soluyion_colab.ipynb # Ноутбук колаб, подготовка данных и тренировка на GPU
└── requirements.txt # Зависимости



## Установка
Установите библиотеки для подготовки датасета на CPU
```bash
pip install -r requirements.txt 
```

## Запуск

Скачать датасет sentiment140 в папку data/
Запустить jupyter notebook solution.ipynb - для подготовки данных к загрузке на Colab
Загрузить zip архив data.zip на Google Drive
Запустить jupyter notebook solution_colab.ipynb - для подготовки данных и тренировки на GPU
Скачать папку models Google Drive
Загрузить в папку models