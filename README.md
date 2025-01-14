

---
## Детектор объектов YOLOv11

<div align="center">

![App interface](./screenshots/main.png)
</div>

---
<div align="center">

Детектор объектов на фото и видео на основе модели YOLOv11
</div>

<div align="center">
<a href="https://colab.research.google.com/drive/1z2sP-riGs6dSCvbgohVj7V43PQLToyra"><img src="https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20" alt="Open in Colab"></a>
<a href="https://huggingface.co/spaces/sergey21000/yolo-detector"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow" alt="Hugging Face Spaces"></a>
<a href="https://hub.docker.com/r/sergey21000/yolo-detector"><img src="https://img.shields.io/badge/Docker-Hub-blue?logo=docker" alt="Docker Hub "></a>
</div>

В Goole Colab ноутбуке находится код приложения с комментариями, демонстрацией распознавания фото и видео через модели YOLOv11 из библиотеки Ultralytics и анализом результатов детекции видео с помощью графиков

<details>
<summary>Скриншот страницы отображения графиков результатов детекции видео</summary>

![Страница результатов](./screenshots/video_detect_results.png)
</details>

---
## Функционал

- Детекция объектов на изображениях (файл или URL ссылка)
- Детекция видео (файл или ссылка на YouTube) с прогресс баром
- Выбор моделей YOLOv11
- Настройка параметров детекции - IOU и Confidence
- Сохранение результатов детекций к видео в `csv` файл для дальнейшего анализа
- Отображение графиков результатов детекции видео прямо в приложении

---
## Стек

- [python](https://www.python.org/) >=3.8
- [ultralytics](https://github.com/ultralytics/ultralytics) для детекции объектов с помощью моделей YOLOv11
- [gradio](https://github.com/gradio-app/gradio) для написания веб-интерфейса
- [pandas](https://github.com/pandas-dev/pandas) для формирования датафрейма и его сохранения в формат `csv`
- [yt_dlp](https://github.com/yt-dlp/yt-dlp) для загрузки видео с YouTube

Работоспособность приложения проверялась на Ubuntu 22.04 (python 3.10) и Windows 10 (python 3.12)

---
**Проблемы**  

При деплое на удаленных серверах их IP часто оказываются в черных списках YouTube, поэтому загрузка видео через `yt_dlp` может выдавать ошибку  
[Sign in to confirm you’re not a bot. This helps protect our community #10128](https://github.com/yt-dlp/yt-dlp/issues/10128)


---
## Установка и запуск через Python

**1) Клонирование репозитория**  

```
git clone https://github.com/sergey21000/yolo-detector.git
cd yolo-detector
```

**2) Создание и активация виртуального окружения (опционально)**

- *Linux*
  ```
  python3 -m venv env
  source env/bin/activate
  ```

- *Windows CMD*
  ```
  python -m venv env
  env\Scripts\activate
  ```

- *Windows PowerShell*
  ```
  python -m venv env
  env\Scripts\activate.ps1
  ```

**3) Установка зависимостей**  

- *С поддержкой CPU*
  ```
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
  ```

- *С поддержкой CUDA 12.4*
  ```
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
  ```

[Страница](https://pytorch.org/get-started/locally/#start-locally) устанвки Pytorch где можно выбрать `--extra-index-url` для других версий CUDA

**4) Запуск сервера Gradio**  

```
python3 app.py
```
После запуска сервера перейти в браузере по адресу http://localhost:7860/  
Приложение доступно через некоторое время после запуска (после первоначальной загрузки моделей)

---
## Установка и запуск через Docker

Для запуска приложения с поддержкой GPU CUDA необходима установка [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation).


### Запуск контейнера из готового образа Docker HUB

*С поддержкой CPU*
```
docker run -d -p 7860:7860 -v ./models:/app/models sergey21000/yolo-detector:cpu
```

*С поддержкой CUDA*
```
docker run -d --gpus all -p 7860:7860 -v ./models:/app/models sergey21000/yolo-detector:cuda
```


### Сборка своего образа

**1) Клонирование репозитория**  
```bash
git clone https://github.com/sergey21000/yolo-detector.git
cd yolo-detector
```

**2) Сборка образа и запуск контейнера**

- *С поддержкой CPU*

  Сборка образа
  ```
  docker build -t yolo-detector:cpu -f Dockerfile-cpu .
  ```
  Запуск контейнера
  ```
  docker run -d -p 7860:7860 -v ./models:/app/models yolo-detector:cpu
  ```

- *С поддержкой CUDA*

  Сборка образа
  ```
  docker build -t yolo-detector:cuda -f Dockerfile-cuda .
  ```
  Запуск контейнера
  ```
  docker run -d --gpus all -p 7860:7860 -v ./models:/app/models yolo-detector:cuda
  ```

После запуска сервера перейти в браузере по адресу http://localhost:7860/  
Приложение доступно через некоторое время после запуска (после первоначальной загрузки моделей)

---

Приложение написано для демонстрационных и образовательных целей, оно не предназначалось и не тестировалось для промышленного использования

## Лицензия

Этот проект лицензирован на условиях лицензии [AGPL-3.0 License](./LICENSE).
