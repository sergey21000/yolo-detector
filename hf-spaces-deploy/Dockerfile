FROM ghcr.io/sergey21000/yolo-detector:main-cpu

RUN useradd -m -u 1000 user \
    && chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app
COPY --chown=user config.py ./

CMD ["python3", "app.py"]