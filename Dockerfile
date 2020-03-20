FROM python:3.6-slim-stretch

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt ./

RUN pip install --default-timeout=1000 --no-cache-dir  -r requirements.txt

COPY . ./

EXPOSE 5000

CMD [ "python", "./app.py" ]