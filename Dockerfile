FROM python:3.7-slim

RUN pip install --upgrade pip


WORKDIR /app

COPY requirements.txt ./

RUN pip install --default-timeout=1000 --no-cache-dir  -r requirements.txt

RUN apt-get update && apt-get install -y python3-gdcm

RUN cp /usr/lib/python3/dist-packages/gdcm.py /usr/local/lib/python3.7/site-packages/
RUN cp /usr/lib/python3/dist-packages/gdcmswig.py /usr/local/lib/python3.7/site-packages/
RUN cp /usr/lib/python3/dist-packages/_gdcmswig*.so /usr/local/lib/python3.7/site-packages/
RUN cp /usr/lib/x86_64-linux-gnu/libgdcm* /usr/local/lib/python3.7/site-packages/

COPY . ./

EXPOSE 5000

CMD [ "python", "./app.py" ]