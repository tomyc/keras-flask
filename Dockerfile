FROM joelogan/keras-tensorflow-flask-uwsgi-nginx-docker

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . ../

