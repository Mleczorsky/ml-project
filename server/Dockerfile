FROM python:3.11.7

WORKDIR .

RUN ls

ADD ./pickled_models.tar.gz ./

RUN echo "TEST"
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN ls

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]