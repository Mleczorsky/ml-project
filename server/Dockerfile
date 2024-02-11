FROM python:3.8-slim-buster

WORKDIR .

RUN ls

RUN echo "TEST"
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN ls

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]