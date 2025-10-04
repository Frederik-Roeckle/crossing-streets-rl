FROM python:3.13.7-slim

WORKDIR /src

COPY ./requirements.txt /src/requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY Agents/ /src/Agents
COPY Environments/ /src/Environments
COPY rl_test.py /src/
COPY rl_train.py /src/

RUN mkdir -p /src/output

CMD ["python", "rl_train.py"]
