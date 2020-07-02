FROM python:3.8

RUN pip install fastapi uvicorn numpy typing

EXPOSE 5000

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]