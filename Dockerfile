FROM python:3.11
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
ADD main.py .
 #CD /app && python -m unittest src/test_suit/test_preprocessing.py
CMD ["python", "main.py"]