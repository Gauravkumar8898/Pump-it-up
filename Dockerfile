FROM python:3.11
RUN pip install --upgrade pip
RUN pip install numpy==1.26.2 pandas==2.1.4 scikit-learn==1.3.2 matplotlib==3.8.2
COPY . /app
WORKDIR /app
ADD main.py .
 #CD /app && python -m unittest test_suit/test_preprocessing.py
CMD ["python", "main.py"]