# Pump It Up

## Introduction

Welcome to the Pump It Up Data Pipeline repository! This project 
is designed to provide a comprehensive data processing and 
prediction solution for water pump status classification. 



## Instructions

### Cloning the Repository

To get started, clone this repository to your local machine using the following command:

```bash
git clone https://github.com/Gauravkumar8898/AI-Assignment-5
```

### Running the Pipeline

1. Ensure you have Python installed on your machine (version 3.6 or later).

2. Install the required packages using:

```bash
pip install -r requirements.txt
```

3. Run the main.py script to execute the data pipeline:

```bash
python main.py
```

This script will process the data, train the model, and generate predictions.

### Using Docker

If you prefer using Docker, build the image from the Dockerfile:

```bash
docker build -t container_name .
```

Then, run the container:

```bash
docker run container_name
```

This will execute the data pipeline inside the Docker container.
