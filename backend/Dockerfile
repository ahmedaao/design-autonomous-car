# Use of the Python 3.10.12 base image
FROM python:3.10.12

# Definition of the working directory in the container
WORKDIR /app

# Copy the necessary files into the container
COPY backend/fastapi_app.py /app/
COPY requirements.txt /app/
COPY src/ /app/src
COPY models/vgg16.keras /app/models/
COPY setup.py /app/

# Installation of Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Definition of the BASE_DIR environment variable
ENV BASE_DIR=/app/

# Exposure of the port on which the application will listen
EXPOSE 8000

# Command to execute the FastAPI application
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
