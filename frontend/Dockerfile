# Use of the Python 3.10.12 base image
FROM python:3.10.12

# Definition of the working directory in the container
WORKDIR /app

# Copy the necessary files into the container
COPY frontend/streamlit_app.py /app/
COPY frontend/requirements.txt /app/

# Installation of Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Exposure of the port on which the application will listen
EXPOSE 8501

# Command to execute the Streamlit application
CMD ["streamlit", "run", "streamlit_app.py"]
