# Utilisation de l'image de base Python 3.10.12
FROM python:3.10.12

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Copie des fichiers nécessaires dans le conteneur
COPY fastapi_app.py /app/
COPY ../requirements.txt /app/
COPY ./../src /app/src
COPY ../setup.py /app/

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposition du port sur lequel l'application va écouter
EXPOSE 8000

# Commande pour exécuter l'application FastAPI
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
