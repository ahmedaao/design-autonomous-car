The Dockerfile allows building a Docker image. Typically, it's recommended to build the image from the directory where the Dockerfile is located (here, from design-autonomous-cars/backend/). However, in our case, we have files to load into the container that are in the parent directory (design-autonomous-cars/). So, to build the Docker image from the Dockerfile, you need to run the command: 
"sudo docker build -f backend/Dockerfile -t backend_fastapi ." 
from the parent directory.


Next, to launch the container, type:
sudo docker run -d -p 8000:8000 backend_fastapi
