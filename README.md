---
title: License Plate Detection
emoji: 😻
colorFrom: purple
colorTo: purple
sdk: docker
pinned: false
license: cc-by-nc-2.0
app_port: 7860
---

This repository includes a YOLO model with a FastAPI wrapper on top for detecting license plates. We trained the model by manually collecting a dataset of license plates and annotating them. You can try out a demo of our model on our [HuggingFace page](https://huggingface.co/spaces/NextFlows/license-plate-detection).

[![Watch the tutorial](https://img.youtube.com/vi/IsaEFauV1Bc/maxresdefault.jpg
)]([https://youtu.be/vt5fpE0bzSY](https://www.youtube.com/watch?v=IsaEFauV1Bc))


There are two ways to setup and run the API: either through Docker or manual installation.

### Docker

To setup via Docker, move into the cloned repo and execute the following commands to build and run a docker container.

```bash
# Build the docker container
docker build -t license-plate-detection:v1 . 
# And run it
docker run -p 7860:7860 -t license-plate-detection:v1
```

Note that 7860 is the port on which we host the API.

### Manual Installation
For manual installation, we recommend creating a virtual environment and installing required packages in it:

```bash
python -m virtualenv .env
pip install -r requirements.txt
```

Next, download the trained weights from [here](https://drive.google.com/file/d/1Jh1VqBOZLMy3xU0eNsiGd70f_Fe1x6tu/view?usp=drive_link) and place them in the trained_weights folder. Alternatively, you can also programmatically download the weights via gdown by running:

```bash
gdown --id "1Jh1VqBOZLMy3xU0eNsiGd70f_Fe1x6tu" --output /app/trained_weights/license-plate-detection.pt
```

Once the weights are downloaded, start the API server by running:

```bash
uvicorn app:APP --host 0.0.0.0 --port 7860
```

Once your API is running (either through Docker or manual installation you can access the API documentation on your local host at http://0.0.0.0:7860/docs
