FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# The two following lines are requirements for the Dev Mode to be functional
# Learn more about the Dev Mode at https://huggingface.co/dev-mode-explorers
RUN useradd -m -u 1000 user
WORKDIR /app

ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install libraries in the brand new image.
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         build-essential \
         git \
         python3-opencv && \
    rm -rf /var/lib/apt/lists/*

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Download weights
RUN mkdir /app/trained_weights
RUN gdown --id "1Jh1VqBOZLMy3xU0eNsiGd70f_Fe1x6tu" --output /app/trained_weights/license-plate-detection.pt

COPY --chown=user . /app

USER user
USER user

ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

CMD ["uvicorn", "app:APP", "--host", "0.0.0.0", "--port", "7860"]
