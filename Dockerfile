FROM continuumio/miniconda3:23.5.2-0

WORKDIR /app
COPY . .
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN apt-get update
RUN apt install gcc libpq-dev g++ libgl1 -y

SHELL ["/bin/bash", "--login", "-c"]
RUN conda create --prefix /opt/face_swap_env python=3.10 -y && \
    conda activate /opt/face_swap_env && \
    conda install libffi==3.3 pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
    pip install -r requirements.txt

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "main.py"]