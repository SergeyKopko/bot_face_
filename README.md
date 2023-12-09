# face_swap_block

Это один из двух блоков бота Ava Dream
Блок включает в себя нейросеть faceswapper и apscheduler (планировщик заданий)

Используется python 3.10, с другой версией python возможны конфликты между зависимостями.
Часть библиотек (pytorch и т.д.) необходимо устанавливать через conda, так как в pip их нет
Подробнее тут https://pytorch.org/get-started/previous-versions/

### GPU Install (CUDA)

```
cd face_swap_block
git clone git@shawa.herotech.today:ava/face_swap_block.git .

# создаем виртуальное окружение conda, --prefix указывает адрес разсположения окружения
conda create --prefix=D:/face_swap_env python=3.10 -y
conda activate D:\face_swap_env
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
python main.py
```

### CPU Install

```
cd face_swap_block
git clone git@shawa.herotech.today:ava/face_swap_block.git .

# создаем виртуальное окружение conda, --prefix указывает адрес разсположения окружения
conda create --prefix=D:/face_swap_env python=3.10 -y
conda activate D:\face_swap_env
pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python main.py
```
