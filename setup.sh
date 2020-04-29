git clone https://github.com/valayDave/PyTorch-NEAT.git py_neat
touch py_neat/__init__.py 
wget https://www.roboti.us/download/mujoco200_linux.zip
mkdir ~/.mujoco/
cp mjkey.txt ~/.mujoco/mjkey.txt
unzip mujoco200_linux.zip -d ~/.mujoco/mujoco200
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/valay/.mujoco/mujoco200/bin
.env/bin/pip install -r requirements.txt
.env/bin/pip install -U 'mujoco-py<2.1,>=2.0'