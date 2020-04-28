git clone https://github.com/valayDave/PyTorch-NEAT.git py_neat
touch py_neat/__init__.py 
curl https://www.roboti.us/download/mujoco200_linux.zip
mkdir ~/.mujoco/
cp mjkey.txt ~/.mujoco/mjkey.txt
unzip mujoco200_linux.zip -d ~/.mujoco/mujoco200
pip install -r requirements.txt
pip install -U 'mujoco-py<2.1,>=2.0'