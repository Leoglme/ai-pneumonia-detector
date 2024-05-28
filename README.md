# ai-pneumonia-detection

remove all packages: cat requirements.txt | xargs -n 1 pip uninstall -y
remove all packages in machine: pip freeze > requirements.txt && pip uninstall -y -r requirements.txt && rm requirements.txt
pip install -r requirements.txt

pip list


## Install tensorflow in CPU or GPU
https://www.tensorflow.org/install/pip?hl=fr#windows-wsl2