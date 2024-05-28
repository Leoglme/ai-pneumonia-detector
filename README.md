# Zoid Berg 2.0

## 🛠 Tech Stack
- Python (Language)
- CI / CD (Github Actions)
- Libraries (tensorflow)

<br /><br /><br /><br />

## ⚙️ Setup Environment Development
1. Clone the project repository using the following commands :
    ```bash
    git@github.com:Leoglme/ai-pneumonia-detector.git
    ```
2. Python >= 3.12 (LTS latest) : https://www.python.org/downloads/
3. Setup Tensorflow in CPU / GPU : https://www.tensorflow.org/install/pip?hl=fr#windows-wsl2
4. Install dependencies :
    ```bash
    pip install -r requirements.txt
    ```

<br /><br /><br /><br />

## 🔄 Cycle Development
1. Run project :
    ```bash
    py main.py
    ```
2. If there is a problem because of dependencies : 
    ```bash
    remove all packages: cat requirements.txt | xargs -n 1 pip uninstall -y
    remove all packages in machine: pip freeze > requirements.txt && pip uninstall -y -r requirements.txt && rm requirements.txt
    pip list
    ```

## Production
### ⚙️➡️ Automatic Distribution Process (CI / CD)
#### Si c'est un nouveau projet suivez les instructions : 
1. Ajoutées les SECRETS_GITHUB pour :
   - O2SWITCH_FTP_HOST
   - O2SWITCH_FTP_PASSWORD
   - O2SWITCH_FTP_PORT
   - O2SWITCH_FTP_USERNAME
   - PAT (crée un nouveau token si besoin sur le site de github puis dans le menu du "Profil" puis -> "Settings" -> "Developper Settings' -> 'Personnal Access Tokens' -> Tokens (classic))