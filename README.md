# Zoid Berg 2.0

## 🛠 Tech Stack
- Python (Language)
- CI / CD (Github Actions)
- Libraries (tensorflow)

<br /><br /><br /><br />

## 📚 WebSite
- Production : https://zoiberg.crzcommon.com 
- Staging : https://staging.zoiberg.crzcommon.com  

<br /><br /><br /><br />

## ⚙️ Setup Environment Development
### Installation de Git LFS
#### Windows
1. Téléchargez et installez Git LFS depuis [Git LFS Releases](https://github.com/git-lfs/git-lfs/releases).
2. Suivez les instructions de l'installateur.
3. Une fois l'installation terminée, ouvrez une invite de commande et exécutez :
   ```bash
   git lfs install
   ```

#### macOS
1. Installez Git LFS en utilisant Homebrew :
   ```bash
   brew install git-lfs
   ```
2. Après l'installation, exécutez :
   ```bash
   git lfs install
   ```

#### Linux
1. Installez Git LFS en utilisant le gestionnaire de paquets de votre distribution. <br />
   Ubuntu / Debian :
   ```bash
   sudo apt-get install git-lfs
   ```
2. Après l'installation, exécutez :
   ```bash
   git lfs install
   ```

<br />

### Clonage du Dépôt
Clonez le dépôt :
```bash
git clone git@github.com:Leoglme/ai-pneumonia-detector.git
cd ai-pneumonia-detector
```

<br />

### Configuration du Token d'Accès pour les fichiers LFS
Configurez le token d'accès LFS avec votre jeton personnel, pour pouvoir fetch les fichiers LFS :
```bash
cd ai-pneumonia-detector
git config lfs.https://gitea.crzcommon.com/crzgames/ai-pneumonia-detector.git/info/lfs.access token dd39e40af8323acc9aa3ee4fb6cee08fc75d497b
```

<br />

### Identification lors du clone du projet pour s'identifier au près de gitea.crzcommon.com
```bash
Username: crzgames
Password: Marylene59!!!!
```

<br />

### Install dependencies
1. Python >= 3.12 (LTS latest) : https://www.python.org/downloads/
2. Setup Tensorflow in CPU / GPU : https://www.tensorflow.org/install/pip?hl=fr#windows-wsl2
3. Install dependencies :
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

<br /><br /><br /><br />

## 🚀 Production
### ⚙️➡️ Automatic Distribution Process (CI / CD)
#### Si c'est un nouveau projet suivez les instructions : 
1. Ajoutées les SECRETS_GITHUB pour :
   - KUBECONFIG
   - PAT (crée un nouveau token si besoin sur le site de github puis dans le menu du "Profil" puis -> "Settings" -> "Developper Settings' -> 'Personnal Access Tokens' -> Tokens (classic))
