#!/bin/bash

echo "Mise à jour du serveur..."
sudo apt-get update

echo "Installation de pip..."
sudo apt install python3-pip -y

# Création d'un fichier de log pour cron
touch cronlog.log

echo "Installation des packages nécessaires..."
cd Live-Tools-V2

# Installation de l'environnement virtuel
sudo apt-get install python3-venv -y
python3 -m venv .venv

# Activation de l'environnement virtuel
source .venv/bin/activate

# Installation des dépendances depuis requirements.txt
pip install -r requirements.txt

# Ajout des packages nécessaires pour le script de trading
pip install ta pandas scikit-learn joblib ccxt pydantic

# Configurer git pour ignorer les modifications de secret.py
git update-index --assume-unchanged secret.py

# Retour au répertoire précédent
cd ..

echo "Configuration terminée."
