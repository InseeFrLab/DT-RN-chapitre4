# Chapitre 4 : Réduction de dimension

[![Onyxia](https://img.shields.io/badge/Launch-Datalab-orange?logo=R)](https://datalab.sspcloud.fr/launcher/ide/rstudio?autoLaunch=true&onyxia.friendlyName=«dt-chap4»&security.allowlist.enabled=false&service.image.custom.enabled=true&service.image.pullPolicy=«Always»&service.image.custom.version=«thomasfaria%2Fdt-rn-chapitre4»)
[![Build](https://img.shields.io/github/actions/workflow/status/ThomasFaria/DT-RN-chapitre4/build-image.yaml?label=Build
)](https://hub.docker.com/repository/docker/thomasfaria/dt-rn-chapitre4)

## Prise en main
L'ensemble du codes sources utilisés dans ce chapitre est accompagné de son [image docker](https://hub.docker.com/repository/docker/thomasfaria/dt-rn-chapitre4) pour assurer une totale reproductibilité des résultats.

Celle-ci peut être utilisée pour vous éviter de télécharger les dépendances nécessaires à ce chapitre. Vous pouvez la récupérer avec la commande suivante :

```
docker pull thomasfaria/dt-rn-chapitre4
```

Il vous est également possible de télécharger les dépendances localement en utilisant le fichier *requirements.txt*, à l'aide de la commande ```pip install -r requirements.txt```.

Cependant nous vous recommendons fortement l'utilisation d'[Onyxia](https://github.com/InseeFrLab/onyxia-web), la plateforme *datascience* développé par l'[Insee](https://www.insee.fr/fr/accueil)). Pour ce faire vous pouvez suivre ces étapes :

- Etape 0: Allez sur [https://datalab.sspcloud.fr/home](https://datalab.sspcloud.fr/home). Cliquer sur **Sign In** et ensuite **create an account** avec votre adresse email institutionnelle ou académique.
- Etape 1: Cliquez [ICI](https://datalab.sspcloud.fr/launcher/ide/rstudio?autoLaunch=true&onyxia.friendlyName=«dt-chap4»&security.allowlist.enabled=false&service.image.custom.enabled=true&service.image.pullPolicy=«Always»&service.image.custom.version=«thomasfaria%2Fdt-rn-chapitre4») ou sur le badge orange en haut de la page pour lancer un service.
- Etape 2: **Ouvrez** le service et suivez les instructions affichées concernant l'**identifiant** et le **mot de passe**.
- Etape 3: **Clonez** le projet grâce à la commande suivant : ```git clone https://github.com/ThomasFaria/DT-RN-chapitre4.git```.

Tous les packages ont déjà été installés, vous devez en mesure de relancer tous les codes présents dans le projet.

## Organisation

Les programmes de cette partie sont écrit en **python**. Le réseau (*autoencoder*) est réalisé avec la librairie **Keras**.

Le code se divise en 4 scripts distincts :

- [1-runDataRetrieval.py](https://github.com/ThomasFaria/DT-RN-chapitre4/blob/main/1-runDataRetrieval.py) : Ce code télécharge automatiquement des données de la Banque Centrale Européenne via l'API de Statistical Data Warehouse (SDW). Nous téléchargeons ici des données d'indice des prix, mais ce script peut être utilisé pour récupérer toutes autres bases de données répertoriées dans SDW.
- [2-runPreprocessing.py](https://github.com/ThomasFaria/DT-RN-chapitre4/blob/main/2-runPreprocessing.py) : Cette partie définit tout d'abord le type de variables (catégorielle ou numérique). Elle procède ensuite à la préparation de la base de données en imputant les données manquantes, en ajustant l'échelle des données et en divisant les données en 3 sous-échantillons (*train*, *test*, *validation*).
- [3-runTraining.py](https://github.com/ThomasFaria/DT-RN-chapitre4/blob/main/3-runTraining.py) : Ce script définit les hyper-paramètres de l'autoencoder et procède à son estimation. On présente également une solution pour enregistrer les résultats en fonction des paramètres spécifiés.
- [4-runACP.py](https://github.com/ThomasFaria/DT-RN-chapitre4/blob/main/4-runACP.py) : Ce code réplique le travail effectué par le script précédent, en utilisant cette fois un algorithme d'Analyse par Composantes Principales (ACP) afin d'en comparer les performances.
