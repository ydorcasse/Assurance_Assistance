# Projet Data - Assurance Assistance Automobile

## DU Python 2025-2026 | Universite de Montpellier

---

## Contexte

Ce projet s'inscrit dans le cadre du **Diplome Universitaire Python 2025-2026** de l'Universite de Montpellier.
En tant que data scientist junior dans une entreprise d'assurance assistance, nous analysons les **dossiers d'assistance automobile** ouverts entre 2021 et 2022 sur le territoire francais.

L'objectif est de mener une etude complete de data science : du traitement des donnees brutes jusqu'a la mise en production d'une application interactive, en passant par l'analyse statistique, la modelisation econometrique et le machine learning.

---

## Participants

| Nom | Prenom | Role |
|-----|--------|------|
| SEYANKAM | Dorcas | Data Scientist Junior |

---

## Organisation du Projet

```
Assurance/
|
|-- data/                              # Donnees du projet
|   |-- raw/                           # Donnees brutes (non modifiees)
|   |   |-- dossier.csv               # Table des dossiers d'assistance (~101 000 lignes)
|   |   |-- temps.csv                 # Table des temps passes par matricule (~431 000 lignes)
|   |   |-- ressources.csv            # Table des ressources/agents (~389 000 lignes)
|   |-- processed/                     # Donnees nettoyees et enrichies
|   |   |-- base_model.csv            # Base TRAIN (duree > 0) pour la modelisation
|   |   |-- base_test.csv             # Base TEST (duree == 0) a predire
|   |   |-- base_dossier.csv          # Base dossier nettoyee
|   |   |-- anomalies_retraitements.csv  # Tableau des anomalies et retraitements
|   |-- models/                        # Modeles sauvegardes
|   |   |-- ml_*.pkl                  # Modeles ML (scikit-learn pipelines)
|   |   |-- dl_*.keras                # Modeles DL (TensorFlow)
|   |   |-- preprocessor.joblib       # Pipeline de preprocessing
|   |   |-- ml_metadata.joblib        # Metadonnees (features, meilleur modele)
|   |   |-- comparaison_modeles.csv   # Tableau comparatif ML vs DL
|   |-- Glossaire.pdf                  # Dictionnaire des donnees (description des variables)
|
|-- traitement_de_donnees/             # Notebook de traitement et nettoyage
|   |-- traitement_donnees_metier.ipynb  # Traitement principal : nettoyage, anomalies, jointures
|
|-- econometrie/                       # Notebook d'econometrie et analyses statistiques
|   |-- econometrie.ipynb             # Analyse descriptive + AFDM + GLM (Gamma, Inv. Gaussien)
|
|-- machine_learning/                  # Notebook de Machine Learning et Deep Learning
|   |-- machine_learning.ipynb        # 6 modeles ML + 4 architectures DL (TensorFlow)
|
|-- mise_en_production/                # Application Streamlit
|   |-- app.py                        # Application web interactive (OOP, pages separees)
|   |-- app_streamlit.py              # Version monolithique de l'application
|   |-- pages/                        # Pages de l'application (architecture OOP)
|       |-- __init__.py
|       |-- accueil.py
|       |-- dataviz.py
|       |-- econometrie.py
|       |-- machine_learning.py
|       |-- exploration.py
|
|-- requirements.txt                   # Dependances Python du projet
|-- README.md                          # Ce fichier
|-- LICENSE                            # Licence du projet
```

---

## Description des Donnees

### Table `dossier.csv` - Dossiers d'Assistance
Ensemble des dossiers d'assistance ouverts de 2021 a 2022 pour du deplacement avec une survenance en France. **1 ligne = 1 dossier**.

| Variable | Description |
|----------|-------------|
| `Numero_dossier_ID` | Identifiant unique du dossier (anonymise) |
| `Client` | Client qui delegue son activite d'assistance (anonymise) |
| `Formule` | Formule d'assistance liee au contrat d'assurance |
| `date.ouverture` | Date d'ouverture du dossier (AAAA/MM/JJ) |
| `heure.ouverture` | Heure d'ouverture (HH:MM:SS) |
| `Matricule.de.traitement` | Personne ayant cree le dossier (171 = creation automatique) |
| `Cause.intervention` | Cause de la demande (panne, accident, vol, etc.) |
| `date.de.survenance` | Date de l'incident |
| `Type.d.energie` | Energie du vehicule (Essence, Diesel, Electrique, etc.) |
| `Outil.d.assistance` | Outil utilise (MCS ou Higgins) |
| `Assistance.ou.Administratif` | Type de dossier (Assistance directe ou Administratif) |
| `TOP.D.R` | Service Depannage/Remorquage (0/1) |
| `TOP.VR` | Service Vehicule de Remplacement (0/1) |
| `TOP.Rappat.valide` | Service Rapatriement d'un valide (0/1) |
| `TOP.Poursuite` | Service Poursuite de voyage (0/1) |
| `TOP.Recup` | Service Recuperation de vehicule (0/1) |
| `TOP.Autres.Garanties` | Autres services d'assistance (0/1) |

### Table `temps.csv` - Temps Dossier
Temps passe par chaque agent (matricule) sur chaque dossier. Le matricule 171 (systeme) n'apparait pas dans cette table.

| Variable | Description |
|----------|-------------|
| `Numero.dossier` | Identifiant du dossier |
| `Matricule` | Agent ayant travaille sur le dossier |
| `Date.debut.traitement` | Date de debut de l'intervention |
| `heure.debut.traitement` | Heure de debut |
| `duree.corrigee` | Duree en secondes passee sur le dossier |

### Table `ressources.csv` - Ressources
Profil des agents (Charges d'Accueil et Charges d'Assistance) pour chaque jour de presence.

| Variable | Description |
|----------|-------------|
| `Matricule` | Identifiant de l'agent |
| `Date.presence` | Jour de presence (JJ/MM/AAAA) |
| `Lieu.travail` | TELE (teletravail) ou SITE (entreprise) |
| `Population` | CAC (Charge d'Accueil) ou CAS (Charge d'Assistance) |
| `Site` | Lieu de rattachement (A ou B) |
| `Type.de.contrat` | CDI, CDD ou CDS (saisonnier) |
| `Duree.travail` | Duree de travail en heures (journee = 7.33h) |
| `Temps.travail` | Temps de presence (100 = temps plein) |
| `Experience` | Jours de presence cumules |

---

## Methodologie

### 1. Traitement des Donnees (`traitement_donnees_metier.ipynb`)

Traitement sequentiel : **Dossier** → **Temps** → **Ressources**, avec 7 etapes par table :

1. Lecture et visualisation
2. Verification et correction des types
3. Retraitement des variables mal encodees
4. Verification de la delimitation temporelle (2021-2022)
5. Detection et retraitement des doublons (logique NB2 : ligne de reference + arbitrage colonne par colonne)
6. Detection et imputation des valeurs manquantes
7. Detection et correction des incoherences

**Sorties** :
- `base_model.csv` : base TRAIN (duree > 0, ~89% des dossiers)
- `base_test.csv` : base TEST (duree == 0, ~11% des dossiers)
- `anomalies_retraitements.csv` : tableau de tracabilite des anomalies et retraitements

### 2. Analyse Statistique et Econometrie (`econometrie.ipynb`)

**Partie 1 — Analyse descriptive**
- Analyse univariee : distributions des variables numeriques et categorielles
- Analyse bivariee : tests statistiques (Kruskal-Wallis, Mann-Whitney, Spearman, Tukey HSD)
- Analyse temporelle : evolution mensuelle, distribution par heure et jour
- Matrice de correlation de Spearman

**Partie 2 — Analyse multivariee**
- AFDM (Analyse Factorielle des Donnees Mixtes) avec 5 composantes (library `prince`)

**Partie 3 — Modelisation econometrique (GLM)**
- GLM Gamma (lien log) avec backward elimination (p < 0.05)
- GLM Inverse Gaussien (lien log) avec backward elimination (p < 0.05)
- Comparaison par AIC + metriques sur validation (MAE, RMSE, MAPE)
- Analyse VIF (multicolinearite) sur les variables retenues
- Diagnostics des residus (QQ-plot, residus vs valeurs ajustees)

### 3. Machine Learning et Deep Learning (`machine_learning_final.ipynb`)

**6 modeles ML classiques** (scikit-learn) :

| Modele | Type | Description |
|--------|------|-------------|
| Regression Lineaire | Baseline | Reference de performance |
| Ridge (L2) | Regularise | Penalite L2 contre la multicolinearite |
| Lasso (L1) | Regularise | Selection automatique de variables |
| Random Forest | Ensemble (Bagging) | Combinaison de 200 arbres |
| Gradient Boosting | Ensemble (Boosting) | Arbres sequentiels correcteurs |
| XGBoost | Ensemble (Boosting) | Etat de l'art du boosting |

**4 architectures Deep Learning** (TensorFlow / Keras) :

| Modele | Architecture | Particularite |
|--------|-------------|---------------|
| MLP Simple | 2 couches (64, 32) | Baseline DL |
| MLP Deep | 4 couches + Dropout + L2 | Regularisation contre l'overfitting |
| Wide & Deep | Branche lineaire + profonde | Capture effets lineaires et non-lineaires |
| Residual MLP | Connexions residuelles | Apprentissage plus stable |

**Demarche** :
- Split 80% train / 20% test sur `base_model.csv`
- Transformation cible : `log(1 + duree)` pour stabiliser la variance
- Metriques en secondes : MAE, RMSE, MAPE
- Courbes de loss (train vs test) pour chaque modele DL
- Importance des variables par permutation
- Sauvegarde de tous les modeles dans `data/models/`

### 4. Mise en Production (Streamlit)

Application Streamlit avec **5 pages** :

| Page | Contenu |
|------|---------|
| Accueil | Chiffres cles, contexte, navigation |
| DataViz | Analyses uni/bivariee, temporelle, correlations |
| Econometrie | Resultats GLM, coefficients, VIF, diagnostics |
| Machine Learning | Comparaison ML vs DL, visualisations |
| Exploration | Filtres interactifs et exploration libre |

Deux versions :
- `app.py` : architecture OOP avec pages separees (`pages/`)
- `app_streamlit.py` : version monolithique

---

## Installation et Execution

### Prerequis
- Python 3.9+
- pip (gestionnaire de paquets Python)

### Installation

```bash
# Cloner le depot
git clone <URL_DU_DEPOT>
cd Assurance

# Creer un environnement virtuel (recommande)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Installer les dependances
pip install -r requirements.txt
```

### Execution des Notebooks

```bash
# Lancer Jupyter
jupyter notebook

# Ouvrir les notebooks dans l'ordre :
# 1. traitement_de_donnees/traitement_donnees_metier.ipynb
# 2. econometrie/econometrie.ipynb
# 3. machine_learning/machine_learning_final.ipynb
```

### Execution de l'Application Streamlit

```bash
cd mise_en_production
streamlit run app.py
# ou : streamlit run app_streamlit.py (version monolithique)
```

L'application sera accessible a l'adresse : `http://localhost:8501`

---

## Resultats Principaux

### Variables les Plus Influentes sur la Duree de Traitement
1. **Nombre d'intervenants** : plus il y a d'agents, plus le dossier est complexe et long
2. **Nombre d'interventions** : indicateur de la complexite du dossier
3. **Services demandes** : VR, Rapatriement et Poursuite de voyage augmentent fortement la duree
4. **Experience des agents** : les agents plus experimentes traitent les dossiers plus vite
5. **Teletravail** : legere reduction de la duree en teletravail

### Facteurs Cles
- Les **pannes mecaniques** et les **problemes de cles/carburant/crevaison** sont traites plus rapidement
- Les **accidents** et **incendies** necessitent des durees de traitement significativement plus longues
- Les **Charges d'Assistance (CAS)** traitent des dossiers plus complexes que les **Charges d'Accueil (CAC)**
- L'outil **Higgins** (nouvel outil) est en phase d'adoption croissante depuis 2021

---

## Technologies Utilisees

| Categorie | Outils |
|-----------|--------|
| Langage | Python 3.9+ |
| Manipulation de donnees | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Statistiques | scipy, statsmodels |
| Machine Learning | scikit-learn, xgboost |
| Deep Learning | TensorFlow / Keras |
| Analyse factorielle | prince |
| Application web | Streamlit |
| Versionnement | Git / GitLab |

---

## Licence

Ce projet est realise dans un cadre academique (DU Python 2025-2026, Universite de Montpellier).
