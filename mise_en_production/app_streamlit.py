# ============================================================
# APPLICATION STREAMLIT - PROJET ASSURANCE ASSISTANCE
# ============================================================
# Cette application presente l'analyse complete des donnees
# d'assurance assistance : DataViz, Econometrie et Machine Learning
# ============================================================

# --- Import des bibliotheques ---
import streamlit as st                # Framework web pour data apps
import pandas as pd                   # Manipulation de DataFrames
import numpy as np                    # Calcul numerique
import matplotlib.pyplot as plt       # Graphiques matplotlib
import seaborn as sns                 # Graphiques statistiques
import plotly.express as px           # Graphiques interactifs Plotly
import plotly.graph_objects as go     # Graphiques avances Plotly
from plotly.subplots import make_subplots  # Sous-graphiques Plotly

# --- Configuration de la page ---
st.set_page_config(
    page_title="Assurance Assistance - Analyse Data",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CHARGEMENT DES DONNEES (en cache pour la performance)
# ============================================================

@st.cache_data
def load_data():
    """Charge les donnees depuis les fichiers CSV.
    base_model.csv = dossiers TRAIN (duree > 0)
    base_test.csv  = dossiers TEST  (duree == 0)
    """
    base = pd.read_csv('../data/processed/base_model.csv')
    base['date.ouverture'] = pd.to_datetime(base['date.ouverture'])
    base['heure'] = pd.to_datetime(base['heure.ouverture'], format='%H:%M:%S').dt.hour

    base_test = pd.read_csv('../data/processed/base_test.csv')
    return base, base_test

@st.cache_data
def load_ml_comparison():
    """Charge le tableau de comparaison ML/DL si disponible."""
    try:
        return pd.read_csv('../data/models/comparaison_modeles.csv', index_col=0)
    except FileNotFoundError:
        return None

# Chargement des donnees
base, base_test = load_data()
ml_comparison = load_ml_comparison()


# ============================================================
# BARRE LATERALE - NAVIGATION
# ============================================================

st.sidebar.markdown("""
# 🚗 Assurance Assistance
### Analyse Data - DU Python 2025-2026
---
""")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Accueil",
        "📊 DataViz - Analyse Descriptive",
        "📈 Econometrie - GLM",
        "🤖 Machine Learning",
        "🔍 Exploration Interactive"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Donnees :**
- Dossiers d'assistance 2021-2022
- Deplacement avec survenance France
""")
st.sidebar.markdown(f"**Base TRAIN (duree > 0) :** {len(base):,}")
st.sidebar.markdown(f"**Base TEST (duree = 0) :** {len(base_test):,}")


# ============================================================
# PAGE 1 : ACCUEIL
# ============================================================

if page == "🏠 Accueil":

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #4a90d9 100%);
                padding: 40px; border-radius: 15px; margin-bottom: 30px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <h1 style="color: white; text-align: center; font-size: 2.5em; margin-bottom: 10px;">
            🚗 Projet Assurance Assistance
        </h1>
        <h3 style="color: #b8d4f0; text-align: center; font-weight: normal; margin-bottom: 5px;">
            Analyse et Prediction de la Duree de Traitement des Dossiers
        </h3>
        <p style="color: #d0e4f7; text-align: center; font-size: 1.1em;">
            DU Python 2025-2026 | Universite de Montpellier
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Metriques principales ---
    st.markdown("### 📋 Chiffres Cles")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Dossiers (TRAIN)", value=f"{len(base):,}")

    with col2:
        duree_moy = base['duree_corrigee_totale'].mean() / 60
        st.metric(label="Duree Moyenne", value=f"{duree_moy:.0f} min")

    with col3:
        st.metric(label="Clients", value=f"{base['Client'].nunique()}")

    with col4:
        st.metric(label="Causes d'intervention", value=f"{base['Cause.intervention'].nunique()}")

    st.markdown("---")

    # --- Description du projet ---
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        ### 📖 Contexte du Projet

        Ce projet s'inscrit dans le cadre du **DU Python 2025-2026** de l'Universite de Montpellier.
        En tant que data scientist junior dans une entreprise d'assurance assistance,
        nous analysons les **dossiers d'assistance automobile** ouverts entre 2021 et 2022.

        #### 🎯 Objectifs

        1. **Traitement des donnees** : Nettoyage, detection d'anomalies, jointures entre les 3 tables
        2. **Analyse descriptive** : Comprendre les distributions et les relations entre variables
        3. **Econometrie** : Modelisation parametrique (GLM Gamma, Inverse Gaussien) avec backward elimination
        4. **Machine Learning** : 6 modeles ML (Ridge, Lasso, RF, GBM, XGBoost) + 4 modeles DL (TensorFlow)
        5. **Mise en production** : Interface interactive Streamlit

        #### 📁 Donnees Disponibles

        | Table | Description | Lignes |
        |-------|------------|--------|
        | `dossier.csv` | Dossiers d'assistance (1 ligne = 1 dossier) | ~101 000 |
        | `temps.csv` | Temps passe par matricule sur chaque dossier | ~431 000 |
        | `ressources.csv` | Profil des agents (contrat, experience, lieu) | ~389 000 |
        """)

    with col_right:
        st.markdown("### 📊 Repartition par Cause d'Intervention")
        cause_counts = base['Cause.intervention'].value_counts()
        fig_cause = px.pie(
            values=cause_counts.values, names=cause_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4
        )
        fig_cause.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_cause, use_container_width=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### ⏱️ Distribution de la Duree de Traitement")
        fig_duree = px.histogram(
            base, x=base['duree_corrigee_totale'] / 60, nbins=100,
            labels={'x': 'Duree (minutes)'}, color_discrete_sequence=['#3498db']
        )
        fig_duree.update_layout(xaxis_title="Duree (minutes)", yaxis_title="Nombre de dossiers",
                                height=350, showlegend=False)
        mediane = base['duree_corrigee_totale'].median() / 60
        fig_duree.add_vline(x=mediane, line_dash="dash", line_color="red",
                            annotation_text=f"Mediane = {mediane:.0f} min")
        st.plotly_chart(fig_duree, use_container_width=True)

    with col_b:
        st.markdown("### 📅 Evolution Mensuelle du Nombre de Dossiers")
        monthly = base.groupby(base['date.ouverture'].dt.to_period('M')).size().reset_index()
        monthly.columns = ['Mois', 'Nombre']
        monthly['Mois'] = monthly['Mois'].astype(str)
        fig_monthly = px.line(monthly, x='Mois', y='Nombre', markers=True,
                              color_discrete_sequence=['#2ecc71'])
        fig_monthly.update_layout(xaxis_title="Mois", yaxis_title="Nombre de dossiers",
                                  height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig_monthly, use_container_width=True)

    # --- Navigation rapide ---
    st.markdown("---")
    st.markdown("### 🚀 Navigation Rapide")

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

    with nav_col1:
        st.markdown("""
        <div style="background: #ecf0f1; padding: 20px; border-radius: 10px; text-align: center;
                    border-left: 5px solid #3498db;">
            <h4>📊 DataViz</h4>
            <p>Analyses graphiques univariees et multivariees</p>
        </div>
        """, unsafe_allow_html=True)

    with nav_col2:
        st.markdown("""
        <div style="background: #ecf0f1; padding: 20px; border-radius: 10px; text-align: center;
                    border-left: 5px solid #2ecc71;">
            <h4>📈 Econometrie</h4>
            <p>GLM Gamma & Inv. Gaussien + backward elimination</p>
        </div>
        """, unsafe_allow_html=True)

    with nav_col3:
        st.markdown("""
        <div style="background: #ecf0f1; padding: 20px; border-radius: 10px; text-align: center;
                    border-left: 5px solid #e74c3c;">
            <h4>🤖 Machine Learning</h4>
            <p>6 modeles ML + 4 modeles Deep Learning (TensorFlow)</p>
        </div>
        """, unsafe_allow_html=True)

    with nav_col4:
        st.markdown("""
        <div style="background: #ecf0f1; padding: 20px; border-radius: 10px; text-align: center;
                    border-left: 5px solid #9b59b6;">
            <h4>🔍 Exploration</h4>
            <p>Filtres interactifs et exploration libre</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE 2 : DATAVIZ - ANALYSE DESCRIPTIVE
# ============================================================

elif page == "📊 DataViz - Analyse Descriptive":

    st.markdown("# 📊 Data Visualisation - Analyse Descriptive")
    st.markdown("Exploration graphique des donnees pour comprendre les distributions et relations entre variables.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Analyse Univariee", "Analyse Bivariee", "Analyse Temporelle", "Correlations"
    ])

    top_vars = ['TOP.D.R', 'TOP.VR', 'TOP.Rappat.valide', 'TOP.Poursuite', 'TOP.Recup', 'TOP.Autres.Garanties']

    # ---- ONGLET 1 : ANALYSE UNIVARIEE ----
    with tab1:
        st.markdown("### Distribution des Variables")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Causes d'Intervention")
            cause_data = base['Cause.intervention'].value_counts().reset_index()
            cause_data.columns = ['Cause', 'Nombre']
            fig = px.bar(cause_data, x='Cause', y='Nombre', color='Cause',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(xaxis_tickangle=-45, showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Type d'Energie")
            energie_data = base['Type.d.energie'].value_counts().reset_index()
            energie_data.columns = ['Energie', 'Nombre']
            fig = px.bar(energie_data, x='Energie', y='Nombre', color='Energie',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Taux d'Activation des Services d'Assistance")
        top_labels = ['Depannage/Remorquage', 'Vehicule Remplacement', 'Rapatriement',
                      'Poursuite Voyage', 'Recuperation', 'Autres']
        top_rates = [base[v].mean() * 100 for v in top_vars]
        fig_top = px.bar(x=top_labels, y=top_rates, labels={'x': 'Service', 'y': 'Taux (%)'},
                         color=top_rates, color_continuous_scale='Blues')
        fig_top.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_top, use_container_width=True)

    # ---- ONGLET 2 : ANALYSE BIVARIEE ----
    with tab2:
        st.markdown("### Relations entre Variables")

        cat_choice = st.selectbox(
            "Variable categorielle pour l'analyse :",
            ['Cause.intervention', 'Type.d.energie', 'Outil.d.assistance',
             'pop_mode', 'site_mode', 'type_contrat_mode']
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"#### Duree par {cat_choice}")
            fig = px.box(base, x=cat_choice, y=np.log1p(base['duree_corrigee_totale']),
                         color=cat_choice, labels={'y': 'log(1 + Duree)'})
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"#### Duree Moyenne par {cat_choice}")
            mean_duree = base.groupby(cat_choice)['duree_corrigee_totale'].mean().sort_values(ascending=False)
            mean_duree_min = mean_duree / 60
            fig = px.bar(x=mean_duree_min.index, y=mean_duree_min.values,
                         labels={'x': cat_choice, 'y': 'Duree moyenne (min)'},
                         color=mean_duree_min.values, color_continuous_scale='Reds')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Nombre d'Intervenants vs Duree de Traitement")
        sample = base.sample(min(5000, len(base)), random_state=42)
        fig = px.scatter(sample, x='nb_intervenants',
                         y=np.log1p(sample['duree_corrigee_totale']),
                         color='Cause.intervention', opacity=0.5,
                         labels={'y': 'log(1 + Duree)', 'x': "Nombre d'intervenants"})
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # ---- ONGLET 3 : ANALYSE TEMPORELLE ----
    with tab3:
        st.markdown("### Analyses Temporelles")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Volume Mensuel de Dossiers")
            monthly = base.groupby([base['date.ouverture'].dt.to_period('M')]).size().reset_index()
            monthly.columns = ['Mois', 'Nombre']
            monthly['Mois'] = monthly['Mois'].astype(str)
            fig = px.area(monthly, x='Mois', y='Nombre', color_discrete_sequence=['#3498db'])
            fig.update_layout(xaxis_tickangle=-45, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Dossiers par Heure de la Journee")
            heure_data = base['heure'].value_counts().sort_index().reset_index()
            heure_data.columns = ['Heure', 'Nombre']
            fig = px.bar(heure_data, x='Heure', y='Nombre', color='Nombre',
                         color_continuous_scale='Viridis')
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Dossiers par Jour de la Semaine")
        jour_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        jour_data = base['date.ouverture'].dt.day_name().value_counts().reindex(jour_order).reset_index()
        jour_data.columns = ['Jour', 'Nombre']
        fig = px.bar(jour_data, x='Jour', y='Nombre', color='Nombre', color_continuous_scale='Sunset')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ---- ONGLET 4 : CORRELATIONS ----
    with tab4:
        st.markdown("### Matrice de Correlation (Spearman)")

        num_cols = ['duree_corrigee_totale', 'nb_interventions', 'nb_intervenants',
                    'prop_tele', 'exp_moy'] + top_vars
        corr = base[num_cols].corr(method='spearman')

        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', aspect='auto')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 3 : ECONOMETRIE - GLM
# ============================================================

elif page == "📈 Econometrie - GLM":

    st.markdown("# 📈 Econometrie - Modeles Lineaires Generalises (GLM)")
    st.markdown("""
    Les GLM permettent de modeliser la duree de traitement en tenant compte de sa distribution
    asymetrique (non-normale). Deux familles sont comparees apres **backward elimination**
    (retrait iteratif des variables non significatives, p > 5%).
    """)

    # --- Resultats des modeles GLM (issus du notebook econometrie.ipynb) ---
    st.markdown("### Comparaison des Modeles GLM")
    st.markdown("""
    **Demarche** : modele complet → backward elimination (p < 0.05) → modele final.
    Les metriques sont calculees sur le **jeu de validation (20%)**, pas sur le train.
    """)

    glm_results = pd.DataFrame({
        'Modele': ['GLM Gamma (Log)', 'GLM Inverse Gaussien (Log)'],
        'Famille': ['Gamma', 'Inverse Gaussienne'],
        'Lien': ['Log', 'Log'],
        'Selection': ['Backward elimination', 'Backward elimination'],
        'Metriques': [
            'AIC + MAE + RMSE + MAPE sur validation',
            'AIC + MAE + RMSE + MAPE sur validation'
        ],
        'Interpretation': [
            'Adapte aux donnees positives continues (variance proportionnelle au carre de la moyenne)',
            'Adapte aux queues lourdes (variance proportionnelle au cube de la moyenne)'
        ]
    })

    st.dataframe(glm_results, use_container_width=True, hide_index=True)

    st.info("""
    **Metriques d'evaluation (sur validation 20%)** :
    - **AIC** : critere d'information (plus bas = meilleur)
    - **MAE** : erreur absolue moyenne (en secondes)
    - **RMSE** : racine de l'erreur quadratique moyenne (en secondes)
    - **MAPE** : erreur absolue moyenne en pourcentage
    """)

    st.markdown("---")

    # --- Coefficients importants ---
    st.markdown("### Coefficients Significatifs (apres backward elimination)")
    st.markdown("""
    Les coefficients du meilleur GLM permettent d'interpreter l'impact de chaque variable
    sur la duree de traitement. Seules les variables significatives (p < 0.05) sont retenues.
    """)

    coef_data = pd.DataFrame({
        'Variable': [
            'nb_intervenants', 'TOP.VR', 'TOP.Rappat.valide', 'TOP.Poursuite',
            'pop_mode_CAS', 'TOP.Autres.Garanties', 'TOP.Recup',
            'Cause: Panne mecanique', 'Cause: Cles/Carburant/Crevaison',
            'Cause: Autres', 'TOP.D.R', 'prop_tele'
        ],
        'Effet': [
            'Augmente', 'Augmente', 'Augmente', 'Augmente',
            'Augmente', 'Augmente', 'Augmente',
            'Reduit', 'Reduit',
            'Reduit', 'Reduit', 'Reduit'
        ],
        'Significativite': ['p < 0.001'] * 12
    })

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🔺 Facteurs qui augmentent la duree
        - **Poursuite de voyage** : service complexe a mettre en place
        - **Rapatriement** : logistique importante de retour
        - **Vehicule de remplacement** : coordination avec un loueur
        - **Population CAS** : dossiers plus complexes
        - **Autres garanties** : prestations supplementaires
        """)

    with col2:
        st.markdown("""
        #### 🔻 Facteurs qui reduisent la duree
        - **Cles/Carburant/Crevaison** : interventions simples et rapides
        - **Causes Autres** : dossiers souvent automatises
        - **Panne mecanique** : processus standardise
        - **Depannage/Remorquage** : service direct sans suite
        - **Teletravail** : agents plus efficaces en teletravail
        """)

    st.markdown("---")

    # --- Analyse VIF ---
    st.markdown("### Analyse de Multicolinearite (VIF)")
    st.markdown("""
    Le **Variance Inflation Factor (VIF)** est calcule sur les variables retenues par chaque
    modele final (apres backward elimination).
    - VIF > 5 : multicolinearite moderee
    - VIF > 10 : multicolinearite forte
    """)

    st.markdown("---")

    # --- Diagnostics des residus ---
    st.markdown("### Diagnostics des Residus")
    st.markdown("""
    Les diagnostics verifient les hypotheses des GLM :
    - **QQ-plot** : les residus suivent-ils la loi attendue ?
    - **Residus vs valeurs ajustees** : detection d'heteroscedasticite ou de patterns residuels
    - **Statistiques** : moyenne, ecart-type, skewness et kurtosis des residus de deviance
    """)


# ============================================================
# PAGE 4 : MACHINE LEARNING
# ============================================================

elif page == "🤖 Machine Learning":

    st.markdown("# 🤖 Machine Learning & Deep Learning")
    st.markdown("""
    Nous avons teste **10 modeles** pour predire la duree de traitement :
    **6 modeles ML classiques** (scikit-learn) et **4 architectures Deep Learning** (TensorFlow).

    - Split : 80% train / 20% test sur `base_model.csv`
    - Cible : `log(1 + duree)` (inversion pour les metriques en secondes)
    - Metriques : **MAE**, **RMSE**, **MAPE** (pas de R2 pour les GLM/DL)
    """)

    # --- Tableau des modeles ---
    st.markdown("### Modeles Testes")

    tab_ml, tab_dl = st.tabs(["Machine Learning (6 modeles)", "Deep Learning (4 modeles)"])

    with tab_ml:
        ml_info = pd.DataFrame({
            'Modele': ['Regression Lineaire', 'Ridge (L2)', 'Lasso (L1)',
                       'Random Forest', 'Gradient Boosting', 'XGBoost'],
            'Type': ['Lineaire', 'Lineaire regularise', 'Lineaire regularise',
                     'Ensemble (Bagging)', 'Ensemble (Boosting)', 'Ensemble (Boosting)'],
            'Avantage': ['Baseline simple', 'Gere la multicolinearite', 'Selection de variables',
                         'Robuste, peu de tuning', 'Tres performant', 'Etat de l\'art']
        })
        st.dataframe(ml_info, use_container_width=True, hide_index=True)

    with tab_dl:
        dl_info = pd.DataFrame({
            'Modele': ['MLP Simple', 'MLP Deep', 'Wide & Deep', 'Residual MLP'],
            'Architecture': ['2 couches (64, 32)', '4 couches + Dropout + L2',
                             'Branche lineaire + profonde', 'Skip connections'],
            'Particularite': ['Baseline DL', 'Regularisation contre l\'overfitting',
                              'Effets lineaires + non-lineaires', 'Apprentissage plus stable']
        })
        st.dataframe(dl_info, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Resultats de la comparaison ---
    st.markdown("### Comparaison des Performances")

    if ml_comparison is not None:
        results_df = ml_comparison.reset_index()
        results_df.columns = ['Modele'] + list(results_df.columns[1:])

        # Determiner le type
        ml_names = ['Regression Lineaire', 'Ridge', 'Lasso', 'Random Forest',
                    'Gradient Boosting', 'XGBoost']

        display_cols = [c for c in results_df.columns if c in
                        ['Modele', 'MAE Train (s)', 'MAE Test (s)', 'RMSE Train (s)',
                         'RMSE Test (s)', 'MAPE Test (%)', 'Type', 'Temps (s)']]

        if display_cols:
            results_sorted = results_df[display_cols].sort_values(
                'RMSE Test (s)' if 'RMSE Test (s)' in display_cols else display_cols[1]
            )
            st.dataframe(results_sorted.round(1), use_container_width=True, hide_index=True)

            best = results_sorted.iloc[0]
            best_name = best['Modele']
            st.success(f"**Meilleur modele : {best_name}** | "
                       f"RMSE = {best.get('RMSE Test (s)', 'N/A')}s | "
                       f"MAE = {best.get('MAE Test (s)', 'N/A')}s | "
                       f"MAPE = {best.get('MAPE Test (%)', 'N/A')}%")
        else:
            st.dataframe(results_df.round(1), use_container_width=True, hide_index=True)
    else:
        st.warning("Le fichier `comparaison_modeles.csv` n'a pas ete trouve. "
                   "Executez d'abord le notebook `machine_learning_final.ipynb`.")

        # Valeurs illustratives
        results_df = pd.DataFrame({
            'Modele': ['XGBoost', 'Gradient Boosting', 'Random Forest', 'MLP Deep',
                       'Wide & Deep', 'Residual MLP', 'MLP Simple',
                       'Ridge', 'Regression Lineaire', 'Lasso'],
            'MAE Test (s)': [330, 340, 350, 360, 365, 370, 380, 400, 402, 405],
            'RMSE Test (s)': [650, 660, 680, 700, 710, 720, 730, 750, 752, 755],
            'MAPE Test (%)': [85, 87, 90, 92, 93, 95, 97, 102, 103, 104],
            'Type': ['ML', 'ML', 'ML', 'DL', 'DL', 'DL', 'DL', 'ML', 'ML', 'ML']
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Graphiques comparatifs ---
    st.markdown("### Visualisation Comparative")

    col1, col2 = st.columns(2)

    with col1:
        if 'RMSE Test (s)' in results_df.columns:
            fig_rmse = px.bar(
                results_df.sort_values('RMSE Test (s)'), x='Modele', y='RMSE Test (s)',
                color='Type' if 'Type' in results_df.columns else 'RMSE Test (s)',
                color_discrete_map={'ML': '#3498db', 'DL': '#e74c3c'},
                title='RMSE Test (plus bas = meilleur)'
            )
            fig_rmse.update_layout(xaxis_tickangle=-45, height=400, showlegend=True)
            st.plotly_chart(fig_rmse, use_container_width=True)

    with col2:
        if 'MAE Test (s)' in results_df.columns:
            fig_mae = px.bar(
                results_df.sort_values('MAE Test (s)'), x='Modele', y='MAE Test (s)',
                color='Type' if 'Type' in results_df.columns else 'MAE Test (s)',
                color_discrete_map={'ML': '#3498db', 'DL': '#e74c3c'},
                title='MAE Test (plus bas = meilleur)'
            )
            fig_mae.update_layout(xaxis_tickangle=-45, height=400, showlegend=True)
            st.plotly_chart(fig_mae, use_container_width=True)

    st.markdown("---")

    # --- Synthese ---
    st.markdown("### Synthese")
    st.markdown("""
    - Les **modeles d'ensemble** (XGBoost, Gradient Boosting, Random Forest) sont generalement les plus performants
    - Le **Deep Learning** (MLP Deep, Wide & Deep) offre des performances comparables mais est moins interpretable
    - Le **nombre d'intervenants** et le **nombre d'interventions** sont les predicteurs les plus forts
    - Les **services d'assistance demandes** (VR, Rapatriement) impactent fortement la duree
    - Tous les modeles sont sauvegardes dans `data/models/` pour utilisation dans l'application
    """)

    st.markdown("---")

    # --- Section Prediction Interactive ---
    st.markdown("### 🔮 Prediction Interactive de la Duree")
    st.markdown("""
    Renseignez les caracteristiques d'un dossier ci-dessous pour obtenir une **estimation
    de la duree de traitement** grace au meilleur modele entraine.
    """)

    # Features (synchronisees avec machine_learning_final.ipynb)
    FEAT_NUM = ['nb_interventions', 'nb_intervenants', 'prop_tele', 'exp_moy']
    FEAT_CAT = ['Cause.intervention', 'Type.d.energie', 'Outil.d.assistance',
                'Assistance.ou.Administratif', 'pop_mode', 'site_mode', 'type_contrat_mode']
    FEAT_BIN = ['TOP.D.R', 'TOP.VR', 'TOP.Rappat.valide', 'TOP.Poursuite',
                'TOP.Recup', 'TOP.Autres.Garanties']

    # --- Formulaire de saisie ---
    st.markdown("#### Caracteristiques du Dossier")

    col1, col2, col3 = st.columns(3)

    with col1:
        cause = st.selectbox(
            "Cause d'intervention",
            sorted(base['Cause.intervention'].unique()), key="pred_cause"
        )
        energie = st.selectbox(
            "Type d'energie",
            sorted(base['Type.d.energie'].dropna().unique()), key="pred_energie"
        )
        outil = st.selectbox(
            "Outil d'assistance",
            sorted(base['Outil.d.assistance'].unique()), key="pred_outil"
        )
        assist = st.selectbox(
            "Assistance ou Administratif",
            sorted(base['Assistance.ou.Administratif'].unique()), key="pred_assist"
        )

    with col2:
        pop = st.selectbox(
            "Population (agent)",
            sorted(base['pop_mode'].dropna().unique()), key="pred_pop"
        )
        site = st.selectbox(
            "Site de travail",
            sorted(base['site_mode'].dropna().unique()), key="pred_site"
        )
        contrat = st.selectbox(
            "Type de contrat",
            sorted(base['type_contrat_mode'].dropna().unique()), key="pred_contrat"
        )

    with col3:
        nb_interventions = st.number_input(
            "Nombre d'interventions", min_value=1, max_value=50, value=3, step=1, key="pred_nb_int"
        )
        nb_intervenants = st.number_input(
            "Nombre d'intervenants", min_value=1, max_value=30, value=2, step=1, key="pred_nb_agent"
        )
        prop_tele = st.slider(
            "Proportion teletravail", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="pred_tele"
        )
        exp_moy = st.number_input(
            "Experience moyenne (jours)", min_value=0.0, max_value=5000.0, value=1500.0,
            step=100.0, key="pred_exp"
        )

    # --- Services d'assistance (toggles) ---
    st.markdown("#### Services d'Assistance Demandes")
    svc1, svc2, svc3 = st.columns(3)

    with svc1:
        top_dr = st.checkbox("Depannage / Remorquage", value=True, key="pred_dr")
        top_vr = st.checkbox("Vehicule de Remplacement", value=False, key="pred_vr")
    with svc2:
        top_rappat = st.checkbox("Rapatriement", value=False, key="pred_rappat")
        top_poursuite = st.checkbox("Poursuite de Voyage", value=False, key="pred_pours")
    with svc3:
        top_recup = st.checkbox("Recuperation", value=False, key="pred_recup")
        top_autres = st.checkbox("Autres Garanties", value=False, key="pred_autres")

    # --- Selection du modele ---
    st.markdown("#### Choix du Modele")

    # Lister les modeles disponibles
    import os, joblib
    MODEL_DIR = '../data/models'

    available_models = {}
    if os.path.exists(MODEL_DIR):
        for f in sorted(os.listdir(MODEL_DIR)):
            if f.endswith('.pkl') and f.startswith('ml_'):
                name = f.replace('ml_', '').replace('.pkl', '').replace('_', ' ')
                available_models[name] = ('ml', os.path.join(MODEL_DIR, f))
            elif f.endswith('.keras') and f.startswith('dl_'):
                name = f.replace('dl_', '').replace('.keras', '').replace('_', ' ')
                available_models[name] = ('dl', os.path.join(MODEL_DIR, f))

    if available_models:
        # Charger les metadonnees pour connaitre le meilleur modele
        meta_path = os.path.join(MODEL_DIR, 'ml_metadata.joblib')
        best_default = 0
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            best_name = meta.get('best_model_name', '')
            # Chercher l'index du meilleur modele
            model_names = list(available_models.keys())
            for idx, n in enumerate(model_names):
                if best_name.lower().replace(' ', '') in n.lower().replace(' ', ''):
                    best_default = idx
                    break

        selected_model = st.selectbox(
            "Modele de prediction",
            list(available_models.keys()),
            index=best_default,
            key="pred_model_choice"
        )
    else:
        selected_model = None
        st.warning("Aucun modele trouve dans `data/models/`. "
                   "Executez d'abord le notebook `machine_learning_final.ipynb`.")

    # --- Bouton de prediction ---
    if st.button("Predire la duree de traitement", type="primary", use_container_width=True):

        # Construction du DataFrame d'entree
        input_data = pd.DataFrame([{
            'nb_interventions': nb_interventions,
            'nb_intervenants': nb_intervenants,
            'prop_tele': prop_tele,
            'exp_moy': exp_moy,
            'Cause.intervention': cause,
            'Type.d.energie': energie,
            'Outil.d.assistance': outil,
            'Assistance.ou.Administratif': assist,
            'pop_mode': pop,
            'site_mode': site,
            'type_contrat_mode': contrat,
            'TOP.D.R': int(top_dr),
            'TOP.VR': int(top_vr),
            'TOP.Rappat.valide': int(top_rappat),
            'TOP.Poursuite': int(top_poursuite),
            'TOP.Recup': int(top_recup),
            'TOP.Autres.Garanties': int(top_autres),
        }])

        prediction = None

        if selected_model and available_models:
            model_type, model_path = available_models[selected_model]

            with st.spinner(f"Chargement du modele {selected_model}..."):
                try:
                    if model_type == 'ml':
                        # Modele sklearn (pipeline complete avec preprocessor)
                        model = joblib.load(model_path)
                        all_features = FEAT_NUM + FEAT_CAT + FEAT_BIN
                        X_pred = input_data[all_features]
                        y_pred_log = model.predict(X_pred)[0]
                        prediction = float(np.expm1(y_pred_log))

                    elif model_type == 'dl':
                        # Modele TensorFlow (necessite le preprocessor)
                        preprocessor_path = os.path.join(MODEL_DIR, 'preprocessor.joblib')
                        if os.path.exists(preprocessor_path):
                            from tensorflow import keras
                            model = keras.models.load_model(model_path)
                            preprocessor = joblib.load(preprocessor_path)
                            all_features = FEAT_NUM + FEAT_CAT + FEAT_BIN
                            X_pred = input_data[all_features]
                            X_pred_p = preprocessor.transform(X_pred)
                            y_pred_log = model.predict(X_pred_p, verbose=0).ravel()[0]
                            prediction = float(np.expm1(y_pred_log))
                        else:
                            st.error("Preprocessor non trouve (`preprocessor.joblib`).")
                except Exception as e:
                    st.error(f"Erreur lors de la prediction : {e}")

        # Fallback : moyenne conditionnelle
        if prediction is None:
            st.warning("Prediction approximative par moyenne conditionnelle (modele non disponible).")
            mask = base['Cause.intervention'] == cause
            if mask.sum() > 0:
                prediction = float(base.loc[mask, 'duree_corrigee_totale'].mean())
            else:
                prediction = float(base['duree_corrigee_totale'].mean())

        prediction = max(prediction, 0.0)

        # --- Affichage du resultat ---
        st.markdown("---")
        st.markdown("#### Resultat de la Prediction")

        res1, res2, res3 = st.columns(3)

        with res1:
            st.metric("Duree estimee (secondes)", f"{prediction:.0f} s")
        with res2:
            st.metric("Duree estimee (minutes)", f"{prediction / 60:.1f} min")
        with res3:
            moyenne = base['duree_corrigee_totale'].mean()
            ecart = ((prediction - moyenne) / moyenne) * 100
            st.metric("Ecart vs moyenne", f"{ecart:+.1f} %")

        # Interpretation contextuelle
        duree_min = prediction / 60
        if duree_min < 10:
            st.success("Ce dossier devrait etre traite **rapidement** (< 10 min).")
        elif duree_min < 30:
            st.info("Duree de traitement **moderee** (10-30 min).")
        elif duree_min < 60:
            st.warning("Duree de traitement **elevee** (30-60 min). Dossier complexe.")
        else:
            st.error("Duree de traitement **tres elevee** (> 1h). Dossier tres complexe.")

        # Details du modele utilise
        if selected_model:
            st.caption(f"Modele utilise : **{selected_model}** | "
                       f"Type : {'ML (scikit-learn)' if available_models[selected_model][0] == 'ml' else 'DL (TensorFlow)'}")


# ============================================================
# PAGE 5 : EXPLORATION INTERACTIVE
# ============================================================

elif page == "🔍 Exploration Interactive":

    st.markdown("# 🔍 Exploration Interactive des Donnees")
    st.markdown("Utilisez les filtres ci-dessous pour explorer les donnees librement.")

    # --- Filtres ---
    st.markdown("### Filtres")

    col1, col2, col3 = st.columns(3)

    with col1:
        causes = ['Toutes'] + sorted(base['Cause.intervention'].unique().tolist())
        cause_filter = st.selectbox("Cause d'intervention", causes)

    with col2:
        energies = ['Toutes'] + sorted(base['Type.d.energie'].dropna().unique().tolist())
        energie_filter = st.selectbox("Type d'energie", energies)

    with col3:
        outils = ['Tous'] + sorted(base['Outil.d.assistance'].unique().tolist())
        outil_filter = st.selectbox("Outil d'assistance", outils)

    # Application des filtres
    filtered = base.copy()
    if cause_filter != 'Toutes':
        filtered = filtered[filtered['Cause.intervention'] == cause_filter]
    if energie_filter != 'Toutes':
        filtered = filtered[filtered['Type.d.energie'] == energie_filter]
    if outil_filter != 'Tous':
        filtered = filtered[filtered['Outil.d.assistance'] == outil_filter]

    st.info(f"**{len(filtered):,} dossiers** correspondent aux filtres selectionnes.")

    st.markdown("---")

    # --- Statistiques du sous-ensemble filtre ---
    st.markdown("### Statistiques du sous-ensemble")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nombre de dossiers", f"{len(filtered):,}")
    with col2:
        st.metric("Duree moyenne", f"{filtered['duree_corrigee_totale'].mean()/60:.0f} min")
    with col3:
        st.metric("Duree mediane", f"{filtered['duree_corrigee_totale'].median()/60:.0f} min")
    with col4:
        st.metric("Nb intervenants moyen", f"{filtered['nb_intervenants'].mean():.1f}")

    # --- Visualisations ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Distribution de la Duree (filtree)")
        fig = px.histogram(filtered, x=filtered['duree_corrigee_totale'] / 60,
                           nbins=80, color_discrete_sequence=['#e74c3c'])
        fig.update_layout(xaxis_title="Duree (minutes)", yaxis_title="Frequence",
                          height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Services Actives (filtres)")
        top_vars = ['TOP.D.R', 'TOP.VR', 'TOP.Rappat.valide', 'TOP.Poursuite',
                    'TOP.Recup', 'TOP.Autres.Garanties']
        top_labels_short = ['D/R', 'VR', 'Rappat.', 'Poursuite', 'Recup.', 'Autres']
        rates_filtered = [filtered[v].mean() * 100 for v in top_vars]
        rates_all = [base[v].mean() * 100 for v in top_vars]

        fig = go.Figure(data=[
            go.Bar(name='Filtre', x=top_labels_short, y=rates_filtered, marker_color='#e74c3c'),
            go.Bar(name='Global', x=top_labels_short, y=rates_all, marker_color='#95a5a6')
        ])
        fig.update_layout(barmode='group', yaxis_title="Taux (%)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # --- Apercu des donnees ---
    st.markdown("### Apercu des Donnees Filtrees")
    st.dataframe(
        filtered[['Numero_dossier_ID', 'Client', 'Cause.intervention', 'Type.d.energie',
                  'duree_corrigee_totale', 'nb_intervenants', 'nb_interventions']].head(20),
        use_container_width=True, hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95a5a6; padding: 20px;">
    <p>Projet Data - DU Python 2025-2026 | Universite de Montpellier</p>
    <p>Donnees : Dossiers d'assistance automobile 2021-2022</p>
</div>
""", unsafe_allow_html=True)
