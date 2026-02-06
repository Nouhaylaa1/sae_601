"""
📝 **Instructions** :
- Installez toutes les bibliothèques nécessaires en fonction des imports présents dans le code, utilisez la commande suivante :conda create -n projet python pandas numpy ..........
- Complétez les sections en écrivant votre code où c’est indiqué.
- Ajoutez des commentaires clairs pour expliquer vos choix.
- Utilisez des emoji avec windows + ;
- Interprétez les résultats de vos visualisations (quelques phrases).
"""
### 1. Importation des librairies et chargement des données
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

# Chargement des données
data = pd.read_csv("/data/ds_salaries.csv")


def main():
    ### 2. Exploration visuelle des données
    st.title("Visualisation des Salaires en Data Science")
    st.markdown("Explorez les tendances des salaires à travers différentes visualisations interactives.")

    if st.checkbox("Afficher un aperçu des données"):
        st.write(data.head(10))

    # Statistiques générales
    st.subheader("Statistiques générales")
    st.write(data.describe())

    ### 3. Distribution des salaires en France
    st.subheader("Distribution des salaires en France")
    france_data = data[data['company_location'] == 'FR']
    if not france_data.empty:
        fig = px.box(france_data, x='experience_level', y='salary_in_usd', color='job_title',
                     title='Distribution des salaires en France par niveau d\'expérience')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas de données pour la France dans ce dataset.")

    ### 4. Analyse des tendances de salaires
    st.subheader("Salaires moyens par catégorie")
    category = st.selectbox("Sélectionnez une catégorie", ['experience_level', 'employment_type', 'job_title', 'company_location'])
    avg_salary = data.groupby(category)['salary_in_usd'].mean().sort_values(ascending=False).head(15)
    fig = px.bar(avg_salary, x=avg_salary.index, y=avg_salary.values,
                 labels={'x': category, 'y': 'Salaire moyen (USD)'},
                 title=f'Salaire moyen par {category}')
    st.plotly_chart(fig, use_container_width=True)

    ### 5. Corrélation entre variables
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    
    st.subheader("Corrélations entre variables numériques")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar_kws={'label': 'Corrélation'})
    st.pyplot(fig)

    ### 6. Analyse des variations de salaire
    st.subheader("Évolution des salaires par métier")
    top_jobs = data['job_title'].value_counts().head(10).index
    job_salary_trend = data[data['job_title'].isin(top_jobs)].groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)
    fig = px.bar(job_salary_trend, 
                 x=job_salary_trend.index, y=job_salary_trend.values,
                 labels={'x': 'Métier', 'y': 'Salaire moyen (USD)'},
                 title='Top 10 des métiers les mieux rémunérés')
    st.plotly_chart(fig, use_container_width=True)

    ### 7. Salaire médian par expérience et taille d'entreprise
    st.subheader("Salaire médian par expérience et taille d'entreprise")
    median_salary = data.groupby(['experience_level', 'company_size'])['salary_in_usd'].median().reset_index()
    fig = px.bar(median_salary, x='experience_level', y='salary_in_usd', color='company_size',
                 barmode='group', labels={'salary_in_usd': 'Salaire médian (USD)', 'experience_level': 'Niveau d\'expérience'},
                 title='Salaire médian par niveau d\'expérience et taille d\'entreprise')
    st.plotly_chart(fig, use_container_width=True)

    ### 8. Filtres dynamiques par plage de salaire
    st.subheader("Filtrer par plage de salaire")
    min_salary = int(data['salary_in_usd'].min())
    max_salary = int(data['salary_in_usd'].max())
    selected_range = st.slider("Sélectionnez une plage de salaire (USD)", min_salary, max_salary, (min_salary, max_salary))
    filtered_data = data[(data['salary_in_usd'] >= selected_range[0]) & (data['salary_in_usd'] <= selected_range[1])]
    st.write(f"Nombre d'employés dans cette plage : {len(filtered_data)}")
    st.dataframe(filtered_data[['job_title', 'experience_level', 'salary_in_usd', 'company_location']].head(20))

    ### 9. Impact du télétravail sur le salaire selon le pays
    st.subheader("Impact du télétravail sur les salaires par pays")
    remote_impact = data.groupby(['company_location', 'remote_ratio'])['salary_in_usd'].mean().reset_index()
    top_countries = data['company_location'].value_counts().head(10).index
    remote_impact_top = remote_impact[remote_impact['company_location'].isin(top_countries)]
    fig = px.bar(remote_impact_top, x='company_location', y='salary_in_usd', color='remote_ratio',
                 barmode='group', title='Salaire moyen par pays et taux de télétravail')
    st.plotly_chart(fig, use_container_width=True)

    ### 10. Filtrage avancé avec deux multiselect
    st.subheader("Filtrage avancé")
    col1, col2 = st.columns(2)

    with col1:
        experience_levels = st.multiselect(
            "Sélectionnez le niveau d'expérience",
            options=data['experience_level'].unique(),
            default=data['experience_level'].unique()
        )

    with col2:
        company_sizes = st.multiselect(
            "Sélectionnez la taille d'entreprise",
            options=data['company_size'].unique(),
            default=data['company_size'].unique()
        )

    advanced_filtered = data[
        (data['experience_level'].isin(experience_levels)) & 
        (data['company_size'].isin(company_sizes))
    ]

    st.write(f"Résultats : {len(advanced_filtered)} employés correspondent aux critères")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Salaire moyen", f"{advanced_filtered['salary_in_usd'].mean():,.0f} USD")
    with col2:
        st.metric("Salaire médian", f"{advanced_filtered['salary_in_usd'].median():,.0f} USD")
    with col3:
        st.metric("Salaire max", f"{advanced_filtered['salary_in_usd'].max():,.0f} USD")

    st.dataframe(advanced_filtered[['job_title', 'experience_level', 'company_size', 'salary_in_usd', 'remote_ratio']].head(30))


if __name__ == "__main__":
    main()