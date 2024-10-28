
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from prettytable import PrettyTable


"""# Préparation des données"""

# Chargement des Données
df = pd.read_csv('Dataframes/clean_dt_pour_.csv', sep=';')

# Remplir les colonnes avec valeurs manquantes
df['Event_coded'] = df['Event_coded'].fillna(0).copy()


"""# Création du model KMeans"""

# Definir les features
features = ['Day', 'Action_Business', 'Day_of_month', 'Cycle_de_rechargement','Jour_ferie', 'Event_coded']

# Definir la transformation des features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Day_of_month', 'Action_Business', 'Cycle_de_rechargement', 'Jour_ferie']),
        ('cat', OneHotEncoder(), ['Event_coded', 'Day'])
    ])

X_scaled = preprocessor.fit_transform(df[features])

# Créer le pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clusterer', KMeans(n_clusters=20, init='k-means++', n_init=20, max_iter=500, random_state=42))  # 43
])

pipeline.fit(df[features])

# Sauvgarder les clusters
df['Cluster'] = pipeline.predict(df[features])

# Calculater les revenus de chaque cluster
cluster_revenu_summary = df.groupby('Cluster')['Revenu'].describe().round(4)
# Créater un objet PrettyTable
table = PrettyTable()

# Ajouter des colonnes
table.field_names = ["Cluster"] + list(cluster_revenu_summary.columns)
for index, row in cluster_revenu_summary.iterrows():
    table.add_row([index] + list(row))

# Afficher la table
print(table)

# Sauvgarder les revenus
revenu_pred = df.groupby('Cluster')['Revenu'].quantile(0.75)

# Enregister le model KMeans entrainer et les moyennes de 'Revenu' de cchaque cluster
joblib.dump(pipeline, 'Model/kmeans_model.pkl')

revenu_pred_df = pd.DataFrame(revenu_pred)
revenu_pred_df.to_csv('Model/prediction_revenu.csv', index=False)