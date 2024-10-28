import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prettytable import PrettyTable
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


"""# Préparation des données :"""

# Chargement des Données
df = pd.read_csv('Dataframes/prepared_dt.csv', sep = ';')

# Remplir les colonnes avec valeurs manquantes
df['Event_coded'] = df['Event_coded'].fillna(0).copy()


"""# Prédiction des revenus avec KMeans"""

# Vérifier que le fichier n'est pas vide puis le télècharger
def load_model(file_path):
    if os.path.getsize(file_path) > 0:
        with open(file_path, 'rb') as f:
            return joblib.load(f)
    else:
        print(f"File {file_path} is empty!")
        return None

# Télècharger le model KMeans
kmeans_model = load_model('Model/kmeans_model.pkl')

# Liste des features utiliser lors de l'apprentissage
features = ['Day', 'Event_coded', 'Action_Business', 'Day_of_month', 'Cycle_de_rechargement', 'Jour_ferie', 'Day_coded']
X_clus = df[features]

# Normaliser les données
preprocessor = kmeans_model.named_steps['preprocessor']
X_clus_scaled = preprocessor.transform(X_clus)

# Prédire les clusters des nouvelles données
clusterer = kmeans_model.named_steps['clusterer']
predicted_clusters = clusterer.predict(X_clus_scaled)

# Ajouter les clusters dans le dataFrame
df['predicted_cluster'] = predicted_clusters

# Télècharger les ravenus
revenu_ = pd.read_csv('Model/prediction_revenu.csv').values.flatten()
# Prédire les revenus par rapport au cluster attribué
df['predicted_revenu'] = revenu_[predicted_clusters]


"""# Etude des résultats :

## Normalisation des revenus:
"""

# La somme des revenus prédit
print("Somme des revenus: ", df['predicted_revenu'].sum().round())

# Calculer le seuil
threshold = 100 - df['predicted_revenu'].sum()

# Déstribuer le seuil sur le mois
threshold_month = threshold / len(df)

df['predicted_revenu'] = (df['predicted_revenu'] + threshold_month).round(4)

# La somme des revenus prédit apés normalisation
print("Somme des revenus: ", df['predicted_revenu'].sum().round(),"%")

# # Sauvgarde des résultats
# df[['Date', 'predicted_revenu']].to_csv('Model/prediction_july.csv', sep=';', index=False)


"""## Comparaison des résultats:"""

# Chargement des données réelles
df_j = pd.read_csv('Dataframes/Real data.csv', sep=';')

df_j["Real %"].sum()

threshold = 100 - df_j["Real %"].sum()
threshold_month = threshold / len(df_j)
df_j["Real %"] = (df_j["Real %"] + threshold_month).round(4)

# Calculer de l'erreur quadratique moyenne
mse = mean_squared_error(df_j['Real %'], df['predicted_revenu'])
print(f'- Mean Squared Error (MSE): {mse:.4f}')

# Calculer de l'erreur moyenne absolut
mae = mean_absolute_error(df_j['Real %'], df['predicted_revenu'])
print(f'- Mean Absolute Error (MAE): {mae:.4f}')

# Calculer de la racine de l'erreur quadratique moyenne
rmse = mean_squared_error(df_j['Real %'], df['predicted_revenu'], squared=False)
print(f'- Root Mean Squared Error (RMSE): {rmse:.4f}')

# Calculer du r quadratique
r2 = r2_score(df_j['Real %'], df['predicted_revenu'])
print(f'- R-squared: {r2:.4f}')

# Définir le seuil de précision (5% de tolerance)
threshold_ = 0.1

# Calculater la précision
accuracy = (abs(df_j['Real %'] - df['predicted_revenu']) <= threshold_ * df_j['Real %']).mean()
# accuracy = (100 - (abs(df_j['Real %'] - df['predicted_revenu']) / df_j['Real %']) * 100).mean()
print(f'- Accuracy: {accuracy*100:.2f}%')

# Calculer l'erreur quadratique de chaque ligne
df['squared_error'] = ((df_j['Real %'] - df['predicted_revenu'])**2).round(6)

# Afficher les résultats
merged_df = pd.merge( df_j[['Date', 'Real %']], df[['Date', 'predicted_revenu', 'squared_error', 'predicted_cluster']], on='Date')

# Créater un objet PrettyTable
table = PrettyTable()

# Ajouter des colonnes
table.field_names = ["Cluster"] + list(merged_df.columns)
for index, row in merged_df.iterrows():
    table.add_row([index] + list(row))

# Afficher la table
print(table)