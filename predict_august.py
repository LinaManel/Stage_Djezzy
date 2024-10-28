import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

"""# Préparation des données :"""

# Chargement des Données
df = pd.read_csv('Dataframes/august_dt.csv', sep = ';')

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
features = ['Day', 'Event_coded', 'Action_Business', 'Day_of_month', 'Cycle_de_rechargement', 'Jour_ferie']
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
print("Somme des revenus: ", df['predicted_revenu'].sum())

# Calculer le seuil
threshold = 100 - df['predicted_revenu'].sum()

# Déstribuer le seuil sur le mois
threshold_month = threshold / len(df)
df['predicted_revenu'] = (df['predicted_revenu'] + threshold_month).round(4)

# La somme des revenus prédit apés normalisation
print("Somme des revenus: ", df['predicted_revenu'].sum().round(),"%")

# # Sauvgarde des résultats
# df[['Date', 'predicted_revenu']].to_csv('Model/prediction_august.csv', sep=';' , index=False)
