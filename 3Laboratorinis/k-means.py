import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from kneed import KneeLocator

# Įkeliami .csv duomenys
df = pd.read_csv('duomenys/tsne_2d_data.csv', sep=';')

X = df['tsne_dim1'].values
Y = df['tsne_dim2'].values

# Naudojamas "Elbow" metodas rasti optimliam klustreių (k) skaičiui
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df[['tsne_dim1', 'tsne_dim2']])
    inertia.append(kmeans.inertia_)

# "Elbow" grafikas
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-', markersize=8)
plt.xlabel('Klasterių skaičius (k)')
plt.ylabel('Kvadratinė paklaida')
plt.title('"Elbow" metodas')
plt.show()

# Pasirenkamas optimalus k naudojant KneeLocator (Nustatoma vieta kurioje kreivė "lenkiasi")
kneedle = KneeLocator(K_range, inertia, curve='convex', direction='decreasing')
optimal_k = kneedle.knee

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df[['tsne_dim1', 'tsne_dim2']])

df['Cluster'] = clusters

# Spausdinami rezultatai pagal t-SNE
plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    plt.scatter(
        df.loc[clusters == cluster, 'tsne_dim1'],
        df.loc[clusters == cluster, 'tsne_dim2'],
        label=f'Cluster {cluster}'
    )

plt.xlabel('')
plt.ylabel('')
plt.title('K-Means klasterių vizualizacija naudojant t-SNE')
plt.legend()
plt.show()