import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from kneed import KneeLocator

# Įkeliami .csv duomenys
df = pd.read_csv('duomenys/atrinkta_aibe.csv', sep=';')
X = df[[col for col in df.columns if col != 'label']].values
Y = df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Naudojamas "Elbow" metodas rasti optimliam klustreių (k) skaičiui
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
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
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

# Taikomas t-SNE dimensijų mažinimui (vizualizacijai)
PERPLEXITY = 50
MAX_ITER = 500
METRIC = 'canberra'
RANDOM_STATE = 42

tsne = TSNE(n_components=2,
    perplexity=PERPLEXITY,
    max_iter=MAX_ITER,
    metric=METRIC,
    random_state=RANDOM_STATE
    )

data_tsne = tsne.fit_transform(X)

# Spausdinami rezultatai pagal t-SNE
plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    plt.scatter(
        data_tsne[clusters == cluster, 0],
        data_tsne[clusters == cluster, 1],
        label=f'Cluster {cluster}'
    )

plt.xlabel('')
plt.ylabel('')
plt.title('K-Means klasterių vizualizacija naudojant t-SNE')
plt.legend()
plt.show()