import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import time
import itertools

# Memuat dataset
data_path = "data.txt"
data = np.loadtxt(data_path, delimiter=" ", usecols=(0, 1))

# Parameter
k = 2
iteration_counter = 0

# Centroid awal
initial_centroids = np.array([[3.4, 3.4], [6.4, 6.4]])

def plot_cluster_result(cluster_members, centroids, iteration, converged, file_name=None):
    """Menampilkan hasil klastering dalam bentuk plot dan menyimpannya jika file_name diberikan."""
    num_clusters = len(cluster_members)
    colors = iter(cm.rainbow(np.linspace(0, 1, num_clusters)))
    plt.figure("Hasil Klastering")
    plt.clf()
    plt.title(f"Iterasi-{iteration}")
    marker = itertools.cycle(['.', '*', '^', 'x', '+'])

    for i in range(num_clusters):
        color = next(colors)
        cluster_data = np.asmatrix(cluster_members[i])
        plt.scatter(np.ravel(cluster_data[:, 0]), np.ravel(cluster_data[:, 1]),
                    marker=next(marker), s=100, c=color, label=f"Kluster-{i + 1}")
    
    for i in range(num_clusters):
        plt.scatter(centroids[i, 0], centroids[i, 1], marker=next(marker),
                    c=color, label=f"Centroid-{i + 1}")

    plt.legend()
    if file_name:
        plt.savefig(file_name)  # Simpan plot sebagai file PNG
    plt.ion()
    plt.show()
    if converged:
        plt.show(block=True)
    else:
        plt.pause(0.1)

def k_means(data, initial_centroids):
    """Mengimplementasikan algoritma K-Means."""
    num_clusters = k
    global iteration_counter
    centroids = np.matrix(initial_centroids)

    while True:
        iteration_counter += 1
        euclidean_matrix_all = np.ndarray(shape=(data.shape[0], 0))

        # Hitung jarak Euclidean ke masing-masing centroid
        for i in range(num_clusters):
            repeated_centroid = np.repeat(centroids[i, :], data.shape[0], axis=0)
            delta_matrix = abs(np.subtract(data, repeated_centroid))
            euclidean_matrix = np.sqrt(np.square(delta_matrix).sum(axis=1))
            euclidean_matrix_all = np.concatenate((euclidean_matrix_all, euclidean_matrix), axis=1)

        # Tentukan anggota klaster berdasarkan jarak minimum
        cluster_indices = np.ravel(np.argmin(np.matrix(euclidean_matrix_all), axis=1))
        cluster_members = [[] for _ in range(k)]

        for i in range(data.shape[0]):
            cluster_members[int(cluster_indices[i])].append(data[i, :])

        # Hitung centroid baru
        new_centroids = np.ndarray(shape=(0, centroids.shape[1]))
        for i in range(num_clusters):
            cluster_data = np.asmatrix(cluster_members[i])
            centroid_cluster = cluster_data.mean(axis=0)
            new_centroids = np.concatenate((new_centroids, centroid_cluster), axis=0)

        print(f"Iterasi: {iteration_counter}")
        print(f"Centroid: {new_centroids}")

        # Simpan hasil iterasi sebagai file PNG
        plot_cluster_result(cluster_members, centroids, iteration_counter, converged=False,
                            file_name=f"iterasi-{iteration_counter}.png")

        # Periksa konvergensi
        if (centroids == new_centroids).all():
            break

        centroids = new_centroids
        time.sleep(1)

    return centroids, cluster_members

# Jalankan algoritma K-Means
final_centroids, cluster_results = k_means(data, initial_centroids)

# Plot hasil akhir
plot_cluster_result(cluster_results, final_centroids, f"{iteration_counter} (Konvergen)", 
                    converged=True, file_name=f"iterasi-{iteration_counter}-konvergen.png")
