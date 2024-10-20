import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class Import_data:
    def both():
        dataset1 = pd.read_csv('data/Mall_Customers.csv')[['Annual Income (k$)', 'Spending Score (1-100)']]
        dataset2 = pd.read_csv('data/iris.csv')[['sepal_width','sepal_length']]
        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the first dataset on the left subplot
        axes[0].scatter(dataset1['Annual Income (k$)'], dataset1['Spending Score (1-100)'], color='blue')
        axes[0].set_title('Mall Customers')
        axes[0].set_xlabel('Annual Income')
        axes[0].set_ylabel('Spending Score (1-100)')

        # Plot the second dataset on the right subplot
        axes[1].scatter(dataset2['sepal_width'], dataset2['sepal_length'], color='blue')
        axes[1].set_title('Sepal Width')
        axes[1].set_xlabel('Sepal Length')
        axes[1].set_ylabel('FinalGrade')

        # Show the plot
        plt.tight_layout()
        plt.show()
        return dataset1, dataset2
    def first_dataset():
        #Task 1: importing and visualising datasets
        dataset1 = pd.read_csv('data/Mall_Customers.csv')

        plt.scatter(dataset1['Annual Income (k$)'], dataset1['Spending Score (1-100)'], color='blue')
        plt.title('Annual Income vs Spending Score')
        plt.show()
        #Type: Connected by Jumpers
        return dataset1
    
    def second_dataset():
        #Task 1: importing and visualising datasets
        dataset1 = pd.read_csv('data/student_performance.csv')

        plt.scatter(dataset1['StudyHoursPerWeek'], dataset1['FinalGrade'], color='blue')
        plt.title('Annual Income vs Spending Score')
        plt.show()
        #Type: Connected by Jumpers
        return dataset1
    
    def plot_elbow_method(dataset1,dataset2,title1,title2):
        distortions = []
        distortions2 = []
        K = range(1, 10)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(dataset1)
            distortions.append(kmeans.inertia_)

        K = range(1, 10)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(dataset2)
            distortions2.append(kmeans.inertia_)

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the first dataset on the left subplot
        axes[0].plot(K, distortions, 'bx-')
        axes[0].set_title(f'Elbow Method for Optimal k: {title1}')
        axes[0].set_xlabel('Number of clusters (k)')
        axes[0].set_ylabel('Inertia')

        # Plot the second dataset on the right subplot
        axes[1].plot(K, distortions2, 'bx-')
        axes[1].set_title(f'Elbow Method for Optimal k: {title2}')
        axes[1].set_xlabel('Number of clusters (k)')
        axes[1].set_ylabel('Inertia')

        # Show the plot
        plt.tight_layout()
        plt.show()
