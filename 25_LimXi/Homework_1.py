import numpy as np
import matplotlib.pyplot as plt

def visualize_clt_bernoulli(p=0.5, sample_size=30, num_samples=10000):

    #Generating samples from a Bernoulli distribution
    samples = np.random.binomial(1, p, (num_samples, sample_size))
    
    #Calculating mean of each sample
    sample_means = np.mean(samples, axis=1)
    
    #Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.hist(sample_means, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.title(f"CLT: Distribution of Sample Means (Bernoulli p={p})")
    plt.xlabel(f"Sample Mean (n={sample_size})")
    plt.ylabel("Frequency")
    
    plt.legend()
    plt.show()

# Run the simulation
visualize_clt_bernoulli(p=0.3, sample_size=20000)