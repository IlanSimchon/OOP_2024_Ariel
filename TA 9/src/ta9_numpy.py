"""
Advanced NumPy Tutorial: Comprehensive Guide
=========================================
This tutorial covers basic to advanced NumPy concepts including array operations,
broadcasting, advanced indexing, statistical functions, and image processing.
"""

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy import stats
from skimage import io


# Part 1: Basic Array Creation
def demonstrate_array_creation():
    """Demonstrates different ways to create NumPy arrays."""
    # Create arrays with zeros and ones
    zeros_array = np.zeros(3)
    ones_array = np.ones(5)
    empty_array = np.empty(3)

    # Create array with specific range
    linspace_array = np.linspace(0, 10, 5)
    arange_array = np.arange(0, 10, 2)  # Start, stop, step

    # Create array from Python list
    list_array = np.array([10, 20, 30])

    # Create identity matrix
    identity_matrix = np.eye(3)

    # Create random arrays
    random_array = np.random.rand(3, 3)  # Uniform distribution [0,1]
    normal_array = np.random.randn(3, 3)  # Normal distribution

    print("Arrays Creation Examples:")
    print("Zeros array:", zeros_array)
    print("Ones array:", ones_array)
    print("Linspace array:", linspace_array)
    print("Arange array:", arange_array)
    print("\nIdentity matrix:\n", identity_matrix)
    print("\nRandom array:\n", random_array)
    print("\nNormal array:\n", normal_array)

    return zeros_array, ones_array, linspace_array, list_array

# Part 2: Broadcasting and Advanced Operations
def demonstrate_broadcasting():
    """Shows NumPy broadcasting rules and advanced operations."""
    # Basic broadcasting
    array_2d = np.array([[1, 2, 3], [4, 5, 6]])
    scalar = 2
    vector = np.array([10, 20, 30])

    print("\nBroadcasting Examples:")
    print("Original 2D array:\n", array_2d)
    print("Broadcasting with scalar:\n", array_2d * scalar)
    print("Broadcasting with vector:\n", array_2d + vector)

    # Advanced broadcasting
    a = np.array([[1, 2, 3], [4, 5, 6]]).reshape(2, 3, 1)
    b = np.array([10, 20]).reshape(1, 1, 2)
    print("\nComplex broadcasting result:\n", (a * b).shape)

    return array_2d, a, b

# Part 3: Advanced Indexing and Masking
def demonstrate_advanced_indexing():
    """Shows advanced indexing techniques and boolean masking."""
    # Create sample array
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # Integer indexing
    indices = np.array([[0, 1], [1, 2]])

    # Boolean masking
    mask = arr > 5

    print("\nAdvanced Indexing:")
    print("Original array:\n", arr)
    print("Using integer indices:\n", arr[indices])
    print("Using boolean mask:\n", arr[mask])

    # Fancy indexing with lists
    rows = [0, 2]
    cols = [1, 3]
    print("\nFancy indexing result:\n", arr[rows][:, cols])

    return arr, mask

# Part 4: Statistical Operations
def demonstrate_statistics():
    """Shows various statistical operations and visualizations in NumPy."""

    # Create multiple datasets for comparison
    np.random.seed(42)  # For reproducibility
    normal_data = np.random.normal(loc=0, scale=1, size=1000)
    uniform_data = np.random.uniform(-3, 3, size=1000)
    bimodal_data = np.concatenate([
        np.random.normal(-2, 0.5, 500),
        np.random.normal(2, 0.5, 500)
    ])

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))

    # 1. Normal Distribution Histogram with Fit
    ax1 = fig.add_subplot(221)
    n, bins, patches = ax1.hist(normal_data, bins=30, density=True, alpha=0.7,
                                color='skyblue', label='Data')

    # Fit a normal distribution and plot it
    mu, sigma = stats.norm.fit(normal_data)
    x = np.linspace(min(bins), max(bins), 100)
    y = stats.norm.pdf(x, mu, sigma)
    ax1.plot(x, y, 'r-', lw=2, label=f'Fit: μ={mu:.2f}, σ={sigma:.2f}')
    ax1.set_title('Normal Distribution with Fit')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.legend()

    # 2. Box Plot Comparison
    ax2 = fig.add_subplot(222)
    box_data = [normal_data, uniform_data, bimodal_data]
    ax2.boxplot(box_data, tick_labels=['Normal', 'Uniform', 'Bimodal'])
    ax2.set_title('Box Plot Comparison')
    ax2.set_ylabel('Value')

    # 3. Distribution Comparison (KDE)
    ax3 = fig.add_subplot(223)
    for data, label, color in zip([normal_data, uniform_data, bimodal_data],
                                  ['Normal', 'Uniform', 'Bimodal'],
                                  ['blue', 'green', 'red']):
        density = stats.gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), 200)
        ax3.plot(xs, density(xs), color=color, label=label)
    ax3.set_title('Kernel Density Estimation')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Density')
    ax3.legend()

    # 4. Q-Q Plot
    ax4 = fig.add_subplot(224)
    stats.probplot(normal_data, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normal)')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Print statistical summaries
    distributions = {
        'Normal': normal_data,
        'Uniform': uniform_data,
        'Bimodal': bimodal_data
    }

    print("\nStatistical Summaries:")
    print("-" * 50)
    for name, data in distributions.items():
        print(f"\n{name} Distribution:")
        print(f"Mean: {np.mean(data):.3f}")
        print(f"Median: {np.median(data):.3f}")
        print(f"Std Dev: {np.std(data):.3f}")
        print(f"Skewness: {stats.skew(data):.3f}")
        print(f"Kurtosis: {stats.kurtosis(data):.3f}")
        percentiles = np.percentile(data, [25, 50, 75])
        print(f"Quartiles: {percentiles[0]:.3f}, {percentiles[1]:.3f}, {percentiles[2]:.3f}")

    return normal_data, uniform_data, bimodal_data

# Part 5: Linear Algebra Operations
def demonstrate_linear_algebra():
    """Demonstrates linear algebra operations."""
    # Create matrices
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print("\nLinear Algebra Operations:")
    print("Matrix multiplication:\n", np.dot(A, B))
    print("Matrix determinant:", np.linalg.det(A))
    print("Matrix inverse:\n", np.linalg.inv(A))

    # Eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(A)
    print("\nEigenvalues:", eigenvals)
    print("Eigenvectors:\n", eigenvecs)

    # Solve linear system: Ax = b
    b = np.array([1, 2])
    x = np.linalg.solve(A, b)
    print("\nSolution to linear system:", x)

    return A, B, eigenvals, eigenvecs

# Part 6: Array Manipulation
def demonstrate_array_manipulation():
    """Shows various array manipulation techniques."""
    arr = np.array([[1, 2, 3], [4, 5, 6]])

    print("\nArray Manipulation:")
    print("Original array:\n", arr)
    print("Transposed:\n", arr.T)
    print("Flattened:", arr.flatten())

    # Stacking arrays
    horizontal_stack = np.hstack((arr, arr))
    vertical_stack = np.vstack((arr, arr))

    print("\nHorizontal stack:\n", horizontal_stack)
    print("Vertical stack:\n", vertical_stack)

    # Splitting arrays
    hsplit = np.hsplit(horizontal_stack, 2)
    vsplit = np.vsplit(vertical_stack, 2)

    print("\nHorizontal split:\n", hsplit[0])
    print("Vertical split:\n", vsplit[0])

    return arr, horizontal_stack, vertical_stack

# Part 7: Advanced Functions and Universal Functions (ufuncs)
def demonstrate_advanced_functions():
    """Shows advanced functions and universal functions in NumPy."""
    x = np.linspace(0, 2*np.pi, 100)

    print("\nAdvanced Functions:")
    # Trigonometric functions
    print("Sin of first 5 elements:", np.sin(x)[:5])
    print("Cos of first 5 elements:", np.cos(x)[:5])

    # Exponential and logarithmic functions
    arr = np.array([1, 2, 3])
    print("\nExponential:", np.exp(arr))
    print("Natural log:", np.log(arr))
    print("Log base 10:", np.log10(arr))

    # Custom ufunc
    def custom_ufunc(x):
        return x**2 + 2*x + 1

    vectorized_func = np.vectorize(custom_ufunc)
    print("\nCustom ufunc result:", vectorized_func(arr))

    return x, arr


def demonstrate_image_blurring(image_path, kernel_size=(5, 5)):
    """
    Demonstrates image blurring using a custom kernel size.

    Parameters:
    image_path (str): Path to the image file
    kernel_size (tuple): Size of the blurring kernel (height, width)
    """
    try:
        # Read the image
        image = io.imread(image_path)
        print(f"\nApplying blur with kernel size {kernel_size}")

        # Create the kernel
        kernel = np.ones(kernel_size) / (kernel_size[0] * kernel_size[1])

        # Apply convolution for each color channel
        blurred = np.zeros_like(image)

        for channel in range(image.shape[2]):
            # Pad the image to handle edges
            pad_h = kernel_size[0] // 2
            pad_w = kernel_size[1] // 2
            padded = np.pad(image[:, :, channel], ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

            # Apply convolution
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    # Extract the region to apply the kernel
                    region = padded[i:i + kernel_size[0], j:j + kernel_size[1]]
                    blurred[i, j, channel] = np.max(region * kernel)

        # Display results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')

        ax2.imshow(blurred.astype(np.uint8))
        ax2.set_title(f'Blurred Image (Kernel: {kernel_size[0]}x{kernel_size[1]})')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

        return blurred

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Part 8: Image Processing (Previous implementation)
def process_image(image_path):
    """Demonstrates image processing operations using NumPy arrays."""
    try:
        # Read the image
        photo = io.imread(image_path)
        print("\nImage Processing:")
        print("Image shape:", photo.shape)

        # Display original and modified images
        image_displays = [
            ("Original Image", photo),
            ("Vertically Flipped", photo[::-1]),
            ("Horizontally Flipped", photo[:,::-1]),
            ("Cropped", photo[100:600,200:600]),
            ("Subsampled", photo[::10,::10]),
            ("try", photo[:,:,::-1]),
            ("Masked", np.where(photo > 100, 255, 0))
        ]

        for title, img in image_displays:
            plt.figure(figsize=(10,10))
            plt.imshow(img)
            plt.title(title)
            plt.show()

    except Exception as e:
        print(f"Error processing image: {e}")


# Part 9: Structured Arrays
def demonstrate_structured_arrays():
    """Shows how to work with structured arrays in NumPy."""
    # Define structured data type
    dt = np.dtype([('name', 'U30'), ('age', 'i4'), ('salary', 'f8')])

    # Create structured array
    employees = np.array([
        ('John Doe', 35, 75000.00),
        ('Jane Smith', 28, 82000.00),
        ('Bob Johnson', 42, 68000.00)
    ], dtype=dt)

    print("\nStructured Arrays:")
    print("Full array:\n", employees)
    print("\nAccessing columns:")
    print("Names:", employees['name'])
    print("Ages:", employees['age'])
    print("Average salary:", np.mean(employees['salary']))

    # Filtering structured arrays
    senior_employees = employees[employees['age'] > 35]
    print("\nSenior employees:\n", senior_employees)

    return employees


# Part 10: Memory Management and Views
def demonstrate_memory_management():
    """Demonstrates memory views and copy operations."""
    # Create original array
    original = np.arange(12).reshape(3, 4)

    # Create different views and copies
    view = original.view()
    shallow_copy = original[:]
    deep_copy = original.copy()

    print("\nMemory Management:")
    print("Original array:\n", original)

    # Modify original
    original[0, 0] = 99
    print("\nAfter modifying original:")
    print("View (affected):\n", view)
    print("Shallow copy (affected):\n", shallow_copy)
    print("Deep copy (unaffected):\n", deep_copy)

    # Memory information
    print("\nMemory information:")
    print("Original data address:", original.__array_interface__['data'][0])
    print("View data address:", view.__array_interface__['data'][0])
    print("Deep copy data address:", deep_copy.__array_interface__['data'][0])

    return original, view, shallow_copy, deep_copy


# Part 11: Performance Optimization
def demonstrate_performance():
    """Shows various performance optimization techniques."""
    import time

    # Compare operations with different approaches
    size = 1000000
    arr = np.random.rand(size)
    list_data = arr.tolist()

    # Test 1: Sum calculation
    print("\nPerformance Comparisons:")

    start = time.time()
    python_sum = sum(list_data)
    python_time = time.time() - start

    start = time.time()
    numpy_sum = np.sum(arr)
    numpy_time = time.time() - start

    print(f"Python sum time: {python_time:.6f} seconds")
    print(f"NumPy sum time: {numpy_time:.6f} seconds")
    print(f"Speed improvement: {python_time / numpy_time:.2f}x")

    # Test 2: Element-wise operations
    start = time.time()
    python_square = [x ** 2 for x in list_data]
    python_time = time.time() - start

    start = time.time()
    numpy_square = arr ** 2
    numpy_time = time.time() - start

    print(f"\nPython square time: {python_time:.6f} seconds")
    print(f"NumPy square time: {numpy_time:.6f} seconds")
    print(f"Speed improvement: {python_time / numpy_time:.2f}x")

    return arr


# Part 12: FFT and Signal Processing
def demonstrate_fft():
    """Demonstrates Fast Fourier Transform capabilities."""
    # Generate a sample signal
    t = np.linspace(0, 1, 1000)
    freq1, freq2 = 10, 50
    signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

    # Compute FFT
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])

    # Plot results
    plt.figure(figsize=(12, 8))

    # Time domain plot
    plt.subplot(211)
    plt.plot(t[:100], signal[:100])
    plt.title('Time Domain Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Frequency domain plot
    plt.subplot(212)
    plt.plot(freqs[:len(freqs) // 2], np.abs(fft_result)[:len(freqs) // 2])
    plt.title('Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()

    return signal, fft_result, freqs


# Part 13: Random Number Generation and Probability
def demonstrate_random():
    """Shows advanced random number generation and probability functions."""
    # Set seed for reproducibility
    np.random.seed(42)

    # Different probability distributions
    distributions = {
        'Normal': np.random.normal(loc=0, scale=1, size=1000),
        'Poisson': np.random.poisson(lam=5, size=1000),
        'Exponential': np.random.exponential(scale=1.0, size=1000),
        'Chi-square': np.random.chisquare(df=5, size=1000),
        'Gamma': np.random.gamma(shape=2, scale=2, size=1000)
    }

    # Plot distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, (name, data) in enumerate(distributions.items()):
        if idx < len(axes):
            axes[idx].hist(data, bins=50, density=True)
            axes[idx].set_title(name)
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Density')

    plt.tight_layout()
    plt.show()

    # Statistical tests
    print("\nDistribution Statistics:")
    for name, data in distributions.items():
        print(f"\n{name} Distribution:")
        print(f"Mean: {np.mean(data):.3f}")
        print(f"Variance: {np.var(data):.3f}")
        print(f"Skewness: {stats.skew(data):.3f}")
        print(f"Kurtosis: {stats.kurtosis(data):.3f}")

    return distributions


def main():
    """Main function to run all demonstrations."""
    print("Advanced NumPy Tutorial Demonstrations\n" + "=" * 40)

    # Original demonstrations
    demonstrate_array_creation()
    demonstrate_broadcasting()
    demonstrate_advanced_indexing()
    demonstrate_statistics()
    demonstrate_linear_algebra()
    demonstrate_array_manipulation()
    demonstrate_advanced_functions()
    demonstrate_structured_arrays()
    demonstrate_memory_management()
    demonstrate_performance()
    demonstrate_fft()
    demonstrate_random()

    # Process image if available
    try:
        process_image('friends.jpg')
        # Add different kernel sizes for blurring
        demonstrate_image_blurring('friends.jpg', kernel_size=(3, 3))  # Small blur
        demonstrate_image_blurring('friends.jpg', kernel_size=(7, 7))  # Medium blur
        demonstrate_image_blurring('friends.jpg', kernel_size=(15, 15))  # Strong blur
    except Exception as e:
        print(f"Could not process image: {e}")


if __name__ == "__main__":
    main()