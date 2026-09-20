import numpy as np
from skimage import color, filters
from skimage import io

class SeamsCalculator:
    def __init__(self, image):
        """
        Initialize seam calculator with an image
        image: RGB image (H x W x 3)
        """
        self.image = image
        self.grey_scale = color.rgb2gray(image)
        self.energy = filters.sobel(self.grey_scale)
        self.rows, self.cols = self.energy.shape

        # DP table: shortest path distances
        self.dp = np.full((self.rows, self.cols), np.inf)

        # Parent table: stores column index of previous pixel
        self.parent = np.zeros((self.rows, self.cols), dtype=int)

    def compute_dp(self):
        """
        Computes SSSP on DAG using vectorized DP
        """
        # Initialization (source -> first row)
        self.dp[0, :] = self.energy[0, :]

        # DAG relaxation (row by row) - VECTORIZED
        for i in range(1, self.rows):
            for j in range(self.cols):
                # Left diagonal
                if j > 0:
                    cost_left = self.dp[i - 1, j - 1]
                else:
                    cost_left = np.inf

                # Straight up
                cost_up = self.dp[i - 1, j]

                # Right diagonal
                if j < self.cols - 1:
                    cost_right = self.dp[i - 1, j + 1]
                else:
                    cost_right = np.inf

                # Find minimum
                costs = [cost_left, cost_up, cost_right]
                min_idx = np.argmin(costs)
                min_cost = costs[min_idx]

                # Update DP and parent
                self.dp[i, j] = self.energy[i, j] + min_cost
                self.parent[i, j] = j + (min_idx - 1)  # -1, 0, or +1

    def compute_dp_vectorized(self):
        """
        Fully vectorized DP computation (faster)
        """
        # Initialization
        self.dp[0, :] = self.energy[0, :]

        for i in range(1, self.rows):
            # Create three shifted versions of previous row
            left = np.concatenate(([np.inf], self.dp[i - 1, :-1]))
            up = self.dp[i - 1, :]
            right = np.concatenate((self.dp[i - 1, 1:], [np.inf]))

            # Stack and find minimum
            stacked = np.vstack([left, up, right])
            self.parent[i, :] = np.argmin(stacked, axis=0) - 1 + np.arange(self.cols)
            self.dp[i, :] = self.energy[i, :] + np.min(stacked, axis=0)

    def get_min_seam(self):
        """
        Backtracks to retrieve the minimum-energy seam
        Returns: list of (row, col)
        """
        seam = []
        
        # Start from minimum in last row
        j = np.argmin(self.dp[-1])
        seam.append((self.rows - 1, j))

        # Backtrack upwards
        for i in range(self.rows - 1, 0, -1):
            j = self.parent[i, j]
            seam.append((i - 1, j))

        seam.reverse()
        return seam


def remove_seam(image, seam):
    """
    Remove a vertical seam from the image
    image: H x W x C numpy array
    seam: list of (row, col)
    returns: image with width reduced by 1
    """
    rows, cols = image.shape[:2]

    if image.ndim == 2:
        # Grayscale / energy map
        new_image = np.zeros((rows, cols - 1), dtype=image.dtype)
    else:
        # Color image
        new_image = np.zeros((rows, cols - 1, image.shape[2]), dtype=image.dtype)

    for i, (_, j) in enumerate(seam):
        # Copy everything except column j (FIXED: use axis=0 for columns)
        if image.ndim == 2:
            new_image[i, :] = np.delete(image[i, :], j)
        else:
            new_image[i, :, :] = np.delete(image[i, :, :], j, axis=0)

    return new_image



def scaling_k(image, k, use_vectorized=True):
    """
    Removes k vertical seams from image
    
    Args:
        image: RGB image array (H x W x 3)
        k: number of seams to remove
        use_vectorized: whether to use vectorized DP (faster)
    
    Returns:
        Resized image with width reduced by k pixels
    """
    current_image = image.copy()
    collected_seams=[]
    for iteration in range(k):
        if iteration % 10 == 0:
            print(f"Removing seam {iteration + 1}/{k}")
        
        sc = SeamsCalculator(current_image)
        
        if use_vectorized:
            sc.compute_dp_vectorized()
        else:
            sc.compute_dp()
        
        seam = sc.get_min_seam()
        collected_seams.append(seam)
        current_image = remove_seam(current_image, seam)
    
    return current_image,collected_seams



def show_comparison(original, resized, k):
    """
    Display side-by-side comparison of original and resized images
    
    Args:
        original: original image
        resized: image after seam carving
        k: number of seams removed
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title(f'Original Image\nSize: {original.shape[1]}x{original.shape[0]}', 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Resized image
    axes[1].imshow(resized)
    axes[1].set_title(f'After Seam Carving\nSize: {resized.shape[1]}x{resized.shape[0]}\n({k} seams removed)', 
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_seam_on_image(image, seams):
    
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)

    img_with_seam = image.copy()

    for seam in seams:
        for row, col in seam:
            img_with_seam[row, col] = [1, 0, 0]  

    return img_with_seam


def show_detailed_comparison(original, resized,seams, k):
    """
    Show detailed comparison with both images and their energy maps
    """
    import matplotlib.pyplot as plt
    
    # Calculate energy maps
    original_gray = color.rgb2gray(original)
    original_energy = filters.sobel(original_gray)
    
    resized_gray = color.rgb2gray(resized)
    resized_energy = filters.sobel(resized_gray)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title(f'Original Image ({original.shape[1]}x{original.shape[0]})', 
                         fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Original energy map
    axes[1, 0].imshow(show_seam_on_image(original_energy,seams))
    axes[1, 0].set_title('Original Energy Map with seams highlighted with red', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Resized image
    axes[0, 1].imshow(resized)
    axes[0, 1].set_title(f'Seam Carved ({resized.shape[1]}x{resized.shape[0]}) - {k} seams removed', 
                         fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Resized energy map
    axes[1, 1].imshow(resized_energy, cmap='gray')
    axes[1, 1].set_title('Resized Energy Map', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    from skimage import io
    import matplotlib.pyplot as plt
    
    # Load image
    image = io.imread('your_image.jpg')
    
    # Remove seams
    k = int(input("K = "))
    result, seams = scaling_k(image, k)
    
    # Show comparison
    #show_comparison(image, result, k)
    
    # Show detailed comparison with energy maps
    #show_detailed_comparison(image, result,seams, k)
    
    # Save result
    io.imsave('result.jpg', result)

    print("Import this module and use scaling_k(image, k) to resize your image")