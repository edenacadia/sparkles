import matplotlib.pyplot as plt


def plot_pca_basis(pca_img, mask_nan, spark_params=""):
    plt.figure(figsize=(7, 3))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"KL mode {i}")
        plt.imshow(pca_img[i]*mask_nan)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.box(False)
    plt.suptitle(f"KL modes {spark_params}", y=0.92)
    plt.tight_layout()
    plt.show()
    return