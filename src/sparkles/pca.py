import numpy as np

def pca_basis(data_ar, klip=3):
    # need to delete the mean here:
    data_ar = data_ar - np.average(data_ar, axis=0)
    # generate a PCA basis from these files
    K, x, y = data_ar.shape
    N = x*y
    image_shape = (x, y)
    reference_lab = data_ar.reshape(K, N)
    # calc covariance
    E = np.cov(reference_lab) * (N - 1)
    # find eigenvalues and eigenvectors
    lambda_values_out, C_out = np.linalg.eigh(E)
    lambda_values = np.flip(lambda_values_out)
    C = np.flip(C_out, axis=1)
    # generate the KL basis 
    Z_KL_lab = reference_lab.T @ (C * np.power(lambda_values, -1/2))
    Z_KL_lab_truc = Z_KL_lab[:,:klip]
    # reshape for actual images
    Z_KL_lab_images = Z_KL_lab.T.reshape((reference_lab.shape[0],) + image_shape)
    Z_KL_lab_images_truc = Z_KL_lab_images[:klip,:,:]
    return Z_KL_lab_truc, Z_KL_lab_images_truc

def pca_projection(data_ar, ref_pca):
    # project each frame of the data onto the PCA basis
    data_ar = data_ar - np.average(data_ar, axis=0)
    # reshaping 
    reference = data_ar.reshape(data_ar.shape[0], data_ar.shape[1]*data_ar.shape[2])
    # project each frame onto PCA basis
    projection = np.array([ref_pca.T @ ref for ref in reference])
    return projection


def proj_norm(data_proj, lab_norm):
    """
    Take in a data array, normalize by the lab_projections
    This is dependent on roll, so we will have 4 options for each PCA basis, per frame
    data_proj: [N, klip]
    data_proj_norm: [N, klip, 4] 
    """
    n = data_proj.shape[0]
    extended_norm = np.array([np.repeat(lab_norm[:,i].reshape(1,4), (n//4 +1), axis=0).flatten()[:n] for i in range(3)]).T

    # TODO: some complicated rolling and selection


def proj_rms(data_proj, rms_n=1000):
    """
    This compares the RMS between lab and sky projections, picked with 
    """
    # TODO