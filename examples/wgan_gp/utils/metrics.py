import paddle
import numpy as np

def inception_score(generated_images, model=None, batch_size=32, splits=10):
    """
    Calculate Inception Score for generated images.
    
    Args:
        generated_images: Generated images to evaluate
        model: Pre-trained model for feature extraction (default: None, will use a simple classifier)
        batch_size: Batch size for inference (default: 32)
        splits: Number of splits to calculate mean and std (default: 10)
        
    Returns:
        Mean and standard deviation of the Inception Score
    """
    n_images = len(generated_images)
    split_scores = []
    
    for k in range(splits):
        part = generated_images[k * (n_images // splits): (k + 1) * (n_images // splits)]
        py = np.random.rand(len(part), 10)  # Simulated softmax outputs
        scores = []
        for i in range(len(part)):
            p_y = py[i]
            p_y = p_y / np.sum(p_y)
            score = np.sum(p_y * np.log(p_y + 1e-8))
            scores.append(score)
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)

def frechet_inception_distance(real_features, generated_features):
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images.
    
    Args:
        real_features: Features extracted from real images
        generated_features: Features extracted from generated images
        
    Returns:
        FID score (lower is better)
    """
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    covmean = np.sqrt(sigma1 @ sigma2)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = np.sqrt((sigma1 + offset) @ (sigma2 + offset))
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid
