"""Stage 1A selects MiniBatchKMeans because full K-means is not GPU-notebook friendly."""

from EncDecPipeline.Codebooks.minibatch_kmeans_vq import MiniBatchKMeansVQ

KMeansVQ = MiniBatchKMeansVQ
