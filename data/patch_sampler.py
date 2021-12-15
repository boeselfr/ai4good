import ee
import tensorflow as tf


class PatchSampler:
    def __init__(
        self,
        kernel_size: int,
        scale: int,
        num_samples_per_shard: int,
        num_shards: int,
    ):
        lists = ee.List.repeat(ee.List.repeat(1, kernel_size), kernel_size)
        self.kernel = ee.Kernel.fixed(kernel_size, kernel_size, lists)

        self.scale = scale
        self.num_shards = num_shards
        self.num_samples_per_shard = num_samples_per_shard

    def sample_image(
        self,
        image: ee.Image,
        region: ee.Geometry,
    ) -> ee.FeatureCollection:
        array_image = image.float().neighborhoodToArray(self.kernel)

        # Export all the training data (in many pieces), with one task per image
        samples = ee.FeatureCollection([])
        for j in range(self.num_shards):
            sample = array_image.sample(
                region=region,
                scale=self.scale,
                numPixels=self.num_samples_per_shard,
                seed=j,
                tileScale=8,
            )
            samples = samples.merge(sample)

        return samples
