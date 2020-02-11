from inferno.io.core import ZipReject, Concatenate
from inferno.io.transform import Compose
from inferno.io.transform.generic import AsTorchBatch
from inferno.io.transform.volume import RandomFlip3D, VolumeAsymmetricCrop
from inferno.io.transform.image import RandomRotate, ElasticTransform
from inferno.utils.io_utils import yaml2dict

from torch.utils.data.dataloader import DataLoader

from neurofire.datasets.loader import RawVolume
from neurofire.datasets.loader import SegmentationVolume
from neurofire.transform.affinities import affinity_config_to_transform
from neurofire.transform.volume import RandomSlide, RejectNonZeroThreshold


class PlatynerisDataset(ZipReject):
    def __init__(self, name, volume_config, slicing_config, master_config=None):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)
        assert 'raw' in volume_config
        assert 'segmentation' in volume_config

        # Get kwargs for raw volume
        raw_volume_kwargs = dict(volume_config.get('raw'))
        raw_volume_kwargs.update(slicing_config)
        # Build raw volume
        self.raw_volume = RawVolume(name=name, **raw_volume_kwargs)

        # Get kwargs for segmentation volume
        segmentation_volume_kwargs = dict(volume_config.get('segmentation'))
        segmentation_volume_kwargs.update(slicing_config)
        self.affinity_config = segmentation_volume_kwargs.pop('affinity_config', None)
        # Build segmentation volume
        self.segmentation_volume = SegmentationVolume(name=name,
                                                      **segmentation_volume_kwargs)

        rejection_threshold = volume_config.get('rejection_threshold', 0.1)
        print("reject at", rejection_threshold)
        # Initialize zipreject
        rejecter = RejectNonZeroThreshold(rejection_threshold)
        super(PlatynerisDataset, self).__init__(self.raw_volume,
                                                self.segmentation_volume,
                                                sync=True,
                                                rejection_dataset_indices=1,
                                                rejection_criterion=rejecter)
        # Set master config (for transforms)
        self.master_config = {} if master_config is None else master_config
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose(RandomFlip3D(),
                             RandomRotate())

        # Elastic transforms can be skipped by setting elastic_transform to false in the
        # yaml config file.
        if self.master_config.get('elastic_transform'):
            elastic_transform_config = self.master_config.get('elastic_transform')
            transforms.add(ElasticTransform(alpha=elastic_transform_config.get('alpha', 2000.),
                                            sigma=elastic_transform_config.get('sigma', 50.),
                                            order=elastic_transform_config.get('order', 0)))

        # TODO doesn't look like we have misalignment, so should be fine
        # if we do not use random slides
        # random slide augmentation
        if self.master_config.get('random_slides', False):
            assert False, "No random slides for now"
            ouput_shape = self.master_config.get('shape_after_slide', None)
            max_misalign = self.master_config.get('max_misalign', None)
            transforms.add(RandomSlide(output_image_size=ouput_shape,
                                       max_misalign=max_misalign))

        # if we compute the affinities on the gpu, or use the feeder for validation only,
        # we don't need to add the affinity transform here
        if self.affinity_config is not None:
            # we apply the affinity target calculation only to the segmentation (1)
            transforms.add(affinity_config_to_transform(apply_to=[1],
                                                        **self.affinity_config))

        # Next: crop invalid affinity labels and elastic augment reflection padding assymetrically
        crop_config = self.master_config.get('crop_after_target', {})
        if crop_config:
            # One might need to crop after elastic transform to avoid edge artefacts of affinity
            # computation being warped into the FOV.
            transforms.add(VolumeAsymmetricCrop(**crop_config))

        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        name = config.get('dataset_name')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        master_config = config.get('master_config')
        return cls(name, volume_config=volume_config,
                   slicing_config=slicing_config,
                   master_config=master_config)


class PlatynerisDatasets(Concatenate):
    def __init__(self, names,
                 volume_config,
                 slicing_config,
                 master_config=None):
        # Make datasets and concatenate
        if names is None:
            datasets = [PlatynerisDataset(name=None,
                                          volume_config=volume_config,
                                          slicing_config=slicing_config,
                                          master_config=master_config)]
        else:
            datasets = [PlatynerisDataset(name=name,
                                          volume_config=volume_config,
                                          slicing_config=slicing_config,
                                          master_config=master_config)
                        for name in names]
        # Concatenate
        super(PlatynerisDatasets, self).__init__(*datasets)
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = AsTorchBatch(3)
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        names = config.get('dataset_names')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        master_config = config.get('master_config')
        return cls(names=names, volume_config=volume_config,
                   slicing_config=slicing_config, master_config=master_config)


def get_platyneris_loaders(config):
    """
    Gets Platyneris loaders given a the path to a configuration file.

    Parameters
    ----------
    config : str or dict
        (Path to) Data configuration.

    Returns
    -------
    torch.utils.data.dataloader.DataLoader
        Data loader built as configured.
    """
    config = yaml2dict(config)
    datasets = PlatynerisDatasets.from_config(config)
    loader = DataLoader(datasets, **config.get('loader_config'))
    return loader
