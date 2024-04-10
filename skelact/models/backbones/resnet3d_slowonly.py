# from mmaction.models.builder import BACKBONES
from .builder import BACKBONES
from .resnet3d import ResNet3d


# ResNet3d = ResNet3d(depth=50,
#                     pretrained=None,
#                     stage_blocks=(4, 6, 3),
#                     pretrained2d=True,
#                     in_channels=3,
#                     num_stages=3,
#                     base_channels=32,
#                     out_indices=(2,),
#                     spatial_strides=(2, 2, 2),
#                     temporal_strides=(1, 1, 2),
#                     conv1_kernel=(3, 7, 7),
#                     conv1_stride=(1, 2),  # 1
#                     pool1_stride=(1, 2),  # 1
#                     advanced=False,
#                     frozen_stages=-1,
#                     inflate=(0, 1, 1),
#                     )

@BACKBONES.register_module()
class ResNet3dSlowOnly(ResNet3d):
    """SlowOnly backbone based on ResNet3d.

    Args:
        conv1_kernel (tuple[int]): Kernel size of the first conv layer. Default: (1, 7, 7).
        inflate (tuple[int]): Inflate Dims of each block. Default: (0, 0, 1, 1).
        **kwargs (keyword arguments): Other keywords arguments for 'ResNet3d'.
    """

    def __init__(self,
                 stage_blocks=(4, 6, 3),  # (4,6)
                 num_stages=3,  # 2
                 in_channels=3,
                 base_channels=32,
                 out_indices=(2,),  # (1,)
                 spatial_strides=(2, 2, 2),  # (2,2)
                 temporal_strides=(1, 1, 2),  # (1,2)
                 conv1_kernel=(1, 3, 3),  # (1, 21, 3),  # (1, 3, 3), (1, 7, 7)
                 inflate=(0, 1, 1), # (0,1)
                 **kwargs):
        super().__init__(
            stage_blocks=stage_blocks,
            num_stages=num_stages,
            in_channels=in_channels,
            base_channels=base_channels,
            out_indices=out_indices,
            spatial_strides=spatial_strides,
            temporal_strides=temporal_strides,
            conv1_kernel=conv1_kernel,
            inflate=inflate,
            **kwargs)

    # def __init__(self,
    #              conv1_kernel=(1, 7, 7),
    #              inflate=(0, 0, 1, 1),
    #              **kwargs):
    #     super().__init__(conv1_kernel=conv1_kernel, inflate=inflate, **kwargs)
