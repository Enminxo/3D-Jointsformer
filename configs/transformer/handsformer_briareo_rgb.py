
_base_ = [
    '../_base_/models/briareo_handsformer.py',
    '../_base_/datasets/briareo_rgb.py',
    '../_base_/schedules/adam_500e.py',
    '../_base_/default_runtime.py'
]


# runtime settings
work_dir = './work_dirs/briareo/RGB/handsformer'
