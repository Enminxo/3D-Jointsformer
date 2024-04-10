# runtime settings
checkpoint_config = dict(interval=5)  # save each 5 epochs
evaluation = dict(interval=5, metrics=['top_k_accuracy'])

# config to register logger hook
# log_config = dict(interval=100, #interval to print th elog
#                   hooks=[dict(type='TextLoggerHook')]) # 500 epochs
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])  # 250 epochs

# hooks= [dict(type='TensorboardLoggerHook') ]

dist_params = dict(backend='nccl')  # Parameters to set up distributed training
log_level = 'INFO'  # The output level of the log.
load_from = None
# Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
resume_from = None
# workflow = [('train', 1)] # workflow means the order of training and val
workflow = [('train', 1), ('val', 1)] 
