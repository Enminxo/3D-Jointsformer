import torch.nn as nn
from utils import train, validate
from dataGen import *
from utils import save_model, save_plots, SaveBestModel
from torchvision.models.video import r3d_18, R3D_18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""video 3D RESNET 18 """
briareo_rgb_train = '/home/enz/code/skelact/data/briareo/xsub/train.pkl'
briareo_rgb_test = '/home/enz/code/skelact/data/briareo/xsub/test.pkl'
briareo_rgb_val = '/home/enz/code/skelact/data/briareo/xsub/val.pkl'

"""Load the dataset"""
# path = 'data/ntu_prueba/nturgb+d_skeletons_60_3d/xsub/train.pkl'
# test_path = 'data/ntu_prueba/nturgb+d_skeletons_60_3d/xsub/val.pkl'
# train_data = CustomDataset(path, data_type='train')
# val_data = CustomDataset(path, data_type='val')
# test_data = CustomDataset(test_path, data_type='test')
train_data = CustomDataset(pickle_path=briareo_rgb_train,
                           data_type='train',
                           centercrop=center_crop(clip_ratio=1),
                           resizeframes=resize_frames(clip_len=35)
                           )
val_data = CustomDataset(pickle_path=briareo_rgb_val,
                         data_type='val',
                         centercrop=center_crop(clip_ratio=1),
                         resizeframes=resize_frames(clip_len=35)
                         )
test_data = CustomDataset(pickle_path=briareo_rgb_test,
                          data_type='test',
                          centercrop=center_crop(clip_ratio=1),
                          resizeframes=resize_frames(clip_len=35)
                          )
BATCH = 256
train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(val_data, batch_size=BATCH, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=BATCH, shuffle=True, num_workers=0)

"""Instantiate the model"""
weights = R3D_18_Weights.DEFAULT

model = r3d_18(weights=weights)
model = model.double().to(device)

# summary(model, (1, 32, 21, 3))
# initialize SaveBestModel class
save_best_model = SaveBestModel()

"""Start training"""
# lists to store per-epoch loss and accuracy values
train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
# start = time.time()

"""Parameters"""
epochs = 30
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# StepLR scheduler starts with a lr and decreases by a factor of 0.1 every 10 epochs.

for epoch in range(epochs):
    print('\n', f"Epoch {epoch + 1} of {epochs}:")
    train_epoch_loss, train_epoch_accuracy = train(device, model, train_dataloader, optimizer, criterion)
    val_epoch_loss, val_epoch_accuracy = validate(device, model, valid_dataloader, criterion)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    scheduler.step()
    print('LR:', optimizer.param_groups[0]['lr'])
    # print('LR:', scheduler.get_last_lr())
    print(f"Train epoch Loss: {train_epoch_loss:.3f}, Train epoch Acc: {train_epoch_accuracy:.2f}")
    print(f'Val epoch Loss: {val_epoch_loss:.3f}, Val epoch Acc: {val_epoch_accuracy:.2f}')
    # save the best model till now if we have the least loss in the current epoch
    save_best_model(
        val_epoch_loss, epoch, model, optimizer, criterion
    )

# end = time.time()
# print('\n', f"Training time: {(end - start) / 60:.2f} minutes")
# save the trained model weights for a final time
output_path = './resnet_briareo_rgb'
if not os.path.exists(output_path):
    os.makedirs(output_path)
save_model(output_path, epochs, model, optimizer, criterion)
# save the loss and accuracy plots
save_plots(output_path, train_accuracy, val_accuracy, train_loss, val_loss)

print('TRAINING COMPLETE')
