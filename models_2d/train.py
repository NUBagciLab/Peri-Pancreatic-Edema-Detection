import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import logging
import numpy as np
from sklearn.metrics import classification_report
from monai.networks.nets.resnet import ResNet,ResNetBlock,ResNetBottleneck
from pancreasdataset import PancreasDataset
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric,Metric
from get2dmodel import get_model

from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    Resize,
)
from monai.utils import set_determinism

def main():
    print_config()
    data_dir = '/data/Ziliang/LinTransUnetGererated/without_denoising/ROI_sclices'
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    modelSave_dir = '/data/Ziliang/Pancreas255_exp_res_2d/Vit_retrain'
    model_name = 'vit'

    set_determinism(seed=50)


    class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
    num_class = len(class_names)
    image_files = [
        [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
        for i in range(num_class)
    ]
    num_each = [len(image_files[i]) for i in range(num_class)]
    image_files_list = []
    image_class = []
    for i in range(num_class):
        image_files_list.extend(image_files[i])
        image_class.extend([i] * num_each[i])
    num_total = len(image_class)

    length = len(image_files_list)
    indices = np.arange(length)
    np.random.shuffle(indices)

    print(f"Total image count: {num_total}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")

    # transforms for train data and test data
    train_transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            Resize((224,224))
        ]
    )
    val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity(),Resize((224,224))])

    # transforms for prediction and label( this is in order to conduct evaluation)
    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])

    # training setting
    
    auc_metric = ROCAUCMetric()
    batch_size = 256
    max_epochs = 100
    val_interval = 1
    val_frac = 0.2
    val_size = int(val_frac * length)
    
    
    # record experiment
    
    logging.basicConfig(filename=modelSave_dir+'/exp_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Experiment started.")
    for i in range(5):
        print('--------------------------------Fold{}----------------------------'.format(i))
        logging.info('--------------------------------Fold{}----------------------------'.format(i))
        
        
        # split dataset
        val_split = int(i * val_frac * length)
        
        # test_indices = indices[:test_split]
        val_indices = indices[val_split:val_split+val_size]
        train_indices = np.concatenate((indices[:val_split],indices[val_split+val_size:]))
        
        train_x = [image_files_list[i] for i in train_indices]
        train_y = [image_class[i] for i in train_indices]
        val_x = [image_files_list[i] for i in val_indices]
        val_y = [image_class[i] for i in val_indices]

        # -------------------------------MODEL DEFINE!!!!--------------------------------------
    
        model = get_model(name=model_name,num_class=num_class).to(device)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), 1e-5)

        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []

        train_ds = PancreasDataset(train_x, train_y, train_transforms)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=10)

        val_ds = PancreasDataset(val_x, val_y, val_transforms)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=10)

        
        # pytorch training epoch
        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                # outputs,_ = model(inputs) # use this line when train VIT
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
                epoch_len = len(train_ds) // train_loader.batch_size
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    y_pred = torch.tensor([], dtype=torch.float32, device=device)
                    y = torch.tensor([], dtype=torch.long, device=device)
                    for val_data in val_loader:
                        val_images, val_labels = (
                            val_data[0].to(device),
                            val_data[1].to(device),
                        )
                        y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                        # y_pred = torch.cat([y_pred, model(val_images)[0]], dim=0) # use this line when train VIT
                        y = torch.cat([y, val_labels], dim=0)
                    y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
                    y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                    auc_metric(y_pred_act, y_onehot)
                    result = auc_metric.aggregate()
                    auc_metric.reset()
                    del y_pred_act, y_onehot
                    metric_values.append(result)
                    acc_value = torch.eq(y_pred.argmax(dim=1), y)
                    acc_metric = acc_value.sum().item() / len(acc_value)
                    if result > best_metric:
                        best_metric = result
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), os.path.join(modelSave_dir, "fold{}_best_metric_model.pth".format(i)))
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                        f" current accuracy: {acc_metric:.4f}"
                        f" best AUC: {best_metric:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )
                    logging.info(" current accuracy: {:.4f}".format(acc_metric),exc_info=True)
                    logging.info("classification_report:\n"+classification_report(y_pred.cpu().argmax(dim=1), y.cpu(), target_names=class_names, digits=4),exc_info=True)
                    print(classification_report(y_pred.cpu().argmax(dim=1), y.cpu(), target_names=class_names, digits=4))
        print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")



if __name__ =='__main__':
    main()