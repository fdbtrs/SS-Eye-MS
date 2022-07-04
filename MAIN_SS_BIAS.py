from tqdm import tqdm
import json
import os
import cv2
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ExponentialLR, StepLR

from util.misc import CSVLogger
from util.database_extended import ScleraSegmentationDataset
from model.modelMSEncoderSingleChannel import *

from util.utils import *
from util.losses import *

torch.cuda.empty_cache()

MODE = 'TEST'

print(f"MODE: {MODE}")
CONFIG_FILE = 'config.json'

# load config
with open(f'config/{CONFIG_FILE}', 'r') as f:
    cfg = json.load(f)

# here we create an identifier for the current training run based on the configuration
augs = []
if cfg['rotation_limit'] > 0:
    augs.append(f"rot{cfg['rotation_limit']}")
if cfg['elastic_transform_prob'] > 0:
    augs.append(f"elastic{int(cfg['elastic_transform_prob']*100)}")
cfg['augmentation'] = '_'.join(augs)

schedule_infos = []
if cfg['schedule'] == 'StepLR':
     schedule_infos.append(str(cfg['step_size']))
schedule_infos = '' if len(schedule_infos) == 0 else ('_s' + '_'.join(schedule_infos) + '_')

training_setup_name = '_'.join(cfg['train_set_names'])
ID = cfg['model'] + '_ext_' + training_setup_name + '_' + cfg['loss'] + '_' + cfg['augmentation'] + schedule_infos + '_c' + str(cfg['channel']) #+ '_r'+ str(cfg['reduction'])
cfg['weights'] = os.path.join(cfg['save_dir'], ID + '.pt')
cfg['test_output'] = os.path.join('result', ID)

# show config
for key, data in cfg.items():
    print(str(key) + ':  ' + str(data))

cudnn.benchmark = True
#torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)



# set model
if cfg['model'] == 'EyeMSResNetBlockMS':
    cnn = EyeMSResNetBlockMS(channel=int(cfg['channel']), reduction=int(cfg['reduction'])).cuda()
elif cfg['model'] == 'EyeMSResNetBlockMSEncoder':
    cnn = EyeMSResNetBlockMSEncoder(channel=int(cfg['channel']), reduction=int(cfg['reduction'])).cuda()

'''
print(cnn)

count_parameters(cnn.cpu())
exit()
'''

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=ScleraSegmentationDataset(mode='TRAIN', channel=int(cfg['channel']), rotation_limit=cfg['rotation_limit'], elastic_transform_prob=cfg['elastic_transform_prob'], set_names=cfg['train_set_names']),
                                           batch_size=int(cfg['batch']),
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=16)

# because no validation data for the extended data is available, we use the MOBIUS test data
val_loader  = torch.utils.data.DataLoader(dataset=ScleraSegmentationDataset(mode='VAL', channel=int(cfg['channel']), set_names=["VAL1"]),
                                          batch_size=1,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
# set loss function
losses = {
    'dice': DiceLoss(),
    'iou': IoULoss(),
    'l1': torch.nn.L1Loss(),
    'mse': torch.nn.MSELoss(),
    'f1': FBetaLoss(beta=1),
    'dicel1_1e-2': DiceL1Loss(weight=1e-2),
    'dicel1_1e-3': DiceL1Loss(weight=1e-3),
    'dicebce_1e-2': DiceBCELoss(weight=1e-2),
    'dicebce_1e-3': DiceBCELoss(weight=1e-3)
}

criterion = losses[cfg['loss'].lower()].cuda()

# set optimizer
# weight decay from https://arxiv.org/pdf/1512.03385.pdf, page 7, 4.2, below table 6

cnn_optimizer =  torch.optim.SGD(cnn.parameters(), lr=float(cfg['learning_rate']), momentum=0.9, nesterov=True, weight_decay=0.0) # 0.0001
#cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=float(cfg['learning_rate']))

if cfg['schedule'] == 'ExponentialLR':
    scheduler = ExponentialLR(cnn_optimizer, gamma=cfg['gamma'])
elif cfg['schedule'] == 'StepLR':
    scheduler = StepLR(cnn_optimizer, gamma=cfg['gamma'], step_size=cfg['step_size'])

# set metrics
eval_metric = FBetaScore(beta=1)

if cfg['weights'] and os.path.exists(cfg['weights']):

    if MODE != 'TEST':
        print("model already exists! either change MODE to TEST if you wanted to test or delete the old weights")
        exit(1)

    # load pretrained model
    cnn.load_state_dict(torch.load(cfg['weights']))
    print('pretrained model: ' + cfg['weights'] + ' loaded.')

def validation(loader, epoch):
    cnn.eval()

    save_path = os.path.join(cfg['test_output'], 'val', str(epoch))

    fscores = []

    for images, labels, _, f, _ in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        fscores.append(eval_metric(pred, labels).item())

        if epoch % 10 == 0:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            pred = pred.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
            img = images.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
            label = labels.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
            segmentation = np.round(pred)

            if cfg['channel'] == 1:
                img = np.stack((img,)*3, axis=-1)
            else:
                img = np.transpose(img, (1, 2, 0))
            label = np.stack((label,)*3, axis=-1)
            segmentation = np.stack((segmentation ,)*3, axis=-1)

            conc = np.concatenate([img, label , segmentation], axis=1)
            cv2.imwrite(os.path.join(save_path, f[0]), conc * 255)

    cnn.train()
    return np.mean(fscores)


def training():
    early_stopping = cfg['early_stopping']
    patience = cfg['patience']

    epochs_no_improvement = 0
    max_val_fscore = 0.0
    best_weights = None
    best_epoch = -1

    # set model to train mode
    cnn.train()

    filename = 'logs/' + ID + '.csv'
    csv_logger = CSVLogger(args=cfg, fieldnames=['epoch', 'train_acc', 'val_acc'], filename=filename)

    for epoch in range(1, 1+int(cfg['epochs'])):
        loss_total = 0.
        fscore_total = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels, _, _, _) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()
           
            cnn.zero_grad()

            pred = cnn(images)
            # pred= torch.round(pred)

            loss = criterion(pred, labels)
            loss.backward()
            cnn_optimizer.step()

            loss_total += loss.item()

            fscore_total += eval_metric(pred, labels).item()

            progress_bar.set_postfix(
                loss='%.5f' % (loss_total / (i + 1)),
                fscore='%.3f' % (fscore_total / (i +1 )))

        train_fscore = fscore_total / (i +1 )
        val_fscore = validation(val_loader, epoch)

        tqdm.write('fscore: %.5f' % (val_fscore))

        # scheduler.step(epoch)  # Use this line for PyTorch <1.4
        scheduler.step()  # Use this line for PyTorch >=1.4

        row = {'epoch': str(epoch), 'train_acc': str(train_fscore), 'val_acc': str(val_fscore)}
        csv_logger.writerow(row)

        if early_stopping:
            if val_fscore > max_val_fscore:
                max_val_fscore = val_fscore
                epochs_no_improvement = 0
                best_weights = cnn.state_dict()
                best_epoch = epoch
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement >= patience:
                print(f"EARLY STOPPING at {best_epoch}: {max_val_fscore}")
                break
        else:
            best_weights = cnn.state_dict()

    torch.save(best_weights, cfg['weights'])
    csv_logger.close()


def testing():
    cnn.eval()

    test_save_path = os.path.join(cfg['test_output'], 'testing')

    test_loader  = torch.utils.data.DataLoader(dataset=ScleraSegmentationDataset(mode='TEST', channel=int(cfg['channel']),
                                                                                 set_names=['MOBIUS', 'SMD', 'SLD']),
                                              batch_size=1,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)

    for images, _ , f, set_name, dir in test_loader:
        images = images.cuda()

        with torch.no_grad():
            pred = cnn(images)

        pred = pred.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
        img = images.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
        seg = np.round(pred)

        if cfg['channel'] == 1:
            img = np.stack((img,)*3, axis=-1)
        else:
            img = np.transpose(img, (1, 2, 0))

        # get grayscale un-thresholded "probability" maps
        pred = np.stack((pred,)*3, axis=-1)

        set_name = set_name[0]

        # crop and resize it to the requested size
        if set_name == "MOBIUS":
            pred = pred[55:-56,:,:]
            pred = cv2.resize(pred, (480, 360), interpolation=cv2.INTER_AREA)
        elif set_name == "SLD" or set_name == "SMD":
            pred = pred[32:-32,:,:]
            pred = cv2.resize(pred, (480, 360), interpolation=cv2.INTER_AREA)

        # round it to get segmentation mask (threshold = 0.5)
        seg = np.round(pred)

        # organizers want to have .png file instead of .jpg
        f = f[0].replace('.jpg', '.png')

        pred_path = os.path.join(test_save_path, set_name, 'Predictions', str(dir.numpy()[0]))

        if not os.path.isdir(pred_path):
            os.makedirs(pred_path)

        bin_path = os.path.join(test_save_path, set_name, 'Binarised', str(dir.numpy()[0]))

        if not os.path.isdir(bin_path):
            os.makedirs(bin_path)

        cv2.imwrite(os.path.join(pred_path, f), pred * 255.0)
        cv2.imwrite(os.path.join(bin_path, f), seg * 255.0 )


if __name__ == '__main__':

    if MODE.upper() == 'TRAIN':
        training()
    if MODE.upper() == 'TEST':
        testing()
