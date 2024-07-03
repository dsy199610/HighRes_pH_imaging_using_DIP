import numpy as np
import argparse
import random
import torch
import pathlib
from utils import logs
import logging
import datetime
import shutil
import sys
from torch.utils.tensorboard import SummaryWriter
from models.UNet_T1_T2_DWI import UNet_T1_T2_DWI
from loader.dataloader import Dataset_pH_mpMRI, interpolate_MRI
from torch.nn import functional as F
import matplotlib.pyplot as plt
import h5py


def create_data_loaders(args):
    Dataset = Dataset_pH_mpMRI(rabbit=args.rabbit, slice_=args.slice)
    loader = torch.utils.data.DataLoader(Dataset, batch_size=1, num_workers=args.num_workers)
    return loader


def save_python_script(args):
    ## copy training
    shutil.copyfile(sys.argv[0], str(args.exp_dir) + '/' + sys.argv[0])
    ## copy model
    shutil.copyfile('models/' + args.model + '.py', str(args.exp_dir) + '/' + args.model + '.py')
    ## copy loader
    shutil.copyfile('loader/dataloader.py', str(args.exp_dir) + '/dataloader.py')


def save_model(args, exp_dir, epoch, model, optimizer):
    logging.info('Saving trained model')

    ## create a models folder if not exists
    if not (args.exp_dir / 'models').exists():
        (args.exp_dir / 'models').mkdir(parents=True, exist_ok=True)

    if epoch == args.num_epochs - 1:
        torch.save({'epoch': epoch, 'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   f=str(exp_dir) + '/models/epoch' + str(epoch) + '.pt')
    logging.info('Done saving model')


def build_model(args):
    if args.model == 'UNet_T1_T2_DWI':
        model = UNet_T1_T2_DWI(in_channels=3, init_features=args.init_channels).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    return model, optimizer


def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    running_loss, running_loss_recon, running_loss_TV = 0, 0, 0
    total_data = len(data_loader)
    for iter, data in enumerate(data_loader):
        pH_clean, T1, T2, DWI, rabbit, slice = data
        pH_clean = pH_clean.float().cuda()
        T1 = T1.float().cuda()
        T2 = T2.float().cuda()
        DWI = DWI.float().cuda()

        if epoch == 0 and iter == 0:
            logging.info('--+' * 10)
            logging.info(f'T1 = {T1.shape}')
            logging.info(f'T2 = {T2.shape}')
            logging.info(f'DWI = {DWI.shape}')
            logging.info(f'GT = {pH_clean.shape} ')
            logging.info('--+' * 10)

        T1, T2, DWI = interpolate_MRI(T1), interpolate_MRI(T2), interpolate_MRI(DWI)

        if epoch == 0 and iter == 0:
            logging.info('--+' * 10)
            logging.info(f'T1 Cropped = {T1.shape}')
            logging.info(f'T2 Cropped = {T2.shape}')
            logging.info(f'DWI Cropped = {DWI.shape}')
            logging.info(f'GT Cropped = {pH_clean.shape} ')
            logging.info('--+' * 10)

        outputs, outputs_SR = model(T1, T2, DWI, (16, 16, 4))
        outputs = outputs.squeeze(-1)
        mask = (pH_clean != 0).float()
        outputs = torch.mul(outputs, mask)
        loss_recon = F.l1_loss(outputs, pH_clean)

        loss_TV = torch.mean(torch.abs(outputs_SR[:, :, :-1, :, :] - outputs_SR[:, :, 1:, :, :])) + \
                  torch.mean(torch.abs(outputs_SR[:, :, :, :-1, :] - outputs_SR[:, :, :, 1:, :])) + \
                  torch.mean(torch.abs(outputs_SR[:, :, :, :, :-1] - outputs_SR[:, :, :, :, 1:]))

        loss = loss_recon + args.TV_weight * loss_TV

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_loss_recon += loss_recon.item()
        running_loss_TV += loss_TV.item()

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{total_data:4d}] '
                f'Loss = {loss.item():.5g} '
                f'Loss Recon = {loss_recon.item():.5g} '
                f'Loss TV = {loss_TV.item():.5g} '
            )

    loss = running_loss / total_data
    loss_recon = running_loss_recon / total_data
    loss_recon_TV = running_loss_TV / total_data
    if writer is not None:
        writer.add_scalar('Dev/Training_Loss', loss, epoch)
        writer.add_scalar('Dev/Training_Loss_Recon', loss_recon, epoch)
        writer.add_scalar('Dev/Training_Loss_TV', loss_recon_TV, epoch)
    return loss


def visualize(args, epoch, model, data_loader, writer):
    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            pH_clean, T1, T2, DWI, rabbit, slice = data
            pH_clean = pH_clean.float().cuda()
            T1 = T1.float().cuda()
            T2 = T2.float().cuda()
            DWI = DWI.float().cuda()

            T1, T2, DWI = interpolate_MRI(T1), interpolate_MRI(T2), interpolate_MRI(DWI)

            outputs, outputs_SR = model(T1, T2, DWI, (16, 16, 4))
            outputs = outputs.squeeze(-1)
            mask = (pH_clean != 0).float()
            outputs = torch.mul(outputs, mask)
            mask_SR = mask.unsqueeze(-1).repeat_interleave(16, dim=2).repeat_interleave(16, dim=3).repeat_interleave(4, dim=4)
            outputs_SR = torch.mul(outputs_SR, mask_SR)

            outputs = outputs.squeeze().cpu().numpy()
            outputs_SR = outputs_SR.squeeze().cpu().numpy()
            pH_clean = pH_clean.squeeze().cpu().numpy()
            T1 = T1.squeeze().cpu().numpy()
            T2 = T2.squeeze().cpu().numpy()
            DWI = DWI.squeeze().cpu().numpy()

            pH_clean = pH_clean.repeat(16, axis=0).repeat(16, axis=1)
            outputs = outputs.repeat(16, axis=0).repeat(16, axis=1)

            (args.exp_dir / 'visualize_epoch').mkdir(parents=True, exist_ok=True)
            output_file = '%s/%s_%s_T1.png' % (str(args.exp_dir / 'visualize_epoch'), rabbit[0], slice[0])
            plt.imsave(output_file, np.concatenate((T1[:, :, 0], T1[:, :, 1], T1[:, :, 2], T1[:, :, 3]), axis=1), cmap='gray')
            output_file = '%s/%s_%s_T2.png' % (str(args.exp_dir / 'visualize_epoch'), rabbit[0], slice[0])
            plt.imsave(output_file, np.concatenate((T2[:, :, 0], T2[:, :, 1], T2[:, :, 2], T2[:, :, 3]), axis=1), cmap='gray')
            output_file = '%s/%s_%s_DWI.png' % (str(args.exp_dir / 'visualize_epoch'), rabbit[0], slice[0])
            plt.imsave(output_file, np.concatenate((DWI[:, :, 0], DWI[:, :, 1], DWI[:, :, 2], DWI[:, :, 3]), axis=1), cmap='gray')
            output_file = '%s/%s_%s_SR_epoch%s.png' % (str(args.exp_dir / 'visualize_epoch'), rabbit[0], slice[0], str(epoch))
            plt.imsave(output_file, np.concatenate((outputs_SR[:, :, 0], outputs_SR[:, :, 1],
                                                    outputs_SR[:, :, 2], outputs_SR[:, :, 3]), axis=1), cmap='jet', vmin=6.5, vmax=7.5)
            output_file = '%s/%s_%s_GT_epoch%s.png' % (str(args.exp_dir / 'visualize_epoch'), rabbit[0], slice[0], str(epoch))
            plt.imsave(output_file, np.concatenate((outputs_SR[:, :, 1], outputs, pH_clean), axis=1), cmap='jet', vmin=6.5, vmax=7.5)


def main_train(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    logs.set_logger(str(args.exp_dir / 'train.log'))
    logging.info('--' * 10)
    logging.info('%s create log file %s' % (datetime.datetime.now().replace(microsecond=0), str(args.exp_dir / 'train.log')))

    save_python_script(args)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    model, optimizer = build_model(args)

    logging.info('--' * 10)
    logging.info(args)
    logging.info('--' * 10)
    logging.info(model)
    logging.info('--' * 10)
    logging.info('Total parameters: %s' % sum(p.numel() for p in model.parameters()))

    loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    logging.info('--' * 10)
    start_training = datetime.datetime.now().replace(microsecond=0)
    logging.info('Start training at %s' % str(start_training))

    for epoch in range(0, args.num_epochs):
        logging.info('Current LR %s' % (scheduler.get_lr()[0]))
        torch.manual_seed(args.seed + epoch)
        train_loss = train_epoch(args, epoch, model, loader, optimizer, writer)
        save_model(args, args.exp_dir, epoch, model, optimizer)
        if epoch % 20 == 1:
            visualize(args, epoch, model, loader, writer)

        scheduler.step(epoch)
        logging.info('Epoch: %s Reduce LR to: %s' % (epoch, scheduler.get_lr()[0]))

    writer.close()


def main_evaluate(args):
    (args.exp_dir / 'results').mkdir(parents=True, exist_ok=True)
    logs.set_logger(str(args.exp_dir / 'eval.log'))
    logging.info('--' * 10)
    logging.info('%s create log file %s' % (datetime.datetime.now().replace(microsecond=0), str(args.exp_dir / 'eval.log')))

    loader = create_data_loaders(args)

    logging.info('--' * 10)
    start_eval = datetime.datetime.now().replace(microsecond=0)
    logging.info('Loading model %s' % str(args.checkpoint))
    logging.info('Start Evaluation at %s' % str(start_eval))

    path = 'data_HighRes/' + str(args.rabbit[0]) + '/' + str(args.slice[0]) + '_pH.mat'
    f = h5py.File(path, 'r')
    var_name, _ = list(f.items())[0]
    pH_HR = f[var_name][()]
    pH_HR = pH_HR[None, :, :]
    pH_HR = torch.from_numpy(pH_HR).float().cuda()

    model, optimizer = build_model(args)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.eval()
    total_data = len(loader)
    with torch.no_grad():
        for iter, data in enumerate(loader):
            pH_clean, T1, T2, DWI, rabbit, slice = data
            pH_clean = pH_clean.float().cuda()
            T1 = T1.float().cuda()
            T2 = T2.float().cuda()
            DWI = DWI.float().cuda()

            T1, T2, DWI = interpolate_MRI(T1), interpolate_MRI(T2), interpolate_MRI(DWI)

            outputs, outputs_SR = model(T1, T2, DWI, (16, 16, 4))
            outputs = outputs.squeeze(-1)
            mask = (pH_clean != 0).float()
            outputs = torch.mul(outputs, mask)
            outputs_HR = outputs_SR
            mask_SR = mask.unsqueeze(-1).repeat_interleave(16, dim=2).repeat_interleave(16, dim=3).repeat_interleave(4, dim=4)
            outputs_SR = torch.mul(outputs_SR, mask_SR)

            outputs_HR = outputs_HR[:, :, 50:-50, 50:-50, :]
            outputs_HR = F.avg_pool3d(outputs_HR, kernel_size=(12, 12, 2), stride=(12, 12, 2))
            outputs_HR = outputs_HR[:, :, :, :, 0]
            mask_HR = (pH_HR != 0).float()
            outputs_HR = torch.mul(outputs_HR, mask_HR)

            pH_inter = F.interpolate(pH_clean.unsqueeze(-1), size=(400, 400, 4), mode='nearest')
            pH_inter = pH_inter[:, :, 50:-50, 50:-50, :]
            pH_inter = F.avg_pool3d(pH_inter, kernel_size=(12, 12, 2), stride=(12, 12, 2))
            pH_inter = pH_inter[:, :, :, :, 0]

            MAE_recon = torch.sum(torch.abs(outputs - pH_clean)) / mask.sum()
            MAE_HighRes = torch.sum(torch.abs(outputs_HR - pH_HR)) / mask_HR.sum()

            MAE_HighRes_mean = torch.abs(outputs_HR - pH_HR)[outputs_HR != 0].mean()
            MAE_HighRes_std = torch.abs(outputs_HR - pH_HR)[outputs_HR != 0].std()
            logging.info(f'{rabbit[0]} {slice[0]} MAE (recon)={MAE_recon} MAE (HighRes)={MAE_HighRes} MAE (HighRes)={MAE_HighRes_mean}+-{MAE_HighRes_std}')

            outputs = outputs.squeeze().cpu().numpy()
            outputs_SR = outputs_SR.squeeze().cpu().numpy()
            pH_clean = pH_clean.squeeze().cpu().numpy()
            pH_HR = pH_HR.squeeze().cpu().numpy()
            outputs_HR = outputs_HR.squeeze().cpu().numpy()
            pH_inter = pH_inter.squeeze().cpu().numpy()
            T1 = T1.squeeze().cpu().numpy()
            T2 = T2.squeeze().cpu().numpy()
            DWI = DWI.squeeze().cpu().numpy()

            output_file = '%s/%s_%s.npz' % (str(args.exp_dir / 'results'), rabbit[0], slice[0])
            np.savez(output_file, pH_clean=pH_clean, pH_recon=outputs, pH_SR=outputs_SR, T1=T1, T2=T2, DWI=DWI, pH_HighRes=pH_HR, pH_HR_recon=outputs_HR)

            pH_clean_plot = pH_clean.repeat(16, axis=0).repeat(16, axis=1)
            outputs_plot = outputs.repeat(16, axis=0).repeat(16, axis=1)
            pH_HR_plot = pH_HR.repeat(12, axis=0).repeat(12, axis=1)
            outputs_HR_plot = outputs_HR.repeat(12, axis=0).repeat(12, axis=1)
            outputs_SR_plot = outputs_SR
            output_file = '%s/%s_%s_T1.png' % (str(args.exp_dir / 'results'), rabbit[0], slice[0])
            plt.imsave(output_file, np.concatenate((T1[:, :, 0], T1[:, :, 1], T1[:, :, 2], T1[:, :, 3]), axis=1), cmap='gray')
            output_file = '%s/%s_%s_T2.png' % (str(args.exp_dir / 'results'), rabbit[0], slice[0])
            plt.imsave(output_file, np.concatenate((T2[:, :, 0], T2[:, :, 1], T2[:, :, 2], T2[:, :, 3]), axis=1), cmap='gray')
            output_file = '%s/%s_%s_DWI.png' % (str(args.exp_dir / 'results'), rabbit[0], slice[0])
            plt.imsave(output_file, np.concatenate((DWI[:, :, 0], DWI[:, :, 1], DWI[:, :, 2], DWI[:, :, 3]), axis=1), cmap='gray')
            output_file = '%s/%s_%s.png' % (str(args.exp_dir / 'results'), rabbit[0], slice[0])
            plt.imsave(output_file, np.concatenate((outputs_SR_plot[:, :, 0], outputs_SR_plot[:, :, 1],
                                                    outputs_SR_plot[:, :, 2], outputs_SR_plot[:, :, 3],
                                                    outputs_plot, pH_clean_plot), axis=1), cmap='jet', vmin=6.7, vmax=7.3)
            output_file = '%s/%s_%s_HR.png' % (str(args.exp_dir / 'results'), rabbit[0], slice[0])
            plt.imsave(output_file, np.concatenate((outputs_HR_plot, pH_HR_plot), axis=1), cmap='jet', vmin=6.7, vmax=7.3)


def create_arg_parser():
    parser = argparse.ArgumentParser()

    ## train
    parser.add_argument('--lr', type=int, default=1e-3, help='learning rate')
    parser.add_argument('--lr-step-size', type=int, default=50, help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--num-epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--report-interval', type=int, default=1, help='Period of printing loss')
    parser.add_argument('--num-workers', default=0, type=int, help='number of works')
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')

    ## method
    parser.add_argument('--model', type=str, default='UNet_T1_T2_DWI', help='method')
    parser.add_argument('--init-channels', type=int, default=16, help='initial channels')
    parser.add_argument('--TV-weight', type=float, default=0.01, help='weight of TV loss')
    parser.add_argument('--rabbit', type=str, default=['Aleksi'], help='Rabbit name')
    parser.add_argument('--slice', type=str, default=['tumor'], help='slice name')
    parser.add_argument('--exp-dir', type=str, default='checkpoints/Aleksi_with_HighRes', help='Path to save models and results')

    ## evaluation
    parser.add_argument('--evaluate', action='store_true', help='If set, test mode')
    parser.add_argument('--checkpoint', type=str, default='epoch199.pt', help='path where the model was saved')

    args = parser.parse_args()

    args.exp_dir = args.exp_dir + '/' + args.model + '_channels' + str(args.init_channels) + '_TV' + str(args.TV_weight)
    args.exp_dir = args.exp_dir + '/' + str(args.rabbit[0]) + '_' + str(args.slice[0])
    args.checkpoint = args.exp_dir + '/models/' + args.checkpoint

    args.exp_dir = pathlib.Path(args.exp_dir)
    return args


if __name__ == '__main__':
    args = create_arg_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.evaluate:
        print('evaluate')
        print('--' * 10)
        main_evaluate(args)
    else:
        print('train')
        print('--' * 10)
        main_train(args)

# tensorboard --logdir='summary'