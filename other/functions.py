from __future__ import print_function, division

import torch
import time
import other.utils as utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from dataset.ColorNormalization import steinseperation


def train_model(model, dataLoaders, dataset_sizes, Map, criterion,
                validation_criterion, optimizer, scheduler, arg):

    since = time.time()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    writer = SummaryWriter(log_dir=arg.log_dir)

    # load model
    start_epoch, best_loss, model, optimizer, scheduler = utils.load_model(arg.load_name,
                                                                           model,
                                                                           optimizer,
                                                                           scheduler)

    for epoch in range(start_epoch, arg.epoch):

        print('Epoch {}/{}'.format(epoch, arg.epoch - 1))
        print('-' * 10, "\n")

        running_loss = 0.0

        for phase in ['Train', 'Validation']:

            if phase == 'Train':
                # Set model to training mode
                model.train(True)

            else:
                # Set model to evaluate mode
                model.train(False)

            # Iterate over data.
            for inputs, heatmaps, epsilons in dataLoaders[phase]:

                inputs, heatmaps, epsilons = inputs.to(device), heatmaps.to(device), epsilons.to(device)
                # forward
                points, h_value = model(inputs)

                ############################    Auto Grad by me   #######################################
                predicted = Map(points,
                                h_value,
                                device,
                                arg.d,
                                arg.heatmap_size,
                                arg.version)

                ############################ Auto Grad by Pytorch #######################################
                # predicted = utils.heat_map_tensor(points, h_value, device, arg.d, arg.out_size)
                # predicted = utils.heat_map_tensor_version2(points, h_value, device, arg.d, arg.out_size)

                predicted = predicted.view(predicted.shape[0], -1)
                heatmaps  = heatmaps.view(heatmaps.shape[0], -1)

                feature  = heatmaps.shape[1]
                epsilons = epsilons.unsqueeze(1)
                epsilons = torch.repeat_interleave(epsilons, repeats=feature, dim=1)

                weights = heatmaps + epsilons

                if phase == 'Train':

                    ###############################################
                    # BCE
                    loss = criterion(predicted, heatmaps)
                    weight_loss = loss * weights

                    ###############################################
                    # SC_CNN
                    # weight_loss = criterion(predicted,
                    #                         heatmaps,
                    #                         weights)

                    # Sum over one data
                    # Average over different data
                    loss = torch.sum(weight_loss, dim=1)
                    loss = torch.mean(loss)
                    # avg_loss = torch.sum(sum_loss)

                else:

                    loss = validation_criterion(predicted,
                                                heatmaps)
                    loss = torch.sqrt(loss)
                    loss = torch.sum(loss, dim=1)
                    loss = torch.mean(loss)
                    # loss = torch.sum(loss)

                running_loss += loss.item() * inputs.shape[0]

                if phase == 'Train':

                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    # Calculate gradient respect to parameters
                    loss.backward()
                    # Update parameters
                    optimizer.step()

                # Empty Catch
                torch.cuda.empty_cache()

            epoch_loss = running_loss / dataset_sizes[phase]

            print(f"{phase} -> Loss: {epoch_loss}", flush=True)

            writer.add_scalars(f"{arg.save_name}/loss",
                               {
                               f"{phase}": epoch_loss
                               },
                               epoch)
            writer.flush()

            if phase == 'Train':
                    scheduler.step()

            if phase == 'Validation' and epoch_loss < best_loss:

                best_loss = epoch_loss
                # Save the best model
                utils.save_model(epoch,
                                 model,
                                 optimizer,
                                 scheduler,
                                 epoch_loss,
                                 arg.save_name)

    writer.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
           time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:4f}'.format(best_loss))



def test_model(image_path, data, model, arg):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    img_transform = utils.transform(arg.version)
    model.train(False)

    _, _, _, stain, _ = steinseperation.stainsep(image_path)
    H_Channel = stain[0]

    cropped, coords = utils.patch_extraction(data,
                                             H_Channel,
                                             arg.patch_size,
                                             arg.stride_size,
                                             arg.version)

    H, W, _ = data.shape
    cell    = np.zeros((H, W))
    count   = np.zeros((H, W))

    [H_prime, W_prime] = arg.heatmap_size

    for img, coord in zip(cropped, coords):

        img = img_transform(np.uint8(img))
        img = img.float()
        img = img.to(device)
        img = img.unsqueeze(0)

        point, h = model(img)

        heat_map = utils.heat_map_tensor(point.view(-1, 2),
                                         h.view(-1, 1),
                                         device,
                                         arg.d,
                                         arg.heatmap_size)
        heatmap  = heat_map.cpu().detach().numpy().reshape((H_prime, W_prime))

        start_H, end_H, start_W, end_W = utils.find_out_coords(coord,
                                                               arg.patch_size,
                                                               arg.heatmap_size)

        cell[start_H:end_H, start_W:end_W] += heatmap

        idx = np.argwhere(heatmap != 0)
        count[idx[:,0]+start_H, idx[:,1]+start_W] += 1

    count[count==0] = 1
    cell = np.divide(cell, count)

    return cell
