from __future__ import print_function, division

import torch
import time
import other.utils as utils


def train_model(model, dataLoaders, dataset_sizes, Map, criterion,
                validation_criterion, optimizer, scheduler, arg):

    since = time.time()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
                    # loss = criterion(predicted, heatmaps)
                    # weight_loss = loss * weights

                    ###############################################
                    # SC_CNN
                    weight_loss = criterion(predicted,
                                            heatmaps,
                                            weights)

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

            print(f"{phase} -> Loss: {epoch_loss}")

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


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
           time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:4f}'.format(best_loss))
