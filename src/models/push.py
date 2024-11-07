import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from .components.find_nearest import to_normalized_tensor
from .components.helpers import makedir, find_continuous_high_activation_crop

from .components.constant import SEASFIRE_ID_MAPPING, SEASFIRE_CATEGORIES

from src.datamodules.seasfire_local_global_datamodule import SeasFireLocalGlobalDataModule

to_tensor = transforms.ToTensor()


# push each prototype to the nearest patch in the training set
def push_prototypes(dataset: SeasFireLocalGlobalDataModule,
                    prototype_network_parallel,  # pytorch network with prototype_vectors
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None,  # if not None, prototypes will be saved here
                    epoch_number=None,  # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True,  # which class the prototype image comes from
                    log=print,
                    pascal=False,
                    input_var=None,
                    prototype_activation_function_in_numpy=None):

    #done defined my mapping
    ID_MAPPING = SEASFIRE_ID_MAPPING
    CATEGORIES = SEASFIRE_CATEGORIES

    cls2name = ID_MAPPING

    if hasattr(prototype_network_parallel, 'module'):
        prototype_network_parallel = prototype_network_parallel.module

    prototype_network_parallel.eval()
    log.info('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.prototype_shape
    n_prototypes = prototype_network_parallel.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-' + str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    num_classes = prototype_network_parallel.num_classes


    log.info(f'Updating prototypes...')

    for push_iter in tqdm(range(len(dataset.data_train)), desc='updating prototypes', total=len(dataset.data_train)):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        img, gt_ann= dataset.data_train[push_iter]

        with torch.no_grad():
            update_prototypes_on_image(dataset,
                                       img,
                                       push_iter,
                                       prototype_network_parallel,
                                       global_min_proto_dist,
                                       global_min_fmap_patches,
                                       proto_rf_boxes,
                                       proto_bound_boxes,
                                       cls2name=cls2name,
                                       img_y=gt_ann,
                                       num_classes=num_classes,
                                       prototype_layer_stride=prototype_layer_stride,
                                       dir_for_saving_prototypes=proto_epoch_dir,
                                       prototype_img_filename_prefix=prototype_img_filename_prefix,
                                       prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                       prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
                                       input_var=input_var)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir,
                             proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log.info('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    prototype_network_parallel.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())

    # de-duplicate prototypes
    _, unique_index = np.unique(prototype_update, axis=0, return_index=True)
    duplicate_idx = [i for i in range(prototype_network_parallel.num_prototypes) if i not in unique_index]

    log.info(f'Removing {len(duplicate_idx)} duplicate prototypes.')
    prototype_network_parallel.prune_prototypes(duplicate_idx)
    os.makedirs(root_dir_for_saving_prototypes, exist_ok=True)
    with open(os.path.join(root_dir_for_saving_prototypes, 'unique_prototypes.json'), 'w') as fp:
        json.dump([int(i) for i in sorted(unique_index)], fp)

    end = time.time()
    log.info('\tpush time: \t{0}'.format(end - start))


# update each prototype for current search image
def update_prototypes_on_image(dataset: SeasFireLocalGlobalDataModule,
                               img: Image,
                               start_index_of_search_batch,
                               ppnet,
                               global_min_proto_dist,  # this will be updated
                               global_min_fmap_patches,  # this will be updated
                               proto_rf_boxes,  # this will be updated
                               proto_bound_boxes,  # this will be updated
                               cls2name,
                               img_y=None,  # required if class_specific == True
                               num_classes=None,  # required if class_specific == True
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None,
                               patch_size=1,
                               input_var=None):
    # segmentation_result = get_image_segmentation(dataset, ppnet, img,
    # window_size=dataset.window_size,
    # window_shift=512,
    # batch_size=4)
    device = torch.device("cuda:0")


    img_y = torch.Tensor(img_y).to(device) 
    img_tensor = torch.Tensor(img).unsqueeze(0)

    img_tensor = img_tensor.to(device)
    conv_features = ppnet.conv_features(img_tensor) #[1, 64, 128, 128]
    # save RAM
    del img_tensor

    logits, distances = ppnet.forward_from_conv_features(conv_features)
    distances = - distances
    model_output_height = conv_features.shape[2]
    model_output_width = conv_features.shape[3]

    img_height = img_y.shape[0]
    img_width = img_y.shape[1]

    #here it is 1
    patch_height = img_height / model_output_height
    patch_width = img_width / model_output_width

    # conv_features = torch.nn.functional.interpolate(conv_features, size=(1024, 2048),
    # mode='bilinear', align_corners=False)
    # distances = torch.nn.functional.interpolate(distances, size=(1024, 2048),
    # mode='bilinear', align_corners=False)

    protoL_input_ = conv_features[0].detach().cpu().numpy()  # [nb_feature_proto, img_size]
    proto_dist_ = distances[0].permute(1, 2, 0).detach().cpu().numpy() #[img_size, nb_proto]

    del conv_features, distances

    class_to_patch_index_dict = {key: set() for key in range(num_classes)}
    res = []

    for pixel_i in range(img_y.shape[0]):
        patch_i = int(pixel_i / patch_height)
        for pixel_j in range(img_y.shape[1]):
            patch_j = int(pixel_j / patch_width)
            pixel_cls = int(img_y[pixel_i, pixel_j].item())

            #if pixel_cls > 0: #done  took off this as i have no void so 0 is no fire 1 is fire and no -1
            class_to_patch_index_dict[pixel_cls].add((patch_i, patch_j))
    # proto 6 71 16 0.22671318 [563, 572, 127, 136]
    # rf_start_h_index = int(patch_i * patch_height)
    # patch_i rf_start_index/patch_height
    # rf_end_h_index = int(patch_i * patch_height + patch_height) + 1

    class_to_patch_index_dict = {k: list(v) for k, v in class_to_patch_index_dict.items()}

    prototype_shape = ppnet.prototype_shape
    n_prototypes = prototype_shape[0]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]  # nb_features
    # get the whole image

    original_img_j = np.transpose(img, (2, 0, 1))  #to do understand why it changed 128,8,128
    original_img_j = to_tensor(original_img_j).detach().cpu()

    original_img_height = original_img_j.shape[0]
    original_img_width = original_img_j.shape[1]

    # get segmentation map
    logits = logits.permute(0, 3, 1, 2)
    logits_inter = torch.nn.functional.interpolate(logits, size=original_img_j.shape[:2],
                                                   mode='bilinear', align_corners=False)
    logits_inter = logits_inter[0]
    pred = torch.argmax(logits_inter, dim=0).cpu().detach().numpy()

    for j in range(n_prototypes):
        # target_class is the class of the class_specific prototype
        target_class = torch.argmax(ppnet.prototype_class_identity[j]).item() #imatrix 1 0 0 1 
        target_class = 1 - target_class # to do so that the 10 first to fire 10 other non fire
        # if there are no pixels of the target_class in this image
        # we go on to the next prototype
        if len(class_to_patch_index_dict[target_class]) == 0:
            print("skip")
            continue
        # proto_dist_.shape = (patches_rows, patches_cols, n_prototypes)
        all_dist = np.asarray([proto_dist_[patch_i, patch_j, j]
                               for patch_i, patch_j in class_to_patch_index_dict[target_class]])  # get the distance for each pixel of the corresponding prototype
        batch_argmin_proto_dist = np.argmin(all_dist)
        batch_min_proto_dist = all_dist[batch_argmin_proto_dist]

        if batch_min_proto_dist < global_min_proto_dist[j]:
            '''
            change the argmin index from the index among
            images of the target class to the index in the entire search
            batch
            '''
            batch_argmin_proto_dist = class_to_patch_index_dict[target_class][batch_argmin_proto_dist]
            patch_i, patch_j = batch_argmin_proto_dist

            # retrieve the corresponding feature map patch
            batch_min_fmap_patch_j = protoL_input_[:, patch_i:patch_i + 1, patch_j:patch_j + 1]

            # batch_min_fmap_patch_j.shape = (64, 1, 1)
            global_min_proto_dist[j] = batch_min_proto_dist
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            # get the receptive field boundary of the image patch
            # that generates the representation
            # protoL_rf_info = ppnet.proto_layer_rf_info
            # rf_prototype_j = compute_rf_prototype((search_batch.shape[2], search_batch.shape[3]),
            # batch_argmin_proto_dist, protoL_rf_info)

            rf_start_h_index = int(patch_i * patch_height)
            rf_end_h_index = int(patch_i * patch_height + patch_height) + 1

            rf_start_w_index = int(patch_j * patch_width)
            rf_end_w_index = int(patch_j * patch_width + patch_width) + 1

            rf_prototype_j = [0, rf_start_h_index, rf_end_h_index, rf_start_w_index, rf_end_w_index]

            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                       rf_prototype_j[3]:rf_prototype_j[4], :]

            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]

            if proto_rf_boxes.shape[1] == 6 and img_y is not None:
                proto_rf_boxes[j, 5] = target_class

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[:, :, j] #image shape

            if ppnet.prototype_activation_function == 'log':
                proto_act_img_j = np.log(
                    (proto_dist_img_j + 1) / (proto_dist_img_j + ppnet.epsilon))
            elif ppnet.prototype_activation_function == 'linear':
                proto_act_img_j = proto_act_img_j = np.log(
                    (proto_dist_img_j + 1) / (proto_dist_img_j + ppnet.epsilon)) #max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)


            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_width, original_img_height),
                                             interpolation=cv2.INTER_CUBIC)

            # high activation area = percentile 95 calculated for activation for all pixels
            threshold = np.percentile(upsampled_act_img_j, 95)

            # show activation map only on the ground truth class
            y_mask = img_y.cpu().detach().numpy() == (target_class)  #to do think about +1
            upsampled_act_img_j_gt = upsampled_act_img_j * y_mask

            proto_bound_j = find_continuous_high_activation_crop(upsampled_act_img_j_gt, rf_prototype_j[1:],
                                                                 threshold=threshold)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                          proto_bound_j[2]:proto_bound_j[3], :]

            # find high activation *only* on ground truth
            threshold_gt = np.percentile(upsampled_act_img_j[y_mask], 95)
            proto_bound_j_gt = find_continuous_high_activation_crop(upsampled_act_img_j_gt, rf_prototype_j[1:],
                                                                    threshold=threshold_gt)
            # crop out the image patch with high activation as prototype image
            proto_img_j_gt = original_img_j[proto_bound_j_gt[0]:proto_bound_j_gt[1],
                             proto_bound_j_gt[2]:proto_bound_j_gt[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and img_y is not None:
                proto_bound_boxes[j, 5] = target_class

            '''
            proto_rf_boxes and proto_bound_boxes column:
            0: image index in the entire dataset
            1: height start index
            2: height end index
            3: width start index
            4: width end index
            5: (optional) class identity
            '''

            rows = 2 
            cols = 4
            if dir_for_saving_prototypes is not None:
                # save the target
                cls_name = cls2name[target_class]
                dir_for_saving_prototypes_cls = os.path.join(dir_for_saving_prototypes, cls_name)
                os.makedirs(dir_for_saving_prototypes_cls, exist_ok=True)

                img_y_np = img_y.cpu().numpy()
                plt.imshow(img_y_np)
                plt.axis('off')

                plt.savefig(os.path.join(dir_for_saving_prototypes_cls,
                                        prototype_img_filename_prefix + f'_{j}-target.png')) #to do change this to get the name of the variable
                plt.close()

                # figure for predicted fire
                fig, axs = plt.subplots(rows, cols, figsize=(16, 8))  # Customize the figure size as needed

                for channel in range(len(input_var)):
                    row_idx = channel // cols
                    col_idx = channel % cols

                    # Plot the original image and prediction on each subplot
                    axs[row_idx, col_idx].imshow(original_img_j[:, :, channel], cmap='gray')
                    axs[row_idx, col_idx].imshow(pred, alpha=0.7, cmap='Reds')  # Add prediction with transparency
                    axs[row_idx, col_idx].set_title(f"Channel {input_var[channel]}")
                    axs[row_idx, col_idx].axis('off')

                plt.tight_layout()

                plt.savefig(os.path.join(dir_for_saving_prototypes_cls,
                                         prototype_img_filename_prefix + f'_{j}_original_segmentation_combined.png'))
                plt.close()

                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes_cls,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    # save the img with prototype location
                    fig, axs = plt.subplots(rows, cols, figsize=(16, 8))  # Create subplots (2x4) and set figure size

                    for channel in range(len(input_var)):

                        row_idx = channel // cols
                        col_idx = channel % cols

                        #original image in the corresponding subplot
                        axs[row_idx, col_idx].imshow(original_img_j[:, :, channel], cmap='gray')

                        # bounding box with red lines
                        axs[row_idx, col_idx].plot([rf_start_w_index, rf_start_w_index], [rf_start_h_index, rf_end_h_index],
                                                   [rf_end_w_index, rf_end_w_index], [rf_start_h_index, rf_end_h_index],
                                                   [rf_start_w_index, rf_end_w_index], [rf_start_h_index, rf_start_h_index],
                                                   [rf_start_w_index, rf_end_w_index], [rf_end_h_index, rf_end_h_index],
                                                   linewidth=2, color='red')
                        axs[row_idx, col_idx].set_title(f"Channel {input_var[channel]}")
                        axs[row_idx, col_idx].axis('off')
                    
                    plt.tight_layout()
                    
                    # Save the whole figure with all subplots
                    plt.savefig(os.path.join(dir_for_saving_prototypes_cls,
                                             prototype_img_filename_prefix + f'_{j}-original_with_boxes_combined.png'))

                    plt.close()


                    # overlay (upsampled) self activation on original image and save the result                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j_gt = upsampled_act_img_j_gt - np.amin(upsampled_act_img_j_gt)
                    rescaled_act_img_j_gt = rescaled_act_img_j_gt / np.amax(rescaled_act_img_j_gt)

                    heatmap_gt = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j_gt), cv2.COLORMAP_JET)
                    heatmap_gt = np.float32(heatmap_gt) / 255
                    heatmap_gt = heatmap_gt[..., ::-1]

                    fig, axs = plt.subplots(rows, cols, figsize=(16, 8))  # Create subplots (2x4 grid)

                    for channel in range(len(input_var)):
                        row_idx = channel // cols
                        col_idx = channel % cols

                        # Create overlayed image (grayscale image + heatmap)
                        axs[row_idx, col_idx].imshow(original_img_j[:, :, channel], cmap='gray',alpha=0.7)
                        axs[row_idx, col_idx].imshow(heatmap_gt, alpha=0.3)

                        # Plot the bounding box with red lines
                        axs[row_idx, col_idx].plot([rf_start_w_index, rf_start_w_index], [rf_start_h_index, rf_end_h_index],
                                                    [rf_end_w_index, rf_end_w_index], [rf_start_h_index, rf_end_h_index],
                                                    [rf_start_w_index, rf_end_w_index], [rf_start_h_index, rf_start_h_index],
                                                    [rf_start_w_index, rf_end_w_index], [rf_end_h_index, rf_end_h_index],
                                                    linewidth=2, color='red')
                        axs[row_idx, col_idx].set_title(f"Channel {input_var[channel]}")
                        axs[row_idx, col_idx].axis('off')
                    
                    plt.tight_layout()

                    plt.savefig(os.path.join(dir_for_saving_prototypes_cls,
                                                prototype_img_filename_prefix + f'_{j}-original_with_self_act_and_boxes_gt_only_combined.png'))
                    
                    plt.close()
                        
                    # overlay (upsampled) self activation on original image and save the result                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)

                    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]

                    fig, axs = plt.subplots(rows, cols, figsize=(16, 8))  # Create subplots (2x4 grid)

                    for channel in range(len(input_var)):
                        row_idx = channel // cols
                        col_idx = channel % cols

                        # Create overlayed image (grayscale image + heatmap)
                        axs[row_idx, col_idx].imshow(original_img_j[:, :, channel], cmap='gray',alpha=0.7)
                        axs[row_idx, col_idx].imshow(heatmap, alpha=0.3)

                        # Plot the bounding box with red lines
                        axs[row_idx, col_idx].plot([rf_start_w_index, rf_start_w_index], [rf_start_h_index, rf_end_h_index],
                                                    [rf_end_w_index, rf_end_w_index], [rf_start_h_index, rf_end_h_index],
                                                    [rf_start_w_index, rf_end_w_index], [rf_start_h_index, rf_start_h_index],
                                                    [rf_start_w_index, rf_end_w_index], [rf_end_h_index, rf_end_h_index],
                                                    linewidth=2, color='red')
                        axs[row_idx, col_idx].set_title(f"Channel {input_var[channel]}")
                        axs[row_idx, col_idx].axis('off')
                    
                    plt.tight_layout()

                    plt.savefig(os.path.join(dir_for_saving_prototypes_cls,
                                                prototype_img_filename_prefix + f'_{j}-original_with_self_act_and_boxes_combined.png'))
                    
                    plt.close()
                    # if img_y.ndim > 2:
                    #     plt.imsave(os.path.join(dir_for_saving_prototypes_cls,
                    #                             prototype_img_filename_prefix + f'_{j}-receptive_field.png'),
                    #                rf_img_j,
                    #                vmin=0.0,
                    #                vmax=1.0)
                    #     overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                    #                          rf_prototype_j[3]:rf_prototype_j[4]]
                    #     plt.imsave(os.path.join(dir_for_saving_prototypes_cls,
                    #                             prototype_img_filename_prefix
                    #                             + f'_{j}-receptive_field_with_self_act.png'),
                    #                overlayed_rf_img_j,
                    #                vmin=0.0,
                    #                vmax=1.0)

                    # # save the prototype image (highly activated region of the whole image)
                    # plt.imsave(os.path.join(dir_for_saving_prototypes_cls,
                    #                         prototype_img_filename_prefix + f'_{j}_inputvar_{input_var[channel]}.png'),
                    #            proto_img_j[:,:,channel],
                    #            vmin=0.0,
                    #            vmax=1.0)

                    # # save the prototype image (highly activated region of the whole image)
                    # plt.imsave(os.path.join(dir_for_saving_prototypes_cls,
                    #                         prototype_img_filename_prefix + f'_{j}_gt_inputvar_{input_var[channel]}.png'),
                    #            proto_img_j_gt[:,:,channel],
                    #            vmin=0.0,
                    #            vmax=1.0)

    del class_to_patch_index_dict
