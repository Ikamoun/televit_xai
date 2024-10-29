import sys
import os
import time
import torch
from tqdm import tqdm
import logging  # Import logging module

from .components.helpers import list_of_distances, make_one_hot

sys.path.append(os.path.abspath("/home/ines/televit_xai"))

from src import utils

# Set up logging configuration
logging.basicConfig(level=logging.WARNING)  # Change to INFO to see more logs
log = utils.get_pylogger(__name__)


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0
    total_avg_separation_cost = 0

    for i, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
        input = image.cuda()
        target = label.cuda()

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, min_distances = model(input)
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)

                l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda() if use_l1_mask else None
                l1 = (model.module.last_layer.weight * l1_mask).norm(p=1) if use_l1_mask else model.module.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item() if class_specific else 0
            total_avg_separation_cost += avg_separation_cost.item() if class_specific else 0

        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Clear GPU memory
        del input, target, output, predicted, min_distances

    end = time.time()

    log.info(f'time: {end - start}')
    log.info(f'cross ent: {total_cross_entropy / n_batches}')
    log.info(f'cluster: {total_cluster_cost / n_batches}')
    if class_specific:
        log.info(f'separation: {total_separation_cost / n_batches}')
        log.info(f'avg separation: {total_avg_separation_cost / n_batches}')
    log.info(f'accu: {n_correct / n_examples * 100}%')
    log.info(f'l1: {model.module.last_layer.weight.norm(p=1).item()}')
    
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log.info(f'p dist pair: {p_avg_pair_dist.item()}')

    return n_correct / n_examples

def train(model, dataloader, optimizer, class_specific=False, coefs=None):
    assert optimizer is not None
    log.info('train')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs)

def test(model, dataloader, class_specific=False):
    log.info('test')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific)

def last_only(model):
    if hasattr(model, 'module'):
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.last_layer.parameters():
        p.requires_grad = True

def warm_only(model):
    # aspp_params = [
    #     model.features.base.aspp.c0.weight,
    #     model.features.base.aspp.c0.bias,
    #     model.features.base.aspp.c1.weight,
    #     model.features.base.aspp.c1.bias,
    #     model.features.base.aspp.c2.weight,
    #     model.features.base.aspp.c2.bias,
    #     model.features.base.aspp.c3.weight,
    #     model.features.base.aspp.c3.bias
    # ]

    if hasattr(model, 'module'):
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True
    # for p in aspp_params:
    #     p.requires_grad = True

def joint(model):
    if hasattr(model, 'module'):
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = True
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True