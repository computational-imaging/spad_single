# Set up training.
import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from depthnet.model import DepthNet, DepthNetWithHints
from depthnet.models.unet_model import UNet
import depthnet.utils as u

def setup_training(opt, device, writer=None):
    # Build model and loss
    # Hyperparameters
    model = None
    if opt.model == "depth":
        model = DepthNet(opt.input_nc, opt.output_nc)
    elif opt.model == "depth_hints":
        model = DepthNetWithHints(DepthNet(opt.input_nc, opt.output_nc), 800//3, 4)
    elif opt.model == "unet":
        model = UNet(opt.input_nc, opt.output_nc)
    if torch.cuda.is_available():
#        print("device: {}".format(torch.cuda.current_device()))
        print(device)
        model.to(device)
    # Split parameters into weight and bias terms (different regularization)
    params = [{"params": []}, {"params": [], "weight_decay": 0.0}]
    for name, param in model.named_parameters():
        if "weight" in name:
            params[0]["params"].append(param)
        elif "bias" in name:
            params[1]["params"].append(param)
        else:
            raise ValueError("Unknown param type: {}".format(name))

    #print(params)
    
    # Checkpointing
    if opt.checkpoint is not None:
        if torch.cuda.is_available():
            checkpoint = torch.load(opt.checkpoint)
        else:
            # Load GPU model on CPU
            checkpoint = torch.load(opt.checkpoint,
                                    map_location=lambda storage,
                                    loc: storage)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer = optim.Adam(params, lr=opt.lr, weight_decay=opt.weight_decay)
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        trainlosses = checkpoint['trainlosses']
        global_it = len(trainlosses)
        vallosses = checkpoint['vallosses']
        if writer is not None:
            for i, trainloss in enumerate(trainlosses): # For tensorboardx
                writer.add_scalar("data/trainloss", trainloss, i)
            for i, valloss in enumerate(vallosses): # For tensorboardx
                writer.add_scalar("data/valloss", valloss, i)
        global_it = len(trainlosses)
        print("=> loaded checkpoint '{}' (trained for {} epoch(s)).".format(opt.checkpoint, start_epoch))
    else:
        start_epoch = 0
        global_it = 0 # Track global iterations
        best_loss = torch.FloatTensor([float('inf')])
        optimizer = optim.Adam(params, lr=opt.lr, weight_decay=opt.weight_decay)
        trainlosses = []
        vallosses = []
        # Initialize weights:
        for name, param in model.named_parameters():
#            print(param.shape)
#            print(len(param.shape))
#            print(name)
            if "conv" in name and "weight" in name and len(param.shape) == 1:
                nn.init.normal_(param)
            elif "conv" in name and "weight" in name:
    #             print(name)
                nn.init.xavier_normal_(param)
                #nn.init.constant_(param, 1)
            elif "norm" in name and "weight" in name:
    #             print(name)
                nn.init.constant_(param, 1)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    # Scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=opt.milestones,
                                               gamma=opt.gamma)
    scheduler.last_epoch = start_epoch - 1

    # Print summary of setup:
    print("loaded opt.checkpoint: {}".format(opt.checkpoint))
    print("start_epoch: {}".format(start_epoch))
    print("global_it: {}".format(global_it))
    print("optimizer: {}".format(optimizer))
    print("batch_size: {}".format(opt.batch_size))
    print("num_epochs: {}".format(opt.num_epochs))
    print("learning rate (initial): {}".format(scheduler.get_lr()))
    print("scheduler: {}".format(scheduler.state_dict()))
    
    out = {"model": model,
           "start_epoch": start_epoch,
           "num_epochs": opt.num_epochs,
           "global_it": global_it,
           "scheduler": scheduler,
           "trainlosses": trainlosses,
           "vallosses": vallosses
          }
    return out

####################
# Run the training #
####################
def train(opt,
          model, loss,
          start_epoch, num_epochs,
          trainlosses, vallosses,
          global_it, scheduler, trainLoader, valLoader=None,
          test_run=False, writer=None, device=None):
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print("epoch: {}".format(epoch))
        data = None
        output = None
        for it, data in enumerate(trainLoader):
            input_ = {}
            for key in data:
                input_[key] = data[key].float()
#                if torch.cuda.is_available():
                input_[key] = input_[key].to(device)

            depth = input_["depth"]
            # New batch
            scheduler.optimizer.zero_grad()
            output = model(input_)
            trainloss = loss(output, depth)
            trainloss.backward()
            scheduler.optimizer.step()
            global_it += 1

            if not (it % 10):
                print("\titeration: {}\ttrain loss: {}".format(it, trainloss.item()))
            trainlosses.append(trainloss.item())
            writer.add_scalar("data/trainloss", trainloss.item(), global_it)
            if test_run: # Stop after 5 batches
                if not ((it + 1) % 5):
                    break
        # Checkpointing
        if valLoader is not None:
            valloss = u.validate(loss, model, valLoader)
            print("End epoch {}\tval loss: {}".format(epoch, valloss))
            vallosses.append(valloss)
            writer.add_scalar("data/valloss", valloss, epoch)

        # Save the last batch output of every epoch
        rgb_input = vutils.make_grid(data["rgb"], nrow=opt.batch_size, normalize=True, scale_each=True)
        writer.add_image('image/rgb_input', rgb_input, epoch)

        depth_truth = vutils.make_grid(data["depth"], nrow=opt.batch_size, normalize=True, scale_each=True)
        writer.add_image('image/depth_truth', depth_truth, epoch)

        depth_output = vutils.make_grid(output, nrow=opt.batch_size, normalize=True, scale_each=True)
        writer.add_image('image/depth_output', depth_output, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), global_it)
    #     save_images(data["rgb"], data["depth"], output, outputDir="images", filename="epoch_{}".format(epoch))

#        is_best = bool(trainloss.data.cpu().numpy() < best_loss.numpy())
        # Get greater Tensor to keep track best acc
#        best_loss = torch.FloatTensor(min(trainloss.data.cpu().numpy(), best_loss.numpy()))
        # Save checkpoint
        u.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
#            'best_loss': best_loss,
            'optim_state_dict': scheduler.optimizer.state_dict(),
            'trainlosses': trainlosses,
            'vallosses': vallosses
        }, False, filename="checkpoints/checkpoint_epoch_{}.pth.tar".format(epoch), always_save=True)
