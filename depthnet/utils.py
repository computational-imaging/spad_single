import torch
from torch.utils.data import Dataset, DataLoader
import torch.cuda as cuda
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os.path
# Helper functions

#################
# Checkpointing #
#################
def save_checkpoint(state, is_best=False, filename='/output/checkpoint.pth.tar', always_save=False):
    """Save checkpoint if a new best is achieved"""
    if is_best or always_save:
        print ("=> Saving checkpoint to: {}".format(filename))
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")

##############
# Validation #
##############
def validate(loss, model, val_loader):
    """Computes the validation error of the model on the validation set.
    val_loader should be a DataLoader.
    
    Returns an ordinary number (i.e. not a tensor)
    
    """
    
    it = None
    losses = []
    for it, data in enumerate(val_loader):
        input_ = {}
        for key in data:
            input_[key] = data[key].float()
            if torch.cuda.is_available():
                input_[key] = input_[key].cuda()

        depth = input_["depth"]
        output = model(input_)
        losses.append(loss(output, depth).item())
    nbatches = it+1
    return sum(losses)/nbatches

##################
# Viewing Images #
##################
def save_images(*batches, outputDir, filename):
    """
    Given a list of tensors of size (B, C, H, W) (batch, channels, height, width) torch.Tensor
    Saves each entry of the batch as an rgb or grayscale image, depending on how many channels
    the image has.
    """
    I = None
    trans = transforms.ToPILImage()
    for batchnum, batch in enumerate(batches):
        if batch.shape[1] == 3:
            pass
        elif batch.shape[1] == 1:
            batch /= torch.max(batch) # normalize to lie in [0, 1]
        else:
            raise ValueError("Unsupported number of channels: {}".format(batch.shape[1]))
        batch = batch.type(torch.float32)
        for img in range(batch.shape[0]):            
            I = trans(batch[img,:,:,:].cpu().detach())
            I.save(os.path.join(outputDir, filename + "_{}_{}.png".format(batchnum, img)))


############
# Plotting #
############
def save_train_val_loss_plots(trainlosses, vallosses, epoch):
    # Train loss
    fig = plt.figure()
    plt.plot(trainlosses)
    plt.title("Train loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("trainloss_epoch{}.png".format(epoch))
    # Train loss
    fig = plt.figure()
    plt.plot(trainlosses)
    plt.title("Val loss")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.savefig("Val loss{}.png".format(epoch))
