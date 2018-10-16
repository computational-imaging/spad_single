from argparse import ArgumentParser

parser = ArgumentParser(description="Load command line options for depthnet.")

# Model
parser.add_argument("model", help="The model.", choices=["depth", "depth_hints"],
                    type=str, default="depth")

# Learning rates:
parser.add_argument("--lr", metavar="F", help="The initial learning rate of the model.",
                    type=float, default=1e-5)
parser.add_argument("--weight-decay", metavar="F", help="The strength of the L2 regularization on the weights.",
                    type=float, default=1e-8)

# Training hyperparameters:
# For reference: scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
parser.add_argument("--num-epochs", metavar="N", help="The number of epochs to train the model.",
                    type=int, default=50)
parser.add_argument("--milestones", metavar="N", help="Learning rate milestone epochs.", nargs="*",
                    type=int, default=[25])
parser.add_argument("--gamma", metavar="F", help="Gamma of multistepLR lr decay.",
                    type=float, default=0.1)
parser.add_argument("--loss", choices=["l2", "l1", "berhu"], help="Loss function.",
                    type=str, default="berhu")
parser.add_argument("--input-nc", help="Number of input channels.",
                    metavar="N", type=int, default=3)
parser.add_argument("--output-nc", help="Number of output channels.",
                    metavar="N", type=int, default=1)

# Data Loading
parser.add_argument("trainFile", help="The location of the text file with the (depth, rgb) pairs.",
                    type=str)
parser.add_argument("trainDir", help="The folder containing the rgb-d training images.",
                    type=str)
parser.add_argument("--valFile", 
                    help="The location of the text file with the (depth, rgb) pairs for the validation dataset.",
                    type=str)
parser.add_argument("--valDir",
                    help="The location of the text file with the (depth, rgb) pairs for the validation dataset.",
                    type=str)
parser.add_argument("--batch-size", metavar="N", help="The batch size for training.",
                    type=int, default=10)
parser.add_argument("--val-batch-size", metavar="N", help="The batch size of the validation set.",
                    type=int, default=10)

# Other
parser.add_argument("--checkpoint", help="The location of a checkpoint file.",
                    type=str)
parser.add_argument("--cuda-device", help="The cuda device index of the gpu to use.",
                    type=int, default=0)
parser.add_argument("--test-run", help="Whether to truncate each epoch for testing.",
                    action="store_true")

opt = parser.parse_args()
print(opt)

