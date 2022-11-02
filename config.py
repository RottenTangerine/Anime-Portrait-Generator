from argparse import ArgumentParser


def get_parser():
    args = ArgumentParser()

    args.add_argument('--dataset', type=str, default='anime', help='name of the dataset')
    args.add_argument('--channel', type=int, default=3, help='number of the channel')
    args.add_argument('--init_type', type=str, default='normal', help='init function')
    args.add_argument('--init_gain', type=float, default=0.02, help='init gain')

    args.add_argument('--noise_features', type=int, default=100, help='size of the random feature')

    # device
    args.add_argument('--cuda', type=bool, default=True, help='use cuda training')

    # training
    args.add_argument('--batch_size', type=int, default=18, help='batch size number')
    args.add_argument('--lr_g', type=float, default=2e-4, help='learning rate of generator')
    args.add_argument('--lr_d', type=float, default=1e-4, help='learning rate of discriminator')
    args.add_argument('--epochs', type=int, default=30, help='number of epochs')
    args.add_argument('--output_channel', type=int, default=3, help='size of the output image channel')

    # print option
    args.add_argument('--print_interval', type=int, default=100, help='print intervel')

    return args.parse_args()
