from slicegan import networks, util

def generate_new_volume(model_path='model/sliceGAN', lz=8):
    ## Data Processing
    # Define image type (colour, grayscale, three-phase or two-phase.
    # n-phase materials must be segmented)
    image_type = 'nphase'

    # img_channels should be number of phases for nphase, 3 for colour, or 1 for grayscale
    img_channels = 4

    ## Network Architectures
    # Training image size, no. channels and scale factor vs raw data
    img_size, scale_factor = 128, 1

    # z vector depth
    z_channels = 32

    # z vector dimension, lz - this determines output size
    # lz size table for reference:
    # lz =  4 -> output dim = 64
    # lz =  6 -> output dim = 128
    # lz =  8 -> output dim = 192
    # lz = 10 -> output dim = 256
    # lz = 12 -> output dim = 320
    # lz = 18 -> output dim = 512
    # lz = 24 -> output dim = 704
    # lz = 32 -> output dim = 960
    # lz = 40 -> output dim = 1216

    # Number of layers in G and D
    lays = 5
    laysd = 6

    # kernel sizes
    dk, gk = [4]*laysd, [4]*lays

    # stride size
    ds, gs = [2]*laysd, [2]*lays

    # Filter sizes
    df = [img_channels, 64, 128, 256, 512, 1]
    gf = [z_channels, 512, 128, 64, 32, img_channels]

    dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]

    ## Create Networks
    _, netG = networks.slicegan_nets(model_path, 0, image_type, dk, ds, df, dp, gk, gs, gf, gp)

    vol = util.generate_volume(model_path, image_type, netG(), nz=z_channels, lz=lz)
    return vol

#print(vol)
