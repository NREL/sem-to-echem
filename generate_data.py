from slicegan import networks, util
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('reduction', type=str)
parser.add_argument('subregionID', type=int)
parser.add_argument('subregion', type=int)

args = parser.parse_args()
r = float(args.reduction)
t = args.subregionID
s = args.subregion

assert (r > 0.) and (r <= 1.), 'ratio must be between (0, 1], r = {} was given'.format(r)
assert (t <= s)


## Path to saved model
model_name = 'slicegan_r{}_s{}{}'.format(str(r).replace('.', 'p'), str(t), str(s))
model_path = 'model/{}/{}'.format(model_name, model_name)

## Path to save images/gif
img_path = 'figs/{}'.format(model_name)

## Data Processing
# Define image type (colour, grayscale, three-phase or two-phase.
# n-phase materials must be segmented)
image_type = 'nphase'

# img_channels should be number of phases for nphase, 3 for colour, or 1 for grayscale
img_channels = 4

## Network Architectures
# Training image size, no. channels and scale factor vs raw data
#img_size, scale_factor = 64, 1

# z vector depth
z_channels = 32

# z vector dimension - this determines output size
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
lz = 8

# Number of layers in G and D
lays = 5
laysd = 6

# kernel sizes
dk, gk = [4]*laysd, [4]*lays

# stide size
ds, gs = [2]*laysd, [2]*lays

# Filter sizes
df = [img_channels, 64, 128, 256, 512, 1]
gf = [z_channels, 512, 128, 64, 32, img_channels]

dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]

## Create Networks
#_, netG = networks.slicegan_rc_nets(model_path, 0, image_type, dk, ds, df, dp, gk, gs, gf, gp)
_, netG = networks.slicegan_nets(model_path, 0, image_type, dk, ds, df, dp, gk, gs, gf, gp)

#vol = util.generate_volume(model_path, image_type, netG(), z_channels, lz=lz)
#print(vol)

util.make_gif(img_path, model_path, image_type, netG(), nz=z_channels, lz=lz)
