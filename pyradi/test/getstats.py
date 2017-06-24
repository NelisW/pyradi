import pyradi.rytarggen as rytarggen
filenames = ['image-Disk-256-256.hdf5','image-rawFile-512-512.hdf5',
'image-StairIR-raw-100-256.hdf5',
'image-Stairslin-10-250-250.hdf5','image-Stairslin-40-100-520.hdf5',
'image-StairslinIR-40-100-520.hdf5','image-Stairslin-LowLight-40-100-520.hdf5',
'image-Uniform-256-256.hdf5','image-Zero-256-256.hdf5'
]
for filename in filenames:
    rytarggen.analyse_HDF5_image(filename,gwidh=18,gheit=12)