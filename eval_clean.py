import astra
import numpy as np
import pylab
import os



test_data = np.load("data/test_data/L506_190_target.npy")
endx = test_data.shape[0]
endy = test_data.shape[1]
img = test_data[0: endx - 1: 4, 0: endy - 1: 4]

vol_geom = astra.create_vol_geom(128, 128)
proj_geom = astra.create_proj_geom('parallel', 1.0, 200, np.linspace(0, np.pi, 512, False))
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
sino_id, sino = astra.create_sino(img, proj_id)

rec_id = astra.data2d.create('-vol', vol_geom)

cfg = astra.astra_dict('FBP_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sino_id
cfg['option'] = {'FilterType': 'Ram-Lak'}
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
rec = astra.data2d.get(rec_id)

np.save(os.path.join("data", "full_fbp"), rec)

pylab.gray()
pylab.imshow(rec)
pylab.show()

