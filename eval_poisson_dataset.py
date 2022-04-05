import numpy as np
import os
import astra


n_split = 2
n_angles = 512
split_angles = np.int(n_angles / n_split)
I0 = np.power(10, 4)

test_data = np.load("data/test_data/L506_190_target.npy")
endx = test_data.shape[0]
endy = test_data.shape[1]
img = test_data[0: endx - 1: 4, 0: endy - 1: 4]

vol_geom = astra.create_vol_geom(128, 128)
proj_geom = astra.create_proj_geom('parallel', 1.0, 200, np.linspace(0, np.pi, n_angles, False))
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
sino_id, sino = astra.create_sino(img, proj_id)
sino_noise = astra.add_noise_to_sino(sino, I0)
sino_noise_id = astra.data2d.create('-sino', proj_geom, sino_noise)
rec_id = astra.data2d.create('-vol', vol_geom)

cfg = astra.astra_dict('FBP_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sino_noise_id
cfg['option'] = {'FilterType': 'Ram-Lak'}
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
rec = astra.data2d.get(rec_id)

np.save(os.path.join("data", "test_poisson"), rec)

for f in range(n_split):
    dir_name = str(f)
    os.makedirs(os.path.join("data", "test_split_poisson", dir_name), exist_ok=True)

for i in range(n_split):
    sub_sino = sino_noise[i::n_split, :]
    sub_proj_geom = astra.create_proj_geom('parallel', 1.0, 200, np.linspace(0, np.pi, split_angles, False))
    sub_proj_id = astra.create_projector('cuda', sub_proj_geom, vol_geom)
    sub_sino_id = astra.data2d.create('-sino', sub_proj_geom, sub_sino)
    sub_rec_id = astra.data2d.create('-vol', vol_geom)

    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = sub_rec_id
    cfg['ProjectionDataId'] = sub_sino_id
    cfg['option'] = {'FilterType': 'Ram-Lak'}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    sub_rec = astra.data2d.get(sub_rec_id)

    np.save(os.path.join("data", "test_split_poisson", str(i), "img"), sub_rec)
