import numpy as np
import glob
import os
from tqdm import tqdm
import astra


# read clean projections
proj_file = sorted(glob.glob(os.path.join("data", "projection_clean", "*.npy")))

# settings of photon counts, number of splits, number of angles
I0 = np.power(10, 4)
n_split = 2
n_angles = 512
split_angles = np.int(n_angles / n_split)

# create directory
for f in range(n_split):
    dir_name = str(f)
    os.makedirs(os.path.join("data", "rec_split_poisson", dir_name), exist_ok=True)

# add poisson noise to projections
for file in tqdm(range(len(proj_file))):
    proj = np.load(proj_file[file])
    proj_noise = astra.add_noise_to_sino(proj, I0)

    # split projections
    for i in range(n_split):
        sub_proj = proj_noise[i::n_split, :]

        vol_geom = astra.create_vol_geom(128, 128)
        proj_geom = astra.create_proj_geom('parallel', 1.0, 200, np.linspace(0, np.pi, split_angles, False))
        proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
        sub_proj_id = astra.data2d.create('-sino', proj_geom, sub_proj)
        rec_id = astra.data2d.create('-vol', vol_geom)
        
        # reconstruct sub-projections
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sub_proj_id
        cfg['option'] = {'FilterType': 'Ram-Lak'}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data2d.get(rec_id)

        # save noisy sub-reconstructions
        np.save(os.path.join("data", "rec_split_poisson", str(i), "img_" + str(file)), rec)
