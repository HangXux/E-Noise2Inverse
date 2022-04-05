import astra
import numpy as np
import os
from tqdm import tqdm
import glob


clean_file = sorted(glob.glob(os.path.join("data", "train_data", "*.npy")))
for file in tqdm(range(len(clean_file))):
    clean_img = np.load(clean_file[file])

    # down sample resolution
    endx = clean_img.shape[0]
    endy = clean_img.shape[1]
    img = clean_img[0: endx - 1: 4, 0: endy - 1: 4]

    # define geometry
    vol_geom = astra.create_vol_geom(128, 128)
    proj_geom = astra.create_proj_geom('parallel', 1.0, 200, np.linspace(0, np.pi, 512, False))
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

    # create sinogram
    sino_id, sino = astra.create_sino(img, proj_id)

    # reconstruct clean
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id
    cfg['option'] = {'FilterType': 'Ram-Lak'}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    rec = astra.data2d.get(rec_id)

    # save clean data
    np.save(os.path.join("data", "projection_clean_full", "proj_clean_" + str(file)), sino)
    np.save(os.path.join("data", "rec_clean_full", "rec_clean_" + str(file)), rec)


