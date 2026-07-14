import scipy.io as sio
import numpy as np
import cv2
import os

DATA_ROOT = 'data/sunrgbd/'
OFFICIAL_ROOT = os.path.join(DATA_ROOT, 'OFFICIAL_SUNRGBD')
TRAINVAL_ROOT = os.path.join(DATA_ROOT, 'sunrgbd_trainval')

START_IDX = 5551
END_IDX = 7050

print(f'Loading meta files...')
meta3d = sio.loadmat(os.path.join(OFFICIAL_ROOT, 'SUNRGBDMeta3DBB_v2.mat'))
meta2d = sio.loadmat(os.path.join(OFFICIAL_ROOT, 'SUNRGBDMeta2DBB_v2.mat'))
SUNRGBDMeta = meta3d['SUNRGBDMeta'][0]
SUNRGBDMeta2DBB = meta2d['SUNRGBDMeta2DBB'][0]

print(f'Processing indices {START_IDX} to {END_IDX}')

depth_dir = os.path.join(TRAINVAL_ROOT, 'depth')
image_dir = os.path.join(TRAINVAL_ROOT, 'image')
calib_dir = os.path.join(TRAINVAL_ROOT, 'calib')
label_dir = os.path.join(TRAINVAL_ROOT, 'label')

for d in [depth_dir, image_dir, calib_dir, label_dir]:
    os.makedirs(d, exist_ok=True)

success = 0; skipped = 0; done_earlier = 0

for imageId in range(START_IDX, END_IDX + 1):
    # Skip if already processed correctly
    mat_path = os.path.join(depth_dir, f'{imageId:06d}.mat')
    cal_path = os.path.join(calib_dir, f'{imageId:06d}.txt')
    if os.path.exists(mat_path) and os.path.exists(cal_path):
        done_earlier += 1
        continue

    try:
        data = SUNRGBDMeta[imageId - 1]
        data2d = SUNRGBDMeta2DBB[imageId - 1]
    except IndexError:
        skipped += 1; continue

    try:
        dp = data['depthpath'][0]
        rp = data['rgbpath'][0]
        if isinstance(dp, (bytes, np.bytes_)): dp = dp.decode()
        if isinstance(rp, (bytes, np.bytes_)): rp = rp.decode()

        dp_full = OFFICIAL_ROOT + dp[16:]
        rp_full = OFFICIAL_ROOT + rp[16:]

        depthVis = cv2.imread(dp_full, cv2.IMREAD_UNCHANGED)
        if depthVis is None:
            skipped += 1; continue

        d16 = depthVis.astype(np.uint16)
        depth_int = np.bitwise_or(np.right_shift(d16, 3),
                                   np.left_shift(d16, 13).astype(np.uint16))
        depth = depth_int.astype(np.float32) / 1000.0
        depth[depth > 8] = 8

        rgb = cv2.imread(rp_full)
        if rgb is None:
            skipped += 1; continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        K = data['K']
        if K.ndim == 1: K = K.reshape(3, 3, order='F')
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]

        H, W = depth.shape
        x, y = np.meshgrid(np.arange(W, dtype=np.float64),
                           np.arange(H, dtype=np.float64), indexing='xy')

        x3 = (x - cx) * depth.astype(np.float64) / fx
        y3 = (y - cy) * depth.astype(np.float64) / fy
        z3 = depth.astype(np.float64)

        pts = np.column_stack([x3.ravel(order='F'),
                                z3.ravel(order='F'),
                                -y3.ravel(order='F')])
        rgb_f = rgb.reshape(-1, 3, order='F')
        invalid = (depth.ravel(order='F') == 0)
        pts[invalid] = np.nan

        Rtilt = data['Rtilt']
        if Rtilt.ndim == 1: Rtilt = Rtilt.reshape(3, 3, order='F')
        pts = (Rtilt @ pts.T).T

        valid = ~np.isnan(pts[:, 0])
        pts = pts[valid]; rgb_f = rgb_f[valid]

        pts_rgb = np.concatenate([pts.astype(np.float32), rgb_f.astype(np.float32)], axis=1)
        sio.savemat(mat_path, {'instance': pts_rgb}, do_compression=False)

        cv2.imwrite(os.path.join(image_dir, f'{imageId:06d}.jpg'),
                     cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        tf = f'{imageId:06d}.txt'
        with open(os.path.join(calib_dir, tf), 'w') as f:
            f.write(' '.join(map(str, Rtilt.ravel(order='F'))) + '\n')
            f.write(' '.join(map(str, K.ravel(order='F'))) + '\n')

        with open(os.path.join(label_dir, tf), 'w') as f:
            gt3d = data['groundtruth3DBB']
            gt3d = gt3d.ravel() if gt3d.ndim > 1 else gt3d
            gt2d_struct = None
            if data2d.size > 0:
                try:
                    gt2d_struct = data2d['groundtruth2DBB'].ravel()
                except (KeyError, AttributeError):
                    pass

            for j in range(len(gt3d)):
                bb3d = gt3d[j]
                cn_arr = bb3d['classname']
                if isinstance(cn_arr, (np.ndarray,)):
                    cn = str(np.asarray(cn_arr).ravel()[0])
                else:
                    cn = str(cn_arr[0]) if hasattr(cn_arr, '__getitem__') else str(cn_arr)
                if isinstance(cn, (bytes, np.bytes_)): cn = cn.decode()

                centroid = bb3d['centroid'].ravel()
                coeffs = np.abs(bb3d['coeffs'].ravel())
                orientation = bb3d['orientation'].ravel()

                if gt2d_struct is not None and j < len(gt2d_struct):
                    bbox2d_arr = gt2d_struct[j]['gtBb2D']
                    bbox2d = np.asarray(bbox2d_arr).ravel()
                else:
                    bbox2d = np.zeros(4)

                f.write(f'{cn} {int(float(bbox2d[0]))} {int(float(bbox2d[1]))} '
                        f'{int(float(bbox2d[2]))} {int(float(bbox2d[3]))} '
                        f'{centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f} '
                        f'{coeffs[0]:.6f} {coeffs[1]:.6f} {coeffs[2]:.6f} '
                        f'{orientation[0]:.6f} {orientation[1]:.6f}\n')

        success += 1

    except Exception as e:
        skipped += 1; continue

    if success % 200 == 0:
        print(f'  [{success}/{END_IDX - START_IDX + 1 - done_earlier - skipped}] processed')

print(f'Done! {success} new, {done_earlier} existed, {skipped} skipped')
