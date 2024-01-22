import numpy as np
import cv2
import sys
import time
import argparse
import torch,os
import torch.optim as optim
from torch import nn
import torch.utils.data as Data
'''
Build pyramid for images
'''
def GaussianPyramid(img, leveln):
    GP = [img]
    for i in range(leveln - 1):
        GP.append(cv2.pyrDown(GP[i]))
    return GP

def LaplacianPyramid(img, leveln):
    LP = []
    for i in range(leveln - 1):
        next_img = cv2.pyrDown(img)
        LP.append(img - cv2.pyrUp(next_img, img.shape[1::-1]))
        img = next_img
    LP.append(img)
    return LP

def blend_pyramid(LPA, LPB, MP):
    blended = []
    for i, M in enumerate(MP):
        blended.append(LPA[i] * M + LPB[i] * (1.0 - M))
    return blended

def reconstruct(LS):
    img = LS[-1]
    for lev_img in LS[-2::-1]:
        img = cv2.pyrUp(img, lev_img.shape[1::-1])
        img += lev_img
    return img

'''
Multiband blending
'''
def multi_band_blending(img1, img2, mask, leveln=6):
    max_leveln = int(np.floor(np.log2(min(img1.shape[0], img1.shape[1],
                                          img2.shape[0], img2.shape[1]))))
    if leveln is None:
        leveln = max_leveln
    if leveln < 1 or leveln > max_leveln:
        print("warning: inappropriate number of leveln")
        leveln = max_leveln

    # Get Gaussian pyramid and Laplacian pyramid
    MP = GaussianPyramid(mask, leveln)
    LPA = LaplacianPyramid(img1.astype(np.float64), leveln)
    LPB = LaplacianPyramid(img2.astype(np.float64), leveln)
    # Blend two Laplacian pyramidspass
    blended = blend_pyramid(LPA, LPB, MP)

    # Reconstruction process
    result = reconstruct(blended)
    result[result > 255] = 255
    result[result < 0] = 0

    return result

def imgLabeling(img1, img2, img3, img4, maskSize, xoffsetL, xoffsetR,
                minloc_old=None):
    if len(img1.shape) == 3:
        errL = np.sum(np.square(img1.astype(np.float64) -
                                img2.astype(np.float64)), axis=2)
        errR = np.sum(np.square(img3.astype(np.float64) -
                                img4.astype(np.float64)), axis=2)
    else:
        errL = np.square(img1.astype(np.float64) - img2.astype(np.float64))
        errR = np.square(img3.astype(np.float64) - img4.astype(np.float64))
    EL = np.zeros(errL.shape, np.float64)
    ER = np.zeros(errR.shape, np.float64)
    EL[0] = errL[0]
    ER[0] = errR[0]
    for i in range(1, maskSize[1]):
        EL[i, 0] = errL[i, 0] + min(EL[i - 1, 0], EL[i - 1, 1])
        ER[i, 0] = errR[i, 0] + min(ER[i - 1, 0], ER[i - 1, 1])
        for j in range(1, EL.shape[1] - 1):
            EL[i, j] = errL[i, j] + \
                       min(EL[i - 1, j - 1], EL[i - 1, j], EL[i - 1, j + 1])
            ER[i, j] = errR[i, j] + \
                       min(ER[i - 1, j - 1], ER[i - 1, j], ER[i - 1, j + 1])
        EL[i, -1] = errL[i, -1] + min(EL[i - 1, -1], EL[i - 1, -2])
        ER[i, -1] = errR[i, -1] + min(ER[i - 1, -1], ER[i - 1, -2])
    minlocL = np.argmin(EL, axis=1) + xoffsetL
    minlocR = np.argmin(ER, axis=1) + xoffsetR
    if minloc_old is None:
        minloc_old = [minlocL, minlocR, minlocL, minlocR]
    minlocL_fin = np.int32(0.4 * minlocL + 0.3 *
                           minloc_old[0] + 0.3 * minloc_old[2])
    minlocR_fin = np.int32(0.4 * minlocR + 0.3 *
                           minloc_old[1] + 0.3 * minloc_old[3])
    mask = np.ones((maskSize[1], maskSize[0], 3), np.float64)
    for i in range(maskSize[1]):
        mask[i, minlocL_fin[i]:minlocR_fin[i]] = 0
        mask[i, minlocL_fin[i]] = 0.5
        mask[i, minlocR_fin[i]] = 0.5
    # cv2.imshow('mask', mask.astype(np.float32))
    return mask, [minlocL, minlocR, minlocL_fin, minlocR_fin]

def verticalBoundary(M, W_remap, W, H):
    """Return vertical boundary of input image."""
    row = np.zeros((W_remap, 3, 1))
    row[:, 2] = 1
    row[:, 0] = np.arange(
        (W - W_remap) / 2, (W + W_remap) / 2).reshape((W_remap, 1))
    product = np.matmul(M, row).reshape((W_remap, 3))
    normed = np.array(list(zip(product[:, 0] / product[:, 2], product[:, 1] / product[:, 2])))
    # list(normed)
    top = np.max(
        normed[
            np.logical_and(
                normed[:, 0] >= W_remap / 2,
                normed[:, 0] < W - W_remap / 2)
        ][:, 1])

    row[:, 1] = H - 1
    product = np.matmul(M, row).reshape((W_remap, 3))
    normed = np.array(
        list(zip(product[:, 0] / product[:, 2], product[:, 1] / product[:, 2])))
    bottom = np.min(normed[np.logical_and(
        normed[:, 0] >= W_remap / 2, normed[:, 0] < W - W_remap / 2)][:, 1])

    return int(top) if top > 0 else 0, int(bottom) if bottom < H else H

'''
1. Get ORB features on the overlaps
2. Match features using template matching
'''
def getMatches_goodtemplmatch(img1, img2, templ_shape, max):

    if not np.array_equal(img1.shape, img2.shape):
        print(("error: inconsistent array dimention", img1.shape, img2.shape))
        sys.exit()
    if not (np.all(templ_shape <= img1.shape[:2]) and
            np.all(templ_shape <= img2.shape[:2])):
        print("error: template shape shall fit img1 and img2")
        sys.exit()

    orb = cv2.ORB_create()
    kps1 = orb.detect(img1, None)
    kps2 = orb.detect(img2, None)
    Hs, Ws = img1.shape[:2]
    Ht, Wt = templ_shape
    matches = []
    for pts1 in kps1:
        yt = pts1.pt[1]
        xt = pts1.pt[0]
        if int(yt) + Ht > Hs or int(xt) + Wt > Ws:
            continue
        result = cv2.matchTemplate(
            img2, img1[int(yt):int(yt) + Ht, int(xt):int(xt) + Wt],
            cv2.TM_CCORR_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        if maxVal > 0.9:
            matches.append((maxVal, (int(xt), int(yt)), maxLoc))
    for pts2 in kps2:
        yt = pts2.pt[1]
        xt = pts2.pt[0]
        if int(yt) + Ht > Hs or int(xt) + Wt > Ws:
            continue
        result = cv2.matchTemplate(
            img1, img2[int(yt):int(yt) + Ht, int(xt):int(xt) + Wt],
            cv2.TM_CCORR_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        if maxVal > 0.9:
            matches.append((maxVal, maxLoc, (int(xt), int(yt))))
    matches.sort(key=lambda e: e[0], reverse=True)
    if len(matches) >= max:
        return np.int32([matches[i][1:] for i in range(max)])
    else:
        return np.int32([c[1:] for c in matches])

'''
Fisheye projection
'''
def equirect_proj(x_proj, y_proj, W, H, fov):
    """Return the equirectangular projection on a unit sphere,
    given cartesian coordinates of the de-warped image."""
    theta_alt = x_proj * fov / W
    phi_alt = y_proj * np.pi / H

    x = np.sin(theta_alt) * np.cos(phi_alt)
    y = np.sin(phi_alt)
    z = np.cos(theta_alt) * np.cos(phi_alt)

    return np.arctan2(y, x), np.arctan2(np.sqrt(x ** 2 + y ** 2), z)

def buildmap(Ws, Hs, Wd, Hd, fov):
    """Return a mapping from de-warped images to fisheye images."""
    fov = fov * np.pi / 180.0

    # cartesian coordinates of the de-warped rectangular image
    ys, xs = np.indices((Hs, Ws), np.float32)
    y_proj = Hs / 2.0 - ys
    x_proj = xs - Ws / 2.0

    # spherical coordinates
    theta, phi = equirect_proj(x_proj, y_proj, Ws, Hs, fov)

    # polar coordinates (of the fisheye image)
    p = Hd * phi / fov

    # cartesian coordinates of the fisheye image
    y_fish = p * np.sin(theta)
    x_fish = p * np.cos(theta)

    ymap = Hd / 2.0 - y_fish
    xmap = Wd / 2.0 + x_fish
    return xmap, ymap, p

'''
Deduplicate same features
'''
def deduplication(inliers):
    pts_1, indexes_1 = np.unique(inliers[:, 1], return_index=True, axis=0)
    label, indexes_2 = np.unique(inliers[:, 0][indexes_1], return_index=True, axis=0)
    train = pts_1[indexes_2]
    return train, label

def stitch(cap_1,cap_2,written_path,fov):
    cap1 = cv2.VideoCapture(cap_1)
    cap2 = cv2.VideoCapture(cap_2)
    count = cap1.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap1.get(cv2.CAP_PROP_FPS)
    size = (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH) * 2), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video = cv2.VideoWriter(written_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    device = torch.device("cpu") # not much difference while using GPU instead

    maxL = 200 # max features to detect on the left overlap
    maxR = 200 # max features to detect on the right overlap

    if cap1.isOpened() and cap2.isOpened():
        i = 1
        print('successful')
        print("The count is %d" % (count))

        # image size paremeters
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        H = frame1.shape[0]
        W_remap = int(fov / 180 * H)
        W = frame1.shape[1] + frame2.shape[1]
        templ_shape = (60, 16)
        offsetYL = int(160 / 1280 * H)
        offsetYR = int(160 / 1280 * H)

        #fishey unwarp
        b_t = time.time()
        xmap, ymap, p = buildmap(Ws=W_remap, Hs=H, Wd=H, Hd=H, fov=fov)
        cam1 = cv2.remap(frame1, xmap, ymap, cv2.INTER_LINEAR)
        cam2 = cv2.remap(frame2, xmap, ymap, cv2.INTER_LINEAR)

        # shift
        cam1_gray = cv2.cvtColor(cam1, cv2.COLOR_BGR2GRAY)
        cam2_gray = cv2.cvtColor(cam2, cv2.COLOR_BGR2GRAY)
        shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
        shifted_cams[H:, int((W - W_remap) / 2):int((W + W_remap) / 2)] = cam1
        shifted_cams[:H, :int(W_remap / 2)] = cam2[:, int(W_remap / 2):]
        shifted_cams[:H, W - int(W_remap / 2):] = cam2[:, :int(W_remap / 2)]

        # collect and match features
        matchesL = getMatches_goodtemplmatch(
            cam1_gray[offsetYL:H - offsetYL, int(W / 2):],
            cam2_gray[offsetYL:H - offsetYL, :W_remap - int(W / 2)],
            templ_shape, maxL)
        matchesR = getMatches_goodtemplmatch(
            cam2_gray[offsetYR:H - offsetYR, int(W / 2):],
            cam1_gray[offsetYR:H - offsetYR, :W_remap - int(W / 2)],
            templ_shape, maxR)
        matchesR = matchesR[:, -1::-1]
        matchesL = matchesL + (int((W - W_remap) / 2), offsetYL)
        matchesR = matchesR + (int((W - W_remap) / 2) + int(W / 2), offsetYR)
        zipped_matches = list(zip(matchesL, matchesR))
        matches = np.int32([e for i in zipped_matches for e in i])
        pts1 = matches[:, 0]
        pts2 = matches[:, 1]

        # homography
        M_current, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)
        num = 0
        # inlier percentage
        for m in range(status.ravel().shape[0]):
            if status.ravel()[m] == 1:
                num += 1
        print("The percentage of interior points is %.2f" % (num / status.ravel().shape[0]))

        top, bottom = verticalBoundary(M_current, W_remap, W, H)
        warped2 = cv2.warpPerspective(shifted_cams[H:], M_current, (W, H))
        warped1 = shifted_cams[0:H, :]
        warped1 = cv2.resize(warped1[top:bottom], (W, H))
        warped2 = cv2.resize(warped2[top:bottom], (W, H))
        EAof2 = np.zeros((H, W, 3), np.uint8)
        EAof2[:, int((W - W_remap) / 2) + 1:int((W + W_remap) / 2) - 1] = 255
        EAof2 = cv2.warpPerspective(EAof2, M_current, (W, H))

        # blend
        b_blend = time.time()
        W_lbl = 120
        blend_level = 4 # up to 7
        # mask for seamless blending
        mask, minloc_old = imgLabeling(
            warped1[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
            warped2[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
            warped1[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
            warped2[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
            (W, H), int(W_remap / 2) - W_lbl, W - int(W_remap / 2))

        warped1[:, int(W_remap / 2):W - int(W_remap /
                                            2)] = warped2[:, int(W_remap / 2):W - int(W_remap / 2)]
        warped2[EAof2 == 0] = warped1[EAof2 == 0]

        # seperately blend left and right overlaps
        blended_l = multi_band_blending(
            warped1[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
            warped2[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
            mask[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
            blend_level)
        blended_r = multi_band_blending(
            warped1[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
            warped2[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
            mask[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
            blend_level)

        # get final stitched result
        blended = np.ones((H, W, 3))
        blended[:, 0:int(W_remap / 2) - W_lbl] = warped1[:, 0:int(W_remap / 2) - W_lbl]
        blended[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)] = blended_l
        blended[:, int(W_remap / 2):W - int(W_remap / 2)] = warped2[:, int(W_remap / 2):W - int(W_remap / 2)]
        blended[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl] = blended_r
        blended[:, W - int(W_remap / 2) + W_lbl:] = warped1[:, W - int(W_remap / 2) + W_lbl:]
        e_t = time.time()
        e_blend = time.time()
        print("Time for blending is %.4f" %(e_blend-b_blend))
        print("Total time is %.4f" %(e_t-b_t) )
        video.write(blended.astype(np.uint8))

        # training setup
        epochs = 40
        # get inliers for training
        index = np.where(status.ravel() == 1)
        inliers = matches[index[0]]

        # build dataset.=
        train, label = deduplication(inliers) # deduplication
        zipped_train = list(zip(train, label))
        dataset = np.int32([e for e in zipped_train])
        np.random.shuffle(dataset) # shuffle dataset
        train = dataset[:, 0] # features in the source image are training data
        label = dataset[:, 1] # features in the dst image are ground truth

        # build test dataset to compare MSE
        b_test = np.ones(int(train[int(train.shape[0] * 0.8):].shape[0]))
        test = np.insert(train[int(train.shape[0] * 0.8):], 2, values=b_test, axis=1)
        label_test = np.insert(label[int(train.shape[0] * 0.8):], 2, values=b_test, axis=1)

        # initialize linear layer
        M = torch.tensor(M_current, requires_grad=True)
        model = nn.Linear(3, 3, bias=False)
        model.weight = nn.Parameter(M)
        criterion = nn.MSELoss()

        #get MSE of current model
        test = torch.tensor(test, requires_grad=False, dtype=torch.float64)
        label_test = torch.tensor(label_test, requires_grad=False, dtype=torch.float64)
        test_output = model(test)
        test_output = test_output / test_output[:, 2].reshape(-1, 1)
        best_loss = criterion(test_output, label_test) / label_test.shape[0]


        print("The loss for frame1 is %.4f" % (best_loss))
        shape_list = []
        shape_list.append(inliers.shape[0])
        while i < 30:
            i += 1
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if ret1 and ret2:

                cam1 = cv2.remap(frame1, xmap, ymap, cv2.INTER_LINEAR)
                cam2 = cv2.remap(frame2, xmap, ymap, cv2.INTER_LINEAR)

                # shift

                shifted_cams = np.zeros((H * 2, W, 3), np.uint8)
                shifted_cams[H:, int((W - W_remap) / 2):int((W + W_remap) / 2)] = cam1
                shifted_cams[:H, :int(W_remap / 2)] = cam2[:, int(W_remap / 2):]
                shifted_cams[:H, W - int(W_remap / 2):] = cam2[:, :int(W_remap / 2)]

                # maintain the total features from only 4 adjacent frames
                if len(shape_list) % 4 == 0:
                    inliers = np.delete(inliers, range(0, shape_list[0]), axis=0)
                    del (shape_list[0])
                # optimize every 60 frames
                if i % 60 == 0:
                    cam1_gray = cv2.cvtColor(cam1, cv2.COLOR_BGR2GRAY)
                    cam2_gray = cv2.cvtColor(cam2, cv2.COLOR_BGR2GRAY)

                    #collect new features
                    matchesL = getMatches_goodtemplmatch(
                        cam1_gray[offsetYL:H - offsetYL, int(W / 2):],
                        cam2_gray[offsetYL:H - offsetYL, :W_remap - int(W / 2)],
                        templ_shape, maxL)
                    matchesR = getMatches_goodtemplmatch(
                        cam2_gray[offsetYR:H - offsetYR, int(W / 2):],
                        cam1_gray[offsetYR:H - offsetYR, :W_remap - int(W / 2)],
                        templ_shape, maxR)
                    matchesR = matchesR[:, -1::-1]
                    matchesL = matchesL + (int((W - W_remap) / 2), offsetYL)
                    matchesR = matchesR + (int((W - W_remap) / 2) + int(W / 2), offsetYR)
                    zipped_matches = list(zip(matchesL, matchesR))
                    matches_append = np.int32([e for i in zipped_matches for e in i])
                    pts1 = matches_append[:, 0]
                    pts2 = matches_append[:, 1]

                    # get inliers
                    M_, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)
                    num = 0
                    for m in range(status.ravel().shape[0]):
                        if status.ravel()[m] == 1:
                            num += 1
                    print("The percentage of interior points is %.2f" % (num / status.ravel().shape[0]))
                    index = np.where(status.ravel() == 1)
                    inliers = np.concatenate((inliers, matches_append[index[0]]), axis=0)
                    shape_list.append(matches_append[index[0]].shape[0])

                    # add new inliers into the dataset
                    train, label = deduplication(inliers)
                    zipped_train = list(zip(train, label))
                    dataset = np.int32([e for e in zipped_train])
                    np.random.shuffle(dataset)
                    train = dataset[:, 0]
                    label = dataset[:, 1]

                    b_test = np.ones(int(train[int(train.shape[0] * 0.8):].shape[0]))
                    test = np.insert(train[int(train.shape[0] * 0.8):], 2, values=b_test, axis=1)
                    label_test = np.insert(label[int(train.shape[0] * 0.8):], 2, values=b_test, axis=1)
                    b = np.ones(int(train.shape[0] * 0.8))
                    train = np.insert(train[:int(train.shape[0] * 0.8)], 2, values=b, axis=1)
                    label = np.insert(label[:int(label.shape[0] * 0.8)], 2, values=b, axis=1)

                    M = torch.tensor(M_current, requires_grad=True)
                    model = nn.Linear(3, 3, bias=False)
                    model.weight = nn.Parameter(M)
                    model.to(device)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)

                    # dataset to tensor
                    inputs = torch.tensor(train, requires_grad=False, dtype=torch.float64)
                    inputs = inputs.to(device)
                    label_t = torch.tensor(label, requires_grad=False, dtype=torch.float64)
                    label_t = label_t.to(device)
                    test = torch.tensor(test, requires_grad=False, dtype=torch.float64)
                    test = test.to(device)
                    label_test = torch.tensor(label_test, requires_grad=False, dtype=torch.float64)
                    label_test = label_test.to(device)
                    torch_dataset = Data.TensorDataset(inputs, label_t)

                    loader = Data.DataLoader(
                        dataset=torch_dataset,
                        batch_size=64,
                        shuffle=True,
                        num_workers=0, )
                    total_step = len(loader)


                    # training
                    t_s = time.time()
                    for epoch in range(epochs):
                        for step, (batch_x, l) in enumerate(loader):
                            outputs = model(batch_x)
                            outputs = outputs / outputs[:, 2].reshape(-1, 1)
                            loss = criterion(outputs, l)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            print(
                                'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, step + 1, total_step,
                                                                                   loss.item() / len(batch_x)))
                    t_e = time.time()
                    print(" Time for training is %.4f ms" % ((t_e - t_s) * 1000))

                    # MSE of trained homography on test dataset
                    test_output = model(test)
                    test_output = test_output / test_output[:, 2].reshape(-1, 1)
                    loss_test = criterion(test_output, label_test) / label_test.shape[0]

                    # MSE of using homography on test dataset
                    M_c = torch.tensor(M_current, requires_grad=True)
                    model_c = nn.Linear(3, 3, bias=False)
                    model_c.weight = nn.Parameter(M_c)
                    model_c.to(device)
                    current_test = model_c(test)
                    current_test = current_test / current_test[:, 2].reshape(-1, 1)
                    current_loss = criterion(current_test, label_test) / label_test.shape[0]

                    print("The loss of M_trained is %.4f, round %d" % (loss_test, i))
                    print("The loss of M_current is %.4f, round %d" % (current_loss, i))


                    if loss_test < current_loss:
                        model = model.cpu()
                        M_current = model.weight.detach().numpy()
                        M_current = M_current / M_current[2][2]
                        best_loss = loss_test
                        print("The best loss is %.4f" % (best_loss))

                # if frames don't require training, directly blend
                top, bottom = verticalBoundary(M_current, W_remap, W, H)
                warped2 = cv2.warpPerspective(shifted_cams[H:], M_current, (W, H))
                warped1 = shifted_cams[0:H, :]
                warped1 = cv2.resize(warped1[top:bottom], (W, H))
                warped2 = cv2.resize(warped2[top:bottom], (W, H))
                EAof2 = np.zeros((H, W, 3), np.uint8)
                EAof2[:, int((W - W_remap) / 2) + 1:int((W + W_remap) / 2) - 1] = 255
                EAof2 = cv2.warpPerspective(EAof2, M_current, (W, H))
                b_blend = time.time()

                # get new mask every 240 frames
                if i % 240 == 0:
                    mask, minloc_old = imgLabeling(
                        warped1[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
                        warped2[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
                        warped1[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
                        warped2[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
                        (W, H), int(W_remap / 2) - W_lbl, W - int(W_remap / 2))

                warped1[:, int(W_remap / 2):W - int(W_remap /
                                                    2)] = warped2[:, int(W_remap / 2):W - int(W_remap / 2)]
                warped2[EAof2 == 0] = warped1[EAof2 == 0]

                blended = warped1 * mask + warped2 * (1 - mask)
                blended_l = multi_band_blending(
                        warped1[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
                        warped2[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
                        mask[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)],
                        blend_level)
                blended_r = multi_band_blending(
                        warped1[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
                        warped2[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
                        mask[:, W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl],
                        blend_level)
                blended = np.ones((H,W,3))
                blended[:, 0:int(W_remap / 2) - W_lbl] = warped1[:, 0:int(W_remap / 2) - W_lbl]
                blended[:, int(W_remap / 2) - W_lbl:int(W_remap / 2)] = blended_l
                blended[:,int(W_remap / 2):W - int(W_remap / 2)] = warped2[:,int(W_remap / 2):W - int(W_remap / 2)]
                blended[:,W - int(W_remap / 2):W - int(W_remap / 2) + W_lbl] = blended_r
                blended[:,W - int(W_remap / 2) + W_lbl:] = warped1[:,W - int(W_remap / 2) + W_lbl:]
                video.write(blended.astype(np.uint8))
                e_blend=time.time()
                print(" Time for blending is %.4f, round is %d" %((e_blend-b_blend),i) )
            else:
                break

    else:
        print('Fail. No videos are found')
    cap1.release()
    cap2.release()
    video.release()

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", dest = "output_path")
parser.add_argument("-i1", "--input1", dest = "input_path_left")
parser.add_argument("-i2", "--input2", dest = "input_path_right")
parser.add_argument("-f", "--fov", dest = "FOV")
args = parser.parse_args()


if __name__ == "__main__":
    stitch(cap_1=args.input_path_left,cap_2=args.input_path_right, written_path=args.output_path,fov=int(args.FOV))