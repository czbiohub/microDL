import nose.tools
import numpy as np

import micro_dl.inference.evaluation_metrics as metrics


im_shape = (20, 25, 15)
target_im = np.ones(im_shape)
for i in range(4):
    target_im[..., i + 1] = i + 1
pred_im = target_im.copy() + 2


def test_mse_metric():
    mse = metrics.mse_metric(target=target_im, prediction=pred_im)
    nose.tools.assert_equal(mse, 4)


def test_mae_metric():
    mae = metrics.mae_metric(target=target_im, prediction=pred_im)
    nose.tools.assert_equal(mae, 2)


def test_r2_metric():
    r2 = metrics.r2_metric(target=target_im, prediction=pred_im)
    nose.tools.assert_less(r2, -1)


def test_corr_metric():
    corr = metrics.corr_metric(target=target_im, prediction=pred_im)
    nose.tools.assert_equal(corr, 1)


def test_ssim_metric():
    ssim = metrics.ssim_metric(target=target_im, prediction=pred_im)
    nose.tools.assert_less(ssim, 1)


def test_xyz_metrics():
    metrics_list = ['ssim', 'corr', 'r2', 'mse', 'mae']
    pred_name = 'test_pred'

    metrics_inst = metrics.MetricsEstimator(
        metrics_list=metrics_list,
        masked_metrics=False,
    )
    metrics_inst.estimate_xyz_metrics(
        target=target_im,
        prediction=pred_im,
        pred_name=pred_name,
    )
    metrics_xyz = metrics_inst.get_metrics_xyz()
    nose.tools.assert_tuple_equal(metrics_xyz.shape, (1, 6))
    metrics_list.append('pred_name')
    nose.tools.assert_list_equal(list(metrics_xyz), metrics_list)


def test_xy_metrics():
    metrics_list = ['ssim', 'corr', 'r2']
    pred_name = 'test_pred'

    metrics_inst = metrics.MetricsEstimator(
        metrics_list=metrics_list,
        masked_metrics=False,
    )
    metrics_inst.estimate_xy_metrics(
        target=target_im,
        prediction=pred_im,
        pred_name=pred_name,
    )
    metrics_xy = metrics_inst.get_metrics_xy()
    nose.tools.assert_tuple_equal(metrics_xy.shape, (im_shape[2], 4))
    metrics_list.append('pred_name')
    nose.tools.assert_list_equal(list(metrics_xy), metrics_list)


def test_xz_metrics():
    metrics_list = ['ssim', 'corr']
    pred_name = 'test_pred'

    metrics_inst = metrics.MetricsEstimator(
        metrics_list=metrics_list,
        masked_metrics=False,
    )
    metrics_inst.estimate_xz_metrics(
        target=target_im,
        prediction=pred_im,
        pred_name=pred_name,
    )
    metrics_xz = metrics_inst.get_metrics_xz()
    nose.tools.assert_tuple_equal(metrics_xz.shape, (im_shape[0], 3))
    metrics_list.append('pred_name')
    nose.tools.assert_list_equal(list(metrics_xz), metrics_list)


def test_yz_metrics():
    metrics_list = ['ssim', 'r2']
    pred_name = 'test_pred'

    metrics_inst = metrics.MetricsEstimator(
        metrics_list=metrics_list,
        masked_metrics=False,
    )
    metrics_inst.estimate_yz_metrics(
        target=target_im,
        prediction=pred_im,
        pred_name=pred_name,
    )
    metrics_yz = metrics_inst.get_metrics_yz()
    nose.tools.assert_tuple_equal(metrics_yz.shape, (im_shape[1], 3))
    metrics_list.append('pred_name')
    nose.tools.assert_list_equal(list(metrics_yz), metrics_list)