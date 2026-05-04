# Generic tests file for use with the pytest module

# Test functions have names starting with "test_"
# These will all be discovered by simply running pytest from the repo root.

# Tests can run python code for the package, or invoke command-line usage.

# Writing good tests is as much an art as a science.
# As the software gets more complex, more complex testing methods are needed.

# But generally, the best tests follow this structure:
# Arrange - set up conditions for the test
# Act - call some function, method or command
# Assert - check that some invariant is true.

import os
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd

from MRISnapshot.create_report import calc_sel_slices, extract_snapshot, parse_config, read_and_check_images
from MRISnapshot.prep_data import prep_data
from MRISnapshot.utils.img_overlays import overlayImageDouble
    
# This simple, system-level test just runs the command line version of the package as in the example script.
# "monkeypatch" is a text fixture that lets you change context scoped to the test.
def test_system_cli(monkeypatch):
    monkeypatch.chdir("./examples/scripts")
    exitcode = os.system("./run_example.sh")
    assert exitcode == 0, "exitcode != 0"


def test_read_and_check_images_accepts_matching_mask_affine(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    affine = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 239.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    data = np.ones((8, 8, 8), dtype=np.float32)

    underlay_file = tmp_path / "scan_rSRI.nii.gz"
    mask_file = tmp_path / "scan_rSRI_SSFinal.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), underlay_file)
    nib.save(nib.Nifti1Image(data, affine), mask_file)

    df_images = pd.DataFrame(
        {
            "ScanID": ["scan"],
            "UnderlayImg": [str(underlay_file)],
            "MaskImg": [str(mask_file)],
        }
    )
    df_conf = pd.DataFrame(
        {
            "ParamName": ["id_col", "ulay_col", "mask_col", "olay_col", "olay_col2"],
            "ParamValue": ["ScanID", "UnderlayImg", "MaskImg", "", ""],
        }
    )

    params = parse_config(df_conf, df_images.columns)
    qc_ok_flag, qc_msg, nii_out, fnames_out = read_and_check_images(
        df_images, params, 0, orient="LPS"
    )

    assert qc_ok_flag == 1
    assert qc_msg == "PASS"
    assert len(nii_out) == 4
    assert fnames_out[:2] == [str(underlay_file), str(mask_file)]


def test_parse_config_reads_overlay2_label_colors():
    df_conf = pd.DataFrame(
        {
            "ParamName": [
                "id_col",
                "ulay_col",
                "mask_col",
                "olay_col",
                "olay_col2",
                "is_edge_olay",
                "is_edge_olay2",
                "olay_color",
                "olay2_label_colors",
            ],
            "ParamValue": [
                "ScanID",
                "UnderlayImg",
                "",
                "OverlayImg",
                "OverlayImg2",
                "0",
                "1",
                "#0066ff",
                "1:#ffcc00+2:#ff3b30+3:#00bcd4",
            ],
        }
    )

    params = parse_config(df_conf, ["ScanID", "UnderlayImg", "OverlayImg", "OverlayImg2"])

    assert params.olay2_label_colors_parsed == {
        1: (255, 204, 0),
        2: (255, 59, 48),
        3: (0, 188, 212),
    }
    assert params.is_edge_olay == 0
    assert params.is_edge_olay2 == 1
    assert params.olay_color_parsed == (0, 102, 255)


def test_overlay_image_double_uses_distinct_colors_for_overlay2_labels():
    img2d_under = np.zeros((12, 12), dtype=float)
    img2d_over1 = np.zeros((12, 12), dtype=float)
    img2d_over2 = np.zeros((12, 12), dtype=float)

    img2d_over2[2:4, 2:4] = 1
    img2d_over2[5:7, 5:7] = 2
    img2d_over2[8:10, 8:10] = 3

    _, pil_fused = overlayImageDouble(
        img2d_under,
        img2d_over1,
        img2d_over2,
        transparency=1,
        is_edge=0,
        over2_label_colors={
            1: (255, 204, 0),
            2: (255, 59, 48),
            3: (0, 188, 212),
        },
    )

    fused = np.array(pil_fused)
    assert tuple(fused[2, 2][:3]) == (255, 204, 0)
    assert tuple(fused[5, 5][:3]) == (255, 59, 48)
    assert tuple(fused[8, 8][:3]) == (0, 188, 212)


def test_overlay_image_double_supports_filled_first_overlay_and_edge_second_overlay():
    img2d_under = np.zeros((16, 16), dtype=float)
    img2d_over1 = np.zeros((16, 16), dtype=float)
    img2d_over2 = np.zeros((16, 16), dtype=float)

    img2d_over1[2:6, 2:6] = 1
    img2d_over2[8:12, 8:12] = 1

    _, pil_fused = overlayImageDouble(
        img2d_under,
        img2d_over1,
        img2d_over2,
        transparency=1,
        is_edge1=0,
        is_edge2=1,
        over1_color=(0, 102, 255),
        over2_label_colors={1: (255, 204, 0)},
    )

    fused = np.array(pil_fused)
    assert tuple(fused[3, 3][:3]) == (0, 102, 255)
    assert tuple(fused[9, 9][:3]) == (0, 0, 0)
    assert tuple(fused[8, 9][:3]) == (255, 204, 0)


def _slice_params(num_slice='', step_size_slice='', min_vox=1, outside_slice_margin=1):
    return SimpleNamespace(
        num_slice=num_slice,
        step_size_slice=step_size_slice,
        min_vox=min_vox,
        outside_slice_margin=outside_slice_margin,
    )


def test_calc_sel_slices_axial_uses_tumor_centered_scheme():
    img_ulay = np.ones((4, 4, 10), dtype=int)
    img_mask = np.zeros((4, 4, 10), dtype=int)

    img_mask[0, 0, 2] = 1
    img_mask[0:4, 0:3, 3] = 1
    img_mask[0:4, 0:4, 4] = 1
    img_mask[0:2, 0:2, 5] = 1
    img_mask[0, 0, 6] = 1

    selected = calc_sel_slices(
        img_ulay,
        img_mask,
        None,
        None,
        _slice_params(),
        0,
        'subj',
        'A',
    )

    assert selected.tolist() == [1, 3, 4, 5, 7]


def test_calc_sel_slices_axial_respects_outside_slice_margin():
    img_ulay = np.ones((4, 4, 12), dtype=int)
    img_mask = np.zeros((4, 4, 12), dtype=int)

    img_mask[0, 0, 2] = 1
    img_mask[0:4, 0:4, 3] = 1
    img_mask[0:2, 0:2, 4] = 1
    img_mask[0, 0, 5] = 1
    img_mask[0, 0, 6] = 1

    selected = calc_sel_slices(
        img_ulay,
        img_mask,
        None,
        None,
        _slice_params(outside_slice_margin=2),
        0,
        'subj',
        'A',
    )

    assert selected.tolist() == [0, 2, 3, 4, 8]


def test_calc_sel_slices_axial_deduplicates_when_targets_overlap():
    img_ulay = np.ones((4, 4, 8), dtype=int)
    img_mask = np.zeros((4, 4, 8), dtype=int)

    img_mask[0:4, 0:4, 2] = 1
    img_mask[0, 0, 3] = 1
    img_mask[0, 0, 4] = 1

    selected = calc_sel_slices(
        img_ulay,
        img_mask,
        None,
        None,
        _slice_params(),
        0,
        'subj',
        'A',
    )

    assert selected.tolist() == [2, 3, 4]


def test_calc_sel_slices_sagittal_mask_uses_tumor_centered_scheme():
    img_ulay = np.ones((4, 4, 10), dtype=int)
    img_mask = np.zeros((4, 4, 10), dtype=int)

    img_mask[0, 0, 2] = 1
    img_mask[0:4, 0:3, 3] = 1
    img_mask[0:4, 0:4, 4] = 1
    img_mask[0:2, 0:2, 5] = 1
    img_mask[0, 0, 6] = 1

    selected = calc_sel_slices(
        img_ulay,
        img_mask,
        None,
        None,
        _slice_params(),
        0,
        'subj',
        'S',
    )

    assert selected.tolist() == [1, 3, 4, 5, 7]


def test_calc_sel_slices_coronal_mask_uses_tumor_centered_scheme():
    img_ulay = np.ones((4, 4, 10), dtype=int)
    img_mask = np.zeros((4, 4, 10), dtype=int)

    img_mask[0, 0, 2] = 1
    img_mask[0:4, 0:3, 3] = 1
    img_mask[0:4, 0:4, 4] = 1
    img_mask[0:2, 0:2, 5] = 1
    img_mask[0, 0, 6] = 1

    selected = calc_sel_slices(
        img_ulay,
        img_mask,
        None,
        None,
        _slice_params(),
        0,
        'subj',
        'C',
    )

    assert selected.tolist() == [1, 3, 4, 5, 7]


def test_calc_sel_slices_non_axial_without_mask_keeps_existing_spacing_logic():
    img_ulay = np.ones((4, 4, 10), dtype=int)
    img_mask = np.zeros((4, 4, 10), dtype=int)
    img_mask[:, :, 1:9] = 1

    selected = calc_sel_slices(
        img_ulay,
        None,
        None,
        None,
        _slice_params(num_slice=4),
        0,
        'subj',
        'S',
    )

    assert selected.tolist() == [2, 4, 5, 7]


def test_extract_snapshot_keeps_numeric_names_for_axial_masked_views(tmp_path):
    img_ulay = np.arange(16, dtype=float).reshape(4, 4, 1)
    params = SimpleNamespace(
        num_olay=0,
        alpha_olay=1,
        is_edge=0,
        is_edge_olay=0,
        is_edge_olay2=0,
        olay_color_parsed=(0, 102, 255),
        olay2_label_colors_parsed={},
    )

    _, axial_name = extract_snapshot(
        img_ulay,
        None,
        None,
        params,
        'A',
        0,
        0,
        'subj',
        str(tmp_path),
        np.array([10, 11, 9, 14, 6]),
    )
    _, sagittal_name = extract_snapshot(
        img_ulay,
        None,
        None,
        params,
        'S',
        0,
        2,
        'subj',
        str(tmp_path),
        np.array([10, 11, 9, 14, 6]),
    )

    assert axial_name == 'subj_A_0'
    assert sagittal_name == 'subj_S_2'


def test_prep_data_writes_outside_slice_margin_default(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()

    underlay = np.ones((4, 4, 4), dtype=np.float32)
    nib.save(nib.Nifti1Image(underlay, np.eye(4)), input_dir / 'subj1_rSRI.nii.gz')

    params = SimpleNamespace(
        in_dir=str(input_dir),
        out_dir=str(output_dir),
        s_ulay='_rSRI.nii.gz',
        s_mask=None,
        s_olay=None,
        s_olay2=None,
    )

    prep_data(params)

    config = pd.read_csv(output_dir / 'config.csv')
    outside_margin = config.loc[config['ParamName'] == 'outside_slice_margin', 'ParamValue']

    assert outside_margin.tolist() == [1]
