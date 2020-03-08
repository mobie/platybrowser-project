from pathlib import Path

import numpy
import torch

from mmpb.segmentation.network.models import UNetAnisotropic
from pybio.core.transformations import apply_transformations
from pybio.spec import load_model
from pybio.spec.utils import get_instance


def test_forward(cache_path):
    spec_path = (
        Path(__file__).parent / "../../segmentation/cells/UNet3DPlatyCellProbs.model/UNet3DPlatyCellProbs.model.yaml"
    )
    assert spec_path.exists(), spec_path.absolute()

    pybio_model = load_model(str(spec_path), cache_path=cache_path)
    assert pybio_model.spec.outputs[0].shape.reference_input == "raw"
    assert pybio_model.spec.outputs[0].shape.scale == (1, 1, 1, 1, 1)
    assert pybio_model.spec.outputs[0].shape.offset == (0, 0, 0, 0, 0)

    model: torch.nn.Module = get_instance(pybio_model)
    assert hasattr(model, "forward")
    assert isinstance(model, UNetAnisotropic)
    model_weights = torch.load(pybio_model.spec.prediction.weights.source, map_location=torch.device("cpu"))
    model.load_state_dict(model_weights)
    pre_transformations = [get_instance(trf) for trf in pybio_model.spec.prediction.preprocess]
    post_transformations = [get_instance(trf) for trf in pybio_model.spec.prediction.postprocess]
    ipt_npz = numpy.load(str(pybio_model.spec.test_input))
    # npz to ndarray
    ipt = [ipt_npz[ipt_npz.files[0]]]
    ipt_npz.close()
    assert len(ipt) == len(pybio_model.spec.inputs)
    assert isinstance(ipt, list)
    assert len(ipt) == 1
    ipt = ipt[0]
    assert ipt.shape == pybio_model.spec.inputs[0].shape

    # Don't test with the real test io, but a smaller test_roi.
    # Because the results differ due to edge effects, load the small test output instead
    # (this one is not linked to in the model.yaml)
    assert len(ipt.shape) == 5, ipt.shape
    test_roi = (slice(0, 1), slice(0, 1), slice(0, 32), slice(0, 32), slice(0, 32))  # to lower test mem consumption
    ipt = ipt[test_roi]
    expected_npz = numpy.load(str(pybio_model.spec.test_output).replace("test_output.npz", "test_output_small.npz"))
    # npz to npy
    expected = expected_npz[expected_npz.files[0]]
    expected_npz.close()

    ipt = apply_transformations(pre_transformations, ipt)
    assert isinstance(ipt, list)
    assert len(ipt) == 1
    out = model.forward(*ipt)
    out = apply_transformations(post_transformations, out)
    assert isinstance(out, list)
    assert len(out) == 1
    out = out[0]
    # assert out.shape == pybio_model.spec.inputs[0].shape  # test_roi makes this test invalid
    assert str(out.dtype).split(".")[-1] == pybio_model.spec.outputs[0].data_type
    assert numpy.allclose(out, expected)
