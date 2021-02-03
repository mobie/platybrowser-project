import os
import numpy as np
import torch

from pybio.spec.utils.transformers import load_and_resolve_spec
from pybio.spec.utils import get_instance


# TODO this is missing the normalization (preprocessing)
def check_model(path):
    """ Convert model weights from format 'pytorch_state_dict' to 'torchscript'.
    """
    spec = load_and_resolve_spec(path)

    with torch.no_grad():
        print("Loading inputs and outputs:")
        # load input and expected output data
        input_data = np.load(spec.test_inputs[0]).astype('float32')
        input_data = torch.from_numpy(input_data)
        expected_output_data = np.load(spec.test_outputs[0]).astype(np.float32)
        print(input_data.shape)

        # instantiate and trace the model
        print("Predicting model")
        model = get_instance(spec)
        state = torch.load(spec.weights['pytorch_state_dict'].source)
        model.load_state_dict(state)

        # check the scripted model
        output_data = model(input_data).numpy()
        assert output_data.shape == expected_output_data.shape
        assert np.allclose(expected_output_data, output_data)
        print("Check passed")


# TODO this is missing the normalization (preprocessing)
def generate_output(path):
    spec = load_and_resolve_spec(path)

    with torch.no_grad():
        print("Loading inputs and outputs:")
        # load input and expected output data
        input_data = np.load(spec.test_inputs[0]).astype('float32')
        input_data = torch.from_numpy(input_data)

        # instantiate and trace the model
        print("Predicting model")
        model = get_instance(spec)
        state = torch.load(spec.weights['pytorch_state_dict'].source)
        model.load_state_dict(state)

        # check the scripted model
        output_data = model(input_data).numpy()
        assert output_data.shape == input_data.shape
        np.save('./test_output.npy', output_data)


def resave_data():
    halo = [32, 48, 48]
    x = np.load('./test_input.npz')['arr_0']
    shape = x.shape[2:]
    bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape, halo))
    bb = (slice(None), slice(None)) + bb
    x = x[bb]
    print(x.shape)
    np.save('./test_input.npy', x)

    y = np.load('./test_output.npz')['arr_0']
    y = y[bb]
    print(y.shape)
    np.save('./test_output.npy', y)


if __name__ == '__main__':
    # resave and crop the older test data
    # resave_data()

    path = os.path.abspath('./UNet3DPlatyCellProbs.model.yaml')

    # generate expected output again
    # generate_output(path)

    # check model predictions against the output
    check_model(path)
