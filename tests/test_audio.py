import sys
from pathlib import Path
from unittest import mock
import numpy as np

class FakeTensor:
    def __init__(self, shape=(100,), numel=100):
        self._shape = shape
        self._numel = numel

    @property
    def shape(self):
        return self._shape

    def numel(self):
        return self._numel

    def squeeze(self, *args, **kwargs):
        if len(self._shape) > 1 and self._shape[0] == 1:
            return FakeTensor(shape=self._shape[1:], numel=self._numel)
        return FakeTensor(shape=self._shape[1:] if len(self._shape)>1 else self._shape, numel=self._numel)

    def unsqueeze(self, *args, **kwargs):
        return FakeTensor(shape=(1,) + self._shape, numel=self._numel)

    def mean(self, *args, **kwargs):
        return FakeTensor(shape=self._shape, numel=self._numel)

    def std(self, *args, **kwargs):
        return FakeTensor(shape=self._shape, numel=self._numel)

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array([0.1, 0.2, 0.3])

    def view(self, *args, **kwargs):
        # A rough emulation
        if args and args[0] == -1:
            return FakeTensor(shape=(self._numel,), numel=self._numel)
        return FakeTensor(shape=(len(args),), numel=self._numel)
        
    def unfold(self, *args, **kwargs):
        return FakeTensor(shape=self._shape, numel=self._numel)

    # Arithmetic
    def __add__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    def __sub__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    def __mul__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    def __truediv__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    def __pow__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    
    def __radd__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    def __rsub__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    def __rmul__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    def __rtruediv__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)

    # Array logic
    def __getitem__(self, idx):
        if isinstance(idx, int) and idx >= self._shape[0]:
            raise IndexError("Index out of bounds")
        if isinstance(idx, int) and idx < 0:
            raise IndexError("Index out of bounds")
        return FakeTensor(shape=self._shape, numel=self._numel)
        
    def __len__(self):
        return self._shape[0] if self._shape else 0



sys.path.append(str(Path(__file__).resolve().parent.parent))

import tests.mock_gpu  # noqa: F401, E402
import shorts  # noqa: E402
from shorts import compute_audio_action_profile  # noqa: E402

def test_compute_audio_action_profile_load_failure():
    shorts.torchaudio.load.side_effect = Exception("Failed")
    t, s = compute_audio_action_profile(Path("dummy.mp4"))
    assert len(t) == 0
    assert len(s) == 0

def test_compute_audio_action_profile_success():
    shorts.torchaudio.load.side_effect = None
    
    waveform = FakeTensor(shape=(2, 48000), numel=96000)
    shorts.torchaudio.load.return_value = (waveform, 48000)
    
    shorts.torch.mean.return_value = FakeTensor(shape=(1, 48000), numel=48000)
    shorts.torch.sqrt.return_value = FakeTensor(shape=(100,), numel=100)
    shorts.torch.cat.side_effect = lambda x, **kwargs: x[0] if x else FakeTensor(shape=(0,), numel=0)
    shorts.torch.abs.return_value = FakeTensor(shape=(100,), numel=100)
    shorts.torch.sum.return_value = FakeTensor(shape=(100,), numel=100)
    shorts.torch.zeros.return_value = FakeTensor(shape=(1025,), numel=1025)
    shorts.torch.ones.return_value = FakeTensor(shape=(21,), numel=21)
    shorts.torch.arange.return_value = FakeTensor(shape=(100,), numel=100)
    shorts.torch.hann_window.return_value = FakeTensor(shape=(2048,), numel=2048)
    
    def pad_mock(tensor, pad, **kwargs):
        return FakeTensor(shape=(1, tensor.shape[1] + pad[0] + pad[1]), numel=tensor.numel() + pad[0] + pad[1])
    shorts.torch.nn.functional = mock.MagicMock()
    shorts.torch.nn.functional.pad.side_effect = pad_mock
    
    def conv1d_mock(*args, **kwargs):
        return FakeTensor(shape=(100,), numel=100)
    shorts.torch.nn.functional.conv1d.side_effect = conv1d_mock
    
    times, score = compute_audio_action_profile(Path("dummy.mp4"), frame_length=2048, hop_length=512)
    
    assert list(times) == [0.1, 0.2, 0.3]
    assert list(score) == [0.1, 0.2, 0.3]
    shorts.torchaudio.load.assert_called()
