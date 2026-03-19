import sys
import types
from unittest import mock

def create_mock_module(name):
    m = types.ModuleType(name)
    m.__path__ = []
    # Provide simple safe defaults for common attributes to avoid pytest introspection loops
    m.__file__ = f"/tmp/{name}_mock.py"
    return m

class FakeTensor:
    def __init__(self, shape=(100,), numel=100):
        self._shape = shape
        self._numel = numel

    @property
    def shape(self):
        return self._shape

    @property
    def device(self):
        return "cpu"

    def numel(self):
        return self._numel

    def dim(self):
        return len(self._shape) if self._shape else 0

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



    def view(self, *args, **kwargs):
        if args and args[0] == -1:
            return FakeTensor(shape=(self._numel,), numel=self._numel)
        return FakeTensor(shape=(len(args),), numel=self._numel)
        
    def float(self): return self

    def permute(self, *dims):
        new_shape = tuple(self._shape[d] for d in dims) if self._shape else ()
        return FakeTensor(shape=new_shape, numel=self._numel)
        
    def contiguous(self): return self
    def byte(self): return self
    def clone(self): return self
    
    def numpy(self):
        import numpy as np
        return np.zeros(self._shape if self._shape else (1,), dtype=np.uint8)
    def __add__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    def __sub__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    def __mul__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    def __truediv__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    def __pow__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    
    def __radd__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    def __rsub__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    def __rmul__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)
    def __rtruediv__(self, other): return FakeTensor(shape=self._shape, numel=self._numel)

    def unfold(self, dimension, size, step):
        # Dummy unfold that returns a 2D tensor
        n_frames = (self._shape[0] - size) // step + 1 if self._shape else 0
        return FakeTensor(shape=(max(1, n_frames), size), numel=max(1, n_frames) * size)
        
    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx >= self._shape[0] or idx < -self._shape[0]:
                raise IndexError("Index out of bounds")
        return FakeTensor(shape=self._shape, numel=self._numel)
        
    def __setitem__(self, idx, value):
        pass # Ignore for testing purposes
        
    def __len__(self):
        return self._shape[0] if self._shape else 0

def setup_mocks():
    if "torch" in sys.modules and isinstance(sys.modules["torch"], types.ModuleType) and hasattr(sys.modules["torch"], "tensor"):
        return # Already mocked

    torch_mock = create_mock_module("torch")
    torch_mock.cuda = create_mock_module("torch.cuda")
    torch_mock.cuda.is_available = lambda: False
    torch_mock.device = lambda x: "cpu"
    torch_mock.tensor = lambda x, **kwargs: FakeTensor()
    torch_mock.abs = lambda x: FakeTensor(shape=x.shape if hasattr(x, 'shape') else (100,))
    torch_mock.mean = lambda x, **kwargs: FakeTensor(shape=x.shape if hasattr(x, 'shape') else (100,))
    torch_mock.sqrt = lambda x: FakeTensor(shape=x.shape if hasattr(x, 'shape') else (100,))
    torch_mock.sum = lambda x, **kwargs: FakeTensor(shape=(100,))
    torch_mock.zeros = lambda x, **kwargs: FakeTensor(shape=x if isinstance(x, tuple) else (x,))
    torch_mock.cat = lambda x, **kwargs: FakeTensor(shape=x[0].shape if hasattr(x[0], 'shape') else (100,)) if isinstance(x, (list, tuple)) and len(x) > 0 else FakeTensor()
    torch_mock.ones = lambda x, **kwargs: FakeTensor(shape=(x,) if isinstance(x, int) else x)
    torch_mock.stack = lambda x, **kwargs: FakeTensor(shape=(len(x),) + x[0].shape if len(x)>0 and hasattr(x[0], 'shape') else (100,))
    torch_mock.arange = lambda x, **kwargs: FakeTensor(shape=(x,) if isinstance(x, int) else x)
    torch_mock.hann_window = lambda x, **kwargs: FakeTensor(shape=(x,) if isinstance(x, int) else x)
    torch_mock.stft = lambda x, **kwargs: FakeTensor(shape=(1025, 100))
    torch_mock.from_numpy = lambda x: x
    torch_mock.no_grad = mock.MagicMock()
    torch_mock.from_dlpack = mock.MagicMock()
    torch_mock.to_dlpack = mock.MagicMock()
    
    torch_mock.utils = create_mock_module("torch.utils")
    torch_mock.utils.dlpack = create_mock_module("torch.utils.dlpack")
    torch_mock.utils.dlpack.from_dlpack = mock.MagicMock()
    
    torch_mock.nn = create_mock_module("torch.nn")
    torch_mock.nn.functional = create_mock_module("torch.nn.functional")
    torch_mock.nn.functional.interpolate = mock.MagicMock()
    torch_mock.nn.functional.pad = mock.MagicMock()
    torch_mock.nn.functional.conv1d = mock.MagicMock()
    
    sys.modules["torch"] = torch_mock

    torchaudio_mock = create_mock_module("torchaudio")
    torchaudio_mock.load = mock.MagicMock()
    sys.modules["torchaudio"] = torchaudio_mock

    nvc_mock = create_mock_module("PyNvCodec")
    nvc_mock.PixelFormat = create_mock_module("PyNvCodec.PixelFormat")
    nvc_mock.PixelFormat.RGB = "RGB"
    nvc_mock.PixelFormat.BGR = "BGR"
    nvc_mock.SeekMode = create_mock_module("PyNvCodec.SeekMode")
    nvc_mock.SeekMode.PREV_KEY_FRAME = "PREV_KEY_FRAME"
    
    class MockDemuxer:
        def __init__(self, *args, **kwargs): pass
        def Width(self): return 1280
        def Height(self): return 720
        def Framerate(self): return 30.0
        def Numframes(self): return 1000
        def Format(self): return "nv12"
        def Codec(self): return "h264"
        def DemuxSinglePacket(self, packet): return False
        def Seek(self, *args, **kwargs): pass
    nvc_mock.PyFFmpegDemuxer = MockDemuxer

    nvc_mock.PyNvDecoder = mock.MagicMock()
    nvc_mock.PySurfaceResizer = mock.MagicMock()
    nvc_mock.PySurfaceConverter = mock.MagicMock()
    
    nvc_mock.Surface = create_mock_module("PyNvCodec.Surface")
    nvc_mock.Surface.Make = mock.MagicMock()
    sys.modules["PyNvCodec"] = nvc_mock
    
    pnvc_mock = create_mock_module("PytorchNvCodec")
    pnvc_mock.make_tensor = mock.MagicMock(return_value=FakeTensor(shape=(3, 720, 1280), numel=3*720*1280))
    sys.modules["PytorchNvCodec"] = pnvc_mock

    cupy_mock = create_mock_module("cupy")
    cupy_mock.asarray = mock.MagicMock(side_effect=lambda x: x)
    cupy_mock.asnumpy = mock.MagicMock(side_effect=lambda x: x)
    cupy_mock.from_dlpack = mock.MagicMock()
    cupy_mock.to_dlpack = mock.MagicMock()
    sys.modules["cupy"] = cupy_mock

    cupyx_mock = create_mock_module("cupyx")
    cupyx_mock.scipy = create_mock_module("cupyx.scipy")
    cupyx_mock.scipy.ndimage = create_mock_module("cupyx.scipy.ndimage")
    cupyx_mock.scipy.ndimage.gaussian_filter = mock.MagicMock()
    sys.modules["cupyx"] = cupyx_mock
    sys.modules["cupyx.scipy"] = cupyx_mock.scipy
    sys.modules["cupyx.scipy.ndimage"] = cupyx_mock.scipy.ndimage

# Execute it once unconditionally when imported
setup_mocks()
