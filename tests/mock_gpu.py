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
        self.dtype = "float32"

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
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return FakeTensor(shape=args[0], numel=self._numel)
        return FakeTensor(shape=args, numel=self._numel)
        
    def expand(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return FakeTensor(shape=args[0], numel=self._numel)
        return FakeTensor(shape=args, numel=self._numel)
        
    def float(self): return self
    def clone(self): return self
    def as_strided(self, *args, **kwargs): return self

    def sum(self, *args, **kwargs):
        return FakeTensor(shape=(1,), numel=1)

    def permute(self, *dims):
        try:
            new_shape = tuple(self._shape[d] for d in dims) if self._shape else ()
            if len(new_shape) != len(dims):
                new_shape = tuple([1] * len(dims))
            return FakeTensor(shape=new_shape, numel=self._numel)
        except Exception:
            return FakeTensor(shape=tuple([1] * len(dims)), numel=self._numel)
        
    def contiguous(self): return self
    def byte(self): return self
    
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
        n_frames = (self._shape[0] - size) // step + 1 if self._shape else 0
        return FakeTensor(shape=(max(1, n_frames), size), numel=max(1, n_frames) * size)
        
    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx >= self._shape[0] or idx < -self._shape[0]:
                raise IndexError("Index out of bounds")
        return FakeTensor(shape=self._shape, numel=self._numel)
        
    def __setitem__(self, idx, value):
        pass
        
    def __len__(self):
        return self._shape[0] if self._shape else 0

def setup_mocks():
    if "torch" in sys.modules and isinstance(sys.modules["torch"], types.ModuleType) and hasattr(sys.modules["torch"], "tensor"):
        return # Already mocked

    torch_mock = create_mock_module("torch")
    torch_mock.cuda = create_mock_module("torch.cuda")
    torch_mock.cuda.is_available = lambda: False
    torch_mock.cuda.empty_cache = mock.MagicMock()
    torch_mock.device = lambda x: "cpu"
    torch_mock.tensor = lambda x, **kwargs: FakeTensor(shape=(len(x),) if isinstance(x, (list, tuple)) else (100,), numel=len(x) if isinstance(x, (list, tuple)) else 100)
    torch_mock.abs = lambda x: FakeTensor(shape=x.shape if hasattr(x, 'shape') else (100,))
    torch_mock.mean = lambda x, **kwargs: FakeTensor(shape=x.shape if hasattr(x, 'shape') else (100,))
    torch_mock.sqrt = lambda x: FakeTensor(shape=x.shape if hasattr(x, 'shape') else (100,))
    torch_mock.sum = lambda x, **kwargs: FakeTensor(shape=(100,))
    torch_mock.zeros = lambda x, **kwargs: FakeTensor(shape=x if isinstance(x, tuple) else (x,))
    torch_mock.cat = lambda x, **kwargs: FakeTensor(shape=x[0].shape if hasattr(x[0], 'shape') else (100,)) if isinstance(x, (list, tuple)) and len(x) > 0 else FakeTensor()
    torch_mock.ones = lambda x, **kwargs: FakeTensor(shape=(x,) if isinstance(x, int) else x)
    torch_mock.stack = lambda x, **kwargs: FakeTensor(shape=(len(x),) + x[0].shape if len(x)>0 and hasattr(x[0], 'shape') else (100,))
    torch_mock.arange = lambda *args, **kwargs: FakeTensor(shape=(args[0],) if len(args)==1 else (args[1]-args[0],))
    torch_mock.hann_window = lambda x, **kwargs: FakeTensor(shape=(x,) if isinstance(x, int) else x)
    torch_mock.stft = lambda x, **kwargs: FakeTensor(shape=(1025, 100))
    torch_mock.exp = lambda x: FakeTensor(shape=x.shape if hasattr(x, 'shape') else (100,))
    torch_mock.from_numpy = lambda x: FakeTensor(shape=x.shape, numel=x.size)
    torch_mock.float32 = "float32"
    torch_mock.float = "float"
    torch_mock.Tensor = FakeTensor
    class DummyNoGrad:
        def __call__(self, func): return func
        def __enter__(self): pass
        def __exit__(self, *args): pass
    torch_mock.no_grad = DummyNoGrad
    torch_mock.from_dlpack = mock.MagicMock()
    torch_mock.to_dlpack = mock.MagicMock()
    
    torch_mock.utils = create_mock_module("torch.utils")
    torch_mock.utils.dlpack = create_mock_module("torch.utils.dlpack")
    torch_mock.utils.dlpack.from_dlpack = mock.MagicMock()
    
    torch_mock.nn = create_mock_module("torch.nn")
    torch_mock.nn.functional = create_mock_module("torch.nn.functional")
    torch_mock.nn.functional.interpolate = mock.MagicMock(side_effect=lambda x, *args, **kwargs: FakeTensor(shape=x.shape if hasattr(x, 'shape') else (100,)))
    torch_mock.nn.functional.pad = mock.MagicMock(side_effect=lambda x, *args, **kwargs: FakeTensor(shape=x.shape if hasattr(x, 'shape') else (100,)))
    torch_mock.nn.functional.conv1d = mock.MagicMock()
    torch_mock.nn.functional.conv2d = mock.MagicMock(side_effect=lambda x, *args, **kwargs: FakeTensor(shape=x.shape if hasattr(x, 'shape') else (100,)))
    
    sys.modules["torch"] = torch_mock

    torchaudio_mock = create_mock_module("torchaudio")
    torchaudio_mock.load = mock.MagicMock()
    class MockAudioInfo:
        sample_rate = 48000
        num_frames = 48000
    torchaudio_mock.info = mock.MagicMock(return_value=MockAudioInfo())
    sys.modules["torchaudio"] = torchaudio_mock

    nvc_mock = create_mock_module("PyNvCodec")
    nvc_mock.PixelFormat = create_mock_module("PyNvCodec.PixelFormat")
    nvc_mock.PixelFormat.RGB = "RGB"
    nvc_mock.PixelFormat.BGR = "BGR"
    nvc_mock.PixelFormat.NV12 = "NV12"
    nvc_mock.PixelFormat.YUV420 = "YUV420"
    nvc_mock.SeekMode = create_mock_module("PyNvCodec.SeekMode")
    nvc_mock.SeekMode.PREV_KEY_FRAME = "PREV_KEY_FRAME"
    
    class MockDemuxer:
        def __init__(self, *args, **kwargs):
            self.mock_w = 1920
            self.mock_h = 1080
            self.mock_fps = 30.0
            self.mock_fmt = nvc_mock.PixelFormat.RGB
        def Format(self): return self.mock_fmt
        def Width(self): return self.mock_w
        def Height(self): return self.mock_h
        def Framerate(self): return self.mock_fps
        def Numframes(self): return 1000
        def Codec(self): return 0
        def Seek(self, *args, **kwargs): pass
        def Timebase(self): return 0.01
        def DemuxSinglePacket(self, packet): return False
        def LastPacketData(self, pkt_data): pass
    nvc_mock.PyFFmpegDemuxer = MockDemuxer

    nvc_mock.PyNvDecoder = mock.MagicMock(side_effect=lambda *args, **kwargs: mock.MagicMock())
    nvc_mock.PySurfaceResizer = mock.MagicMock(side_effect=lambda *args, **kwargs: mock.MagicMock())
    nvc_mock.PySurfaceConverter = mock.MagicMock(side_effect=lambda *args, **kwargs: mock.MagicMock())
    
    nvc_mock.SeekContext = mock.MagicMock()
    nvc_mock.PacketData = mock.MagicMock
    nvc_mock.ColorspaceConversionContext = mock.MagicMock()
    nvc_mock.ColorSpace = create_mock_module("PyNvCodec.ColorSpace")
    nvc_mock.ColorSpace.BT_601 = "BT_601"
    nvc_mock.ColorRange = create_mock_module("PyNvCodec.ColorRange")
    nvc_mock.ColorRange.MPEG = "MPEG"
    
    nvc_mock.Surface = create_mock_module("PyNvCodec.Surface")
    nvc_mock.Surface.Make = mock.MagicMock()
    sys.modules["PyNvCodec"] = nvc_mock
    
    pnvc_mock = create_mock_module("PytorchNvCodec")
    pnvc_mock.make_tensor = mock.MagicMock(return_value=FakeTensor(shape=(100,)))
    pnvc_mock.DptrToTensor = mock.MagicMock(return_value=FakeTensor(shape=(1080, 1920, 3)))
    sys.modules["PytorchNvCodec"] = pnvc_mock

setup_mocks()
