"""Optional CPU geometry kernel. Compilation is explicit startup work, never a frame operation."""
import ctypes
import hashlib
import os
import platform
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np

_kernel=None
_library=None
def prepare_route_projection():
    global _kernel, _library
    if _kernel is not None:return True
    compiler=shutil.which('gcc')
    if compiler is None:return False
    source=Path(__file__).with_name('route_projection_kernel.c')
    flags=['-O3','-fno-fast-math','-ffp-contract=off','-shared','-fPIC']
    identity = (str(flags), platform.machine(), platform.system())
    key=hashlib.sha256(source.read_bytes()+str(identity).encode()).hexdigest()[:20]
    cache=Path(tempfile.gettempdir())/f'autodrive_geometry_{os.getuid()}'
    cache.mkdir(mode=0o700,exist_ok=True)
    if cache.stat().st_uid != os.getuid() or cache.stat().st_mode & 0o022:
        raise RuntimeError('Unsafe geometry cache ownership or permissions')
    binary=cache/f'route_projection_{key}.so'
    if not binary.exists():
        with tempfile.TemporaryDirectory(dir=cache) as build:
            pending=Path(build)/'kernel.so'
            subprocess.run([compiler,*flags,str(source),'-lm','-o',str(pending)],check=True,capture_output=True,timeout=30)
            pending.replace(binary)
    library=ctypes.CDLL(str(binary));kernel=library.project_route
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    kernel.argtypes=[array]*5+[ctypes.c_size_t]*2+[array]*3;kernel.restype=ctypes.c_int
    _library=library
    _kernel=kernel
    return True

def projection_available():return _kernel is not None

def project_points_to_route(xyz,road,lengths=None):
    if _kernel is None:raise RuntimeError('CPU projection must be prepared at startup')
    xyz=np.ascontiguousarray(xyz,dtype=np.float64);road=np.ascontiguousarray(road,dtype=np.float64)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or road.ndim!=2 or road.shape[1]!=3 or len(road)<2:
        raise ValueError('Projection requires XYZ arrays and at least two route vertices')
    lengths=np.ascontiguousarray(np.linalg.norm(np.diff(road[:,:2],axis=0),axis=1) if lengths is None else lengths,dtype=np.float64)
    if lengths.shape!=(len(road)-1,) or not np.isfinite(lengths).all() or np.any(lengths<=0):
        raise ValueError('Projection segment lengths must be finite and positive')
    if not np.isfinite(xyz).all() or not np.isfinite(road).all():raise ValueError('Nonfinite projection input')
    squared=np.array([length**2 for length in lengths]);offsets=np.r_[0.,np.cumsum(lengths)[:-1]]
    outputs=tuple(np.empty(len(xyz),dtype=np.float64) for _ in range(3))
    code=_kernel(xyz,road,lengths,squared,offsets,len(xyz),len(lengths),*outputs)
    if code:raise ValueError('Projection size outside supported bounds')
    return outputs
