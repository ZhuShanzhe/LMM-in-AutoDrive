"""Optional exact support screening; explicit initialization before inference."""
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


def prepare_support_precheck():
    global _kernel,_library
    if _kernel is not None:return True
    compiler=shutil.which('gcc')
    if compiler is None:return False
    source=Path(__file__).with_name('radar_support_kernel.c')
    flags=['-O3','-fno-fast-math','-ffp-contract=off','-shared','-fPIC']
    identity=(str(flags),platform.machine(),platform.system())
    key=hashlib.sha256(source.read_bytes()+str(identity).encode()).hexdigest()[:20]
    cache=Path(tempfile.gettempdir())/f'autodrive_geometry_{os.getuid()}'
    cache.mkdir(mode=0o700,exist_ok=True)
    if cache.stat().st_uid!=os.getuid() or cache.stat().st_mode & 0o022:
        raise RuntimeError('Unsafe geometry cache ownership or permissions')
    binary=cache/f'radar_support_{key}.so'
    if not binary.exists():
        with tempfile.TemporaryDirectory(dir=cache) as folder:
            pending=Path(folder)/'kernel.so'
            subprocess.run([compiler,*flags,
                str(source),'-lm','-o',str(pending)],check=True,capture_output=True,timeout=30)
            pending.replace(binary)
    _library=ctypes.CDLL(str(binary));_kernel=_library.screen_support
    double=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    integer=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    output=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    _kernel.argtypes=[double,double,double,integer,integer,ctypes.c_size_t,ctypes.c_size_t,output]
    _kernel.restype=ctypes.c_int
    return True


def screen_support(cloud,points,neighborhoods,matrix):
    if _kernel is None:return None
    cloud=np.ascontiguousarray(cloud,dtype=np.float64)
    points=np.ascontiguousarray(points,dtype=np.float64)
    matrix=np.ascontiguousarray(matrix,dtype=np.float64)
    if not len(points):return np.empty(0,dtype=np.int32)
    if (cloud.ndim!=2 or cloud.shape[1]!=3 or points.shape!=(len(neighborhoods),3) or matrix.shape!=(4,4)
            or not np.isfinite(cloud).all() or not np.isfinite(points).all()
            or not np.isfinite(matrix).all()):return None
    offsets=np.r_[0,np.cumsum([len(x) for x in neighborhoods],dtype=np.int64)]
    indices=np.concatenate([np.asarray(x,dtype=np.int64) for x in neighborhoods])
    result=np.zeros(len(points),dtype=np.int32)
    status=_kernel(cloud,points,matrix,indices,offsets,len(points),len(cloud),result)
    if status:raise ValueError('Invalid ground support indices')
    return result
