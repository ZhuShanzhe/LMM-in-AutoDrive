"""Sample host/process resources without attributing whole-GPU usage to a model."""
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


def gpu_snapshot():
    executable=shutil.which('nvidia-smi')
    if executable is None:
        return dict(status='UNAVAILABLE',reason='nvidia-smi not found',devices=[])
    try:
        result=subprocess.run([executable,'--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu',
                               '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or 'nvidia-smi failed')
        devices=[]
        for row in csv.reader(io.StringIO(result.stdout)):
            if not row:
                continue
            if len(row)!=5:
                raise ValueError('unexpected nvidia-smi output')
            def numeric(value):
                try:
                    number=float(value.strip())
                    return number if math.isfinite(number) and number>=0 else None
                except ValueError:
                    return None
            devices.append(dict(uuid=row[0].strip(),name=row[1].strip(),
                                used_mib=numeric(row[2]),total_mib=numeric(row[3]),utilization_pct=numeric(row[4])))
        return dict(status='AVAILABLE' if devices else 'UNAVAILABLE',scope='whole_device_all_processes',devices=devices)
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as error:
        return dict(status='UNAVAILABLE',reason=str(error),devices=[])


class ResourceSampler:
    def __init__(self,path,interval_s=1.0,pids=None):
        if not math.isfinite(interval_s) or interval_s<.2:
            raise ValueError('resource interval must be finite and at least 0.2s')
        self.path=Path(path)
        self.interval=interval_s
        self.pids=dict(pids or {'model_host':os.getpid()})
        self.stop_event=threading.Event()
        self.thread=None
        self.samples=0
        self.errors=[]
        self.rss_peaks={}
        self.gpu_peaks={}
        self.gpu_available_samples=0
        self.processes={}
        self.process_error=None
        try:
            import psutil
            for name,pid in self.pids.items():
                process=psutil.Process(pid)
                process.cpu_percent(None)
                self.processes[name]=process
        except Exception as error:
            self.process_error=str(error)

    def start(self):
        if self.thread is not None:
            raise RuntimeError('resource sampler already started')
        self.stream=self.path.open('x',encoding='utf-8')
        self.started=time.monotonic()
        self.thread=threading.Thread(target=self._loop,name='resource-sampler',daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        processes={}
        for name,pid in self.pids.items():
            try:
                process=self.processes[name]
                rss=process.memory_info().rss
                processes[name]=dict(pid=pid,status='AVAILABLE',rss_bytes=rss,
                    cpu_pct=process.cpu_percent(None) if self.samples else None,
                    cpu_scope='process; may exceed 100 percent across cores')
                self.rss_peaks[name]=max(self.rss_peaks.get(name,0),rss)
            except Exception as error:
                processes[name]=dict(pid=pid,status='UNAVAILABLE',reason=self.process_error or str(error))
        gpu=gpu_snapshot()
        self.gpu_available_samples+=int(gpu['status']=='AVAILABLE')
        for device in gpu['devices']:
            if device['used_mib'] is not None:
                key=device['uuid']
                self.gpu_peaks[key]=max(self.gpu_peaks.get(key,0),device['used_mib'])
        allocator={'status':'UNAVAILABLE','scope':'current_process_pytorch_allocator_only'}
        torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            allocator.update(status='AVAILABLE',devices=[dict(index=i,
                allocated_bytes=torch.cuda.memory_allocated(i),reserved_bytes=torch.cuda.memory_reserved(i))
                for i in range(torch.cuda.device_count())])
        return dict(elapsed_wall_s=time.monotonic()-self.started,processes=processes,
                    gpu=gpu,pytorch_allocator=allocator)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                row=self._sample()
                self.stream.write(json.dumps(row,allow_nan=False)+'\n')
                self.stream.flush()
                self.samples+=1
            except Exception as error:
                self.errors.append(str(error))
                break
            self.stop_event.wait(self.interval)

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
            self.stream.close()
        return dict(schema_version='resource_summary/1.0',sample_count=self.samples,
                    interval_s=self.interval,process_rss_sampled_peak_bytes=self.rss_peaks,
                    whole_gpu_sampled_peak_mib=self.gpu_peaks,gpu_available_samples=self.gpu_available_samples,
                    errors=self.errors,process_probe_error=self.process_error,
                    limitations=['sampled peaks may miss short spikes','whole GPU includes CARLA and unrelated processes',
                                 'PyTorch allocator excludes other CUDA libraries','initialization before sampler start excluded'])


class OnlineResourceJournal:
    """Best-effort telemetry that never changes the controller's action path."""

    def __init__(self, output_path):
        base = Path(output_path)
        self.path = base.with_name(base.stem + '_resources.jsonl')
        self.summary_path = base.with_name(base.stem + '_resources_summary.json')
        self.sampler = None
        self.attempted = False
        self.result = None
        self.error = None

    def start(self):
        if self.attempted:
            return
        self.attempted = True
        try:
            self.sampler = ResourceSampler(self.path).start()
        except Exception as error:
            self.error = str(error)

    def close(self):
        if self.result is not None:
            return self.result
        try:
            summary = self.sampler.close() if self.sampler is not None else {}
            self.result = dict(status='RECORDED' if summary.get('sample_count', 0)
                               and not summary.get('errors') else 'UNAVAILABLE',
                               scope='online_control_loop; model initialization excluded',
                               sampling=summary, error=self.error)
        except Exception as error:
            self.result = dict(status='UNAVAILABLE', error=str(error))
        try:
            with self.summary_path.open('x', encoding='utf-8') as stream:
                json.dump(self.result, stream, indent=2, allow_nan=False)
        except Exception as error:
            self.result['summary_write_error'] = str(error)
        return self.result
