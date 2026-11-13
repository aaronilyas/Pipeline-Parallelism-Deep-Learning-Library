# Baseline

Workload W is a 12-layer MLP, `y = gelu(x @ W + b)`, width 4096, float32.
One step is 8 microbatches of 256 rows, mean-squared error (`torch.nn.MSELoss`),
and SGD at `1e-3`. Samples/sec is `2048 / median(step time)` over 50 steps after 5 warmup steps.
The median of an even count is the average of the two central values.
Pipeline step time is the max across ranks after a gloo barrier.
Byte counts are the totals sent during the measured steps, one record per stage boundary.
AMP is off. TF32 is PyTorch's default, recorded in each JSON and not changed.
Torch release `2.14.1` is installed on every training host.
The GTX 1080 host uses the CUDA 12.6 wheel (`--index-url https://download.pytorch.org/whl/cu126`).
The Mac uses the macOS arm64 wheel of the same release. No number here is estimated.

## Hub

Coordinator `omarchy` is not a training rank. CPU `AMD Athlon(tm) X4 860K Quad Core Processor`. GPU `NVIDIA GeForce GTX 960, 5.2, 2048 MiB`.

## Single host

| ssh | hostname | device | samples/sec | median step s | status | command | commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| macmini | Aarons-Mac-mini-2.local | mps | 1836.7688640466938 | 1.1150014790036948 | ok | `ssh macmini ~/ppdl-w-bench/venv/bin/python ~/ppdl-w-bench/pytorch/single_host.py ~/ppdl-w-bench/jobs/single-macmini-mps.json` | `51b57a4b6babe8e1ea47e753e5e44b2f0050178a` |
| ubuntu | aaron-B250M-DS3H | cuda | 4509.8415858222115 | 0.4541179465013556 | ok | `ssh ubuntu ~/ppdl-w-bench/venv/bin/python ~/ppdl-w-bench/pytorch/single_host.py ~/ppdl-w-bench/jobs/single-ubuntu-cuda.json` | `51b57a4b6babe8e1ea47e753e5e44b2f0050178a` |
| windows | Aaron | cuda | 6827.089946088578 | 0.29998140000679996 | ok | `ssh windows & $env:USERPROFILE\ppdl-w-bench\venv\Scripts\python.exe $env:USERPROFILE\ppdl-w-bench\pytorch\single_host.py $env:USERPROFILE\ppdl-w-bench\jobs\single-windows-cuda.json` | `51b57a4b6babe8e1ea47e753e5e44b2f0050178a` |

## Pipeline

Gloo did not form a group on the full host set. Attempts:

```
[
  {
    "port": 29631,
    "ranks": [
      {
        "ssh": "windows",
        "status": "not run",
        "error": "C:\\Users\\aaron\\ppdl-w-bench\\venv\\Lib\\site-packages\\torch\\_subclasses\\functional_tensor.py:368: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\torch\\csrc\\utils\\tensor_numpy.cpp:84.)\n  cpu = _conversion_method_template(device=torch.device(\"cpu\"))\n[W1112 19:41:44.000000000 socket.cpp:764] [c10d] The client socket has failed to connect to [aaron-B250M-DS3H.lan]:29631 (system error: 10049 - The requested address is not valid in its context.).\nFatal Python error: Aborted\n\nCurrent thread 0x00004de8 (most recent call first):\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\venv\\Lib\\site-packages\\torch\\distributed\\distributed_c10d.py\", line 669 in _create_gloo_process_group\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\venv\\Lib\\site-packages\\torch\\distributed\\distributed_c10d.py\", line 2874 in _new_process_group_helper\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\venv\\Lib\\site-packages\\torch\\distributed\\distributed_c10d.py\", line 2553 in init_process_group\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\venv\\Lib\\site-packages\\torch\\distributed\\c10d_logger.py\", line 97 in wrapper\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\venv\\Lib\\site-packages\\torch\\distributed\\c10d_logger.py\", line 83 in wrapper\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\pytorch\\rendezvous.py\", line 39 in init_gloo\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\pytorch\\gloo_probe.py\", line 33 in main\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\pytorch\\gloo_probe.py\", line 75 in <module>\n\nExtension modules: torch._C, torch._C._dynamo.autograd_compiler, torch._C._dynamo.eval_frame, torch._C._dynamo.guards, torch._C._dynamo.utils, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special (total: 11)\n"
      },
      {
        "ssh": "macmini",
        "status": "not run",
        "error": "/Users/aaronilyas/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:368: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /Users/runner/work/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)\n  cpu = _conversion_method_template(device=torch.device(\"cpu\"))\nlibc++abi: terminating due to uncaught exception of type gloo::EnforceNotMet: [enforce fail at /Users/runner/work/pytorch/pytorch/third_party/gloo/gloo/transport/uv/pair.cc:554] state_ != CONNECTING. 1 vs 1. Cannot close pair while waiting on connection\nFatal Python error: Aborted\n\nCurrent thread 0x00000001f8e81300 (most recent call first):\n  File \"/Users/aaronilyas/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 669 in _create_gloo_process_group\n  File \"/Users/aaronilyas/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 2874 in _new_process_group_helper\n  File \"/Users/aaronilyas/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 2553 in init_process_group\n  File \"/Users/aaronilyas/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/c10d_logger.py\", line 97 in wrapper\n  File \"/Users/aaronilyas/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/c10d_logger.py\", line 83 in wrapper\n  File \"/Users/aaronilyas/ppdl-w-bench/pytorch/rendezvous.py\", line 39 in init_gloo\n  File \"/Users/aaronilyas/ppdl-w-bench/pytorch/gloo_probe.py\", line 33 in main\n  File \"/Users/aaronilyas/ppdl-w-bench/pytorch/gloo_probe.py\", line 75 in <module>\n\nExtension modules: torch._C, torch._C._dynamo.autograd_compiler, torch._C._dynamo.eval_frame, torch._C._dynamo.guards, torch._C._dynamo.utils, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special (total: 11)\n"
      },
      {
        "ssh": "ubuntu",
        "status": "not run",
        "error": "/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:368: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /__w/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)\n  cpu = _conversion_method_template(device=torch.device(\"cpu\"))\nFatal Python error: Segmentation fault\n\nCurrent thread 0x00007fa1bc9ba780 (most recent call first):\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 669 in _create_gloo_process_group\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 2874 in _new_process_group_helper\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 2553 in init_process_group\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/c10d_logger.py\", line 97 in wrapper\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/c10d_logger.py\", line 83 in wrapper\n  File \"/home/aaron/ppdl-w-bench/pytorch/rendezvous.py\", line 39 in init_gloo\n  File \"/home/aaron/ppdl-w-bench/pytorch/gloo_probe.py\", line 33 in main\n  File \"/home/aaron/ppdl-w-bench/pytorch/gloo_probe.py\", line 75 in <module>\n\nExtension modules: torch._C, torch._C._dynamo.autograd_compiler, torch._C._dynamo.eval_frame, torch._C._dynamo.guards, torch._C._dynamo.utils, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special, cuda.bindings._bindings.cydriver, cuda.bindings.cydriver, cuda.bindings.driver, cuda.bindings._bindings.cyruntime_ptds, cuda.bindings._bindings.cyruntime, cuda.bindings.cyruntime, cuda.bindings.runtime (total: 18)\n"
      }
    ]
  }
]
```

| policy | hosts | devices | split | samples/sec | median step s | status | commit |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Subset groups

These groups were tried after the full set did not produce a finished 50-step pipeline.

- `windows-ubuntu` status `not run` port `None`

```
[
  {
    "port": 29801,
    "ranks": [
      {
        "error": "Traceback (most recent call last):\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\pytorch\\gloo_probe.py\", line 33, in main\n    init_gloo(\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\pytorch\\rendezvous.py\", line 39, in init_gloo\n    dist.init_process_group(\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\venv\\Lib\\site-packages\\torch\\distributed\\c10d_logger.py\", line 83, in wrapper\n    return func(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\venv\\Lib\\site-packages\\torch\\distributed\\c10d_logger.py\", line 97, in wrapper\n    func_return = func(*args, **kwargs)\n                  ^^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\venv\\Lib\\site-packages\\torch\\distributed\\distributed_c10d.py\", line 2553, in init_process_group\n    default_pg, _ = _new_process_group_helper(\n                    ^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\venv\\Lib\\site-packages\\torch\\distributed\\distributed_c10d.py\", line 2874, in _new_process_group_helper\n    backend_result = creator_fn(dist_backend_opts, backend_options)\n                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\Users\\aaron\\ppdl-w-bench\\venv\\Lib\\site-packages\\torch\\distributed\\distributed_c10d.py\", line 669, in _create_gloo_process_group\n    backend_class = ProcessGroupGloo(\n                    ^^^^^^^^^^^^^^^^^\nRuntimeError: [enforce fail at C:\\actions-runner\\_work\\pytorch\\pytorch\\third_party\\gloo\\gloo\\transport\\uv\\address.cc:27] sizeof(impl_) == bytes.size(). 136 vs 184\n",
        "hostname": "Aaron",
        "rank": 0,
        "status": "not run",
        "world": 2,
        "ssh": "windows"
      },
      {
        "ssh": "ubuntu",
        "status": "not run",
        "error": "/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:368: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /__w/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)\n  cpu = _conversion_method_template(device=torch.device(\"cpu\"))\nFatal Python error: Segmentation fault\n\nCurrent thread 0x00007fd8acbdf780 (most recent call first):\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 669 in _create_gloo_process_group\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 2874 in _new_process_group_helper\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 2553 in init_process_group\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/c10d_logger.py\", line 97 in wrapper\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/c10d_logger.py\", line 83 in wrapper\n  File \"/home/aaron/ppdl-w-bench/pytorch/rendezvous.py\", line 39 in init_gloo\n  File \"/home/aaron/ppdl-w-bench/pytorch/gloo_probe.py\", line 33 in main\n  File \"/home/aaron/ppdl-w-bench/pytorch/gloo_probe.py\", line 75 in <module>\n\nExtension modules: torch._C, torch._C._dynamo.autograd_compiler, torch._C._dynamo.eval_frame, torch._C._dynamo.guards, torch._C._dynamo.utils, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special, cuda.bindings._bindings.cydriver, cuda.bindings.cydriver, cuda.bindings.driver, cuda.bindings._bindings.cyruntime_ptds, cuda.bindings._bindings.cyruntime, cuda.bindings.cyruntime, cuda.bindings.runtime (total: 18)\n"
      }
    ]
  }
]
```

- `windows-macmini` status `not run` port `None`

```
[
  {
    "port": 29841,
    "ranks": [
      {
        "ssh": "windows",
        "status": "not run",
        "error": "C:\\Users\\aaron\\ppdl-w-bench\\venv\\Lib\\site-packages\\torch\\_subclasses\\functional_tensor.py:368: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\torch\\csrc\\utils\\tensor_numpy.cpp:84.)\n  cpu = _conversion_method_template(device=torch.device(\"cpu\"))\n[W1112 19:43:03.000000000 socket.cpp:764] [c10d] The client socket has failed to connect to [Aaron.lan]:29841 (system error: 10049 - The requested address is not valid in its context.).\n\nTIMEOUT"
      },
      {
        "ssh": "macmini",
        "status": "not run",
        "error": "/Users/aaronilyas/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:368: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /Users/runner/work/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)\n  cpu = _conversion_method_template(device=torch.device(\"cpu\"))\n[W1112 19:44:17.213186000 socket.cpp:939] [c10d] The server socket on [::ffff:192.168.1.246]:29841 has timed out, will retry.\n[W1112 19:45:33.895788000 socket.cpp:939] [c10d] The server socket on [::ffff:192.168.1.246]:29841 has timed out, will retry.\n\nTIMEOUT"
      }
    ]
  }
]
```

- `macmini-ubuntu` status `not run` port `None`

```
[
  {
    "port": 29881,
    "ranks": [
      {
        "error": "Traceback (most recent call last):\n  File \"/Users/aaronilyas/ppdl-w-bench/pytorch/gloo_probe.py\", line 33, in main\n    init_gloo(\n  File \"/Users/aaronilyas/ppdl-w-bench/pytorch/rendezvous.py\", line 39, in init_gloo\n    dist.init_process_group(\n  File \"/Users/aaronilyas/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/c10d_logger.py\", line 83, in wrapper\n    return func(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/aaronilyas/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/c10d_logger.py\", line 97, in wrapper\n    func_return = func(*args, **kwargs)\n                  ^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/aaronilyas/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 2553, in init_process_group\n    default_pg, _ = _new_process_group_helper(\n                    ^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/aaronilyas/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 2874, in _new_process_group_helper\n    backend_result = creator_fn(dist_backend_opts, backend_options)\n                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/aaronilyas/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 669, in _create_gloo_process_group\n    backend_class = ProcessGroupGloo(\n                    ^^^^^^^^^^^^^^^^^\nRuntimeError: [enforce fail at /Users/runner/work/pytorch/pytorch/third_party/gloo/gloo/transport/uv/address.cc:27] sizeof(impl_) == bytes.size(). 136 vs 184\n",
        "hostname": "Aarons-Mac-mini-2.local",
        "rank": 0,
        "status": "not run",
        "world": 2,
        "ssh": "macmini"
      },
      {
        "ssh": "ubuntu",
        "status": "not run",
        "error": "/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:368: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /__w/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)\n  cpu = _conversion_method_template(device=torch.device(\"cpu\"))\nFatal Python error: Segmentation fault\n\nCurrent thread 0x00007f711a274780 (most recent call first):\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 669 in _create_gloo_process_group\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 2874 in _new_process_group_helper\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py\", line 2553 in init_process_group\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/c10d_logger.py\", line 97 in wrapper\n  File \"/home/aaron/ppdl-w-bench/venv/lib/python3.12/site-packages/torch/distributed/c10d_logger.py\", line 83 in wrapper\n  File \"/home/aaron/ppdl-w-bench/pytorch/rendezvous.py\", line 39 in init_gloo\n  File \"/home/aaron/ppdl-w-bench/pytorch/gloo_probe.py\", line 33 in main\n  File \"/home/aaron/ppdl-w-bench/pytorch/gloo_probe.py\", line 75 in <module>\n\nExtension modules: torch._C, torch._C._dynamo.autograd_compiler, torch._C._dynamo.eval_frame, torch._C._dynamo.guards, torch._C._dynamo.utils, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special, cuda.bindings._bindings.cydriver, cuda.bindings.cydriver, cuda.bindings.driver, cuda.bindings._bindings.cyruntime_ptds, cuda.bindings._bindings.cyruntime, cuda.bindings.cyruntime, cuda.bindings.runtime (total: 18)\n"
      }
    ]
  }
]
```


## Chosen baseline

No pipeline finished. The baseline is the fastest PyTorch configuration that ran: windows cuda at 6827.089946088578 samples/sec.

## linalg-pipeline

Status: not run.

not run. Workload W is gelu(x @ W + b), mean-squared error, and SGD at 1e-3. linalg_pipeline::train applies gelu(x @ W) and has no bias add. It requires dy before the forward that produces y, so the mean-squared-error gradient of those predictions cannot be formed inside the step. the only multi-rank transport constructor is linalg_transport::localhost_mesh, which built 2 endpoints on 127.0.0.1

Command: `cargo run -p linalg-bench -- bench/runs/linalg-pipeline.json`
Commit: `51b57a4b6babe8e1ea47e753e5e44b2f0050178a`

Per-step times and byte counts are in `bench/runs/`.
