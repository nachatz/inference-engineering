# CPU preload benchmark

Written by `just bench`. Medians of 3 runs per row. Each run is a 10 s headless locust run: 16 users with no wait time, 1 row(s) per request, a new connection per request. The load generator runs on the same machine.

- Machine: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.41, AMD Ryzen 7 9800X3D 8-Core Processor, 8 CPUs, 16.3 GB RAM
- Python 3.13.16, ONNX Runtime 1.31.0
- Model: `model/model.onnx`, 352.6 MB

| workers | preload | req/s | p50 ms | p99 ms | errors | PSS total MB | RSS total MB | private per worker MB | first response s | all workers ready s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | no | 142.9 | 110.0 | 130.0 | 0 | 458 | 499 | 408.9 | 0.60 | 0.60 |
| 1 | yes | 145.4 | 110.0 | 120.0 | 0 | 457 | 893 | 15.6 | 0.61 | 0.61 |
| 2 | no | 161.7 | 100.0 | 180.0 | 0 | 870 | 954 | 406.2 | 0.66 | 0.73 |
| 2 | yes | 236.2 | 62.0 | 150.0 | 0 | 473 | 1326 | 14.8 | 0.61 | 0.67 |
| 4 | no | 168.0 | 85.0 | 250.0 | 0 | 1684 | 1861 | 405.7 | 0.86 | 1.04 |
| 4 | yes | 302.8 | 42.0 | 140.0 | 0 | 502 | 2190 | 14.4 | 0.61 | 0.77 |
| 8 | no | 166.9 | 80.0 | 250.0 | 0 | 3304 | 3674 | 405.0 | 0.98 | 1.57 |
| 8 | yes | 291.5 | 46.0 | 150.0 | 0 | 555 | 3914 | 13.7 | 0.61 | 0.97 |
