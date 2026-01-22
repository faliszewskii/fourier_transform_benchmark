# Fast Fourier Transform C++ Benchmark

Benchmark of the following Fourier Transform implementations in C++:
- Readily available C/C++ FFTW library,
- Naive DFT running in O(N^2) time,
- Cooley Tukey Fast Fourier Transform,
- Cooley Tukey FFT with OpenMP multithreaded optimization.
- Nvidia cuFFT library

The implementations are allowed for preinitialization of the state for a given input size (called a plan) and then tested for the plan's execution time and throughput.
The following benchmark was performed on 4 byte float type.

## Results

<p float="left">
  <img height="300" alt="time" src="https://github.com/user-attachments/assets/cad259dd-6717-40d1-a6bd-22d0a63f9d23" />
  <img height="300" alt="items" src="https://github.com/user-attachments/assets/e0732862-907e-4b09-8487-795971f4a6ad" />
</p>

The charts above show how big is the difference between the naive approach O(N^2) and Fast Fourier Transforms O(NlogN). 

OpenMP optimization of Cooley Tukey FFT is lagging behind at the small data sizes because of thread parallelization overhead but easily overtakes classic FFT implementation coming close to the most optimized FFTW library implementation. The best approach seems to be a hybrid approach using single threaded implementations for small N  with parallelization beginning at N=~2^16.

The same logic applies to cuFFT library altough it performs much better than FFTW with benchmark showing it nearing O(N) time. 

## Hardware and system configuration

- Architecture:                x86_64
- CPU Model name:              13th Gen Intel(R) Core(TM) i7-13620H
- CPUs:                        16
- Thread(s) per core:          2
- CPU max MHz:                 4900.0000
- Caches (sum of all):         
  - L1d:                       416 KiB (10 instances)
  - L1i:                       448 KiB (10 instances)
  - L2:                        9.5 MiB (7 instances)
  - L3:                        24 MiB (1 instance)
- GPU Model name:              NVIDIA GeForce RTX 4060 Laptop GPU
- RAM size:                    2x16Gi
- RAM speed:                   5600 MT/s
- OS:                          Ubuntu 24.04.3 LTS

## Building

```bash
git clone git@github.com:faliszewskii/fourier_transform_benchmark.git
cd fourier_transform_benchmark
mkdir build
cd build
cmake ../
make
```

## Running

Running GTests
```
cd fourier_transform_benchmark
cd build
./fft
```

Running benchmark
```
cd fourier_transform_benchmark
cd build
make run_fft_benchmark
```

Plotting the latest benchmark results
```
cd fourier_transform_benchmark
cd build
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r ../python/requirements.txt
python ../python/plot_benchmark.py
```

## Project Structure

  - `benchmarks` - Json results of Google Benchmark
  - `external` - External libraries 
  - `cpp` - All of the C++ code
    - `src` - Discrete Fourier Transform implementations
      - `fourier_transform.h` - C++ concept for the Fourier Transform interface.
      - `fftw_wrapper.h`
      - `naive_dft.h`
      - `cooley_tukey_fft.h`
      - `openmp_cooley_tukey_fft.h`
      - `cu_fft.cuh`
    - `tests` - GTests for correctness and Google Benchmark for performance
      - `ft_analytical_tests.cpp` - Correctness tests based on analytical properties of Fourier Transform.    
      - `ft_performance_tests.cpp` - Performance tests for random input.
      - `ft_random_tests.cpp` - Correctness tests with random input with FFTW as ground truth.
      - `ft_backends.cpp` - Utils for GTest Typed Parametrized Tests.
      - `test_utils.cpp`
  - `python` - All of the python code (data plotting)
    - `get_latest_stamp.py` - for a given directory returns the file with the latest time stamp in the name.
    - `plot_benchmark.py` - Given a Google Benchmark json file plot time/N and items/s/N.

