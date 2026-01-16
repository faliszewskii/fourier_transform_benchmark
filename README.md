# Fast Fourier Transform Benchmark

Benchmark of the following Fourier Transform implementations:
- Readily available C/C++ FFTW library,
- Naive DFT running in O(N^2) time,
- Cooley Tukey Fast Fourier Transform,
- Cooley Tukey FFT with OpenMP multithreaded optimization.

The implementations are allowed for preinitialization of the state for a given input size (called a plan) and then tested for the plan's execution time and throughput for different random input data.

## Results

<p float="left">
<img width="400" alt="time" src="https://github.com/user-attachments/assets/6f2c908e-4e98-40b0-891f-d4e965ad3bcc" />
<img width="400" alt="items" src="https://github.com/user-attachments/assets/3a35e367-9e8d-4350-a0ce-c05d6a5f226a" />
</p>

The charts above show how big is the difference between the naive approach O(N^2) and Fast Foureir Transforms O(NlogN). 
OpenMP optimization of Cooley Tukey FFT is lagging behind at the small data sizes because of thread parallelization overhead but easily overtakes classic FFT implementation coming close to the most optimized FFTW library implementation. The best approach seems to be to use single threaded implementations for small N  with parallelization beginning at N=~2^17.


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
    - `tests` - GTests for correctness and Google Benchmark for performance
      - `ft_analytical_tests.cpp` - Correctness tests based on analytical properties of Fourier Transform.    
      - `ft_performance_tests.cpp` - Performance tests for random input.
      - `ft_random_tests.cpp` - Correctness tests with random input with FFTW as ground truth.
      - `ft_backends.cpp` - Utils for GTest Typed Parametrized Tests.
      - `test_utils.cpp`
  - `python` - All of the python code (data plotting)
    - `get_latest_stamp.py` - for a given directory returns the file with the latest time stamp in the name.
    - `plot_benchmark.py` - Given a Google Benchmark json file plot time/N and items/s/N.

