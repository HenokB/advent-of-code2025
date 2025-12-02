# Compilation and Execution

To compile and run the codes, you can simply do this:

## Part 1

```bash
nvcc -O2 --extended-lambda -std=c++17 day2_cuda.cu -o day2_cuda
./day2_cuda input.txt
```

## Part 2

```bash
nvcc -O2 --extended-lambda -std=c++17 day2_part2.cu -o day2_part2
./day2_part2 input.txt
```
