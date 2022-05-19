## Cmake build on Windows

1. Install ```mingw-w64``` and set the enviroment variable.
2. clone the project.
3. use the following command

```bash
$ mkdir build
$ cd build
$ cmake .. -G "MinGW Makefiles"
$ mingw32-make
```
    
## Cmake build on Linux

1. clone the project.
2. use the following command

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Cmake cross-build on Linux targeting windows

1. Install ```mingw-w64``` and set the enviroment variable.
2. clone the project.
3. use the following command

```bash
$ mkdir build
$ cd build
$ cmake .. -DNN_CROSS_COMPILE=ON
$ make
```
