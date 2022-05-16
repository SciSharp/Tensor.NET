## Cmake build on Windows

1. Install ```mingw-w64``` and set the enviroment variable.
2. clone the project.
3. use the following command
    ```
    $ mkdir build
    $ cd build
    $ cmake .. -G "MinGW Makefiles"
    $ mingw32-make
    ```