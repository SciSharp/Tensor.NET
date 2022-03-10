## Num.NET

### TODO

| Event | Description | Status |
| ----- | -------------- | ----- |
| Complete the smallest framework of c++ part | Construct a smallest framework of c++ part which could run naive implemention of Matmul successfully. In this stage, extensibility should be taken into consideration, but details could be ignored. | Complete ✅ |
| Complete naive implemention of Matmul | The results of naive implemention of matmul are wrong. Besides, more test cases need to be added, including more shapes, more dtypes and more data. | On going 🚀 |
| Add Benchmark test for Matmul | Add some bechmark tests for Matmul to evaluate its effeciency | Waiting 🔵 |
| Add calculator to decide the output shape of ops | Checking if the shapes are matched in the body of op is not a good choice. A calculator for it is needed | Waiting On going 🚀 |
| Add broadcast | Add broadcast with stride and wrap it | On going 🚀 |
| Add script to auto build and test | Write a script on linux to build and run all tests automatically | Waiting 🔵 |
| Add naive op ```dot``` | Add naive implementation for op ```dot``` | Waiting 🔵 |
| Add naive op ```reshape``` | Add naive implementation for op ```reshape``` | Waiting 🔵 |
| Add naive op ```transpose``` | Add naive implementation for op ```transpose``` | Waiting 🔵 |
| Add naive op ```add``` | Add naive implementation for op ```add``` | Waiting 🔵 |
| Add naive op ```sub``` | Add naive implementation for op ```sub``` | Waiting 🔵 |
| Add naive op ```mul``` | Add naive implementation for op ```mul``` | Waiting 🔵 |
| Add naive op ```div``` | Add naive implementation for op ```div``` | Waiting 🔵 |
| Add compiling condition for nn_assert and nn_throw | Add macro to seperate release and debug mode. Note that options should be added to cmake files. | Waiting 🔵 |
| Add ```type_deduce``` | Add deduce method for type of layout to decide the ```dtype``` of the output Array. | Ongoing 🚀 |
| Add ```Status``` | Add a status struct to tell the caller if the call success and return error message if failed. | Waiting 🔵 |


✅   ❌   🚀   🔵