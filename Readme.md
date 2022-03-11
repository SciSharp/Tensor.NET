## Num.NET

### TODO

| Event | Description | Priority | Status |
| ----- | --------------| --- | ----- |
| Complete the smallest framework of c++ part | Construct a smallest framework of c++ part which could run naive implemention of Matmul successfully. In this stage, extensibility should be taken into consideration, but details could be ignored. | p0 | Complete ✅ |
| Complete naive implemention of Matmul | The results of naive implemention of matmul are wrong. Besides, more test cases need to be added, including more shapes, more dtypes and more data. | p0 | Complete ✅ |
| Add Benchmark test for Matmul | Add some bechmark tests for Matmul to evaluate its effeciency | p1 | Waiting 🔵 |
| Add deduce to decide the output shape of ops | Checking if the shapes are matched in the body of op is not a good choice. A calculator for it is needed | p0 | Complete ✅ |
| Add broadcast | Add broadcast with stride and wrap it | p0 | Complete ✅ |
| Add script to auto build and test | Write a script on linux to build and run all tests automatically | p2 | Waiting 🔵 |
| Add naive op ```dot``` | Add naive implementation for op ```dot``` | p0 | Waiting 🔵 |
| Add naive op ```reshape``` | Add naive implementation for op ```reshape``` | p0 | On going 🚀 |
| Add naive op ```transpose``` | Add naive implementation for op ```transpose``` | p0 | On going 🚀 |
| Add naive op ```add``` | Add naive implementation for op ```add``` | p0 | Waiting 🔵 |
| Add naive op ```sub``` | Add naive implementation for op ```sub``` | p0 | Waiting 🔵 |
| Add naive op ```mul``` | Add naive implementation for op ```mul``` | p0 | Waiting 🔵 |
| Add naive op ```div``` | Add naive implementation for op ```div``` | p0 | Waiting 🔵 |
| Add ```type_deduce``` | Add deduce method for type of layout to decide the ```dtype``` of the output Array. | p0 | Waiting 🔵 |
| Add ```Status``` | Add a status struct to tell the caller if the call success and return error message if failed. | p0 | On going 🚀 |
| Define actions on Debug and Release mode | define different actions for one expr on different mode. For instance, ```nn_assert``` should not take action on release mode | p0 | Waiting 🔵 |
| Add ```Checker``` for test | Add a checker class to judge if the pred and result are matched. | p0 | On going 🚀 |
| Add Process for overflow | Add process to deal with overflow, show user NAN instead of completely wrong data. | p0 | Waiting 🔵 |
| Automatically squeeze the shape of scalar to one-dim | Mainly for result of matmul. | p2 | Waiting 🔵 |


✅   ❌   🚀   🔵
