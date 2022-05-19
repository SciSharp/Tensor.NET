# Tensor.NET

Tensor.NET is a lightweight and high-performance tensor library which provides numpy-like operations but .NET style interfaces. It supports generic tensor, Linq, C# native slices and so on. It is young so that it may may lack some features or have some BUGs. Please tell us on github or via email, thank you!

## Why using Tensor.NET

Currently, there has been some repositories to support tensor operation, providing numpy-like interfaces, such as [NumSharp](https://github.com/SciSharp/NumSharp) and [TensorFlow.NET](https://github.com/SciSharp/TensorFlow.NET). I also learned a lot from them and thanks for the authors of the two repositories sincerely!

However, everyone has his strong and weak points, compared with the previous works, ```Tensor.NET``` provides the features below.

- Faster speed: using c++ to implement operations.

    ```Tensor.NET``` puts its implmentation of operations in c++ and interop with native api. Then a encapsulation is applied to hide the interop from users. It could provide better performance than pure C# implementation.

- Better extensibility: using generic class ```Tensor<T>```.

    Erasing type in the class design could provide convinience for python-style interfaces. However, using generic class could better integrate with other .NET code.

- More friendly apis for .NETer
    
    Rather than providing python-style and numpy-like apis, ```Tensor.NET``` aims to design .NET style apis. It implements ```IEnumerable<T>``` interface, supports C# index and slice, and use generic to constriant the operations between tensors of different data types. ```Tensor.NET``` provides the main features in ```Numpy```, but provides .NET style interfaces.

The following features will be provided in the future.

- Optimize the algorithm of operations on x86.

- Support multi-thread to accelerate the operations.

- Support reading and writing tensor from image, excel, csv and other more formats.

## API map from Numpy to Tensor.NET (WIP)

```Tensor.NET``` implements the main features of ```Numpy```. The mapping from Numpy API to Tensor.NET API could be found [here](./doc/API_Mapping.md).

```Tensor.NET``` is a lightweight library so that some features may not been covered so far. Please submit an issue or PR to tell us the feature you need.

## Installation

```
PM> Install-Package Tensor.NET
```


## Documention

Currently, most of the docs are in the making. Please wait for some days, thank you.

For how to build from source, please refer to [this doc](./doc/Build.md)

For how to use, please refer to [the user guide]() (WIP)

For examples, please refer to [this repository]()

## Benchmarks (WIP)

Benchmarks will be privided later.

## Simple usages

```cs
using Tensornet;
using Tensornet.Common;
using Tensornet.Math;
using System.Text;

// Matrix multiply
// Create Tensor from arrays.
Tensor<int> a = Tensor.FromArray<int>(new int[]{1, 2, 3, 4, 5, 6}, new int[] { 1, 3, 2 });
Tensor<double> b = Tensor.FromArray<double>(new double[]{1.2, 2.6, 3.9, 4.1, 5.0, 6.5, 1.7, 2, 3, 4, 5, 6, 1.2, 2.6, 
                    3.9, 4.1, 5.0, 6.5, 1.7, 2, 3, 4, 5, 6, 5.0, 6.5, 1.7, 2, 3, 4, 5, 6}, new int[] { 4, 2, 4 });
var c = a.Matmul(b);
// Print the tensor
Console.WriteLine(c);

// Reshape, permute and transpose
c = c.Reshape(new int[] { 6, 8 }).Permute(new int[] { 1, 0 }).Transpose(0, 1);

// Indexer
c[0, 0] = 999.0;
c[5, 7] = -888.0;

// Slice
var s = c[1..3, ^5..^2];


// Random generations
var gaussian = Tensor.Random.Normal<int>(new TensorShape(5, 6), 0, 5);
var avg = Tensor.Random.Uniform<int>(new TensorShape(5, 6), -5, 5);

// Other creation methods for Tensor
var eye = Tensor.Eye<int>(4, 5, 2);
eye.Fill(258);
var linspace = Tensor.Linspace<double>(0, 10, 5);
var ones = Tensor.Ones<int>(new int[] { 2, 3 });

// Squeeze and unsqueeze
ones = ones.Unsqueeze(1);
ones = ones.Unsqueeze(3);
ones = ones.Squeeze();

// Using Linq APIs
foreach(var item in eye){
    Console.WriteLine($"{item} ");
}

// Bool index
var bidx = Tensor.Random.Uniform<bool>(new int[] { 3, 3 }, 0, 1);
var bres = ta[ta > 20];
var twhere = ta.Where(x => x > 20, 999, -555);
var trefer = Tensor.Random.Uniform<int>(new int[] { 2, 3, 3 }, -100, -50);
ta[ta > 20] = trefer;

// Serialization and Deserialization
TensorWriter.Write("test.tn", ta);
var tr = TensorReader.Read<float>("test.tn");

// argmax and argmin
var src = Tensor.Random.Uniform<int>(new int[] { 4, 5 }, 0, 50);
var amax = src.Argmax(0);
var amin = src.Argmin(1);
```

## The major shortcoming

Despite the advantages above, ```Tensor.NET``` does have some shortcomings, the main shortcoming is its basic arithment with operator ```+ - % /```.

Since ```Tensor.NET``` use generic class in its design and operations between tensors of different types are supposed, there's no way to provide these operators in .NET 6 and C# 10 except the operations between tensors of the same data type. For tensors with different data types, ```Add, Sub, Mod, Div``` methods are provided. However, still, there will be some inconvinience.

This problem is supposed to be solved in .NET 7 with its new features. Thanks for the using and waiting!
