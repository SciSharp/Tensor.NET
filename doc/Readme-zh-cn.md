中文 | [English](../)

# Tensor.NET
我为其编写了一点点东西

Tensor.NET是一个轻量级、高性能的.NET平台下的张量运算库，用c++作为底层实现，提供与Numpy相似的功能，但是封装成.NET风格的接口，使用泛型张量，支持Linq接口，C#原生的slice方式等等。项目还处于比较新的阶段，可能会缺少一些功能或者有一些BUG，但作者会持续维护该项目，如果发现功能缺失或者BUG可以发起issue或PR，感谢您的参与！

作者本人在早期学习中也受益于一些中文资料和项目，本项目将提供中文版的说明和文档，也建立QQ群以供交流，作者恳请广大同胞可以：
- 有问题优先发Issue：QQ群的聊天记录进行查询和检索会很困难，而在Issue中查询会方便许多，这样有问题也方便后来者查询。
- Issue尽量使用英文：Github毕竟是来自世界各地的程序员的社区，英文是最通用的语言，而对于中国人自己来讲，恐怕也不希望查询issue的时候要用中英文查两遍，所以Issue中还请优先使用英文。

QQ群：206663843

## Tensor.NET的优势

当前已经有了一些张量计算相关的.NET库，比如[NumSharp](https://github.com/SciSharp/NumSharp)和[TensorFlow.NET](https://github.com/SciSharp/TensorFlow.NET)。作者也从这两个库中学习到了很多，在此向这两个库的作者们表示感谢！

然而，本项目创建的原因便是，这两个库多少在某些方面很优秀，在另外一些方面存在不足，TensorFlow.NET是一个非常优秀的.NET深度学习库，但是对于普通的张量运算来讲太过庞大了；NumSharp实现了Numpy风格的接口，对Python直接向.NET转换比较友好，但是底层实现的效率不高，且对泛型的擦除使得一些功能在强类型语言下受限。

本项目的特性具体如下：

- **更快的运算速度**

    Tensor.NET采用C++作为底层实现，然后使用C#与其进行Interop并提供封装，后续会逐步优化各个算子的实现来提速。

- **泛型张量**

    本项目的设计理念与```System.Numerics.Tensor```比较相似，采用泛型类而非类型擦除，与其它大多数情况下的.NET特性保持一致，这样更方便进行扩展，也对未接触过Numpy等弱类型语言张量框架的开发者友好一些。

- **.NET风格的API**

    本项目提供Numpy库中的主要功能部分，但是并不封装成Python风格的API，而是采用.NET风格，例如```Tensor<T>```继承并实现了```IEnumerable<T>```接口，可以直接用foreach迭代、转换成List、使用Linq接口等等，支持C#原生的切片方式，比如```t[..3, 2..^1]```。

在未来，本项目计划提供以下特性：

- 针对x86平台进行算子优化

- 支持多线程计算

- 支持在图像、Excel、CSV等文件中读写张量

## 安装方式

使用如下命令

```
PM> Install-Package Tensor.NET
```
或
```
$ dotnet add package Tensor.NET
```
也可以在VS的Nuget包管理器中搜索Tensor.NET并安装。

## 文档

目前，大多数的文档都还在制作中，还请耐心等待，欢迎参与文档编写与翻译工作。

- 关于如何构建c++库，请参考[此文档](./doc/Build.md)

- 关于如何使用Tensor.NET，请参考 [用户手册]() (制作中)

- 关于Tensor.NET的例程，请参考[此项目](https://github.com/AsakusaRinne/Tensor.NET.Examples)

## Benchmarks (制作中)

后续会加入基准测试

## 简单的使用

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

## 主要的缺陷

虽然Tensor.NET在某些方面有优点，但不可否认其有一些明显的缺点，最主要的缺点就是基础运算符，因为.NET6和C# 10目前不支持运算符的泛型定义，也不支持针对运算符进行扩展方法那样的泛型方法特化，所以无法对加减乘除以及取模这几个基础运算符支持所有类型张量之间的操作。

当前只能在同类型张量之间使用这些基础运算符，或者使用```Add```, ```Div```等函数接口来进行不同类型张量之间的运算，这可能会带来一些不方便。

这一点有望在.NET7正式发布后得到解决，利用其新特性对该问题进行优化。
