using NUnit.Framework;
using Numnet;
using System;
using System.Collections.Generic;

namespace TensorBasicTest;

public class SliceTest
{
    private List<Tuple<Tensor<int>, Tensor<int>>> _map;
    [SetUp]
    public void Setup()
    {
        _map = new List<Tuple<Tensor<int>, Tensor<int>>>(new Tuple<Tensor<int>, Tensor<int>>[]{
            new Tuple<Tensor<int>, Tensor<int>>(Tensor.FromArray<int>(new int[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}), 
                        Tensor.FromArray<int>(new int[]{1, 2, 3, 4, 5, 6}, new int[]{3, 2})),
            new Tuple<Tensor<int>, Tensor<int>>(
                Tensor.FromArray<int>(new int[]{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, new int[]{3, 8}), 
                Tensor.FromArray<int>(new int[]{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, new int[]{2, 2, 6}))
        });
    }

    [Test]
    public void Test1()
    {
        var src = Tensor.FromArray(new int[] { 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6 }, new int[] { 3, 8 });
        var dst = src[new Slice(1, 3), new Slice(3, 6)];
        Tensor<int> truth = Tensor.FromArray<int>(new int[] { 6, 1, 2, 2, 3, 4 }, new int[] { 2, 3 });
        Assert.IsTrue(TensorUtils.IsValueEqual(dst, truth));
    }

    [Test]
    public void Test2()
    {
        var src = Tensor.FromArray(new int[] { 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6 }, new int[] { 3, 8 });
        var dst = src[new Slice(0, 3, 2), new Slice(1, 8, 3)];
        Tensor<int> truth = Tensor.FromArray<int>(new int[] { 2, 5, 2, 6, 3, 6 }, new int[] { 2, 3 });
        Assert.IsTrue(TensorUtils.IsValueEqual(dst, truth));
    }

    [Test]
    public void Test3()
    {
        var src = Tensor.FromArray(new int[] { 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6 }, new int[] { 3, 8 });
        var dst = src[new Slice(1, 3), new Slice(5)];
        Tensor<int> truth = Tensor.FromArray<int>(new int[] { 2, 4 }, new int[] { 2 });
        Assert.IsTrue(TensorUtils.IsValueEqual(dst, truth));
    }
}