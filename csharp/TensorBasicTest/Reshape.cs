using NUnit.Framework;
using Numnet;
using System;
using System.Collections.Generic;

namespace TensorBasicTest;

public class ReshapeTest
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
        foreach(var (inp, truth) in _map){
            Assert.IsTrue(TensorUtils.IsValueEqual(inp.Reshape(truth.Shape), truth));
        }
    }
}