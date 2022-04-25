using NUnit.Framework;
using Numnet;
using System;
using System.Collections.Generic;

namespace TensorBasicTest;

public class BroadcastTest
{
    private List<Tuple<Tensor<int>, Tensor<int>>> _map;
    [SetUp]
    public void Setup()
    {
        _map = new List<Tuple<Tensor<int>, Tensor<int>>>(new Tuple<Tensor<int>, Tensor<int>>[]{
            new Tuple<Tensor<int>, Tensor<int>>(Tensor.FromArray<int>(new int[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}), 
                        Tensor.FromArray<int>(new int[]{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, new int[]{4, 2, 3})),
            new Tuple<Tensor<int>, Tensor<int>>(
                Tensor.FromArray<int>(new int[]{1, 2, 3, 4, 5, 6}, new int[]{2, 1, 3}),
                Tensor.FromArray<int>(new int[]{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6}, new int[]{2, 4, 3}))
        });
    }

    [Test]
    public void Test1()
    {
        foreach(var (inp, truth) in _map){
            Console.WriteLine(inp.Broadcast(truth.Shape));
            Assert.IsTrue(TensorUtils.IsValueEqual(inp.Broadcast(truth.Shape), truth));
        }
    }
}