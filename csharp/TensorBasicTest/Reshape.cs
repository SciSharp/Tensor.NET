using NUnit.Framework;
using Numnet;
using System;
using System.Collections.Generic;

namespace TensorBasicTest;

public class ReshapeTest
{
    private List<Tuple<Tensor, Tensor>> _map;
    [SetUp]
    public void Setup()
    {
        _map = new List<Tuple<Tensor, Tensor>>(new Tuple<Tensor, Tensor>[]{
            new Tuple<Tensor, Tensor>(Tensor.FromArray(new int[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}), 
                        Tensor.FromArray(new int[]{1, 2, 3, 4, 5, 6}, new int[]{3, 2})),
            new Tuple<Tensor, Tensor>(
                Tensor.FromArray(new int[]{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, new int[]{3, 8}), 
                Tensor.FromArray(new int[]{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, new int[]{2, 2, 6}))
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