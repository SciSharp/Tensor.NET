using NUnit.Framework;
using Numnet;
using Numnet.Common;
using Numnet.Manipulation;

namespace TensorOpTest;

public class EyeTest
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void Test1()
    {
        var t = Tensor.FromArray<int>(
            new int[] { 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, new TensorShape(4, 5));
        var p = Tensor.Eye<int>(4, 5, 2);
        Assert.IsTrue(TensorUtils.IsValueEqual(t, p));
    }

    [Test]
    public void Test2()
    {
        var t = Tensor.FromArray<double>(
            new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                           0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
                           0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 
                           0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }, new TensorShape(8, 6));
        var p = Tensor.Eye<double>(8, 6, -1);
        Assert.IsTrue(TensorUtils.IsValueEqual(t, p));
    }
}