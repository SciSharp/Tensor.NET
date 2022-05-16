using NUnit.Framework;
using Tensornet;
using Tensornet.Common;

namespace TensorOpTest;

public class FillTest
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void Test1()
    {
        var t = Tensor.FromArray<int>(
            new int[] { 0, 0, 1, 0, 0, 0 }, new TensorShape(3, 2));
        var truth = Tensor.FromArray<int>(
            new int[] { 123, 123, 123, 123, 123, 123 }, new TensorShape(3, 2));
        System.Diagnostics.Debugger.Launch();
        t.Fill(123);
        Assert.IsTrue(TensorUtils.IsValueEqual(t, truth));
    }
}