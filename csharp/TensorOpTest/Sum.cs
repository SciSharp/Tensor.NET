using NUnit.Framework;
using Tensornet;

namespace TensorOpTest;

public class SumTest
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void Test1()
    {
        var s = Tensor.FromArray<int>(
            new int[] { 18, 4, 35, 22, 34, 18, 15, 5, 13, 49, 41, 3, 
                        4, 35, 14, 22, 38, 29, 42, 47, 30, 4, 9, 0, 
                        35, 0, 43, 27, 44, 29, 2, 17, 16, 6, 36, 41, 
                        1, 22, 3, 30, 35, 13, 41, 45, 38, 44, 38, 21, 
                        42, 41, 19, 24, 40, 39, 37, 24, 28, 17, 22, 21, 
                        15, 23, 35, 16, 31, 3, 29, 28, 0, 15, 17, 10 }, new TensorShape(3, 4, 2, 3));
        var t = Tensor.FromArray<int>(
            new int[] { 1734 }, new TensorShape(1));
        var p = s.Sum();
        Assert.IsTrue(TensorUtils.IsValueEqual(t, p));
    }

    // [Test]
    // public void Test2()
    // {
    //     var s = Tensor.FromArray<int>(
    //         new int[] { -151, -46,  -9,   -62,  -158, -74,  35,   -10,  -123, -94,  -122, 58,
    //                     -124, 139,  -173, -137, -178, 116,  52,   -92,  -14,  -176, -133, -109,
    //                     -114, -157, -186, 46,   -78,  -144, 155,  -60,  47,   150,  -133, -58,
    //                     -17,  -161, -36,  11,   133,  -170, -149, -155, 10,   -118, -112, -103,
    //                     -110, 183,  29,   21,   189,  -85,  83,   -186, -114, -104, -171, -116,
    //                     -110, 88,   -130, 42,   106,  120,  -94,  -77,  49,   74,   96,   -28 }, new TensorShape(6, 3, 2, 2));
    //     var t = Tensor.FromArray<int>(
    //         new int[] { -151, -124, -114, -17,  -110, -110, -158, -178, -78,  133,  189,  106,
    //                     -123, -14,  47,   10,   -114, 49,   -46,  139,  -157, -161, 183,  88,
    //                     -74,  116,  -144, -170, -85,  120,  -94,  -176, 150,  -118, -104, 74,
    //                     -9,   -173, -186, -36,  29,   -130, 35,   52,   155,  -149, 83,   -94,
    //                     -122, -133, -133, -112, -171, 96,   -62,  -137, 46,   11,   21,   42,
    //                     -10,  -92,  -60,  -155, -186, -77,  58,   -109, -58,  -103, -116, -28 }, new TensorShape(2, 2, 3, 6));
    //     var p = s.Sum(2, 3, 1, 0);
    //     Assert.IsTrue(TensorUtils.IsValueEqual(t, p));
    // }

    // [Test]
    // public void Test3()
    // {
    //     var s = Tensor.FromArray<double>(
    //         new double[] { -144.88911173213074, -177.9092558668978 }, new TensorShape(1, 2));
    //     var t = Tensor.FromArray<double>(
    //         new double[] { -144.88911173213074, -177.9092558668978 }, new TensorShape(2, 1));
    //     var p = s.Sum(1, 0);
    //     Assert.IsTrue(TensorUtils.IsValueEqual(t, p));
    // }
}