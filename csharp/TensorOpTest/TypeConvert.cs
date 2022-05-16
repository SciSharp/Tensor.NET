using NUnit.Framework;
using Tensornet;
using System;

namespace TensorOpTest;

public class TypeConvertTest
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void Test1()
    {
        var inp = Tensor.FromArray<double>(
                new double[]{-184.47836822007747, -135.77934833640774, -131.5692045780147,
                            -139.88637591883548, 7.378481161197385,   -194.85200685695676,
                            124.72624465928914,  135.32088057645922,  -149.63309464031306,
                            5.068530645297187,   -52.609905261884194, -5.83477677509191,
                            -121.05109868593935, -133.2874615589557,  -108.08990561152387,
                            26.347047463464122,  -65.70184395799768,  97.66633377665073,
                            68.78419424070898,   181.3987014174778,   -17.19642705834329,
                            -177.23101755020787, -157.03624051832935, 16.655507860783615}, 
                            new int[]{2, 3, 4})[0..2, 1..3, 2..4];
        var pred = inp.ToTensor<int>();
        var truth = Tensor.FromArray<int>(new int[] { 124, 135, -52, -5, 68, 181, -157, 16 }, new int[] { 2, 2, 2 });
        Assert.IsTrue(TensorUtils.IsValueEqual(pred, truth));
    }
    [Test]
    public void Test2()
    {
        var inp = Tensor.FromArray<double>(
                new double[]{-184.47836822007747, -135.77934833640774, -131.5692045780147,
                            -139.88637591883548, 7.378481161197385,   -194.85200685695676,
                            .0,  135.32088057645922,  -149.63309464031306,
                            5.068530645297187,   -52.609905261884194, -5.83477677509191,
                            -121.05109868593935, -133.2874615589557,  -108.08990561152387,
                            26.347047463464122,  -65.70184395799768,  97.66633377665073,
                            68.78419424070898,   181.3987014174778,   -17.19642705834329,
                            -177.23101755020787, -157.03624051832935, .0}, 
                            new int[]{2, 3, 4})[0..2, 1..3, 2..4];
        var pred = inp.ToTensor<bool>();
        var truth = Tensor.FromArray<bool>(new bool[] { false, true, true, true, true, true, true, false }, new int[] { 2, 2, 2 });
        Assert.IsTrue(TensorUtils.IsValueEqual(pred, truth));
    }
}