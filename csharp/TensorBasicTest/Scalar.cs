using NUnit.Framework;
using Tensornet;
using System;
using System.Collections.Generic;

namespace TensorBasicTest;

public class ScalarTest
{
    [SetUp]
    public void Setup()
    {

    }

    [Test]
    public void TestConvert()
    {
        Scalar a = new Scalar<int>(123);
        Scalar b = a.AsScalar<double>();
        b += 0.001;
        Assert.AreEqual(b.GetValue<float>(), 123.001f);
        b.SetValue<int>(456);
        Assert.AreEqual(b.GetValue<long>(), 456L);
    }

    [Test]
    public void TestEqual()
    {
        Scalar a = new Scalar<int>(123);
        Scalar b = a.AsScalar<double>();
        b += 0.9;
        Assert.IsTrue(b == 123.9);
        Assert.IsFalse(a >= 124);
        a.SetValue<double>(123.9);
        Assert.IsFalse(a.Equals(b));
        Assert.AreEqual(a.GetValue<double>(), 124.0);
    }
}