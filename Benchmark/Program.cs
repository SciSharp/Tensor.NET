using BenchmarkDotNet.Attributes;
using NumSharp;
using BenchmarkDotNet.Running;
using System.Linq;
using NN.Core;

var summary = BenchmarkRunner.Run(typeof(Program).Assembly);
Console.WriteLine(summary);

public static class BenchmarkParam
{
    public static int TrainEpochs = 100;
    public static int DataLength = 4000;
    public static int FeatureCount = 1000;
}