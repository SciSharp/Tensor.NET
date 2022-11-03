using BenchmarkDotNet.Running;
using NN.Native.Operators;
using BenchmarkDotNet.Attributes;
using System.Reflection;
using NN.Native.Data;
using NN.Native.Basic;
using NN.Native.Operators.Naive;
using NN.Native.Basic.DType;
using System.Runtime.Remoting;

var summary = BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
Console.WriteLine(summary);


public class MatmulFloat32Benchmark
{
    public IEnumerable<int[][]> DimensionValues => new int[][][] {
            new int[][]{ new int[] { 3, 4 }, new int[] { 4, 5} },
            new int[][]{ new int[] { 32, 64 }, new int[] { 64, 48} },
            new int[][]{ new int[] { 256, 512 }, new int[] { 512, 192} },
            new int[][]{ new int[] { 8, 1024 }, new int[] { 1024, 1} },
        };
    [ParamsSource(nameof(DimensionValues))]
    public int[][] Dimensions { get; set; }

    NativeArray<float> ArrayLeft { get; set; }
    NativeArray<float> ArrayRight { get; set; }
    [GlobalSetup(Targets = new[] { nameof(TensornetMatmul) })]
    public void TensornetSetup()
    {
        ArrayLeft = NativeArray.Random.Normal<float>(new NativeLayout(Dimensions[0]), -100, 100);
        ArrayRight = NativeArray.Random.Normal<float>(new NativeLayout(Dimensions[1]), -100, 100);
    }

    [Benchmark]
    public void TensornetMatmul()
    {
        //NativeArray<float> res = new(new NativeLayout(new int[] { Dimensions[0][0], Dimensions[1][1] }), new DefaultNativeMemoryManager());
        //MatmulOperator<float, float, float, FloatHandler>.Exec(ArrayLeft.Span, ArrayRight.Span, res.Span, ArrayLeft._layout, ArrayRight._layout, res._layout);
    }
}