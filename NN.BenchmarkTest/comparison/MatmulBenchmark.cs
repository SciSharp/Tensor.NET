using NN.Native.Operators;
using BenchmarkDotNet.Attributes;
using System.Reflection;
using NN.Native.Data;
using NN.Native.Basic;
using static Tensorflow.Binding;
using NN.Native.Operators.Naive;
using NN.Native.Basic.DType;

[MemoryDiagnoser]
[BenchmarkCategory("Comparison-MatmulFloat32")]
internal class MatmulFloat32Benchmark
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
    NumSharp.NDArray NDArrayLeft { get; set; }
    NumSharp.NDArray NDArrayRight { get; set; }
    Tensorflow.NumPy.NDArray TFNetTensorLeft { get; set; }
    Tensorflow.NumPy.NDArray TFNetTensorRight { get; set; }
    [GlobalSetup(Targets = new[] { nameof(GenericMathMatmul) })]
    public void TensornetSetup()
    {
        ArrayLeft = NativeArray.Random.Normal<float>(new NativeLayout(Dimensions[0]), -100, 100);
        ArrayRight = NativeArray.Random.Normal<float>(new NativeLayout(Dimensions[1]), -100, 100);
    }
    [GlobalSetup(Targets = new[] { nameof(NumSharpMatmul) })]
    public void NumSharpSetup()
    {
        NDArrayLeft = NumSharp.np.random.uniform(-100, 100, new NumSharp.Shape(Dimensions[0]));
        NDArrayRight = NumSharp.np.random.uniform(-100, 100, new NumSharp.Shape(Dimensions[1]));
    }
    [GlobalSetup(Targets = new[] { nameof(TFNetMatmul) })]
    public void TFNetSetup()
    {
        TFNetTensorLeft = Tensorflow.NumPy.np.random.normal(0, 100, new Tensorflow.Shape(Dimensions[0]));
        TFNetTensorRight = Tensorflow.NumPy.np.random.normal(0, 100, new Tensorflow.Shape(Dimensions[1]));
    }

    [Benchmark]
    public void GenericMathMatmul()
    {
        //NumSharp.np.matmul(NDArrayLeft, NDArrayRight);
        NativeArray<float> res = new(new NativeLayout(new int[] { Dimensions[0][0], Dimensions[1][1] }), new DefaultNativeMemoryManager());
#if NET7_0_OR_GREATER
        MatmulOperator<float>.Exec(ArrayLeft.Span, ArrayRight.Span, res.Span, ArrayLeft._layout, ArrayRight._layout, res._layout);
#else
        new MatmulOperator<float, Float32Handler>().Exec(ArrayLeft.Span, ArrayRight.Span, res.Span, ArrayLeft._layout, ArrayRight._layout, res._layout);
#endif
    }
    [Benchmark]
    public void NumSharpMatmul()
    {
        NumSharp.np.matmul(NDArrayLeft, NDArrayRight);
    }
    [Benchmark]
    public void TFNetMatmul()
    {
        tf.matmul(TFNetTensorLeft, TFNetTensorRight);
    }
}