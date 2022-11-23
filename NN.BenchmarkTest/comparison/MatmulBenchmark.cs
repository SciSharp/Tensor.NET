using NN.Native.Operators;
using BenchmarkDotNet.Attributes;
using System.Reflection;
using NN.Native.Data;
using NN.Native.Basic;
using static Tensorflow.Binding;
using NN.Native.Operators.Naive;
using NN.Native.Basic.DType;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Jobs;
using NN.BenchmarkTest.Helper;
using Tensorflow.NumPy;

// [MemoryDiagnoser]
[BenchmarkCategory("Comparison-MatmulFloat32")]
// [SimpleJob(RunStrategy.Throughput, runtimeMoniker: RuntimeMoniker.Net60)]
[SimpleJob(RunStrategy.Throughput, runtimeMoniker: RuntimeMoniker.Net70)]
internal class MatmulFloat32Benchmark
{
    public IEnumerable<int[][]> DimensionValues => new int[][][] {
            //new int[][]{ new int[] { 8, 8 }, new int[] { 8, 8} },
            //new int[][]{ new int[] { 20, 20 }, new int[] { 20, 20} },
            //new int[][]{ new int[] { 64, 64 }, new int[] { 64, 64} },
            //new int[][]{ new int[] { 128, 128 }, new int[] { 128, 128} },
            //new int[][]{ new int[] { 192, 192 }, new int[] { 192, 192} },
            //new int[][]{ new int[] { 256, 256 }, new int[] { 256, 256} },
            //new int[][]{ new int[] { 384, 384 }, new int[] { 384, 384} },
            //new int[][]{ new int[] { 512, 512 }, new int[] { 512, 512} },
            //new int[][]{ new int[] { 800, 800 }, new int[] { 800, 800} },
            //new int[][]{ new int[] { 1000, 1000 }, new int[] { 1000, 1000} },

            new int[][]{ new int[] { 125, 125 }, new int[] { 125, 125} },
            new int[][]{ new int[] { 2, 939 }, new int[] { 939, 50} },

            new int[][]{ new int[] { 32, 64 }, new int[] { 64, 96} },
            new int[][]{ new int[] { 256, 512 }, new int[] { 512, 768} },
            new int[][]{ new int[] { 500, 500 }, new int[] { 500, 500} },
        };
    [ParamsSource(nameof(DimensionValues))]
    public int[][] Dimensions { get; set; }

    NativeArray<float> ArrayLeft { get; set; }
    NativeArray<float> ArrayRight { get; set; }
    //NumSharp.NDArray NDArrayLeft { get; set; }
    //NumSharp.NDArray NDArrayRight { get; set; }
    Tensorflow.NumPy.NDArray TFNetTensorLeft { get; set; }
    Tensorflow.NumPy.NDArray TFNetTensorRight { get; set; }
    [GlobalSetup(Targets = new[] { nameof(GenericMathMatmul) })]
    public void TensornetSetup()
    {
        ArrayLeft = NativeArray.Random.Normal<float>(new NativeLayout(Dimensions[0]), -100, 100);
        ArrayRight = NativeArray.Random.Normal<float>(new NativeLayout(Dimensions[1]), -100, 100);
    }
    //[GlobalSetup(Targets = new[] { nameof(NumSharpMatmul) })]
    //public void NumSharpSetup()
    //{
    //    NDArrayLeft = NumSharp.np.random.uniform(-100, 100, new NumSharp.Shape(Dimensions[0]));
    //    NDArrayRight = NumSharp.np.random.uniform(-100, 100, new NumSharp.Shape(Dimensions[1]));
    //}
    [GlobalSetup(Targets = new[] { nameof(TFNetMatmul) })]
    public void TFNetSetup()
    {
        TFNetTensorLeft = Tensorflow.NumPy.np.random.normal(0, 100, new Tensorflow.Shape(Dimensions[0])).astype(Tensorflow.TF_DataType.TF_FLOAT);
        TFNetTensorRight = Tensorflow.NumPy.np.random.normal(0, 100, new Tensorflow.Shape(Dimensions[1])).astype(Tensorflow.TF_DataType.TF_FLOAT);
    }

    [Benchmark]
    public void GenericMathMatmul()
    {
#if NET7_0_OR_GREATER
        var res = NN.Native.Operators.X86.MatmulOperator<float>.Exec(ArrayLeft, ArrayRight);
#else
        var res = new NN.Native.Operators.X86.MatmulOperator<float, Float32Handler>().Exec(ArrayLeft, ArrayRight);
#endif
    }
    //[Benchmark]
    //public void NumSharpMatmul()
    //{
    //    NumSharp.np.matmul(NDArrayLeft, NDArrayRight);
    //}
    [Benchmark]
    public void TFNetMatmul()
    {
        tf.matmul(TFNetTensorLeft, TFNetTensorRight);
    }
}