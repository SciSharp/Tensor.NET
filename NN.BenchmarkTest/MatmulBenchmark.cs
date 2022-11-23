using BenchmarkDotNet.Attributes;
using NN.Native.Basic.DType;
using NN.Native.Basic;
using NN.Native.Data;
using NN.Native.Operators.Naive;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Jobs;
using NN.BenchmarkTest.Helper;

namespace NN.BenchmarkTest
{
    [MemoryDiagnoser] 
    [BenchmarkCategory("Matmul")]
    [SimpleJob(RunStrategy.Throughput, runtimeMoniker: RuntimeMoniker.NetCoreApp31, baseline: true)]
    [SimpleJob(RunStrategy.Throughput, runtimeMoniker: RuntimeMoniker.Net60)]
    [SimpleJob(RunStrategy.Throughput, runtimeMoniker: RuntimeMoniker.Net70)]
    //[BenchmarkDotNet.Attributes.RyuJitX64Job]
    public class MatmulFloat32Benchmark
    {
        public IEnumerable<DimensionHelper> DimensionValues => new DimensionHelper[] {
            //new DimensionHelper( new int[] { 6, 7 }, new int[] { 7, 11} ),
            //new DimensionHelper( new int[] { 30, 65 }, new int[] { 65, 91} ),
            new DimensionHelper(new int[] { 125, 125 }, new int[] { 125, 125} ),
            ////new DimensionHelper(new int[] { 259, 511 }, new int[] { 511, 777} ),
            ////new DimensionHelper(new int[] { 999, 998 }, new int[] { 998, 1013} ),
            //new DimensionHelper(new int[] { 50, 939 }, new int[] { 939, 1} ),
            new DimensionHelper(new int[] { 2, 939 }, new int[] { 939, 50} ),
            //new DimensionHelper(new int[] { 1023, 1 }, new int[] { 1, 50} ),
            //new DimensionHelper(new int[] { 67, 1 }, new int[] { 1, 789} ),

            new DimensionHelper( new int[] { 32, 64 }, new int[] { 64, 96} ),
            new DimensionHelper(new int[] { 256, 512 }, new int[] { 512, 768} ),
            new DimensionHelper(new int[] { 500, 500 }, new int[] { 500, 500} ),
            //new DimensionHelper(new int[] { 1000, 1000 }, new int[] { 1000, 1000} ),
        };
        [ParamsSource(nameof(DimensionValues))]
        public DimensionHelper Dimensions { get; set; }

        NativeArray<double> ArrayLeft { get; set; }
        NativeArray<double> ArrayRight { get; set; }
        [GlobalSetup(Targets = new[] { nameof(X86Matmul), nameof(NaiveMatmul) })]
        public void GlobalSetup()
        {
            ArrayLeft = NativeArray.Random.Normal<double>(new NativeLayout(Dimensions[0]), -100, 100);
            ArrayRight = NativeArray.Random.Normal<double>(new NativeLayout(Dimensions[1]), -100, 100);
        }
        [Benchmark]
        public void NaiveMatmul()
        {
            NativeArray<double> res = new(new NativeLayout(new int[] { Dimensions[0][0], Dimensions[1][1] }), new DefaultNativeMemoryManager());
#if NET7_0_OR_GREATER
            MatmulOperator<double>.Exec(ArrayLeft, ArrayRight, ref res);
#else
            new MatmulOperator<double, Float64Handler>().Exec(ArrayLeft, ArrayRight, ref res);
#endif
        }
        [Benchmark]
        public unsafe void X86Matmul()
        {
            NativeArray<double> res = new(new NativeLayout(new int[] { Dimensions[0][0], Dimensions[1][1] }), new DefaultNativeMemoryManager());
#if NET7_0_OR_GREATER
            NN.Native.Operators.X86.MatmulOperator<double>.Exec(ArrayLeft, ArrayRight, ref res);
#else
            new NN.Native.Operators.X86.MatmulOperator<double, Float64Handler>().Exec(ArrayLeft, ArrayRight, ref res);
#endif
        }
    }
}
