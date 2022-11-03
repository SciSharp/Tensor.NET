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
    //[SimpleJob(RunStrategy.Throughput, runtimeMoniker: RuntimeMoniker.NetCoreApp31, baseline: true)]
    //[SimpleJob(RunStrategy.Throughput, runtimeMoniker: RuntimeMoniker.Net60)]
    [SimpleJob(RunStrategy.Throughput, runtimeMoniker: RuntimeMoniker.Net70)]
    //[BenchmarkDotNet.Attributes.RyuJitX64Job]
    public class MatmulFloat32Benchmark
    {
        public IEnumerable<DimensionHelper> DimensionValues => new DimensionHelper[] {
            new DimensionHelper( new int[] { 6, 8 }, new int[] { 8, 11} ),
            new DimensionHelper( new int[] { 32, 64 }, new int[] { 64, 96} ),
            new DimensionHelper(new int[] { 256, 512 }, new int[] { 512, 768} ),
            new DimensionHelper(new int[] { 1000, 1000 }, new int[] { 1000, 1000} ),
            new DimensionHelper(new int[] { 2, 1024 }, new int[] { 1024, 1} ),
            new DimensionHelper(new int[] { 1024, 1 }, new int[] { 2, 1024} )
        };
        [ParamsSource(nameof(DimensionValues))]
        public DimensionHelper Dimensions { get; set; }

        NativeArray<float> ArrayLeft { get; set; }
        NativeArray<float> ArrayRight { get; set; }
        [GlobalSetup(Targets = new[] { nameof(X86Matmul), nameof(NaiveMatmul) })]
        public void GlobalSetup()
        {
            ArrayLeft = NativeArray.Random.Normal<float>(new NativeLayout(Dimensions[0]), -100, 100);
            ArrayRight = NativeArray.Random.Normal<float>(new NativeLayout(Dimensions[1]), -100, 100);
        }
        [Benchmark]
        public void NaiveMatmul()
        {
            NativeArray<float> res = new(new NativeLayout(new int[] { Dimensions[0][0], Dimensions[1][1] }), new DefaultNativeMemoryManager());
#if NET7_0_OR_GREATER
            MatmulOperator<float>.Exec(ArrayLeft.Span, ArrayRight.Span, res.Span, ArrayLeft._layout, ArrayRight._layout, res._layout);
#else
            new MatmulOperator<float, Float32Handler>().Exec(ArrayLeft.Span, ArrayRight.Span, res.Span, ArrayLeft._layout, ArrayRight._layout, res._layout);
#endif
        }
        [Benchmark]
        public unsafe void X86Matmul()
        {
            NativeArray<float> res = new(new NativeLayout(new int[] { Dimensions[0][0], Dimensions[1][1] }), new DefaultNativeMemoryManager());
#if NET7_0_OR_GREATER
            NN.Native.Operators.X86.MatmulOperator<float>.Exec((float*)ArrayLeft.Pin().Pointer, (float*)ArrayRight.Pin().Pointer
                , (float*)res.Pin().Pointer, ArrayLeft._layout, ArrayRight._layout, res._layout);
#else
            new NN.Native.Operators.X86.MatmulOperator<float, Float32Handler>().Exec((float*)ArrayLeft.Pin().Pointer, (float*)ArrayRight.Pin().Pointer
                , (float*)res.Pin().Pointer, ArrayLeft._layout, ArrayRight._layout, res._layout);
#endif
        }
    }
}
