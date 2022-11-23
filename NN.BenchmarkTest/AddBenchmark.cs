using BenchmarkDotNet.Attributes;
using NN.Native.Basic.DType;
using NN.Native.Basic;
using NN.Native.Data;
using NN.Native.Operators.Naive;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Jobs;
using NN.BenchmarkTest.Helper;
using System.Xml.Linq;
using System.Reflection.Metadata;

namespace NN.BenchmarkTest
{
    [MemoryDiagnoser]
    [BenchmarkCategory("Add")]
    [SimpleJob(RunStrategy.Throughput, runtimeMoniker: RuntimeMoniker.NetCoreApp31, baseline: true)]
    [SimpleJob(RunStrategy.Throughput, runtimeMoniker: RuntimeMoniker.Net60)]
    [SimpleJob(RunStrategy.Throughput, runtimeMoniker: RuntimeMoniker.Net70)]
    //[BenchmarkDotNet.Attributes.RyuJitX64Job]
    internal class AddFloat32Benchmark
    {
        public IEnumerable<DimensionHelper> DimensionValues => new DimensionHelper[] {
            new DimensionHelper(new int[] { 125, 125 }, new int[] { 125, 125} ),
            new DimensionHelper(new int[] { 2, 939 }, new int[] { 939, 50} ),
            new DimensionHelper( new int[] { 32, 64 }, new int[] { 64, 96} ),
            new DimensionHelper(new int[] { 256, 512 }, new int[] { 512, 768} ),
            new DimensionHelper(new int[] { 500, 500 }, new int[] { 500, 500} ),
            new DimensionHelper(new int[] { 1000, 1000 }, new int[] { 1000, 1000} ),
        };
        [ParamsSource(nameof(DimensionValues))]
        public DimensionHelper Dimensions { get; set; }

        NativeArray<float> ArrayLeft { get; set; }
        NativeArray<float> ArrayRight { get; set; }
        [GlobalSetup(Targets = new[] { nameof(X86Add), nameof(NaiveAdd) })]
        public void GlobalSetup()
        {
            ArrayLeft = NativeArray.Random.Normal<float>(new NativeLayout(Dimensions[0]), -100, 100);
            ArrayRight = NativeArray.Random.Normal<float>(new NativeLayout(Dimensions[1]), -100, 100);
        }
        [Benchmark]
        public void NaiveAdd()
        {
            NativeArray<float> res = new(new NativeLayout(new int[] { Dimensions[0][0], Dimensions[1][1] }), new DefaultNativeMemoryManager());
#if NET7_0_OR_GREATER
            TernaryElemWiseOperator<float, float, float>.Exec(ArrayLeft, ArrayRight, ref res, (x, y, offset) => x + y);
#else
            new TernaryElemWiseOperator<float, float, float>().Exec(ArrayLeft, ArrayRight, ref res, (x, y, offset) => x + y);
#endif
        }
        [Benchmark]
        public unsafe void X86Add()
        {
            NativeArray<float> res = new(new NativeLayout(new int[] { Dimensions[0][0], Dimensions[1][1] }), new DefaultNativeMemoryManager());
#if NET7_0_OR_GREATER
            Native.Operators.X86.Arithmetic.AddOperator<float>.Exec(ArrayLeft, ArrayRight, ref res);
#else
            new Native.Operators.X86.Arithmetic.AddOperator<float, Float32Handler>().Exec(ArrayLeft, ArrayRight, ref res);
#endif
        }
    }
}
