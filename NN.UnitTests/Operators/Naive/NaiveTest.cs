using NN.Native.Data;
using NN.Native.Extensions;
using NN.Native.Basic;
using NN.Native.Operators.Naive;
using NN.Native.Basic.DType;
using NN.Native.Abstraction.Common;
using NN.Native.Abstraction;
using Xunit.Abstractions;
using NN.Native.Abstraction.Operators;

namespace NN.UnitTests.Naive
{
    public class NaiveTest
    {
        private readonly ITestOutputHelper _output;

        public NaiveTest(ITestOutputHelper output)
        {
            _output = output;
        }
        [Fact]
        public void TypeConvertTest()
        {
            var array = NativeArray.FromArray(new NativeLayout(new int[] { 3, 4 }), new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
            var truth = NativeArray.FromArray(new NativeLayout(new int[] { 4, 3 }), new float[] { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 });
#if NET7_0_OR_GREATER
            var res = TypeConvertOperator<int, float, Float32Handler>.Exec(array, new TypeConvertParam()
            {
                Transpose = true,
                DimA = 0,
                DimB = 1
            });
#else
            var res = new TypeConvertOperator<int, float, Float32Handler>().Exec(array, new TypeConvertParam()
            {
                Transpose = true,
                DimA = 0,
                DimB = 1
            });
#endif
            Assert.True(res.IsElementsEqualWith(truth));
        }
        [Fact]
        public void TransposeTest()
        {
            var array = NativeArray.FromArray(new NativeLayout(new int[] { 3, 4 }), new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
            var truth = NativeArray.FromArray(new NativeLayout(new int[] { 4, 3 }), new int[] { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 });
#if NET7_0_OR_GREATER
            var res = TransposeOperator<int>.Exec(array, new TransposeParam() { DimA = 0, DimB = 1 });
#else
            var res = new TransposeOperator<int>().Exec(array, new TransposeParam(){ DimA = 0, DimB = 1 });
#endif
            Assert.True(res.IsElementsEqualWith(truth));
        }
        [Fact]
        public void MatmulTest()
        {
            var a = NativeArray.FromArray(new NativeLayout(new int[] { 3, 4 }), new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
            var b = NativeArray.FromArray(new NativeLayout(new int[] { 4, 2 }), new float[] { 1, 2, 1, 2, 1, 1, 2, 3 });
            var truth = NativeArray.FromArray(new NativeLayout(new int[] { 3, 2 }), new float[] { 14, 21, 34, 53, 54, 85 });
#if NET7_0_OR_GREATER
            var c = Native.Operators.Naive.MatmulOperator<float>.Exec(a, b);
#else
            var c = new Native.Operators.Naive.MatmulOperator<float, Float32Handler>().Exec(a, b);
#endif
            Assert.True(c.IsElementsEqualWith(truth));
        }
        [Fact]
        public void BinaryElemWiseTest()
        {
            var array = NativeArray.FromArray(new NativeLayout(new int[] { 2, 3 }), new int[] { 1, 2, 3, 4, 5, 6 });
            var truth = NativeArray.FromArray(new NativeLayout(new int[] { 2, 3 }), new double[] { 2.5, 5.0, 7.5, 10.0, 12.5, 15.0});
#if NET7_0_OR_GREATER
            var res = BinaryElemWiseOperator<int, double>.Exec(array, (x, idx) => 2.5 * x);
#else
            var res = new BinaryElemWiseOperator<int, double>().Exec(array, (x, idx) => x * 2.5);
#endif
            Assert.True(res.IsElementsEqualWith(truth));
        }
        [Fact]
        public void UnaryElemWiseTest()
        {
            var array = NativeArray.FromArray(new NativeLayout(new int[] { 2, 3 }), new int[] { 1, 2, 3, 4, 5, 6 });
            var truth = NativeArray.FromArray(new NativeLayout(new int[] { 2, 3 }), new int[] { 2, 5, 8, 11, 14, 17 });
#if NET7_0_OR_GREATER
            UnaryElemWiseOperator<int>.Exec(ref array, (x, idx) => 2 * x + idx);
#else
            new UnaryElemWiseOperator<int>().Exec(ref array, (x, idx) => x * 2 + idx);
#endif
            Assert.True(array.IsElementsEqualWith(truth));
        }
        [Fact]
        public void TernaryElemWiseTest()
        {
            var a = NativeArray.FromArray(new NativeLayout(new int[] { 2, 3 }), new int[] { 1, 2, 3, 4, 5, 6 });
            var b = NativeArray.FromArray(new NativeLayout(new int[] { 2, 3 }), new float[] { 6, 5, 4, 3, 2, 1 });
            var truth = NativeArray.FromArray(new NativeLayout(new int[] { 2, 3 }), new double[] { -5, -3, -1, 1, 3, 5});
#if NET7_0_OR_GREATER
            var res = TernaryElemWiseOperator<int, float, double>.Exec(a, b, (x, y, idx) => x - y);
#else
            var res = new TernaryElemWiseOperator<int, float, double>().Exec(a, b, (x, y, idx) => x - y);
#endif
            Assert.True(res.IsElementsEqualWith(truth));
        }
        [Fact]
        public void ReduceSumTest()
        {
            var array = NativeArray.FromArray(new NativeLayout(new int[] { 3, 2, 3 }), new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 });
            var truth = NativeArray.FromArray(new NativeLayout(new int[] { 3 }), new int[] { 51, 57, 63 });
#if NET7_0_OR_GREATER
            var res = ReduceOperator<int, ReduceSumOpHandler<int>>.Exec(array, new ReduceParam(new int[] { 0, 1 }));
#else
            var res = new ReduceOperator<int, ReduceSumOpHandler<int, Int32Handler>>().Exec(array, new ReduceParam(new int[] { 0, 1 }));
#endif
            _output.WriteLine(res.Print());
            res._layout.RemoveDanglingAxesInplace();
            Assert.True(res.IsElementsEqualWith(truth));
        }
        [Fact]
        public void ReduceMaxTest()
        {
            var array = NativeArray.FromArray(new NativeLayout(new int[] { 3, 2, 3 }), new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 });
            var truth = NativeArray.FromArray(new NativeLayout(new int[] { 3, 2 }), new int[] { 3, 6, 9, 12, 15, 18 });
#if NET7_0_OR_GREATER
            var res = ReduceOperator<int, ReduceMaxOpHandler<int>>.Exec(array, new ReduceParam(new int[] { 2 }));
#else
            var res = new ReduceOperator<int, ReduceMaxOpHandler<int, Int32Handler>>().Exec(array, new ReduceParam(new int[] { 2 }));
#endif
            _output.WriteLine(res.Print());
            res._layout.RemoveDanglingAxesInplace();
            Assert.True(res.IsElementsEqualWith(truth));
        }
        [Fact]
        public void ReduceMinTest()
        {
            var array = NativeArray.FromArray(new NativeLayout(new int[] { 3, 2, 3, 2 }), 
                new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 }
            );
            var truth = NativeArray.FromArray(new NativeLayout(new int[] { 2, 2 }), new int[] { 1, 2, 7, 8 });
#if NET7_0_OR_GREATER
            var res = ReduceOperator<int, ReduceMinOpHandler<int>>.Exec(array, new ReduceParam(new int[] { 0, 2 }));
#else
            var res = new ReduceOperator<int, ReduceMinOpHandler<int, Int32Handler>>().Exec(array, new ReduceParam(new int[] { 0, 2 }));
#endif
            _output.WriteLine(res.Print());
            res._layout.RemoveDanglingAxesInplace();
            Assert.True(res.IsElementsEqualWith(truth));
        }

        [Fact]
        public void ReduceMeanTest()
        {
            var array = NativeArray.FromArray(new NativeLayout(new int[] { 3, 2, 3, 2 }),
                new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 }
            );
            var truth = NativeArray.FromArray(new NativeLayout(new int[] { 2, 3 }), new int[] { 13, 15, 17, 19, 21, 23 });
#if NET7_0_OR_GREATER
            var res = ReduceOperator<int, ReduceMeanOpHandler<int>>.Exec(array, new ReduceParam(new int[] { 0, 3 }));
#else
            var res = new ReduceOperator<int, ReduceMeanOpHandler<int, Int32Handler>>().Exec(array, new ReduceParam(new int[] { 0, 3 }));
#endif
            _output.WriteLine(res.Print());
            res._layout.RemoveDanglingAxesInplace();
            Assert.True(res.IsElementsEqualWith(truth));
        }
    }
}