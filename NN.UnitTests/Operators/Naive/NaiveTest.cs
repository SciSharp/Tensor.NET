using NN.Native.Data;
using NN.Native.Extensions;
using NN.Native.Basic;
using NN.Native.Operators.Naive;
using NN.Native.Basic.DType;
using NN.Native.Operators.Common.Params;
using Xunit.Abstractions;

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
            _output.WriteLine(array.Print());
            var convertedArray = new NativeArray<float>(new NativeLayout(new int[] { 4, 3 }), new DefaultNativeMemoryManager());
            TypeConvertOperator<int, float, Float32Handler>.Exec(array.Span, convertedArray.Span, array._layout, convertedArray._layout, new TypeConvertParam()
            {
                Transpose = true,
                DimA = 0,
                DimB = 1
            });
            _output.WriteLine(convertedArray.Print());
        }
        [Fact]
        public void MatmulTest()
        {
            var a = NativeArray.FromArray(new NativeLayout(new int[] { 3, 4 }), new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
            var b = NativeArray.FromArray(new NativeLayout(new int[] { 4, 2 }), new float[] { 1, 2, 1, 2, 1, 1, 2, 3 });
            var c = new NativeArray<float>(new NativeLayout(new int[] { 3, 2 }), new DefaultNativeMemoryManager());
            Native.Operators.Naive.MatmulOperator<float>.Exec(a.Span, b.Span, c.Span, a._layout, b._layout, c._layout);
            _output.WriteLine(c.Print());
        }
    }
}