using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NN.Native.Data;
using NN.Native.Basic;
using NN.Native.Basic.DType;
using System.Runtime.InteropServices;
using NN.Native.Abstraction.DType;
using System.Numerics;

namespace NN.UnitTests.Operators.X86
{
    public class MatmulTest
    {
        public (int[], int[])[] Shapes { get; set; } = new(int[], int[])[] {
                (new int[] {2, 4}, new int[] {4, 4}),
                (new int[] {3, 7}, new int[] {7, 16}),
                (new int[] {1, 20}, new int[] {20, 1}),
                (new int[] {10, 1}, new int[] {1, 18}),
                (new int[] {16, 99}, new int[] {99, 128}),
                //(new int[] {1024, 1023}, new int[] {1023, 1026}),
                (new int[] {20, 28}, new int[] {28, 36}),
                (new int[] {2, 939}, new int[] {939, 50}),
            };

        [Fact]
        public unsafe void TestAllTypes()
        {
            GenericTest<float, Float32Handler>(x => x, x => x);
            GenericTest<double, Float64Handler>(x => x, x => x);
            GenericTest<int, Int32Handler>(x => x, x => x);
        }

        private unsafe void GenericTest<TData, THandler>(Func<int, TData> numberConvertFunc, Func<TData, double> resultConvertFunc)
#if NET7_0_OR_GREATER
            where TData: unmanaged, INumberBase<TData>
#else
            where TData: unmanaged
#endif
            where THandler: INativeDTypeHandler<TData>, new()
        {
            foreach (var shape in Shapes)
            {
                var a = NativeArray.Random.Normal<TData>(new NativeLayout(shape.Item1), numberConvertFunc(-10), numberConvertFunc(10));
                var b = NativeArray.Random.Normal<TData>(new NativeLayout(shape.Item2), numberConvertFunc(-10), numberConvertFunc(10));
#if NET7_0_OR_GREATER
                var naiveResult = Native.Operators.Naive.MatmulOperator<TData>.Exec(a, b);
                var x86Result = Native.Operators.X86.MatmulOperator<TData>.Exec(a, b);
#else
                var naiveResult = new Native.Operators.Naive.MatmulOperator<TData, THandler>().Exec(a, b);
                var x86Result = new Native.Operators.X86.MatmulOperator<TData, THandler>().Exec(a, b);
#endif
                var enumeratorNaive = NativeLayout.GetIndexEnumerator(naiveResult._layout);
                var enumeratorX86 = NativeLayout.GetIndexEnumerator(x86Result._layout);
                var naiveSpan = naiveResult.Span;
                var x86SPan = x86Result.Span;
                for (int i = 0; i < naiveResult._layout.TotalElemCount(); i++)
                {
                    //Assert.Equal(naiveSpan[enumeratorNaive.MoveNext()], x86SPan[enumeratorX86.MoveNext()]);
                    var diff = resultConvertFunc(naiveSpan[enumeratorNaive.MoveNext()]) - resultConvertFunc(x86SPan[enumeratorX86.MoveNext()]);
                    Assert.True(diff > -0.01 && diff < 0.01);
                }
            }
        }
    }
}
