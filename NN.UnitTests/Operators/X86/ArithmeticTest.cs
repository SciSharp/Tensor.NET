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
    public class ArithmeticTest
    {
        public (int[], int[])[] Shapes { get; set; } = new (int[], int[])[] {
                (new int[] {2, 4}, new int[] {2, 4}),
                (new int[] {7, 16}, new int[] {7, 16}),
                (new int[] {20, 1}, new int[] {20, 1}),
                (new int[] {1, 18}, new int[] {1, 18}),
                (new int[] {99, 128}, new int[] {99, 128}),
                (new int[] {939, 50}, new int[] {939, 50}),
            };

        [Fact]
        public unsafe void TestAddOperatorAllTypes()
        {
            GenericTernaryTest<float>(
#if NET7_0_OR_GREATER
                (x, y) => NN.Native.Operators.X86.Arithmetic.AddOperator<float>.Exec(x, y),
#else
                (x, y) => new NN.Native.Operators.X86.Arithmetic.AddOperator<float, Float32Handler>().Exec(x, y),
#endif
                (x, y) => x + y,
                x => x,
                -100, 100
            );
            GenericTernaryTest<double>(
#if NET7_0_OR_GREATER
                (x, y) => NN.Native.Operators.X86.Arithmetic.AddOperator<double>.Exec(x, y),
#else
                (x, y) => new NN.Native.Operators.X86.Arithmetic.AddOperator<double, Float64Handler>().Exec(x, y),
#endif
                (x, y) => x + y,
                x => x,
                -100, 100
            );
            GenericTernaryTest<int>(
#if NET7_0_OR_GREATER
                (x, y) => NN.Native.Operators.X86.Arithmetic.AddOperator<int>.Exec(x, y),
#else
                (x, y) => new NN.Native.Operators.X86.Arithmetic.AddOperator<int, Int32Handler>().Exec(x, y),
#endif
                (x, y) => x + y,
                x => x,
                -100, 100
            );
        }

        [Fact]
        public unsafe void TestSubtractOperatorAllTypes()
        {
            GenericTernaryTest<float>(
#if NET7_0_OR_GREATER
                (x, y) => NN.Native.Operators.X86.Arithmetic.SubtractOperator<float>.Exec(x, y),
#else
                (x, y) => new NN.Native.Operators.X86.Arithmetic.SubtractOperator<float, Float32Handler>().Exec(x, y),
#endif
                (x, y) => x - y,
                x => x,
                -100, 100
            );
            GenericTernaryTest<double>(
#if NET7_0_OR_GREATER
                (x, y) => NN.Native.Operators.X86.Arithmetic.SubtractOperator<double>.Exec(x, y),
#else
                (x, y) => new NN.Native.Operators.X86.Arithmetic.SubtractOperator<double, Float64Handler>().Exec(x, y),
#endif
                (x, y) => x - y,
                x => x,
                -100, 100
            );
            GenericTernaryTest<int>(
#if NET7_0_OR_GREATER
                (x, y) => NN.Native.Operators.X86.Arithmetic.SubtractOperator<int>.Exec(x, y),
#else
                (x, y) => new NN.Native.Operators.X86.Arithmetic.SubtractOperator<int, Int32Handler>().Exec(x, y),
#endif
                (x, y) => x - y,
                x => x,
                -100, 100
            );
        }

        [Fact]
        public unsafe void TestMultiplyOperatorAllTypes()
        {
            GenericTernaryTest<float>(
#if NET7_0_OR_GREATER
                (x, y) => NN.Native.Operators.X86.Arithmetic.MultiplyOperator<float>.Exec(x, y),
#else
                (x, y) => new NN.Native.Operators.X86.Arithmetic.MultiplyOperator<float, Float32Handler>().Exec(x, y),
#endif
                (x, y) => x * y,
                x => x,
                -100, 100
            );
            GenericTernaryTest<double>(
#if NET7_0_OR_GREATER
                (x, y) => NN.Native.Operators.X86.Arithmetic.MultiplyOperator<double>.Exec(x, y),
#else
                (x, y) => new NN.Native.Operators.X86.Arithmetic.MultiplyOperator<double, Float64Handler>().Exec(x, y),
#endif
                (x, y) => x * y,
                x => x,
                -100, 100
            );
            GenericTernaryTest<int>(
#if NET7_0_OR_GREATER
                (x, y) => NN.Native.Operators.X86.Arithmetic.MultiplyOperator<int>.Exec(x, y),
#else
                (x, y) => new NN.Native.Operators.X86.Arithmetic.MultiplyOperator<int, Int32Handler>().Exec(x, y),
#endif
                (x, y) => x * y,
                x => x,
                -100, 100
            );
        }

        [Fact]
        public unsafe void TestDivideOperatorAllTypes()
        {
            GenericTernaryTest<float>(
#if NET7_0_OR_GREATER
                (x, y) => NN.Native.Operators.X86.Arithmetic.DivideOperator<float>.Exec(x, y),
#else
                (x, y) => new NN.Native.Operators.X86.Arithmetic.DivideOperator<float, Float32Handler>().Exec(x, y),
#endif
                (x, y) => x / y,
                x => x,
                1, 1000
            );
            GenericTernaryTest<double>(
#if NET7_0_OR_GREATER
                (x, y) => NN.Native.Operators.X86.Arithmetic.DivideOperator<double>.Exec(x, y),
#else
                (x, y) => new NN.Native.Operators.X86.Arithmetic.DivideOperator<double, Float64Handler>().Exec(x, y),
#endif
                (x, y) => x / y,
                x => x,
                1, 1000
            );
            GenericTernaryTest<int>(
#if NET7_0_OR_GREATER
                (x, y) => NN.Native.Operators.X86.Arithmetic.DivideOperator<int>.Exec(x, y),
#else
                (x, y) => new NN.Native.Operators.X86.Arithmetic.DivideOperator<int, Int32Handler>().Exec(x, y),
#endif
                (x, y) => x / y,
                x => x,
                1, 1000
            );
        }

        [Fact]
        public unsafe void TestNegativeOperatorAllTypes()
        {
            GenericBinaryTest<float>(
#if NET7_0_OR_GREATER
                x => NN.Native.Operators.X86.Arithmetic.NegativeOperator<float>.Exec(x),
#else
                x => new NN.Native.Operators.X86.Arithmetic.NegativeOperator<float, Float32Handler>().Exec(x),
#endif
                x => -x,
                x => x,
                1, 1000
            );
            GenericBinaryTest<double>(
#if NET7_0_OR_GREATER
                x => NN.Native.Operators.X86.Arithmetic.NegativeOperator<double>.Exec(x),
#else
                x => new NN.Native.Operators.X86.Arithmetic.NegativeOperator<double, Float64Handler>().Exec(x),
#endif
                x => -x,
                x => x,
                1, 1000
            );
            GenericBinaryTest<int>(
#if NET7_0_OR_GREATER
                x => NN.Native.Operators.X86.Arithmetic.NegativeOperator<int>.Exec(x),
#else
                x => new NN.Native.Operators.X86.Arithmetic.NegativeOperator<int, Int32Handler>().Exec(x),
#endif
                x => -x,
                x => x,
                1, 1000
            );
        }

        private unsafe void GenericTernaryTest<TData>(
            Func<NativeArray<TData>, NativeArray<TData>, NativeArray<TData>> x86ExecuteFunc, 
            Func<TData, TData, TData> ternaryFunc, Func<TData, double> resultConvertFunc, 
            TData lowerRandomBorder, TData upperRandomBorder, double error = 0.01)
#if NET7_0_OR_GREATER
            where TData : unmanaged, INumberBase<TData>
#else
            where TData : unmanaged
#endif
        {
            foreach (var shape in Shapes)
            {
                var a = NativeArray.Random.Normal<TData>(new NativeLayout(shape.Item1), lowerRandomBorder, upperRandomBorder);
                var b = NativeArray.Random.Normal<TData>(new NativeLayout(shape.Item2), lowerRandomBorder, upperRandomBorder);
#if NET7_0_OR_GREATER
                var naiveResult = Native.Operators.Naive.TernaryElemWiseOperator<TData, TData, TData>.Exec(a, b, (x, y, offset) => ternaryFunc(x, y));
#else
                var naiveResult = new Native.Operators.Naive.TernaryElemWiseOperator<TData, TData, TData>().Exec(a, b, (x, y, offset) => ternaryFunc(x, y));
#endif
                var x86Result = x86ExecuteFunc(a, b);
                var enumeratorNaive = NativeLayout.GetIndexEnumerator(naiveResult._layout);
                var enumeratorX86 = NativeLayout.GetIndexEnumerator(x86Result._layout);
                var naiveSpan = naiveResult.Span;
                var x86SPan = x86Result.Span;
                for (int i = 0; i < naiveResult._layout.TotalElemCount(); i++)
                {
                    //Assert.Equal(naiveSpan[enumeratorNaive.MoveNext()], x86SPan[enumeratorX86.MoveNext()]);
                    var diff = resultConvertFunc(naiveSpan[enumeratorNaive.MoveNext()]) - resultConvertFunc(x86SPan[enumeratorX86.MoveNext()]);
                    Assert.True(diff > -error && diff < error);
                }
            }
        }

        private unsafe void GenericBinaryTest<TData>(
            Func<NativeArray<TData>, NativeArray<TData>> x86ExecuteFunc,
            Func<TData, TData> referenceFunc, Func<TData, double> resultConvertFunc,
            TData lowerRandomBorder, TData upperRandomBorder, double error = 0.01)
#if NET7_0_OR_GREATER
            where TData : unmanaged, INumberBase<TData>
#else
            where TData : unmanaged
#endif
        {
            foreach (var shape in Shapes)
            {
                var a = NativeArray.Random.Normal<TData>(new NativeLayout(shape.Item1), lowerRandomBorder, upperRandomBorder);
#if NET7_0_OR_GREATER
                var naiveResult = Native.Operators.Naive.BinaryElemWiseOperator<TData, TData>.Exec(a, (x, offset) => referenceFunc(x));
#else
                var naiveResult = new Native.Operators.Naive.BinaryElemWiseOperator<TData, TData>().Exec(a, (x, offset) => referenceFunc(x));
#endif
                var x86Result = x86ExecuteFunc(a);
                var enumeratorNaive = NativeLayout.GetIndexEnumerator(naiveResult._layout);
                var enumeratorX86 = NativeLayout.GetIndexEnumerator(x86Result._layout);
                var naiveSpan = naiveResult.Span;
                var x86SPan = x86Result.Span;
                for (int i = 0; i < naiveResult._layout.TotalElemCount(); i++)
                {
                    //Assert.Equal(naiveSpan[enumeratorNaive.MoveNext()], x86SPan[enumeratorX86.MoveNext()]);
                    var diff = resultConvertFunc(naiveSpan[enumeratorNaive.MoveNext()]) - resultConvertFunc(x86SPan[enumeratorX86.MoveNext()]);
                    Assert.True(diff > -error && diff < error);
                }
            }
        }
    }
}
