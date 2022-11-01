using NN.Native.Abstraction.Operators;
using NN.Native.Basic;
using NN.Native.Abstraction.DType;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;

namespace NN.Native.Operators.X86
{
#if NET7_0_OR_GREATER
    public class MatmulOperator<T> : IMatmulOperator<T> where T : unmanaged, INumberBase<T>
    {
        public static bool IsThreadSafe { get => false; }
        public static OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#else
    public class MatmulOperator<T, U>: IMatmulOperator<T> where T : unmanaged where U: INativeDTypeHandler<T>, new()
    {
        private static U _handler = new();
        public bool IsThreadSafe { get => false; }
        public OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#endif
#if NET7_0_OR_GREATER
        //[MethodImpl(MethodImplOptions.AggressiveOptimization)]
        static
#endif
        public unsafe void Exec(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> c, in NativeLayout layoutA, in NativeLayout layoutB, in NativeLayout layoutC)
        {
            // The array should be contiguous here
            int aRows = layoutA._shape[0];
            int aCols = layoutA._shape[1];
            int bCols = layoutB._shape[1];

            for(int j = 0; j <= bCols - 4; j += 4)
            {
                for(int i = 0; i < aRows; i++)
                {
                    for(int k = 0; k < aCols; k++)
                    {
#if NET7_0_OR_GREATER
                        c[i * bCols + j] += a[i * aCols + k] * b[k * bCols + j];
                        c[i * bCols + j + 1] += a[i * aCols + k] * b[k * bCols + j + 1];
                        c[i * bCols + j + 2] += a[i * aCols + k] * b[k * bCols + j + 2];
                        c[i * bCols + j + 3] += a[i * aCols + k] * b[k * bCols + j + 3];
#endif
                    }
                }
            }
            for(int j = bCols / 4 * 4; j < bCols; j++)
            {
                for (int i = 0; i < aRows; i++)
                {
                    for (int k = 0; k < aCols; k++)
                    {
#if NET7_0_OR_GREATER
                        c[i * bCols + j] += a[i * aCols + k] * b[k * bCols + j];
#endif
                    }
                }
            }
        }
    }
}
