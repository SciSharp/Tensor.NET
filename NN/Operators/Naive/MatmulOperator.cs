using NN.Native.Abstraction.DType;
using NN.Native.Abstraction.Operators;
using NN.Native.Basic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Operators.Naive
{

#if NET7_0_OR_GREATER
    public class MatmulOperator<T>: IMatmulOperator<T> where T : unmanaged, INumberBase<T>
    {
        public static bool IsThreadSafe { get => false; }
        public static OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#else
    public class MatmulOperator<T, U> where T : unmanaged where U: INativeDTypeHandler<T>, new()
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
            for (int i = 0; i < aRows; i++)
            {
                for (int j = 0; j < bCols; j++)
                {
#if NET7_0_OR_GREATER
                    var res = T.Zero;
#else
                    var res = _handler.Zero;
#endif
                    for (int k = 0; k < aCols; k++)
                    {
#if NET7_0_OR_GREATER
                        res += a[i * aCols + k] * b[k * bCols + j];
#else
                        res = _handler.MultiplyAndAdd(a[i * aCols + k], b[k * bCols + j], res);
#endif
                    }
                    c[i * bCols + j] = res;
                }
            }
        }
    }
}
