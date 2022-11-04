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
using System.ComponentModel.Design.Serialization;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

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

            int aIdx = 0, bIdx = 0;

            for (int i = 0; i <= aRows - 8; i += 8)
            {
                bIdx = 0;
                for (int j = 0; j <= bCols - 8; j += 8)
                {
                    Kernel32b8x8(a.Slice(aIdx), b.Slice(bIdx), c.Slice(i * bCols + j), aRows, aCols, bCols);
                    bIdx += 8;
                }
                aIdx += 8 * aCols;
            }
            for (int i = aRows / 8 * 8; i < aRows; i++)
            {
                for (int j = 0; j < bCols; j++)
                {
                    for (int k = 0; k < aCols; k++)
                    {
#if NET7_0_OR_GREATER
                        c[i * bCols + j] += a[i * aCols + k] * b[k * bCols + j];
#endif
                    }
                }
            }
            for (int i = 0; i < aRows / 8 * 8; i++)
            {
                for (int j = bCols / 8 * 8; j < bCols; j++)
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

        public static unsafe void Kernel32b8x8(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> c, int aRows, int aCols, int bCols)
        {
            Vector<T> c_0_v = new Vector<T>();
            Vector<T> c_1_v = new Vector<T>();
            Vector<T> c_2_v = new Vector<T>();
            Vector<T> c_3_v = new Vector<T>();
            Vector<T> c_4_v = new Vector<T>();
            Vector<T> c_5_v = new Vector<T>();
            Vector<T> c_6_v = new Vector<T>();
            Vector<T> c_7_v = new Vector<T>();

            Vector<T> a_0_v, a_1_v, a_2_v, a_3_v, a_4_v, a_5_v, a_6_v, a_7_v;
            Vector<T> b_v;
            int offset = 0;

            for(int k = 0; k < aCols; k++)
            {
                a_0_v = new Vector<T>(a[k]);
                a_1_v = new Vector<T>(a[aCols + k]);
                a_2_v = new Vector<T>(a[2 * aCols + k]);
                a_3_v = new Vector<T>(a[3 * aCols + k]);
                a_4_v = new Vector<T>(a[4 * aCols + k]);
                a_5_v = new Vector<T>(a[5 * aCols + k]);
                a_6_v = new Vector<T>(a[6 * aCols + k]);
                a_7_v = new Vector<T>(a[7 * aCols + k]);

                b_v = new Vector<T>(b.Slice(offset));
                offset += bCols;

                c_0_v += a_0_v * b_v;
                c_1_v += a_1_v * b_v;
                c_2_v += a_2_v * b_v;
                c_3_v += a_3_v * b_v;
                c_4_v += a_4_v * b_v;
                c_5_v += a_5_v * b_v;
                c_6_v += a_6_v * b_v;
                c_7_v += a_7_v * b_v;
            }
            c_0_v.CopyTo(c);
            c_1_v.CopyTo(c.Slice(bCols));
            c_2_v.CopyTo(c.Slice(2 * bCols));
            c_3_v.CopyTo(c.Slice(3 * bCols));
            c_4_v.CopyTo(c.Slice(4 * bCols));
            c_5_v.CopyTo(c.Slice(5 * bCols));
            c_6_v.CopyTo(c.Slice(6 * bCols));
            c_7_v.CopyTo(c.Slice(7 * bCols));
        }
    }
}
