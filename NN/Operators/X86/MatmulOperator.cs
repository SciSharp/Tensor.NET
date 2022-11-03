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
        public unsafe void Exec(T* a, T* b, T* c, in NativeLayout layoutA, in NativeLayout layoutB, in NativeLayout layoutC)
        {
            // The array should be contiguous here
            int aRows = layoutA._shape[0];
            int aCols = layoutA._shape[1];
            int bCols = layoutB._shape[1];

            T c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33;
            T b00, b01, b02, b03;

            for (int i = 0; i <= aRows - 4; i += 4)
            {
                for (int j = 0; j <= bCols - 4; j += 4)
                {
#if NET7_0_OR_GREATER
                    c00 = c01 = c02 = c03 = c10 = c11 = c12 = c13 = c20 = c21 = c22 = c23 = c30 = c31 = c32 = c33 = T.Zero;
                    b00 = b01 = b02 = b03 = T.Zero;
#else
                    c00 = c01 = c02 = c03 = c10 = c11 = c12 = c13 = c20 = c21 = c22 = c23 = c30 = c31 = c32 = c33 = _handler.Zero;
                    b00 = b01 = b02 = b03 = _handler.Zero;
#endif
                    int idx0 = i * aCols, idx1 = (i + 1) * aCols, idx2 = (i + 2) * aCols, idx3 = (i + 3) * aCols;

                    for (int k = 0; k < aCols; k++)
                    {
                        b00 = b[k * bCols + j];
                        b01 = b[k * bCols + j + 1];
                        b02 = b[k * bCols + j + 2];
                        b03 = b[k * bCols + j + 3];
#if NET7_0_OR_GREATER
                        c00 += b00 * a[idx0];
                        c10 += b00 * a[idx1];
                        c20 += b00 * a[idx2];
                        c30 += b00 * a[idx3];

                        c01 += b01 * a[idx0];
                        c11 += b01 * a[idx1];
                        c21 += b01 * a[idx2];
                        c31 += b01 * a[idx3];

                        c02 += b02 * a[idx0];
                        c12 += b02 * a[idx1];
                        c22 += b02 * a[idx2];
                        c32 += b02 * a[idx3];

                        c03 += b03 * a[idx0++];
                        c13 += b03 * a[idx1++];
                        c23 += b03 * a[idx2++];
                        c33 += b03 * a[idx3++];
#endif
                    }
                    c[i * bCols + j] = c00;
                    c[i * bCols + j + 1] = c01;
                    c[i * bCols + j + 2] = c02;
                    c[i * bCols + j + 3] = c03;
                    c[(i + 1) * bCols + j] = c10;
                    c[(i + 1) * bCols + j + 1] = c11;
                    c[(i + 1) * bCols + j + 2] = c12;
                    c[(i + 1) * bCols + j + 3] = c13;
                    c[(i + 2) * bCols + j] = c20;
                    c[(i + 2) * bCols + j + 1] = c21;
                    c[(i + 2) * bCols + j + 2] = c22;
                    c[(i + 2) * bCols + j + 3] = c23;
                    c[(i + 3) * bCols + j] = c30;
                    c[(i + 3) * bCols + j + 1] = c31;
                    c[(i + 3) * bCols + j + 2] = c32;
                    c[(i + 3) * bCols + j + 3] = c33;
                }
            }
            for (int i = aRows / 4 * 4; i < aRows; i++)
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
            for (int i = 0; i < aRows / 4 * 4; i++)
            {
                for (int j = bCols / 4 * 4; j < bCols; j++)
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

#if NET7_0_OR_GREATER
        //[MethodImpl(MethodImplOptions.AggressiveOptimization)]
        static
#endif
        public unsafe void Kernel32b8x8(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> c, in NativeLayout layoutA, in NativeLayout layoutB, in NativeLayout layoutC)
        {
            int aRows = layoutA._shape[0];
            int aCols = layoutA._shape[1];
            int bCols = layoutB._shape[1];

            Vector256<T> c_0_v = new Vector256<T>();
            Vector256<T> c_1_v = new Vector256<T>();
            Vector256<T> c_2_v = new Vector256<T>();
            Vector256<T> c_3_v = new Vector256<T>();
            Vector256<T> c_4_v = new Vector256<T>();
            Vector256<T> c_5_v = new Vector256<T>();
            Vector256<T> c_6_v = new Vector256<T>();
            Vector256<T> c_7_v = new Vector256<T>();

            
        }
    }
}
