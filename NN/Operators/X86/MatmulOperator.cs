using NN.Native.Abstraction.Operators;
using NN.Native.Basic;
using NN.Native.Abstraction.DType;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NN.Native.Data;
using NN.Native.Abstraction.Data;
using System.Linq.Expressions;

namespace NN.Native.Operators.X86
{
#if NET7_0_OR_GREATER
    public class MatmulOperator<T> : IMatmulOperator<T> where T : unmanaged, INumberBase<T>
    {
        public static bool IsThreadSafe { get => false; }
        public static bool RequireContiguousArray { get => true; }
        public static OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#else
    public class MatmulOperator<T, THandler>: IMatmulOperator<T> where T : unmanaged where THandler: INativeDTypeHandler<T>, new()
    {
        private static THandler _handler = new();
        public bool IsThreadSafe { get => false; }
        public bool RequireContiguousArray { get => true; }
        public OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#endif
        private static readonly int _bw = 64;
        private static readonly int _bh = 64;

#if NET7_0_OR_GREATER
        static
#endif
        public NativeArray<T> Exec(in NativeArray<T> a, in NativeArray<T> b, INativeMemoryManager? memoryManager = null)
        {
            var layoutA = new NativeLayout(a._layout);
            var layoutB = new NativeLayout(b._layout);
            var res = new NativeArray<T>(IMatmulOperator<T>.DeduceLayout(ref layoutA, ref layoutB), memoryManager);
            ExecInternal(a.Span, b.Span, res.Span, layoutA, layoutB, res._layout);
            return res;
        }

        private unsafe static void ExecInternal(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> c, in NativeLayout layoutA, in NativeLayout layoutB, in NativeLayout layoutC)
        {
            // The array should be contiguous here
            int aRows = layoutA._shape[0];
            int aCols = layoutA._shape[1];
            int bCols = layoutB._shape[1];
            int bIdx, cIdx = 0;

            var packedB = new T[_bh * _bw];
            var packedSpanB = packedB.AsSpan();
            if(typeof(T) == typeof(float))
            {
                fixed (T* ptrA = a, ptrB = b, ptrC = c, ptrPackedB = packedSpanB)
                {
                    float* fa = (float*)ptrA, fb = (float*)ptrB, fc = (float*)ptrC, fPackedB = (float*)ptrPackedB;
                    for (int j = 0; j < bCols; j += _bw, cIdx += _bw)
                    {
                        int width = (bCols - j) > _bw ? _bw : (bCols - j);
                        for (int k = 0; k < aCols; k += _bh)
                        {
                            int height = (aCols - k) > _bh ? _bh : (aCols - k);
                            bIdx = j + k * bCols;
                            Float32MicroKernel.ExecBlock(fa + k, fb + bIdx, fc + cIdx, aRows, height, width, aRows, aCols, bCols, fPackedB);
                        }
                    }
                }
            }
            else if(typeof(T) == typeof(double))
            {
                fixed (T* ptrA = a, ptrB = b, ptrC = c, ptrPackedB = packedSpanB)
                {
                    double* da = (double*)ptrA, db = (double*)ptrB, dc = (double*)ptrC, dPackedB = (double*)ptrPackedB;
                    for (int j = 0; j < bCols; j += _bw, cIdx += _bw)
                    {
                        int width = (bCols - j) > _bw ? _bw : (bCols - j);
                        for (int k = 0; k < aCols; k += _bh) // exchange order 
                        {
                            int height = (aCols - k) > _bh ? _bh : (aCols - k);
                            bIdx = j + k * bCols;
                            Float64MicroKernel.ExecBlock(da + k, db + bIdx, dc + cIdx, aRows, height, width, aRows, aCols, bCols, dPackedB);
                        }
                    }
                }
            }
            else
            {
                fixed (T* ptrA = a, ptrB = b, ptrC = c, ptrPackedB = packedSpanB)
                {
                    int length = Vector<T>.Count;
                    for (int j = 0; j < bCols; j += _bw, cIdx += _bw)
                    {
                        int width = (bCols - j) > _bw ? _bw : (bCols - j);
                        for (int k = 0; k < aCols; k += _bh) // exchange order 
                        {
                            int height = (aCols - k) > _bh ? _bh : (aCols - k);
                            bIdx = j + k * bCols;
                            GenericMicroKernel.ExecBlock(ptrA + k, ptrB + bIdx, ptrC + cIdx, aRows, height, width, aRows, aCols, bCols, ptrPackedB, length);
                        }
                    }
                }
            }
        }

        private static class Utils
        {
            public unsafe static void PackMatrixBWithPxQ<U>(U* src, U* dst, int p, int q, int bCols) where U: unmanaged
            {
#if NET7_0_OR_GREATER
            Vector256<U> data;
            for (int k = 0; k < p; k++, dst += q, src += bCols)
            {
                data = Vector256.Load(src);
                data.Store(dst);
            }
#else
                for (int k = 0; k < p; k++, src += bCols)
                {
                    for(int j = 0; j < q; j++)
                    {
                        *(dst++) = src[j];
                    }
                }
#endif
            }
        }

        private static class Float32MicroKernel
        {
            // (m, p) * (p, n)
            public static unsafe void ExecBlock(float* a, float* b, float* c, int m, int p, int n, int aRows, int aCols, int bCols, float* packedB)
            {
                int aIdx, bIdx, cIdx;

                // 8x8 blocks
                for (int i = 0; i <= m - 8; i += 8)
                {
                    cIdx = i * bCols;
                    aIdx = i * aCols;
                    for (int j = 0; j <= n - 8; j += 8, cIdx += 8)
                    {
                        bIdx = j * p;
                        if (i == 0) Utils.PackMatrixBWithPxQ(b + j, packedB + bIdx, p, 8, bCols);
                        Kernel8x8(a + aIdx, packedB + bIdx, c + cIdx, p, 8, aRows, aCols, bCols);
                    }
                }
                // The left rows and every 8 columns
                int iStart = m / 8 * 8;
                aIdx = iStart * aCols;
                bool hasPacked = m >= 8 && n >= 8;
                for (int i = iStart; i < m; i++, aIdx += aCols)
                {
                    cIdx = i * bCols;
                    for (int j = 0; j <= n - 8; j += 8, cIdx += 8)
                    {
                        if (!hasPacked && i == iStart) Utils.PackMatrixBWithPxQ(b + j, packedB + j * p, p, 8, bCols);
                        Kernel1x8(a + aIdx, packedB + j * p, c + cIdx, p, 8, aRows, aCols, bCols);
                    }
                }

                // The left columns and every 4 rows
                int jStart = n / 8 * 8;
                float c0, c1, c2, c3;
                for (int i = 0; i <= m - 4; i += 4)
                {
                    for (int j = jStart; j < n; j++)
                    {
                        int aIdx0 = i * aCols, aIdx1 = aIdx0 + aCols, aIdx2 = aIdx1 + aCols, aIdx3 = aIdx2 + aCols;
                        cIdx = i * bCols + j;
                        c0 = c[cIdx];
                        cIdx += bCols;
                        c1 = c[cIdx];
                        cIdx += bCols;
                        c2 = c[cIdx];
                        cIdx += bCols;
                        c3 = c[cIdx];

                        for (int k = 0; k < p; k++)
                        {
                            var bValue = b[k * bCols + j];
                            c0 += a[aIdx0++] * bValue;
                            c1 += a[aIdx1++] * bValue;
                            c2 += a[aIdx2++] * bValue;
                            c3 += a[aIdx3++] * bValue;
                        }
                        cIdx = i * bCols + j;
                        c[cIdx] = c0;
                        cIdx += bCols;
                        c[cIdx] = c1;
                        cIdx += bCols;
                        c[cIdx] = c2;
                        cIdx += bCols;
                        c[cIdx] = c3;
                    }
                }
                // The last 1-3 rows and the left columns
                for (int i = m / 4 * 4; i < m; i++)
                {
                    for (int k = 0; k < p; k++)
                    {
                        cIdx = i * bCols + jStart;
                        bIdx = k * bCols + jStart;
                        var aValue = a[i * aCols + k];
                        for (int j = jStart; j < n; j++)
                        {
                            c[cIdx++] += aValue * b[bIdx++];
                        }
                    }
                }
            }
            private static unsafe void Kernel8x8(float* a, float* b, float* c, int p, int n, int aRows, int aCols, int bCols)
            {
                var cBackup = c;

                var c_0_v = Avx2.LoadVector256(c);
                c += bCols;
                var c_1_v = Avx2.LoadVector256(c);
                c += bCols;
                var c_2_v = Avx2.LoadVector256(c);
                c += bCols;
                var c_3_v = Avx2.LoadVector256(c);
                c += bCols;
                var c_4_v = Avx2.LoadVector256(c);
                c += bCols;
                var c_5_v = Avx2.LoadVector256(c);
                c += bCols;
                var c_6_v = Avx2.LoadVector256(c);
                c += bCols;
                var c_7_v = Avx2.LoadVector256(c);
                c = cBackup;

                Vector256<float> b_v;

                for (int k = 0; k < p; k++)
                {
                    b_v = Avx2.LoadVector256(b);
                    b += n;
                    var ta = a + k;

                    c_0_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_0_v);
                    ta += aCols;
                    c_1_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_1_v);
                    ta += aCols;
                    c_2_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_2_v);
                    ta += aCols;
                    c_3_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_3_v);
                    ta += aCols;
                    c_4_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_4_v);
                    ta += aCols;
                    c_5_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_5_v);
                    ta += aCols;
                    c_6_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_6_v);
                    ta += aCols;
                    c_7_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_7_v);
                }

                Avx2.Store(c, c_0_v);
                c += bCols;
                Avx2.Store(c, c_1_v);
                c += bCols;
                Avx2.Store(c, c_2_v);
                c += bCols;
                Avx2.Store(c, c_3_v);
                c += bCols;
                Avx2.Store(c, c_4_v);
                c += bCols;
                Avx2.Store(c, c_5_v);
                c += bCols;
                Avx2.Store(c, c_6_v);
                c += bCols;
                Avx2.Store(c, c_7_v);
            }
            private static unsafe void Kernel1x8(float* a, float* b, float* c, int p, int n, int aRows, int aCols, int bCols)
            {
                var c_0_v = Avx2.LoadVector256(c);

                Vector256<float> b_v;

                for (int k = 0; k < p; k++, b += n)
                {
                    b_v = Avx2.LoadVector256(b);

                    c_0_v = Fma.MultiplyAdd(b_v, Vector256.Create(*a++), c_0_v);
                }

                Avx2.Store(c, c_0_v);
            }
        }

        private static class Float64MicroKernel
        {
            public static unsafe void ExecBlock(double* a, double* b, double* c, int m, int p, int n, int aRows, int aCols, int bCols, double* packedB)
            {
                int aIdx, bIdx, cIdx;

                for (int i = 0; i <= m - 8; i += 8)
                {
                    cIdx = i * bCols;
                    aIdx = i * aCols;
                    for (int j = 0; j <= n - 4; j += 4, cIdx += 4)
                    {
                        bIdx = j * p;
                        if (i == 0) Utils.PackMatrixBWithPxQ(b + j, packedB + bIdx, p, 4, bCols);
                        Kernel64b8x4(a + aIdx, packedB + bIdx, c + cIdx, p, 4, aRows, aCols, bCols);
                    }
                }
                int iStart = m / 8 * 8;
                aIdx = iStart * aCols;
                bool hasPacked = m >= 8 && n >= 8;
                for (int i = iStart; i < m; i++, aIdx += aCols)
                {
                    cIdx = i * bCols;
                    for (int j = 0; j <= n - 4; j += 4, cIdx += 4)
                    {
                        if (!hasPacked && i == iStart) Utils.PackMatrixBWithPxQ(b + j, packedB + j * p, p, 4, bCols);
                        Kernel64b1x4(a + aIdx, packedB + j * p, c + cIdx, p, 4, aRows, aCols, bCols);
                    }
                }
                int jStart = n / 4 * 4;
                double c0, c1, c2, c3;
                for (int i = 0; i <= m - 4; i += 4)
                {
                    for (int j = jStart; j < n; j++)
                    {
                        int aIdx0 = i * aCols, aIdx1 = aIdx0 + aCols, aIdx2 = aIdx1 + aCols, aIdx3 = aIdx2 + aCols;
                        cIdx = i * bCols + j;
                        c0 = c[cIdx];
                        cIdx += bCols;
                        c1 = c[cIdx];
                        cIdx += bCols;
                        c2 = c[cIdx];
                        cIdx += bCols;
                        c3 = c[cIdx];

                        for (int k = 0; k < p; k++)
                        {
                            var bValue = b[k * bCols + j];
                            c0 += a[aIdx0++] * bValue;
                            c1 += a[aIdx1++] * bValue;
                            c2 += a[aIdx2++] * bValue;
                            c3 += a[aIdx3++] * bValue;
                        }
                        cIdx = i * bCols + j;
                        c[cIdx] = c0;
                        cIdx += bCols;
                        c[cIdx] = c1;
                        cIdx += bCols;
                        c[cIdx] = c2;
                        cIdx += bCols;
                        c[cIdx] = c3;
                    }
                }
                for (int i = m / 4 * 4; i < m; i++)
                {
                    for (int k = 0; k < p; k++)
                    {
                        cIdx = i * bCols + jStart;
                        bIdx = k * bCols + jStart;
                        var aValue = a[i * aCols + k];
                        for (int j = jStart; j < n; j++)
                        {
                            c[cIdx++] += aValue * b[bIdx++];
                        }
                    }
                }
            }
            private static unsafe void Kernel64b8x4(double* a, double* b, double* c, int p, int n, int aRows, int aCols, int bCols)
            {
                var cBackup = c;

                var c_0_v = Avx2.LoadVector256(c);
                c += bCols;
                var c_1_v = Avx2.LoadVector256(c);
                c += bCols;
                var c_2_v = Avx2.LoadVector256(c);
                c += bCols;
                var c_3_v = Avx2.LoadVector256(c);
                c += bCols;
                var c_4_v = Avx2.LoadVector256(c);
                c += bCols;
                var c_5_v = Avx2.LoadVector256(c);
                c += bCols;
                var c_6_v = Avx2.LoadVector256(c);
                c += bCols;
                var c_7_v = Avx2.LoadVector256(c);
                c = cBackup;

                Vector256<double> b_v;

                for (int k = 0; k < p; k++)
                {
                    b_v = Avx2.LoadVector256(b);
                    b += n;
                    var ta = a + k;

                    c_0_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_0_v);
                    ta += aCols;
                    c_1_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_1_v);
                    ta += aCols;
                    c_2_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_2_v);
                    ta += aCols;
                    c_3_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_3_v);
                    ta += aCols;
                    c_4_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_4_v);
                    ta += aCols;
                    c_5_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_5_v);
                    ta += aCols;
                    c_6_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_6_v);
                    ta += aCols;
                    c_7_v = Fma.MultiplyAdd(b_v, Vector256.Create(*ta), c_7_v);
                }

                Avx2.Store(c, c_0_v);
                c += bCols;
                Avx2.Store(c, c_1_v);
                c += bCols;
                Avx2.Store(c, c_2_v);
                c += bCols;
                Avx2.Store(c, c_3_v);
                c += bCols;
                Avx2.Store(c, c_4_v);
                c += bCols;
                Avx2.Store(c, c_5_v);
                c += bCols;
                Avx2.Store(c, c_6_v);
                c += bCols;
                Avx2.Store(c, c_7_v);
            }
            private static unsafe void Kernel64b1x4(double* a, double* b, double* c, int p, int n, int aRows, int aCols, int bCols)
            {
                var c_0_v = Avx2.LoadVector256(c);

                Vector256<double> b_v;

                for (int k = 0; k < p; k++, b += n)
                {
                    b_v = Avx2.LoadVector256(b);

                    c_0_v = Fma.MultiplyAdd(b_v, Vector256.Create(*a++), c_0_v);
                }

                Avx2.Store(c, c_0_v);
            }
        }

        private static class GenericMicroKernel
        {
            // (m, p) * (p, n)
            public static unsafe void ExecBlock(T* a, T* b, T* c, int m, int p, int n, int aRows, int aCols, int bCols, T* packedB, int length)
            {
                int aIdx, bIdx, cIdx;

                // 8 x <length> blocks
                for (int i = 0; i <= m - 8; i += 8)
                {
                    cIdx = i * bCols;
                    aIdx = i * aCols;
                    for (int j = 0; j <= n - length; j += length, cIdx += length)
                    {
                        bIdx = j * p;
                        if (i == 0) Utils.PackMatrixBWithPxQ(b + j, packedB + bIdx, p, length, bCols);
                        Kernel8xL(a + aIdx, packedB + bIdx, c + cIdx, p, length, aRows, aCols, bCols, length);
                    }
                }
                // The left rows and every <length> columns
                int iStart = m / 8 * 8;
                aIdx = iStart * aCols;
                bool hasPacked = m >= 8 && n >= length;
                for (int i = iStart; i < m; i++, aIdx += aCols)
                {
                    cIdx = i * bCols;
                    for (int j = 0; j <= n - length; j += length, cIdx += length)
                    {
                        if (!hasPacked && i == iStart) Utils.PackMatrixBWithPxQ(b + j, packedB + j * p, p, length, bCols);
                        Kernel1xL(a + aIdx, packedB + j * p, c + cIdx, p, length, aRows, aCols, bCols, length);
                    }
                }

                // The left columns and every 4 rows
                int jStart = n / length * length;
                T c0, c1, c2, c3;
                for (int i = 0; i <= m - 4; i += 4)
                {
                    for (int j = jStart; j < n; j++)
                    {
                        int aIdx0 = i * aCols, aIdx1 = aIdx0 + aCols, aIdx2 = aIdx1 + aCols, aIdx3 = aIdx2 + aCols;
                        cIdx = i * bCols + j;
                        c0 = c[cIdx];
                        cIdx += bCols;
                        c1 = c[cIdx];
                        cIdx += bCols;
                        c2 = c[cIdx];
                        cIdx += bCols;
                        c3 = c[cIdx];

                        for (int k = 0; k < p; k++)
                        {
                            var bValue = b[k * bCols + j];
#if NET7_0_OR_GREATER
                            c0 += a[aIdx0++] * bValue;
                            c1 += a[aIdx1++] * bValue;
                            c2 += a[aIdx2++] * bValue;
                            c3 += a[aIdx3++] * bValue;
#else
                            c0 = _handler.MultiplyAndAdd(a[aIdx0++], bValue, c0);
                            c1 = _handler.MultiplyAndAdd(a[aIdx1++], bValue, c1);
                            c2 = _handler.MultiplyAndAdd(a[aIdx2++], bValue, c2);
                            c3 = _handler.MultiplyAndAdd(a[aIdx3++], bValue, c3);
#endif
                        }
                        cIdx = i * bCols + j;
                        c[cIdx] = c0;
                        cIdx += bCols;
                        c[cIdx] = c1;
                        cIdx += bCols;
                        c[cIdx] = c2;
                        cIdx += bCols;
                        c[cIdx] = c3;
                    }
                }
                // The last 1-3 rows and the left columns
                for (int i = m / 4 * 4; i < m; i++)
                {
                    for (int k = 0; k < p; k++)
                    {
                        cIdx = i * bCols + jStart;
                        bIdx = k * bCols + jStart;
                        var aValue = a[i * aCols + k];
                        for (int j = jStart; j < n; j++)
                        {
#if NET7_0_OR_GREATER
                            c[cIdx++] += aValue * b[bIdx++];
#else
                            var value = c[cIdx];
                            c[cIdx++] = _handler.MultiplyAndAdd(aValue, b[bIdx++], value);
#endif
                        }
                    }
                }
            }
            private static unsafe void Kernel8xL(T* a, T* b, T* c, int p, int n, int aRows, int aCols, int bCols, int length)
            {
                var cBackup = c;

#if NET7_0_OR_GREATER
                var c_0_v = Vector256.Load(c);
                c += bCols;
                var c_1_v = Vector256.Load(c);
                c += bCols;
                var c_2_v = Vector256.Load(c);
                c += bCols;
                var c_3_v = Vector256.Load(c);
                c += bCols;
                var c_4_v = Vector256.Load(c);
                c += bCols;
                var c_5_v = Vector256.Load(c);
                c += bCols;
                var c_6_v = Vector256.Load(c);
                c += bCols;
                var c_7_v = Vector256.Load(c);
                Vector256<T> b_v;
#else
                var c_0_v = new Vector<T>(new Span<T>(c, length));
                c += bCols;
                var c_1_v = new Vector<T>(new Span<T>(c, length));
                c += bCols;
                var c_2_v = new Vector<T>(new Span<T>(c, length));
                c += bCols;
                var c_3_v = new Vector<T>(new Span<T>(c, length));
                c += bCols;
                var c_4_v = new Vector<T>(new Span<T>(c, length));
                c += bCols;
                var c_5_v = new Vector<T>(new Span<T>(c, length));
                c += bCols;
                var c_6_v = new Vector<T>(new Span<T>(c, length));
                c += bCols;
                var c_7_v = new Vector<T>(new Span<T>(c, length));
                Vector<T> b_v;
#endif

                c = cBackup;

                for (int k = 0; k < p; k++)
                {
#if NET7_0_OR_GREATER
                    b_v = Vector256.Load(b);
#else
                    b_v = new Vector<T>(new Span<T>(b, length));
#endif
                    b += n;
                    var curA = a + k;

                    c_0_v += b_v * *curA;
                    curA += aCols;
                    c_1_v += b_v * *curA;
                    curA += aCols;
                    c_2_v += b_v * *curA;
                    curA += aCols;
                    c_3_v += b_v * *curA;
                    curA += aCols;
                    c_4_v += b_v * *curA;
                    curA += aCols;
                    c_5_v += b_v * *curA;
                    curA += aCols;
                    c_6_v += b_v * *curA;
                    curA += aCols;
                    c_7_v += b_v * *curA;
                }

#if NET7_0_OR_GREATER
                c_0_v.Store(c);
                c += bCols;
                c_1_v.Store(c);
                c += bCols;
                c_2_v.Store(c);
                c += bCols;
                c_3_v.Store(c);
                c += bCols;
                c_4_v.Store(c);
                c += bCols;
                c_5_v.Store(c);
                c += bCols;
                c_6_v.Store(c);
                c += bCols;
                c_7_v.Store(c);
#else
                c_0_v.CopyTo(new Span<T>(c, length));
                c += bCols;
                c_1_v.CopyTo(new Span<T>(c, length));
                c += bCols;
                c_2_v.CopyTo(new Span<T>(c, length));
                c += bCols;
                c_3_v.CopyTo(new Span<T>(c, length));
                c += bCols;
                c_4_v.CopyTo(new Span<T>(c, length));
                c += bCols;
                c_5_v.CopyTo(new Span<T>(c, length));
                c += bCols;
                c_6_v.CopyTo(new Span<T>(c, length));
                c += bCols;
                c_7_v.CopyTo(new Span<T>(c, length));
#endif
            }
            private static unsafe void Kernel1xL(T* a, T* b, T* c, int p, int n, int aRows, int aCols, int bCols, int length)
            {
#if NET7_0_OR_GREATER
                var c_0_v = Vector256.Load(c);
                Vector256<T> b_v;
#else
                var spanC = new Span<T>(c, length);
                var c_0_v = new Vector<T>(spanC);
                Vector<T> b_v;
#endif

                for (int k = 0; k < p; k++, b += n)
                {
#if NET7_0_OR_GREATER
                    b_v = Vector256.Load(b);
#else
                    b_v = new Vector<T>(new Span<T>(b, length));
#endif
                    c_0_v += b_v * *a++;
                }

#if NET7_0_OR_GREATER
                c_0_v.Store(c);
#else
                c_0_v.CopyTo(spanC);
#endif
            }
        }
    }
}
