using NN.Native.Abstraction.Operators;
using NN.Native.Abstraction.Operators.Arithmetic;
using NN.Native.Basic;
using NN.Native.Data;
using NN.Native.Abstraction.DType;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;
using NN.Native.Abstraction.Data;

namespace NN.Native.Operators.X86.Arithmetic
{
#if NET7_0_OR_GREATER
    public class MultiplyOperator<T> : IMultiplyOperator<T> where T : unmanaged, IAdditionOperators<T, T, T>, INumberBase<T>
    {
        public static bool IsThreadSafe { get => false; }
        public static bool RequireContiguousArray { get => true; }
        public static OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#else
    public class MultiplyOperator<T, THandler>: IMultiplyOperator<T> where T : unmanaged where THandler: INativeDTypeHandler<T>, new()
    {
        private static THandler _handler = new();
        public bool IsThreadSafe { get => false; }
        public bool RequireContiguousArray { get => true; }
        public OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#endif

#if NET7_0_OR_GREATER
        static
#endif
        public NativeArray<T> Exec(in NativeArray<T> a, in NativeArray<T> b, INativeMemoryManager? memoryManager = null)
        {
            var res = new NativeArray<T>(IMultiplyOperator<T>.DeduceLayout(a._layout, b._layout), memoryManager);
            ExecInternal(a.Span, b.Span, res.Span, a._layout, b._layout, res._layout);
            return res;
        }
#if NET7_0_OR_GREATER
        static
#endif
        public NativeArray<T> Exec(in NativeArray<T> a, T b, INativeMemoryManager? memoryManager = null)
        {
            var res = new NativeArray<T>(IMultiplyOperator<T>.DeduceLayout(a._layout), memoryManager);
            ExecInternal(a.Span, b, res.Span, a._layout, res._layout);
            return res;
        }
        private static unsafe void ExecInternal(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> c, in NativeLayout layoutA, in NativeLayout layoutB, in NativeLayout layoutC)
        {
            Debug.Assert(layoutA.IsSameShape(layoutB));
            Debug.Assert(layoutA.IsSameShape(layoutC));
            int length = Vector<T>.Count;
            int totalElems = layoutC.TotalElemCount();
            int groupCount = totalElems / length;
            MicroKernel8xN(a, b, c, groupCount, length);
            for (int i = groupCount * length; i < totalElems; i++)
            {
#if NET7_0_OR_GREATER
                c[i] = a[i] * b[i];
#else
                c[i] = _handler.Multiply(a[i], b[i]);
#endif
            }
        }

        private static unsafe void ExecInternal(ReadOnlySpan<T> a, T b, Span<T> c, in NativeLayout layoutA, in NativeLayout layoutC)
        {
            Debug.Assert(layoutA.IsSameShape(layoutC));
            int length = Vector<T>.Count;
            int totalElems = layoutC.TotalElemCount();
            int groupCount = totalElems / length;
            MicroKernel8xN(a, b, c, groupCount, length);
            for (int i = groupCount * length; i < totalElems; i++)
            {
#if NET7_0_OR_GREATER
                c[i] = a[i] * b;
#else
                c[i] = _handler.Multiply(a[i], b);
#endif
            }
        }

        private static unsafe void MicroKernel8xN(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> c, int loops, int length)
        {
            Vector<T> a_v, b_v, c_v;
            int offset = 0;
            for (int i = 0; i < loops; i++, offset += length)
            {
                a_v = new Vector<T>(a.Slice(offset));
                b_v = new Vector<T>(b.Slice(offset));
                c_v = Vector.Multiply(a_v, b_v);
                c_v.CopyTo(c.Slice(offset));
            }
        }

        private static unsafe void MicroKernel8xN(ReadOnlySpan<T> a, T b, Span<T> c, int loops, int length)
        {
            Vector<T> a_v, b_v, c_v;
            b_v = new Vector<T>(b);
            int offset = 0;
            for (int i = 0; i < loops; i++, offset += length)
            {
                a_v = new Vector<T>(a.Slice(offset));
                c_v = Vector.Multiply(a_v, b_v);
                c_v.CopyTo(c.Slice(offset));
            }
        }
    }
}