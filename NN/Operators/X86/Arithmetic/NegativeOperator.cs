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
using System.Buffers;

namespace NN.Native.Operators.X86.Arithmetic
{
#if NET7_0_OR_GREATER
    public class NegativeOperator<T>: INegativeOperator<T> where T : unmanaged, IAdditionOperators<T, T, T>, INumberBase<T>
    {
        public static bool IsThreadSafe { get => false; }
        public static bool RequireContiguousArray { get => true; }
        public static OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#else
    public class NegativeOperator<T, THandler>: INegativeOperator<T> where T : unmanaged where THandler: INativeDTypeHandler<T>, new()
    {
        private static THandler _handler = new();
        public bool IsThreadSafe { get => false; }
        public bool RequireContiguousArray { get => true; }
        public OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#endif

#if NET7_0_OR_GREATER
        static
#endif
        public NativeArray<T> Exec(in NativeArray<T> src, INativeMemoryManager? memoryManager = null)
        {
            var res = new NativeArray<T>(IAddOperator<T>.DeduceLayout(src._layout), memoryManager);
            ExecInternal(src.Span, res.Span, src._layout, res._layout);
            return res;
        }

        private static unsafe void ExecInternal(ReadOnlySpan<T> src, Span<T> dst, in NativeLayout layoutSrc, in NativeLayout layoutDst)
        {
            Debug.Assert(layoutSrc.IsSameShape(layoutDst));
            Debug.Assert(layoutSrc.IsSameShape(layoutDst));
            int length = Vector<T>.Count;
            int totalElems = layoutDst.TotalElemCount();
            int groupCount = totalElems / length;
            MicroKernel8xN(src, dst, groupCount, length);
            for (int i = groupCount * length; i < totalElems; i++)
            {
#if NET7_0_OR_GREATER
                dst[i] = -src[i];
#else
                dst[i] = _handler.GetNegative(src[i]);
#endif
            }
        }


        private static unsafe void MicroKernel8xN(ReadOnlySpan<T> src, Span<T> dst, int loops, int length)
        {
            Vector<T> src_v, dst_v;
            int offset = 0;
            for (int i = 0; i < loops; i++, offset += length)
            {
                src_v = new Vector<T>(src.Slice(offset));
                dst_v = -src_v;
                dst_v.CopyTo(dst.Slice(offset));
            }
        }
    }
}