using NN.Native.Abstraction.Operators;
using NN.Native.Basic;
using NN.Native.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection.Metadata;
using System.Text;
using System.Threading.Tasks;
using NN.Native.Extensions;
using System.Diagnostics;
using NN.Native.Abstraction.DType;
using NN.Native.Abstraction.Common;
using NN.Native.Abstraction.Data;

namespace NN.Native.Operators.Naive
{
#if NET7_0_OR_GREATER
    public class TransposeOperator<T> : ITransposeOperator<T> where T : unmanaged, INumberBase<T>
    {
        public static bool IsThreadSafe { get => false; }
        public static bool RequireContiguousArray { get => false; }
        public static OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#else
    public class TransposeOperator<T> : ITransposeOperator<T> where T : unmanaged
    {
        public bool IsThreadSafe { get => false; }
        public bool RequireContiguousArray { get => false; }
        public OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#endif
#if NET7_0_OR_GREATER
        static
#endif
        public NativeArray<T> Exec(in NativeArray<T> src, in TransposeParam param, INativeMemoryManager? memoryManager = null)
        {
            var res = new NativeArray<T>(ITransposeOperator<T>.DeduceLayout(src._layout, param), memoryManager);
            ExecInternal(src.Span, res.Span, src._layout, res._layout, param);
            return res;
        }

        public static unsafe void ExecInternal(ReadOnlySpan<T> src, Span<T> dst, in NativeLayout layoutSrc, in NativeLayout layoutDst, in TransposeParam param)
        {
            Debug.Assert(layoutSrc._ndim == layoutDst._ndim);
            var indices = stackalloc int[layoutSrc._ndim];
            int totalElems = layoutSrc.TotalElemCount();
            for (int offset = 0; offset < totalElems; offset++)
            {
                layoutSrc.OffsetToIndices(indices, offset);
                indices[param.DimA] ^= indices[param.DimB];
                indices[param.DimB] ^= indices[param.DimA];
                indices[param.DimA] ^= indices[param.DimB];
                dst[layoutDst.IndicesToOffset(indices)] = src[offset];
            }
        }
    }
}
