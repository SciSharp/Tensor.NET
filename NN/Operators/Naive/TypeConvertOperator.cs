using NN.Native.Abstraction.Common;
using NN.Native.Abstraction.Data;
using NN.Native.Abstraction.DType;
using NN.Native.Abstraction.Operators;
using NN.Native.Basic;
using NN.Native.Data;
using NN.Native.Extensions;

namespace NN.Native.Operators.Naive
{
    public class TypeConvertOperator<TA, TB, THandler>: ITypeConvertOperator<TA, TB> where TA: unmanaged where TB: unmanaged where THandler: INativeConvertible<TA, TB>, new()
    {
#if NET7_0_OR_GREATER
        public static bool IsThreadSafe { get => false; }
        public static bool RequireContiguousArray { get => false; }
        public static OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#else
        private static THandler _handler = new();
        public bool IsThreadSafe { get => false; }
        public bool RequireContiguousArray { get => false; }
        public OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#endif
#if NET7_0_OR_GREATER
        static
#endif
        public NativeArray<TB> Exec(in NativeArray<TA> a, in TypeConvertParam param, INativeMemoryManager? memoryManager = null)
        {
            var res = new NativeArray<TB>(ITypeConvertOperator<TA, TB>.DeduceLayout(a._layout, param), memoryManager);
            ExecInternal(a.Span, res.Span, a._layout, res._layout, param);
            return res;
        }
        public static unsafe void ExecInternal(ReadOnlySpan<TA> a, Span<TB> b, in NativeLayout layoutA, in NativeLayout layoutB, in TypeConvertParam param)
        {
            int totalElems = layoutA.TotalElemCount();
            if (!param.Transpose)
            {
                var indexEnumeratorA = NativeLayout.GetIndexEnumerator(layoutA);
                var indexEnumeratorB = NativeLayout.GetIndexEnumerator(layoutB);
                for (int i = 0; i < totalElems; i++)
                {
#if NET7_0_OR_GREATER
                    b[indexEnumeratorB.MoveNext()] = THandler.Convert(a[indexEnumeratorA.MoveNext()]);
#else
                    b[indexEnumeratorB.MoveNext()] = _handler.Convert(a[indexEnumeratorA.MoveNext()]);
#endif
                }
            }
            else
            {
                int* indices = stackalloc int[layoutB._ndim];
                for(int offset = 0; offset < totalElems; offset++)
                {
                    layoutA.OffsetToIndices(indices, offset);
                    indices[param.DimA] ^= indices[param.DimB];
                    indices[param.DimB] ^= indices[param.DimA];
                    indices[param.DimA] ^= indices[param.DimB];
#if NET7_0_OR_GREATER
                    b[layoutB.IndicesToOffset(indices)] = THandler.Convert(a[offset]);
#else
                    b[layoutB.IndicesToOffset(indices)] = _handler.Convert(a[offset]);
#endif
                }
            }
        }
    }
}
