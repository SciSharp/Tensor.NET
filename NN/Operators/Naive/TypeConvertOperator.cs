using NN.Native.Abstraction.DType;
using NN.Native.Abstraction.Operators;
using NN.Native.Basic;
using NN.Native.Extensions;
using NN.Native.Operators.Common;
using NN.Native.Operators.Common.Params;

namespace NN.Native.Operators.Naive
{
    public class TypeConvertOperator<TA, TB, THandler>: TypeConvertOperatorBase where TA: unmanaged where TB: unmanaged where THandler: INativeConvertible<TA, TB>
    {
        public static bool IsThreadSafe { get => false; }
        public static OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
        public static unsafe void Exec(ReadOnlySpan<TA> a, Span<TB> b, in NativeLayout layoutA, in NativeLayout layoutB, in TypeConvertParam param)
        {
            int totalElems = layoutA.TotalElemCount();
            if (!param.Transpose)
            {
                var indexEnumeratorA = NativeLayout.GetIndexEnumerator(layoutA);
                var indexEnumeratorB = NativeLayout.GetIndexEnumerator(layoutB);
                for (int i = 0; i < totalElems; i++)
                {
                    THandler.Convert(a[indexEnumeratorA.MoveNext()], ref b[indexEnumeratorB.MoveNext()]);
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
                    THandler.Convert(a[offset], ref b[layoutB.IndicesToOffset(indices)]);
                }
            }
        }
    }
}
