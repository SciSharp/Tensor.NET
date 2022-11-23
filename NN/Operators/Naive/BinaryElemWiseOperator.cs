using NN.Native.Abstraction.Operators;
using NN.Native.Basic;
using NN.Native.Data;
using NN.Native.Abstraction.Common;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NN.Native.Abstraction.Data;

namespace NN.Native.Operators.Naive
{
#if NET7_0_OR_GREATER
    public class BinaryElemWiseOperator<TA, TB> : IBinaryElemWiseOperator<TA, TB> where TA : unmanaged where TB: unmanaged
    {
        public static bool IsThreadSafe { get => false; }
        public static bool RequireContiguousArray { get => false; }
        public static OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#else
    public class BinaryElemWiseOperator<TA, TB>: IBinaryElemWiseOperator<TA, TB> where TA : unmanaged where TB: unmanaged
    {
        public bool IsThreadSafe { get => false; }
        public bool RequireContiguousArray { get => false; }
        public OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#endif
#if NET7_0_OR_GREATER
        static
#endif
        public NativeArray<TB> Exec(in NativeArray<TA> src, Func<TA, int, TB> elemFunc, INativeMemoryManager? memoryManager = null)
        {
            var res = new NativeArray<TB>(IBinaryElemWiseOperator<TA, TB>.DeduceLayout(src._layout), memoryManager);
            ExecInternal(src.Span, res.Span, src._layout, res._layout, elemFunc); 
            return res;
        }

        private static void ExecInternal(ReadOnlySpan<TA> src, Span<TB> dst, in NativeLayout layoutSrc, in NativeLayout layoutDst, Func<TA, int, TB> elemFunc)
        {
            Debug.Assert(layoutSrc.IsSameShape(layoutDst));
            int totalElems = layoutDst.TotalElemCount();
            var enumratorA = NativeLayout.GetIndexEnumerator(layoutSrc);
            for (int i = 0; i < totalElems; i++)
            {
                dst[i] = elemFunc(src[enumratorA.MoveNext()], i);
            }
        }
    }
}
