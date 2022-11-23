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
    public class UnaryElemWiseOperator<T> : IUnaryElemWiseOperator<T> where T : unmanaged
    {
        public static bool IsThreadSafe { get => false; }
        public static bool RequireContiguousArray { get => false; }
        public static OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#else
    public class UnaryElemWiseOperator<T>: IUnaryElemWiseOperator<T> where T : unmanaged
    {
        public bool IsThreadSafe { get => false; }
        public bool RequireContiguousArray { get => false; }
        public OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#endif
#if NET7_0_OR_GREATER
        static
#endif
        public void Exec(ref NativeArray<T> src, Func<T, int, T> elemFunc)
        {
            ExecInternal(src.Span, src._layout, elemFunc);
        }

        static private unsafe void ExecInternal(Span<T> src, in NativeLayout layoutSrc, Func<T, int, T> elemFunc)
        {
            Debug.Assert(!layoutSrc.IsEmpty());
            int totalElems = layoutSrc.TotalElemCount();
            var enumratorA = NativeLayout.GetIndexEnumerator(layoutSrc);
            for (int i = 0; i < totalElems; i++)
            {
                var idx = enumratorA.MoveNext();
                src[idx] = elemFunc(src[idx], i);
            }
        }
    }
}
