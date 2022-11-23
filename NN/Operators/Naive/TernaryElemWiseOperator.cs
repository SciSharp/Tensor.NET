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
    public class TernaryElemWiseOperator<TA, TB, TC> : ITernaryElemWiseOperator<TA, TB, TC> where TA : unmanaged where TB : unmanaged where TC : unmanaged
    {
        public static bool IsThreadSafe { get => false; }
        public static bool RequireContiguousArray { get => false; }
        public static OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#else
    public class TernaryElemWiseOperator<TA, TB, TC>: ITernaryElemWiseOperator<TA, TB, TC> where TA : unmanaged where TB: unmanaged where TC : unmanaged
    {
        public bool IsThreadSafe { get => false; }
        public bool RequireContiguousArray { get => false; }
        public OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#endif
#if NET7_0_OR_GREATER
        static
#endif
        public NativeArray<TC> Exec(in NativeArray<TA> a, in NativeArray<TB> b, Func<TA, TB, int, TC> elemFunc, INativeMemoryManager? memoryManager = null)
        {
            var res = new NativeArray<TC>(ITernaryElemWiseOperator<TA, TB, TC>.DeduceLayout(a._layout, b._layout), memoryManager);
            ExecInternal(a.Span, b.Span, res.Span, a._layout, b._layout, res._layout, elemFunc);
            return res;
        }

#if NET7_0_OR_GREATER
        static
#endif
        public NativeArray<TC> Exec(in NativeArray<TA> a, TB b, Func<TA, TB, int, TC> elemFunc, INativeMemoryManager? memoryManager = null)
        {
            var res = new NativeArray<TC>(ITernaryElemWiseOperator<TA, TB, TC>.DeduceLayout(a._layout), memoryManager);
            ExecInternal(a.Span, b, res.Span, a._layout, res._layout, elemFunc);
            return res;
        }

#if NET7_0_OR_GREATER
        static
#endif
        public NativeArray<TC> Exec(TA a, in NativeArray<TB> b, Func<TA, TB, int, TC> elemFunc, INativeMemoryManager? memoryManager = null)
        {
            var res = new NativeArray<TC>(ITernaryElemWiseOperator<TA, TB, TC>.DeduceLayout(b._layout), memoryManager);
            ExecInternal(a, b.Span, res.Span, b._layout, res._layout, elemFunc);
            return res;
        }

        static private unsafe void ExecInternal(ReadOnlySpan<TA> a, ReadOnlySpan<TB> b, Span<TC> dst, in NativeLayout layoutA, in NativeLayout layoutB, in NativeLayout layoutC, Func<TA, TB, int, TC> elemFunc)
        {
            Debug.Assert(layoutA.IsSameShape(layoutB));
            Debug.Assert(layoutA.IsSameShape(layoutC));
            int totalElems = layoutC.TotalElemCount();
            var enumeratorA = NativeLayout.GetIndexEnumerator(layoutA);
            var enumeratorB = NativeLayout.GetIndexEnumerator(layoutB);
            for (int i = 0; i < totalElems; i++)
            {
                dst[i] = elemFunc(a[enumeratorA.MoveNext()], b[enumeratorB.MoveNext()], i);
            }
        }

        static private unsafe void ExecInternal(ReadOnlySpan<TA> a, TB b, Span<TC> dst, in NativeLayout layoutA, in NativeLayout layoutC, Func<TA, TB, int, TC> elemFunc)
        {
            Debug.Assert(layoutA.IsSameShape(layoutC));
            int totalElems = layoutC.TotalElemCount();
            var enumeratorA = NativeLayout.GetIndexEnumerator(layoutA);
            for (int i = 0; i < totalElems; i++)
            {
                dst[i] = elemFunc(a[enumeratorA.MoveNext()], b, i);
            }
        }

        static private unsafe void ExecInternal(TA a, ReadOnlySpan<TB> b, Span<TC> dst, in NativeLayout layoutB, in NativeLayout layoutC, Func<TA, TB, int, TC> elemFunc)
        {
            Debug.Assert(layoutB.IsSameShape(layoutC));
            int totalElems = layoutC.TotalElemCount();
            var enumeratorB = NativeLayout.GetIndexEnumerator(layoutB);
            for (int i = 0; i < totalElems; i++)
            {
                dst[i] = elemFunc(a, b[enumeratorB.MoveNext()], i);
            }
        }
    }
}
