using NN.Native.Abstraction.Data;
using NN.Native.Basic;
using NN.Native.Data;
using NN.Native.Exceptions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.Operators
{
    public interface ITernaryElemWiseOperator<TA, TB, TC>: ITernaryOperator where TA: unmanaged where TB: unmanaged where TC : unmanaged
    {
#if NET7_0_OR_GREATER
        static abstract
#endif
        NativeArray<TC> Exec(in NativeArray<TA> a, in NativeArray<TB> b, Func<TA, TB, int, TC> elemFunc, INativeMemoryManager? memoryManager = null);
#if NET7_0_OR_GREATER
        static abstract
#endif
        NativeArray<TC> Exec(in NativeArray<TA> a, TB b, Func<TA, TB, int, TC> elemFunc, INativeMemoryManager? memoryManager = null);
#if NET7_0_OR_GREATER
        static abstract
#endif
        NativeArray<TC> Exec(TA a, in NativeArray<TB> b, Func<TA, TB, int, TC> elemFunc, INativeMemoryManager? memoryManager = null);

        public static NativeLayout DeduceLayout(in NativeLayout a, in NativeLayout b)
        {
            if (!a.IsSameShape(b))
            {
                throw new InvalidShapeException();
            }
            return NativeLayout.ShapeLike(a);
        }
        public static NativeLayout DeduceLayout(in NativeLayout a)
        {
            return NativeLayout.ShapeLike(a);
        }
    }
}
