using NN.Native.Data;
using NN.Native.Abstraction.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NN.Native.Abstraction.Data;
using NN.Native.Basic;

namespace NN.Native.Abstraction.Operators
{
    public interface IBinaryElemWiseOperator<TA, TB>: IBinaryOperator where TA: unmanaged where TB : unmanaged
    {
#if NET7_0_OR_GREATER
        static abstract
#endif
        NativeArray<TB> Exec(in NativeArray<TA> src, Func<TA, int, TB> elemFunc, INativeMemoryManager? memoryManager = null);

        internal static NativeLayout DeduceLayout(in NativeLayout layoutSrc)
        {
            return NativeLayout.ShapeLike(layoutSrc);
        }
    }
}
