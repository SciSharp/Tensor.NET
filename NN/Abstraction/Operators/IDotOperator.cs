using NN.Native.Abstraction.Data;
using NN.Native.Basic;
using NN.Native.Data;
using NN.Native.Exceptions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.Operators
{
    public unsafe interface IDotOperator<T> : ITernaryOperator where T : unmanaged
    {
#if NET7_0_OR_GREATER
        static abstract
#endif
        NativeArray<T> Exec(in NativeArray<T> a, in NativeArray<T> b, INativeMemoryManager? memoryManager = null);

        internal static NativeLayout DeduceLayout(ref NativeLayout a, ref NativeLayout b)
        {
            //TODO: broadcast
            if (!a.IsSameShape(b))
            {
                throw new InvalidShapeException();
            }
            return NativeLayout.ShapeLike(a);
        }
    }
}
