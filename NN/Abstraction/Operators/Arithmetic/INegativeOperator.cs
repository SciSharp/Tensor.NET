using NN.Native.Abstraction.Data;
using NN.Native.Basic;
using NN.Native.Data;
using NN.Native.Exceptions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.Operators.Arithmetic
{
    public interface INegativeOperator<T> : ITernaryOperator where T : unmanaged
    {
#if NET7_0_OR_GREATER
        static abstract
#endif
        NativeArray<T> Exec(in NativeArray<T> src, INativeMemoryManager? memoryManager = null);

        public static NativeLayout DeduceLayout(in NativeLayout src)
        {
            return NativeLayout.ShapeLike(src);
        }
    }
}
