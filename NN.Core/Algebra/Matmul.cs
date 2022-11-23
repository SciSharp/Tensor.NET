using NN.Native.Abstraction.Common;
using NN.Native.Basic;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using NN.Native.Data;
using NN.Native.Abstraction.DType;
using NN.Native.Abstraction.Operators;
using NN.Native.Operators.X86;
using NN.Native.Abstraction.Data;

namespace NN.Core.Algebra
{
#if NET7_0_OR_GREATER
    internal class Matmul<T> where T : unmanaged, INumber<T>
#else
    internal class Matmul<T, THandler> where T: unmanaged where THandler: INativeDTypeHandler<T>, new()
#endif
    {
        internal static NDArray<T> Exec(NDArray<T> a, NDArray<T> b, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = MatmulOperator<T>.Exec(a._nativeArray, b._nativeArray, memoryManager);
#else
            var res = new MatmulOperator<T, THandler>().Exec(a._nativeArray, b._nativeArray, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }
    }
}
