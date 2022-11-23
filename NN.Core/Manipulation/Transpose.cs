using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NN.Native.Abstraction.Common;
using NN.Native.Abstraction.Data;
using NN.Native.Abstraction.Operators;
using NN.Native.Operators.Naive;

namespace NN.Core.Manipulation
{
#if NET7_0_OR_GREATER
    internal class Transpose<T> where T : unmanaged, INumber<T>
#else
    internal class Transpose<T> where T: unmanaged
#endif
    {
        internal static NDArray<T> Exec(NDArray<T> src, in TransposeParam param, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = TransposeOperator<T>.Exec(src._nativeArray, param, memoryManager);
#else
            var res = new TransposeOperator<T>().Exec(src._nativeArray, param, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }
    }
}
