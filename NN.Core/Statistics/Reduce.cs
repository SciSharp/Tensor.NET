using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NN.Native.Operators.Naive;
using NN.Native.Abstraction.Operators;
using NN.Native.Abstraction.Data;
using NN.Native.Abstraction.Common;
using NN.Native.Abstraction.DType;

namespace NN.Core.Statistics
{
#if NET7_0_OR_GREATER
    internal class Reduce<T> where T : unmanaged, INumber<T>
#else
    internal class Reduce<T, THandler> where T: unmanaged where THandler: INativeDTypeHandler<T>, new()
#endif
    {
        internal static NDArray<T> Sum(NDArray<T> src, in ReduceParam param, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = ReduceOperator<T, ReduceSumOpHandler<T>>.Exec(src._nativeArray, param, memoryManager);
#else
            var res = new ReduceOperator<T, ReduceSumOpHandler<T, THandler>>().Exec(src._nativeArray, param, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }

        internal static NDArray<T> Mean(NDArray<T> src, in ReduceParam param, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = ReduceOperator<T, ReduceMeanOpHandler<T>>.Exec(src._nativeArray, param, memoryManager);
#else
            var res = new ReduceOperator<T, ReduceMeanOpHandler<T, THandler>>().Exec(src._nativeArray, param, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }

        internal static NDArray<T> Max(NDArray<T> src, in ReduceParam param, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = ReduceOperator<T, ReduceMaxOpHandler<T>>.Exec(src._nativeArray, param, memoryManager);
#else
            var res = new ReduceOperator<T, ReduceMaxOpHandler<T, THandler>>().Exec(src._nativeArray, param, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }

        internal static NDArray<T> Min(NDArray<T> src, in ReduceParam param, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = ReduceOperator<T, ReduceMinOpHandler<T>>.Exec(src._nativeArray, param, memoryManager);
#else
            var res = new ReduceOperator<T, ReduceMinOpHandler<T, THandler>>().Exec(src._nativeArray, param, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }
    }
}
