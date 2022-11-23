using NN.Native.Abstraction.Operators;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NN.Native.Operators.Naive;

namespace NN.Core.Statistics
{
#if NET7_0_OR_GREATER
    internal class UnaryElemWise<T> where T : unmanaged, INumber<T>
#else
    internal class UnaryElemWise<T> where T : unmanaged
#endif
    {
        internal static void Exec(NDArray<T> src, Func<T, int, T> elemFunc, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            UnaryElemWiseOperator<T>.Exec(ref src._nativeArray, elemFunc);
#else
            new UnaryElemWiseOperator<T>().Exec(ref src._nativeArray, elemFunc);
#endif
        }
    }

#if NET7_0_OR_GREATER
    internal class BinaryElemWise<TA, TB> where TA : unmanaged, INumber<TA> where TB : unmanaged, INumber<TB>
#else
    internal class BinaryElemWise<TA, TB> where TA : unmanaged where TB : unmanaged
#endif
    {
        internal static NDArray<TB> Exec(NDArray<TA> src, Func<TA, int, TB> elemFunc, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = BinaryElemWiseOperator<TA, TB>.Exec(src._nativeArray, elemFunc);
#else
            var res = new BinaryElemWiseOperator<TA, TB>().Exec(src._nativeArray, elemFunc);
#endif
            return NDArray<TB>.FromNativeArray(res);
        }
    }

#if NET7_0_OR_GREATER
    internal class TernaryElemWise<TA, TB, TC> where TA : unmanaged, INumber<TA> where TB : unmanaged, INumber<TB> where TC : unmanaged, INumber<TC>
#else
    internal class TernaryElemWise<TA, TB, TC> where TA : unmanaged where TB : unmanaged where TC : unmanaged
#endif
    {
        internal static NDArray<TC> Exec(NDArray<TA> a, NDArray<TB> b, Func<TA, TB, int, TC> elemFunc, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = TernaryElemWiseOperator<TA, TB, TC>.Exec(a._nativeArray, b._nativeArray, elemFunc);
#else
            var res = new TernaryElemWiseOperator<TA, TB, TC>().Exec(a._nativeArray, b._nativeArray, elemFunc);
#endif
            return NDArray<TC>.FromNativeArray(res);
        }
    }
}
