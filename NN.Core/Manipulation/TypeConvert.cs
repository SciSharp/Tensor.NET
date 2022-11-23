using NN.Native.Abstraction.DType;
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
    internal class TypeConvert<TA, TB, THandler> where TA : unmanaged, INumber<TA> where TB : unmanaged, INumber<TB> where THandler : INativeConvertible<TA, TB>, new()
#else
    internal class TypeConvert<TA, TB, THandler> where TA : unmanaged where TB : unmanaged where THandler : INativeConvertible<TA, TB>, new()
#endif
    {
        internal static NDArray<TB> Exec(NDArray<TA> src, in TypeConvertParam param, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = TypeConvertOperator<TA, TB, THandler>.Exec(src._nativeArray, param);
#else
            var res = new TypeConvertOperator<TA, TB, THandler>().Exec(src._nativeArray, param);
#endif
            return NDArray<TB>.FromNativeArray(res);
        }
    }
}
