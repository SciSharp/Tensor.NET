using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NN.Native.Abstraction.Data;
using NN.Native.Abstraction.Operators;
using NN.Native.Operators.X86;
using NN.Native.Operators.X86.Arithmetic;
using NN.Native.Abstraction.DType;

namespace NN.Core.Arithmetic
{
#if NET7_0_OR_GREATER
    internal class BasicArithmetic<T> where T : unmanaged, INumber<T>
#else
    internal class BasicArithmetic<T, THandler> where T: unmanaged where THandler: INativeDTypeHandler<T>, new()
#endif
    {
        internal static NDArray<T> Add(NDArray<T> a, NDArray<T> b, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = AddOperator<T>.Exec(a._nativeArray, b._nativeArray, memoryManager);
#else
            var res = new AddOperator<T, THandler>().Exec(a._nativeArray, b._nativeArray, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }
        internal static NDArray<T> Add(NDArray<T> a, T b, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = AddOperator<T>.Exec(a._nativeArray, b, memoryManager);
#else
            var res = new AddOperator<T, THandler>().Exec(a._nativeArray, b, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }
        internal static NDArray<T> Add(T a, NDArray<T> b, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = AddOperator<T>.Exec(b._nativeArray, a, memoryManager);
#else
            var res = new AddOperator<T, THandler>().Exec(b._nativeArray, a, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }


        internal static NDArray<T> Subtract(NDArray<T> a, NDArray<T> b, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = SubtractOperator<T>.Exec(a._nativeArray, b._nativeArray, memoryManager);
#else
            var res = new SubtractOperator<T, THandler>().Exec(a._nativeArray, b._nativeArray, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }
        internal static NDArray<T> Subtract(NDArray<T> a, T b, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = SubtractOperator<T>.Exec(a._nativeArray, b, memoryManager);
#else
            var res = new SubtractOperator<T, THandler>().Exec(a._nativeArray, b, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }
        internal static NDArray<T> Subtract(T a, NDArray<T> b, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = SubtractOperator<T>.Exec(a, b._nativeArray, memoryManager);
#else
            var res = new SubtractOperator<T, THandler>().Exec(a, b._nativeArray, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }

        internal static NDArray<T> Multiply(NDArray<T> a, NDArray<T> b, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = MultiplyOperator<T>.Exec(a._nativeArray, b._nativeArray, memoryManager);
#else
            var res = new MultiplyOperator<T, THandler>().Exec(a._nativeArray, b._nativeArray, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }
        internal static NDArray<T> Multiply(NDArray<T> a, T b, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = MultiplyOperator<T>.Exec(a._nativeArray, b, memoryManager);
#else
            var res = new MultiplyOperator<T, THandler>().Exec(a._nativeArray, b, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }
        internal static NDArray<T> Multiply(T a, NDArray<T> b, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = MultiplyOperator<T>.Exec(b._nativeArray, a, memoryManager);
#else
            var res = new MultiplyOperator<T, THandler>().Exec(b._nativeArray, a, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }

        internal static NDArray<T> Divide(NDArray<T> a, NDArray<T> b, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = DivideOperator<T>.Exec(a._nativeArray, b._nativeArray, memoryManager);
#else
            var res = new DivideOperator<T, THandler>().Exec(a._nativeArray, b._nativeArray, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }
        internal static NDArray<T> Divide(NDArray<T> a, T b, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = DivideOperator<T>.Exec(a._nativeArray, b, memoryManager);
#else
            var res = new DivideOperator<T, THandler>().Exec(a._nativeArray, b, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }
        internal static NDArray<T> Divide(T a, NDArray<T> b, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = DivideOperator<T>.Exec(a, b._nativeArray, memoryManager);
#else
            var res = new DivideOperator<T, THandler>().Exec(a, b._nativeArray, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }

        internal static NDArray<T> Negative(NDArray<T> src, INativeMemoryManager? memoryManager = null, OperatorHandlerType handlerType = OperatorHandlerType.Default)
        {
#if NET7_0_OR_GREATER
            var res = NegativeOperator<T>.Exec(src._nativeArray, memoryManager);
#else
            var res = new NegativeOperator<T, THandler>().Exec(src._nativeArray, memoryManager);
#endif
            return NDArray<T>.FromNativeArray(res);
        }
    }
}
