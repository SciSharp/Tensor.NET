using NN.Native.Abstraction.Common;
using NN.Native.Basic;
using NN.Native.Data;
using NN.Native.Abstraction.DType;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NN.Native.Abstraction.Data;

namespace NN.Native.Abstraction.Operators
{
    public interface IReduceOperator<T, TReduceHandler> : IBinaryOperator where T : unmanaged where TReduceHandler : IReduceOpHandler<T>
    {
#if NET7_0_OR_GREATER
        static abstract
#endif
        NativeArray<T> Exec(in NativeArray<T> src, in ReduceParam param, INativeMemoryManager? memoryManager = null);

        public static NativeLayout DeduceLayout(in NativeLayout src, in ReduceParam param)
        {
            int[] shape = new int[src._ndim];
            var span = shape.AsSpan();
            src._shape.AsSpan(0, src._ndim).CopyTo(span);
            for(int i = 0; i < param.Dims.Length; i++)
            {
                span[param.Dims[i]] = 1;
            }
            return new NativeLayout(shape);
        }
    }
    public interface IReduceOpHandler<T> where T: unmanaged
    {
#if NET7_0_OR_GREATER
        static abstract T GetValue(T x);
        static abstract T Apply(T x, T y);
        static abstract T SetValue(T x, T size);
#else
        T GetValue(T x);
        T Apply(T x, T y);
        T SetValue(T x, T size);
#endif
    }

    #region ReduceOpHandler Implementations
#if NET7_0_OR_GREATER
    public class ReduceMaxOpHandler<TData> : IReduceOpHandler<TData> where TData : unmanaged, IComparisonOperators<TData, TData>
    {
        static
#else
    public class ReduceMaxOpHandler<TData, TDataHandler> : IReduceOpHandler<TData> where TData : unmanaged where TDataHandler: INativeDTypeHandler<TData>, new()
    {
        private static TDataHandler _handler = new TDataHandler();
#endif
        public TData GetValue(TData x) => x;

#if NET7_0_OR_GREATER
        public static TData Apply(TData x, TData y)
        {
            return x > y ? x : y;
        }
#else
        public TData Apply(TData x, TData y)
        {
            return _handler.IsGraeter(x, y) ? x : y;
        }
#endif

#if NET7_0_OR_GREATER
        static
#endif
        public TData SetValue(TData x, TData _) => x;
    }

#if NET7_0_OR_GREATER
    public class ReduceMinOpHandler<TData> : IReduceOpHandler<TData> where TData : unmanaged, IComparisonOperators<TData, TData>
    {
        static
#else
    public class ReduceMinOpHandler<TData, TDataHandler> : IReduceOpHandler<TData> where TData : unmanaged where TDataHandler: INativeDTypeHandler<TData>, new()
    {
        private static TDataHandler _handler = new TDataHandler();
#endif
        public TData GetValue(TData x) => x;

#if NET7_0_OR_GREATER
        public static TData Apply(TData x, TData y)
        {
            return x < y ? x : y;
        }
#else
        public TData Apply(TData x, TData y)
        {
            return _handler.IsLess(x, y) ? x : y;
        }
#endif

#if NET7_0_OR_GREATER
        static
#endif
        public TData SetValue(TData x, TData _) => x;
    }

#if NET7_0_OR_GREATER
    public class ReduceMeanOpHandler<TData> : IReduceOpHandler<TData> where TData : unmanaged, IAdditionOperators<TData, TData, TData>, IDivisionOperators<TData, TData, TData>
    {
        static
#else
    public class ReduceMeanOpHandler<TData, TDataHandler> : IReduceOpHandler<TData> where TData : unmanaged where TDataHandler: INativeDTypeHandler<TData>, new()
    {
        private static TDataHandler _handler = new TDataHandler();
#endif
        public TData GetValue(TData x) => x;

#if NET7_0_OR_GREATER
        public static TData Apply(TData x, TData y)
        {
            return x + y;
        }
#else
        public TData Apply(TData x, TData y)
        {
            return _handler.Add(x, y);
        }
#endif

#if NET7_0_OR_GREATER
        public static TData SetValue(TData x, TData size) => x / size;
#else
        public TData SetValue(TData x, TData size) => _handler.Divide(x, size);
#endif
    }

#if NET7_0_OR_GREATER
    public class ReduceSumOpHandler<TData> : IReduceOpHandler<TData> where TData : unmanaged, IAdditionOperators<TData, TData, TData>
    {
        static
#else
    public class ReduceSumOpHandler<TData, TDataHandler> : IReduceOpHandler<TData> where TData : unmanaged where TDataHandler: INativeDTypeHandler<TData>, new()
    {
        private static TDataHandler _handler = new TDataHandler();
#endif
        public TData GetValue(TData x) => x;

#if NET7_0_OR_GREATER
        public static TData Apply(TData x, TData y)
        {
            return x + y;
        }
#else
        public TData Apply(TData x, TData y)
        {
            return _handler.Add(x, y);
        }
#endif

#if NET7_0_OR_GREATER
        static
#endif
        public TData SetValue(TData x, TData _) => x;
    }

    #endregion
}
