using NN.Native.Abstraction.Common;
using NN.Native.Abstraction.Data;
using NN.Native.Abstraction.Operators;
using NN.Native.Basic;
using NN.Native.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Operators.Naive
{
    public class ReduceOperator<T, TReduceHandler>: IReduceOperator<T, TReduceHandler> where T : unmanaged where TReduceHandler : IReduceOpHandler<T>, new()
    {
#if NET7_0_OR_GREATER
        public static bool IsThreadSafe { get => false; }
        public static bool RequireContiguousArray { get => false; }
        public static OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#else
        private static TReduceHandler _reduceHandler = new();
        public bool IsThreadSafe { get => false; }
        public bool RequireContiguousArray { get => false; }
        public OperatorHandlerType HandlerType { get => OperatorHandlerType.Naive; }
#endif

#if NET7_0_OR_GREATER
        static
#endif
        public NativeArray<T> Exec(in NativeArray<T> src, in ReduceParam param, INativeMemoryManager? memoryManager = null)
        {
            var res = new NativeArray<T>(IReduceOperator<T, TReduceHandler>.DeduceLayout(src._layout, param), memoryManager);
            var reducedLayout = new NativeLayout();
            var keepedLayout = new NativeLayout();
            bool[] reducedDims = new bool[src._layout._ndim];

            Debug.Assert(param.Dims.Length <= src._layout._ndim);
            for (int i = 0; i < param.Dims.Length; i++)
            {
                Debug.Assert(param.Dims[i] < src._layout._ndim);
                reducedDims[param.Dims[i]] = true;
            }
            for(int i = 0; i < reducedDims.Length; i++)
            {
                if (reducedDims[i])
                {
                    reducedLayout.AddAxisInplace(reducedLayout._ndim, src._layout._shape[i], src._layout._stride[i]);
                }
                else
                {
                    keepedLayout.AddAxisInplace(keepedLayout._ndim, src._layout._shape[i], src._layout._stride[i]);
                }
            }

            int reducedTotalElems = reducedLayout.TotalElemCount();
            int keepedTotalElems = keepedLayout.TotalElemCount();
            var enumeratorKeeped = NativeLayout.GetIndexEnumerator(keepedLayout);
            var enumeratorReduced = NativeLayout.GetIndexEnumerator(reducedLayout);
            var srcSpan = src.Span;
            var dstSpan = res.Span;
            T size = ConvertIntToT(reducedTotalElems);
            if (reducedTotalElems == 0) return res;
            for(int i = 0, offset = 0; i < keepedTotalElems; i++, offset++)
            {
                var keepedPartIdx = enumeratorKeeped.MoveNext();
                enumeratorReduced.Clear();
                var value = srcSpan[keepedPartIdx + enumeratorReduced.MoveNext()];
                for(int j = 1; j < reducedTotalElems; j++)
                {
#if NET7_0_OR_GREATER
                    value = TReduceHandler.Apply(value, TReduceHandler.GetValue(srcSpan[keepedPartIdx + enumeratorReduced.MoveNext()]));
#else
                    value = _reduceHandler.Apply(value, _reduceHandler.GetValue(srcSpan[keepedPartIdx + enumeratorReduced.MoveNext()]));
#endif
                }
#if NET7_0_OR_GREATER
                dstSpan[offset] = TReduceHandler.SetValue(value, size);
#else
                dstSpan[offset] = _reduceHandler.SetValue(value, size);
#endif
            }
            return res;
        }

        private static T ConvertIntToT(int size)
        {
            if(typeof(T) == typeof(int))
            {
                return (T)(object)size;
            }
            else if (typeof(T) == typeof(float))
            {
                return (T)(object)(float)size;
            }
            else if (typeof(T) == typeof(double))
            {
                return (T)(object)(double)size;
            }
            else
            {
                throw new NotImplementedException();
            }
        }
    }
}
