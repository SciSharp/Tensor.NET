
using NN.Native.Abstraction.Common;
using NN.Native.Abstraction.Data;
using NN.Native.Basic;
using NN.Native.Data;

namespace NN.Native.Abstraction.Operators
{
    public interface ITypeConvertOperator<TA, TB>: IBinaryOperator where TA: unmanaged where TB : unmanaged
    {
#if NET7_0_OR_GREATER
        static abstract
#endif
        NativeArray<TB> Exec(in NativeArray<TA> src, in TypeConvertParam param, INativeMemoryManager? memoryManager = null);

        public static NativeLayout DeduceLayout(in NativeLayout src, in TypeConvertParam param)
        {
            if (!param.Transpose)
            {
                return NativeLayout.ShapeLike(src);
            }
            int[] shape = new int[src._ndim];
            src._shape.AsSpan(0, src._ndim).CopyTo(shape);
            var temp = shape[param.DimA];
            shape[param.DimA] = shape[param.DimB];
            shape[param.DimB] = temp;
            return new NativeLayout(shape);
        }
    }
}
