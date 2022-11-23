using NN.Native.Abstraction.Common;
using NN.Native.Abstraction.Data;
using NN.Native.Basic;
using NN.Native.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.Operators
{
    public interface ITransposeOperator<T>: IBinaryOperator where T: unmanaged
    {
#if NET7_0_OR_GREATER
        static abstract
#endif
        NativeArray<T> Exec(in NativeArray<T> src, in TransposeParam param, INativeMemoryManager? memoryManager = null);

        public static NativeLayout DeduceLayout(in NativeLayout src, in TransposeParam param)
        {
            int[] shape = new int[src._ndim];
            src.Shape.AsSpan(0, src._ndim).CopyTo(shape);
            var temp = shape[param.DimA];
            shape[param.DimA] = shape[param.DimB];
            shape[param.DimB] = temp;
            return new NativeLayout(shape);
        }
    }
}
