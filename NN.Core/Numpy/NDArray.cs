using NN.Core.Exceptions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text; 
using System.Threading.Tasks;

namespace NN.Core.Numpy
{
    public abstract class NDArray
    {
        public abstract ReadOnlySpan<int> Shape { get; }
        public abstract int Ndim { get; }
        public abstract Type DType { get; }

        public abstract object GetAt(params int[] indices);

        public T At<T>(params int[] indices)
        {
            return (T)GetAt(indices);
        }
#if NET7_0_OR_GREATER
        public NDArray<T> As<T>() where T : unmanaged, INumber<T>
#else
        public NDArray<T> As<T>() where T: unmanaged
#endif
        {
            if(DType == typeof(T))
            {
                return (NDArray<T>)(object)this;
            }
            else
            {
                throw new MismatchedTypeException();
            }
        }
    }
}
