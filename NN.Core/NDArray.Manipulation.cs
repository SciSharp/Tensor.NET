using NN.Core.Manipulation;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace NN.Core
{
    public partial class NDArray<T>
    {
        public virtual NDArray<T> Transpose(int axisA, int axisB)
        {
#if NET7_0_OR_GREATER
            return Transpose<T>.Exec(this, new Native.Abstraction.Common.TransposeParam(axisA, axisB));
#else
            throw new NotImplementedException();
#endif
        }
#if NET7_0_OR_GREATER
        public virtual NDArray<TTo> TypeConvert<TTo>(bool transpose = false, int axisA = -1, int axisB = -1) where TTo: unmanaged, INumber<TTo>
#else
        public virtual NDArray<TTo> TypeConvert<TTo>(bool transpose = false, int axisA = -1, int axisB = -1) where TTo: unmanaged
#endif
        {
            throw new NotImplementedException();
        }
    }
}
