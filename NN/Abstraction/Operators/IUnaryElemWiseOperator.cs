using NN.Native.Abstraction.Data;
using NN.Native.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.Operators
{
    public interface IUnaryElemWiseOperator<T>: IUnaryOperator where T: unmanaged
    {
#if NET7_0_OR_GREATER
        static abstract
#endif
        void Exec(ref NativeArray<T> src, Func<T, int, T> elemFunc);
    }
}
