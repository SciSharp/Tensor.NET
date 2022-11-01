using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.DType
{
    public interface INativeDTypeHandler<T> :
        IBinaryDTypeHandler<T>, IUnaryDTypeHandler<T>, ITernaryDTypeHandler<T> where T : unmanaged
    {

    }
}
