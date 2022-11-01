using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.DType
{
    public interface INativeConvertible<TA, TB> where TA: unmanaged where TB: unmanaged
    {
#if NET7_0_OR_GREATER
        static abstract TB Convert(TA a);
#else
        TB Convert(TA a);
#endif
    }
}
