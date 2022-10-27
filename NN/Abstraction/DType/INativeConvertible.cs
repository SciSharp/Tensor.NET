using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.DType
{
    public interface INativeConvertible<TA, TB> where TA: unmanaged where TB: unmanaged
    {
        static abstract void Convert(in TA a, ref TB b);
    }
}
