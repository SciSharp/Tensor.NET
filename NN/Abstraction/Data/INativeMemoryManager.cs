using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.Data
{
    public interface INativeMemoryManager
    {
        Memory<T> AllocateMemory<T>(int elems) where T: unmanaged;
    }
}
