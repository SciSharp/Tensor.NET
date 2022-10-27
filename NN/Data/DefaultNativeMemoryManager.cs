using NN.Native.Abstraction.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Buffers;

namespace NN.Native.Data
{
    public class DefaultNativeMemoryManager: INativeMemoryManager
    {
        public Memory<T> AllocateMemory<T>(int elems) where T: unmanaged
        {
            return new(new T[elems * Marshal.SizeOf<T>()]);
        }
    }
}
