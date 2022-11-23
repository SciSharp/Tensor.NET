using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NN.Native.Data;

namespace NN.Native.Abstraction.Data
{
    public interface INativeMemoryManager
    {
        private static readonly INativeMemoryManager _defaultMemoryManager = new DefaultNativeMemoryManager();
        public static INativeMemoryManager Default { get => _defaultMemoryManager; }
        Memory<T> AllocateMemory<T>(int elems) where T: unmanaged;
    }
}
