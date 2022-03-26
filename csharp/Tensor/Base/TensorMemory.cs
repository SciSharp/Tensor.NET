using System.Runtime.InteropServices;
using System.Buffers;

namespace Numnet.Base{
    internal class TensorMemory{
        private Memory<byte> _memory;
        public TensorMemory(ulong length, int size){
            _memory = new byte[length * (ulong)size];
        }

        public Span<T> AsSpan<T>() where T:struct{
            return MemoryMarshal.Cast<byte, T>(_memory.Span);
        }

        public void Pin(out MemoryHandle handle){
            handle = _memory.Pin();
        }
    }
}