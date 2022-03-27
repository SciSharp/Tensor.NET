using System.Runtime.InteropServices;
using System.Buffers;
using System.Linq;

namespace Numnet.Base{

    public class TensorMemory<T> where T :struct
    {
        private  Memory<T> _memory;
        public TensorMemory(ulong length){
            _memory = new T[length];
        }
        public TensorMemory(T[] data){
            _memory = data.AsMemory<T>();
        }

        public Span<T> AsSpan(){
            return _memory.Span;
        }

        public void Pin(out MemoryHandle handle){
            handle = _memory.Pin();
        }
    }
}