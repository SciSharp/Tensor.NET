using System.Runtime.InteropServices;
using System.Buffers;
using System.Linq;

namespace Numnet.Tensor.Base{

    public class TensorMemory<T> where T :struct
    {
        private  Memory<T> _memory;
        public TensorMemory(int length){
            _memory = new T[length];
        }
        public TensorMemory(T[] data){
            _memory = data.AsMemory<T>();
        }
        public TensorMemory(Array data){
            if(data.GetType().GetElementType() != typeof(T)){

            }
            var array = new T[data.Length];
        }

        public Span<T> AsSpan(){
            return _memory.Span;
        }

        public void Pin(out MemoryHandle handle){
            handle = _memory.Pin();
        }
    }
}