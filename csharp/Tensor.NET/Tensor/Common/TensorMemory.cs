using System.Runtime.InteropServices;
using System.Buffers;
using System.Linq;
using Tensornet.Native;

namespace Tensornet.Common{
    internal interface ITensorMemory{
        void Pin(out MemoryHandle handle);
    }
    internal class TensorMemory<T> : ITensorMemory where T : struct{
        private Memory<T> _memory;
        public TensorMemory(int length){
            _memory = new T[length];
        }
        public TensorMemory(T[] data, bool reuse = false){
            if(reuse){
                _memory = data;
            }
            else{
                _memory = new T[data.Length];
                data.CopyTo(AsSpan());
            }
        }
        public TensorMemory(Span<T> data){
            data.CopyTo(AsSpan());
        }
        public Span<T> AsSpan(){
            return _memory.Span;
        }
        public Span<Byte> AsByteSpan(){
            return MemoryMarshal.Cast<T, byte>(_memory.Span);
        }
        public void Pin(out MemoryHandle handle){
            handle = _memory.Pin();
        }
    }
}