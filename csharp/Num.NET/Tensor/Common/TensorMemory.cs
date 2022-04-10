using System.Runtime.InteropServices;
using System.Buffers;
using System.Linq;
using Numnet.Native;

namespace Numnet.Common{
    internal class TensorMemory{
        private Memory<byte> _memory;
        public TensorMemory(int length, DType dtype){
            var size = TensorTypeInfo.GetTypeSize(dtype);
            _memory = new byte[length * size];
        }
        public Span<T> AsSpan<T>() where T: struct{
            return MemoryMarshal.Cast<byte, T>(_memory.Span);
        }
        internal Span<byte> AsSpan(){
            return _memory.Span;
        }
        public void Pin(out MemoryHandle handle){
            handle = _memory.Pin();
        }
    }
    internal class TensorMemory<T>:TensorMemory where T :struct
    {
        public TensorMemory(T[] data):base(data.Length, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype){
            data.CopyTo(AsSpan<T>());
        }
        public TensorMemory(Span<T> data):base(data.Length, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype){
            data.CopyTo(AsSpan<T>());
        }
    }
}