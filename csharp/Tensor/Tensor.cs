using System.Buffers;
using Numnet.Base;
using Numnet.Common;

namespace Numnet{
    public sealed partial class Tensor<T>: TensorBase, ITensor<T> where T:struct{
        public TensorMemory<T> TMemory{ get; private set; }
        protected override void Pin(out MemoryHandle handle){
            TMemory.Pin(out handle);
        }
        public Tensor(Span<ulong> shape){
            var dtypeInfo = TensorTypeInfo.GetTypeInfo(typeof(T));
            TLayout = new TensorLayout(dtypeInfo._dtype, shape);
            TMemory = new TensorMemory<T>(TLayout.total_elems());
        }

        public Tensor(IEnumerable<T> data, Span<ulong> shape){
            var dtypeInfo = TensorTypeInfo.GetTypeInfo(typeof(T));
            TLayout = new TensorLayout(dtypeInfo._dtype, shape);
            TMemory = new TensorMemory<T>(data.ToArray());
        }

        public Tensor(T[] data, Span<ulong> shape){
            var dtypeInfo = TensorTypeInfo.GetTypeInfo(typeof(T));
            TLayout = new TensorLayout(dtypeInfo._dtype, shape);
            TMemory = new TensorMemory<T>(data);
        }

        public Span<T> AsSpan(){
            return TMemory.AsSpan();
        }
    }
}