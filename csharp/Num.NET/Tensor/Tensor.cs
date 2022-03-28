using System.Buffers;
using Numnet.Tensor.Base;

namespace Numnet.Tensor{
    public sealed partial class Tensor<T>: TensorBase, ITensor<T> where T:struct{
        public TensorMemory<T> TMemory{ get; private set; }
        protected override void Pin(out MemoryHandle handle){
            TMemory.Pin(out handle);
        }

        internal Tensor(int[] shape){
            var dtypeInfo = TensorTypeInfo.GetTypeInfo(typeof(T));
            TLayout = new TensorLayout(dtypeInfo._dtype, shape);
            int length = 1;
            foreach(var s in shape){
                length *= s;
            }
            TMemory = new TensorMemory<T>(length);
        }

        public Tensor(IEnumerable<T> data, Span<int> shape){
            var dtypeInfo = TensorTypeInfo.GetTypeInfo(typeof(T));
            TLayout = new TensorLayout(dtypeInfo._dtype, shape);
            TMemory = new TensorMemory<T>(data.ToArray());
        }

        public Tensor(T[] data, Span<int> shape){
            var dtypeInfo = TensorTypeInfo.GetTypeInfo(typeof(T));
            TLayout = new TensorLayout(dtypeInfo._dtype, shape);
            TMemory = new TensorMemory<T>(data);
        }

        public Tensor(Array data){
            var ndim = data.Rank;
            int[] shape = new int[ndim];
            for (int i = 0; i < ndim; i++){
                shape[i] = data.GetLength(i);
            }
            var dtypeInfo = TensorTypeInfo.GetTypeInfo(data.GetType().GetElementType());
            TLayout = new TensorLayout(dtypeInfo._dtype, shape);
            TMemory = new TensorMemory<T>(data);
        }

        public Span<T> AsSpan(){
            return TMemory.AsSpan();
        }
    }
}