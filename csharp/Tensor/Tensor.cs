using Numnet.Base;

namespace Numnet{
    public class Tensor<T>: TensorBase{
        public Tensor(Span<ulong> shape){
            var dtypeInfo = TensorTypeInfo.GetTypeInfo(typeof(T));
            _layout = new TensorLayout(dtypeInfo._dtype, shape);
            _dataHandle = new TensorMemory(_layout.total_elems(), dtypeInfo._size);
        }
    }
}