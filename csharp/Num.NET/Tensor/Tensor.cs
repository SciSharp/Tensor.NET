using System.Buffers;
using Numnet.Tensor.Base;
using System.Text;

namespace Numnet.Tensor{
    public sealed partial class Tensor<T>: TensorBase, ITensor<T> where T:struct{
        public TensorMemory<T> TMemory{ get; private set; }
        protected override void Pin(out MemoryHandle handle){
            TMemory.Pin(out handle);
        }

        public Tensor(IEnumerable<T> data, Span<int> shape):base(new TensorLayout(TensorTypeInfo.GetTypeInfo(typeof(T))._dtype, shape)){
            TMemory = new TensorMemory<T>(data.ToArray());
        }

        public Tensor(T[] data, Span<int> shape):base(new TensorLayout(TensorTypeInfo.GetTypeInfo(typeof(T))._dtype, shape)){
            TMemory = new TensorMemory<T>(data);
        }

        public Tensor(Array data):base(new TensorLayout()){
            var ndim = data.Rank;
            int[] shape = new int[ndim];
            for (int i = 0; i < ndim; i++){
                shape[i] = data.GetLength(i);
            }
            var dtypeInfo = TensorTypeInfo.GetTypeInfo(data.GetType().GetElementType()!);
            TLayout.NDim = ndim;
            TLayout.DType = dtypeInfo._dtype;
            TLayout.Shape = shape;
            TLayout.InitContiguousLayout();
            TMemory = new TensorMemory<T>(data);
        }

        internal Tensor(int[] shape):base(new TensorLayout(TensorTypeInfo.GetTypeInfo(typeof(T))._dtype, shape)){
            int length = 1;
            foreach(var s in shape){
                length *= s;
            }
            TMemory = new TensorMemory<T>(length);
        }

        internal Tensor(TensorMemory<T> memory, TensorLayout layout):base(layout){
            if(TensorTypeInfo.GetTypeInfo(typeof(T))._dtype != layout.DType){
                throw new NotImplementedException();
            }
            TMemory = memory;
        }

        public Span<T> AsSpan(){
            return TMemory.AsSpan();
        }

        public override string ToString()
        {
            Func<int, int> getRealPos = idx => {
                int res = 0;
                for (int i =  TLayout.NDim - 1; i >= 0; i--) {
                    int mod = TLayout.Stride[i];
                    if (mod <= 0)
                        mod = TLayout.Shape[i] * (i > 0 ? TLayout.Stride[i - 1] : 1);
                    else
                        res += idx / mod * mod;
                    idx %= mod;
                    if (idx <= 0) break;
                }
                return res;
            };

            StringBuilder r = new StringBuilder($"Tensor({TLayout.GetInfoString()}):\n");
            var data = TMemory.AsSpan();
            for (int i = 0; i < TLayout.TotalElemCount(); i++) {
                int mod = 1;
                for (int j = 0; j < TLayout.NDim; j++) {
                    mod *= TLayout.Shape[j];
                    if (i % mod == 0) {
                        r.Append("[");
                    } else {
                        break;
                    }
                }
                r.Append(" ").Append(data[getRealPos(i)]);

                if ((i + 1) % TLayout.Shape[0] != 0) r.Append(",");

                r.Append(" ");
                mod = 1;
                int hit_times = 0;
                for (int j = 0; j < TLayout.NDim; j++) {
                    mod *= TLayout.Shape[j];
                    if ((i + 1) % mod == 0) {
                        r.Append("]");
                        hit_times++;
                    } else {
                        break;
                    }
                }
                if (hit_times > 0 && hit_times < TLayout.NDim) {
                    r.Append(",\n");
                    for (int j = 0; j < TLayout.NDim - hit_times; j++) {
                        r.Append(" ");
                    }
                }
            }
            // r.Append("\n");
            return r.ToString();
        }
    }
}